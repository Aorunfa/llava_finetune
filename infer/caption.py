import os
import sys
__dir__ = '/'.join(os.path.dirname(__file__).split('/')[:-1])
sys.path.append(__dir__)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import torch
import cv2
from PIL import Image
import pandas as pd
from logger import Logger

"""
pipeline:
预训练模型加载
数据集加载dataloader
推理：输入处理 tokenizer 初始化推理参数 output处理 返回数据
"""

class CaptiopnLava(object):
    """
    使用LLava对图片进行标注
    对给定的dataset进行扫描标注
    存储标注结果
    """
    log_path = r'./cap_log.log'

    def __init__(self, model_path, model_base=None, load_8bit=False, load_4bit=False, 
                 device='cuda', conv_mode=None, temperature=0.2, max_new_tokens=512, top_p=None):
        # 初始化模型
        self._init_model(model_path=model_path, model_base=model_base,load_8bit=load_8bit, load_4bit=load_4bit, 
                         device=device, conv_mode=conv_mode)

        # 初始化图像提示
        inp = "describe this image briefly. The answer should start with: The image caption is: "
        if self.model.config.mm_use_im_start_end:
            self.inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            self.inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
    
    def _init_model(self, **args):
        disable_torch_init()
        self.model_name = get_model_name_from_path(args['model_path'])
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                                                                                 args['model_path'], args['model_base'],
                                                                                 self.model_name, args['load_8bit'], 
                                                                                 args['load_4bit'],  args['device'])          
        self.conv_mode = args['conv_mode'] # 对话模式初始化
        if not self.conv_mode:
            if 'llama-2' in self.model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"
        return

    def _setup_log(self):
        self.log = Logger(self.log_path, mode='a')


    def _data_process_batch(self, batch):
        # batch 数据预处理
        images = batch['pixel_array'].clone().detach().cpu()
        images = [Image.fromarray(torch.clamp(image, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy()) 
                  for image in images]

        image_tensor = process_images(images, self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        return image_tensor
    
    @staticmethod
    def _process_output(outputs):
        # 推理结果处理
        mid_frames= []
        for output in outputs:
            mid_frame = output.replace('The image shows ', f'')
            mid_frame = mid_frame.replace('The image shows that ', f'')
            mid_frame = mid_frame.replace('The image caption is: ', f'')
            mid_frame = mid_frame.replace('</s>', '')
            mid_frame = mid_frame.replace('\n', '')
            print('mid_frame: ', mid_frame)
            if mid_frame.startswith(' '):
                mid_frame = mid_frame[1:]
            mid_frames.append(mid_frame)
        return mid_frames

    def _save_caption_csv(self, caption_batch, csv_result_path):
        # 存储标注信息
        if caption_batch:
            df = pd.DataFrame(data={'videoid': caption_batch[0], 'caption_llava': caption_batch[1]})
            df.to_csv(csv_result_path, 
                      index=False, 
                      mode='a', 
                      header=not pd.io.common.file_exists(csv_result_path))

    @torch.inference_mode()
    def detect_by_batch(self, batch):
        # 检测一个batch, 返回标注结果 
        # check TODO need log record
        broken = False
        for link in batch['link']:
            cap = cv2.VideoCapture(link)
            fps = cap.get(5)
            if fps == 0:
                print('broken video: ', link)
                broken = True
                continue
        if broken:
            return [batch['videoid'], [''] * len(batch['videoid'])]
    
        image_tensor = self._data_process_batch(batch)
        batchsize = len(batch['link']) 

        # 建立对话
        conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        conv.append_message(roles[0], self.inp)
        conv.append_message(roles[1], None)
        prompt = [conv.get_prompt() for i in range(batchsize)]
        input_ids = torch.cat(
                    [tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda() for p in prompt], 0
                )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        num_beams = 1

        output_ids = self.model.generate(input_ids,
                                         images=image_tensor,
                                         do_sample=True if self.temperature > 0 else False,
                                         temperature=self.temperature,
                                         top_p=self.top_p,
                                         num_beams=num_beams,
                                         max_new_tokens=self.max_new_tokens,
                                         use_cache=True,
                                         stopping_criteria=[stopping_criteria],
                                        )
        # 输出后处理
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], 
                                              skip_special_tokens=True
                                              )
        caption_batch = self._process_output(outputs)
        return [batch['videoid'], caption_batch]
    
    def __call__(self, dataloader, csv_result_path, lock=None):
        # dataloader from Dataloader(Dataset...)
        # lock is torch.multiprocessing.Lock() which is used for multi gups inference
        self._setup_log()
        self.log(f'caption batch total {len(dataloader)}')

        for batch in dataloader:
            bid = batch['videoid']
            self.log(rf'{bid} star')
            try:
                caption_batch = self.detect_by_batch(batch)
                if lock:
                    with lock:
                        self._save_caption_csv(caption_batch, csv_result_path)
                else:
                    self._save_caption_csv(caption_batch, csv_result_path)
                self.log(rf'{bid} sucess')
            except KeyboardInterrupt:
                print('主动终止程序')
                self._clear()
            except Exception as e:
                print(e)
                caption_batch = None
                self.log(f'{bid} failed')
        self._clear()
        return 

    
    def _clear(self):
        self.log('清理程序---')
        for attr in list(self.__dict__.keys()):
            if attr != 'log':
                delattr(self, attr)
        self.log('清理结束---')


if __name__ == '__main__':
    # artlist 数据测试
    from dataset_base import get_dataloader
    num_gpus = 1
    model_path='/highres_new/high_res/project/data_process/0_code_caption/llava-v1.5-7b'

    s3_local_dir_replace = ('s3://morph-dataself-tag/download_artlist_1080p/artlist_20', '/highres_new/high_res/project/data_process/0_code_caption/0_data')
    video_col_indx = ('videoid', 'link')
    csv_path='/highres_new/high_res/project/data_process/0_code_caption/artlist.csv'
    csv_result_path = r'/highres_new/high_res/project/data_process/0_code_caption/artlist_caption.csv'


    batch_size = 6
    num_workers = 10

    dl = get_dataloader(batch_size=batch_size,
                   num_workers=num_workers,
                   num_gpus=num_gpus,
                   s3_local_dir_replace=s3_local_dir_replace,
                   video_col_indx=video_col_indx,
                   csv_path=csv_path,
                   csv_result_path=csv_result_path)

    model = CaptiopnLava(model_path=model_path)
    model.log_path = '/highres_new/high_res/project/data_process/0_code_caption/cap_log.log'
    model(dl, csv_result_path)


    """
    huggingface-cli download --resume-download liuhaotian/llava-v1.6-mistral-7b --local-dir /dev/shm/chaofeng/llava-v1.6-mistral-7b
    """