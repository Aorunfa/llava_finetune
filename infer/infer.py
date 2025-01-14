import argparse
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import threading
import copy
import queue
import pandas as pd
import numpy as np
from logger import Logger

def split_ls(ls, n):
    res = []
    l = int(len(ls) / n)
    for i in range(n):
        if i == n - 1:
            res.append(ls[i * l: ])
            continue
        res.append(ls[i * l: (i + 1) * l])
    return res

def append_txt(df: pd.DataFrame, txt_path):
    with open(txt_path, 'a') as file:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_csv(file,
                sep='\t',
                index=False,
                header=not os.stat(txt_path).st_size > 0)

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def thread_reading(images_path, image_processor, model_cfg, queue: queue.Queue):
    for image_path in images_path:
        try:
            images = load_images([image_path])
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model_cfg
            )           ### split for 5 chuncks

            data = {'images_tensor': images_tensor, 'image_path': image_path, 'image_sizes': image_sizes}
            queue.put(data)
        except:
            continue
    queue.put(None)


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    print(conv_mode)

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


def infer(args, images_path, save_txt, device, log_path):
    # Model
    disable_torch_init()
    log = Logger(log_path, mode='w')
    log('star rank; task num %d; first %s' % (len(images_path), images_path[0]))

    model_name = get_model_name_from_path(args.model_path)  # llm
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=device
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # input_ids = (
    #     tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #     .unsqueeze(0)
    #     .cuda()
    # )
    #### get input ids
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    
    ### get input in thread
    queue_img = queue.Queue(maxsize=124)
    thread = threading.Thread(target=thread_reading,
                             kwargs={'images_path': images_path,
                                     'image_processor': image_processor,
                                     'model_cfg': model.config,
                                     'queue': queue_img}
                             )
    thread.start()
    result_tol = []
    while True:
        data = queue_img.get()
        if data is None:
            break
        images_tensor = data['images_tensor'].to(model.device, dtype=torch.float16)
        image_sizes = data['image_sizes']
        image_path = data['image_path']

        #image_files = image_parser(args)
        #images = load_images(image_files)
        # image_sizes = [x.size for x in images]
        # images_tensor = process_images(
        #     images,
        #     image_processor,
        #     model.config
        # ).to(model.device, dtype=torch.float16)

        log('star -- %s' % image_path)


        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    copy.deepcopy(input_ids),
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # print(outputs)
            result_tol.append({'image_path': image_path, 'caption': outputs})
            log('sucess -- %s' % image_path)
        except Exception as e:
            log('failed -- %s' % image_path)
            log(e)

        if len(result_tol) > 10:
            append_txt(result_tol, save_txt)
            result_tol = []
    
    append_txt(result_tol, save_txt)
    thread.join()



if __name__ == "__main__":
    ckpt = '/dev/shm/chaofeng/llava-v1.6-mistral-7b'
    image_file = '/home/chaofeng/BLIP/test.png'
    image_file = '/morph-chaofeng/stock/real/ice-2799109_1280.png'
    #qurey = 'describe this picture shortly.'
    # qurey = 'caption this image'
    # qurey = 'describe main information of the image as shortly as possible'
    # qurey = 'caption the main information of the image'
    
    #qurey = 'describe main information so shortly'
    #qurey = "What is in this image?, the answer should be limitted less than 20 words"
    qurey = "What is in this image?, the answer should starts with 'The image shows' and should less than 15 words"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=ckpt)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=image_file)
    parser.add_argument("--query", type=str, default=qurey)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()


    ### 单卡单个推理
    links = ['/home/chaofeng/LLaVA/images/llava_v1_5_radar.jpg']
    save_txt = '/home/chaofeng/LLaVA/self/test/t.csv'
    device = 'cuda'
    log_path = '/home/chaofeng/LLaVA/self/test/l.log'
    res = infer(args, links, save_txt, device, log_path)
    print(res)
    ###########

    # # eval_model(args)

    # # for stock images caption
    # save_dir = '/home/chaofeng/video_stock/lava_feature2'
    # image_dir = '/morph-chaofeng/stock/real'
    # log_dir = '/home/chaofeng/video_stock/log'
    # images_path = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('png')]
    # print(len(images_path))
    # print(images_path[0])
    # done_files = os.listdir(save_dir)
    # if len(done_files) > 0:
    #     df = pd.DataFrame({'link': images_path})
    #     for f in done_files:
    #         f = os.path.join(save_dir, f)
    #         print(f)
    #         done = pd.read_csv(f, sep='\t')
    #         df = df[~df['link'].isin(done['image_path'])]
    #         print(df.shape)
    #     images_path = df['link'].to_list()
    # print(len(images_path))
    
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')

    # # shuffle
    # np.random.seed(0)
    # np.random.shuffle(images_path)
    
    # # start task
    # MAX_PROCESS = 1
    # pools = []
    # links = split_ls(images_path, MAX_PROCESS) # 8 parts
    # for rank in range(MAX_PROCESS):
    #     log_path = os.path.join(log_dir, f'log_{rank}.log')
    #     save_txt = os.path.join(save_dir, f'caption_{rank}.csv')
    #     device = 'cuda:%d' % rank
    #     pools.append(multiprocessing.Process(target=infer, 
    #                                          args=(args, 
    #                                                links[rank], 
    #                                                save_txt, 
    #                                                device,
    #                                                log_path),
    #                                   ))
    
    # for p in pools:
    #     p.start()
    
    # for p in pools:
    #     p.join()
    

    
    """
    nohup /var/lib/anaconda3/envs/llava/bin/python /home/chaofeng/LLaVA/self/infer.py > /home/chaofeng/video_stock/n_lava.log 2>&1 &
    """





