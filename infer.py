
# from llava_model.builder_llm import load_pretrained_model
# model_path = "/dev/shm/chaofeng/llava-v1.6-mistral-7b"
# model_base = "/home/chaofeng/llava_finetune/ckpt"
# model_name = model_path.split("/")[-1]
# # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
# #                                                                        model_base=None, #model_base, 
# #                                                                        model_name=model_name)

# # 简单处理加载方式
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from peft import PeftModel
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
# print(f"Loading LoRA weights from {model_base}")
# model = PeftModel.from_pretrained(model, model_base)
# print(f"Merging weights")
# model = model.merge_and_unload()
# print('Convert to FP16...')
# model.to(torch.float16)

################
import argparse
import torch
import os
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

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from llava_model.utils.constants import(
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava_model.utils.conversation import conv_templates, SeparatorStyle
from llava_model.builder_llm import load_pretrained_model
from llava_model.utils.utils import disable_torch_init
from llava_model.utils.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


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

def log(*args):
    for i in args:
        print(i)


def read_input(image_path, image_processor, model_cfg):
    images = load_images([image_path])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
                images,
                image_processor,
                model_cfg
            )    

    data = {'images_tensor': images_tensor, 'image_path': image_path, 'image_sizes': image_sizes}
    return data

def infer(args, images_path, device):
    # Model
    disable_torch_init()
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

    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # import torch
    # from peft import PeftModel
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    # model_lora = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
    # model_base = '/home/chaofeng/llava_finetune/ckpt'
    # print(f"Loading LoRA weights from {model_base}")
    # model_lora = PeftModel.from_pretrained(model_lora, model_base)
    # print(f"Merging weights")
    # model_lora = model_lora.merge_and_unload()
    # print('Convert to FP16...')
    # model_lora.to('cuda:1')
    # model_lora.to(torch.float16)



    data = read_input(images_path, image_processor, model.config)
    # images_tensor = data['images_tensor'].to(model.device, dtype=torch.float16)
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
    return outputs
    



if __name__ == "__main__":
    ##### 预训练模型单卡推理 ###########
    ckpt = '/dev/shm/chaofeng/llava-v1.6-mistral-7b'
    ckpt = '/home/chaofeng/llava_finetune/llava-v1.6-mistral-7b-new'
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
    image_path = '/home/chaofeng/LLaVA/images/llava_v1_5_radar.jpg'
    device = 'cuda:0'
    log_path = '/home/chaofeng/LLaVA/self/test/l.log'
    res = infer(args, image_path, device)
    print(res)



    # #### 加载lora模型
    # from llava_model.builder_llm import load_pretrained_model
    # model_path = "/home/chaofeng/llava_finetune/lora-checkpoint"
    # model_base = "/dev/shm/chaofeng/llava-v1.6-mistral-7b"
    # model_name = 'lora_' + 'llava-v1.6-mistral-7b' 
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
    #                                                                        model_base=model_base, 
    #                                                                        model_name=model_name)



    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # import torch
    # from peft import PeftModel
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    # model_lora = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
    # model_base = '/home/chaofeng/llava_finetune/ckpt'
    # print(f"Loading LoRA weights from {model_base}")
    # model_lora = PeftModel.from_pretrained(model_lora, model_base)
    # print(f"Merging weights")
    # model_lora = model_lora.merge_and_unload()
    # print('Convert to FP16...')
    # model_lora.to('cuda:1')
    # model_lora.to(torch.float16)






