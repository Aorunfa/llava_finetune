import torch
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re
import copy
from llava.utils.constants import(
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.utils.conversation import conv_templates
from llava.builder_llm import load_pretrained_model
from llava.utils.utils import disable_torch_init
from llava.utils.mm_utils import (
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
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # #### mistrial

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

    #get input ids
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # model.config.image_aspect_ratio = 'pad'
    
    data = read_input(images_path, image_processor, model.config)
    images_tensor = data['images_tensor'].to(model.device, dtype=torch.float16)
    image_sizes = data['image_sizes']
    image_path = data['image_path']
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
    log('end -- %s' % image_path)
    return outputs
    



if __name__ == "__main__":
    import argparse
    ckpt = '/local/dev1/chaofeng/llava-v1.5-7b'
    qurey = "What is in this image?, the answer should starts with 'The image shows' and should less than 15 words"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=ckpt)
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, default=image_file)
    parser.add_argument("--query", type=str, default=qurey)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()


    # 单卡推理    
    image_path = 'llava_finetune/doc/test.png'
    device = 'cuda:0'
    res = infer(args, image_path, device)
    print(res)