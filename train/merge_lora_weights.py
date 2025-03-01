import argparse
from llava.builder_llm import load_pretrained_model
from llava.utils.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/home/chaofeng/llava_finetune/ckpt') # lora path
    parser.add_argument("--model-base", type=str, default="/dev/shm/chaofeng/llava-v1.6-mistral-7b") # base path
    parser.add_argument("--save-model-path", type=str, default='/home/chaofeng/llava_finetune/llava-v1.6-mistral-7b-lora')

    args = parser.parse_args()

    merge_lora(args)
