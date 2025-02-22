import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from peft import get_peft_model, LoraConfig
from torch.amp.autocast_mode import autocast
from train.utils import (
                    build_dataloader, 
                    build_optimizer_scheduler, 
                    prepare_inputs, 
                    print_loss, 
                    save_metric)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/local/dev1/chaofeng/llava-v1.5-7b")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default='/local/dev1/chaofeng/clip-vit-large-patch14-336')   # clip 图片编码器
    mm_vision_select_layer: Optional[int] = field(default=-2)                                        # 选择clip第几层特征进行返回，最佳实验原则
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)                   
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')                                    # mm adapter的类型，这里选择多层
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default='/local/dev1/chaofeng/LLaVA-CC3M-Pretrain-595K/chat.json',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/local/dev1/chaofeng/LLaVA-CC3M-Pretrain-595K/images')
    image_aspect_ratio: str = 'pad'                                                    # 图片处理方式, anyres划分四个象限和中间 一张图会得到五张图, anyres v1.6改进

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    
    # lora config
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=True)

    # modified
    output_dir: str = field(default='/home/chaofeng/workhome/chaofeng/llava_finetune/result')
    metric_csv: str = '/home/chaofeng/workhome/chaofeng/llava_finetune/result/train_loss.csv'
    
    fp16: bool = False
    bf16: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 128  # per_device_train_batch_size * gradient_accumulation_steps should be 128
    evaluation_strategy: str =  "no"
    save_strategy: str = "steps"
    save_steps: str = 50000
    save_total_limit: str = 1
    learning_rate: float = 2e-6
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str =  "cosine"  
    logging_steps: int = 1
    tf32: bool = True
    model_max_length: int =  2048
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

def train(model, train_loader, training_args:TrainingArguments):
    model.train()
    optimizer, lr_scheduler = build_optimizer_scheduler(training_args, model)
    loss_item_avg = 0
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            step += 1 + epoch * len(train_loader)            
            with autocast('cuda', enabled=training_args.bf16, dtype=torch.bfloat16):
                batch = prepare_inputs(batch, model)
                loss = model(**batch, return_dict=True)['loss']
                loss_item = loss.data.item()
                loss_item_avg += loss_item

            loss.backward()
            if step % training_args.gradient_accumulation_steps == 0:
                loss_item_avg = loss_item_avg / training_args.gradient_accumulation_steps                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()

                current_lr = optimizer.param_groups[0]['lr']
                print_loss(step, loss_item_avg, current_lr)
                
                # save csv
                data = {'step': step, 'loss_item': loss_item_avg, 'current_lr': current_lr}
                save_metric(data, training_args.metric_csv)
                loss_item_avg = 0

            lr_scheduler.step()
            save_peft_lora_model(model, training_args)
            return 

    save_peft_lora_model(model, training_args)

def save_peft_lora_model(model, training_args):
    from train.utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
    model.config.use_cache = True
    # get lora
    state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                    )
    # get no lora, e.g mm_proj_adapter
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                    )
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))


def load_peft_lora__model(model_path,         # lora or pretrain dir
                          model_base,         # pretrained dir if set lora model in model_path
                          model_merge_path=None,
                          load_8bit=False, 
                          load_4bit=False, 
                          device_map="auto", # 多gpu时指定模型如何在设备上进行分片
                          device="cuda", 
                          use_flash_attn=False, 
                          **kwargs):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    from llava.utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.llm import LlavaConfig, LlavaLlamaForCausalLM

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    
    #from llm_llama import LlavaConfig
    lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    print('Loading LLaVA from base model...')
    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs) # lora_cfg_pretrained should same as base
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    #adjust ouputhead an embeddig matrix
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    none_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
    if os.path.exists(none_lora_path):
        non_lora_trainables = torch.load(none_lora_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

    # peft load lora state dict
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')


    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    if model_merge_path is not None:
        model.save_pretrained(model_merge_path)
        tokenizer.save_pretrained(model_merge_path)
        return


    return tokenizer, model, image_processor, context_len

if __name__ == '__main__':
    """
    使用lora微调llava-v1.5-7b
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    from llava.llm.llava_llama import LlavaLlamaForCausalLM
    
    # model 
    attn_implementation = None # or 'flash_attention_2'
    model = LlavaLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype
                )
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )    
    tokenizer.pad_token = tokenizer.unk_token
    
    # clip encoder
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length


    # train lora
    if training_args.lora_enable:
        from train.utils import find_all_linear_names
        target_modules = find_all_linear_names(model)    
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=target_modules
        )

        model = get_peft_model(model, peft_config)
    

    # train mm_mlp_adapter
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        #model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    
    model.print_trainable_parameters()
    model = model.cuda()

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_args.model_version = model_args.version
    train_loader, _ = build_dataloader(data_args, training_args, tokenizer)
    training_args.num_training_steps = training_args.num_train_epochs * len(train_loader)
    
    train(model, train_loader, training_args)


    # merge
    load_peft_lora__model(model_path='/home/chaofeng/workhome/chaofeng/llava_finetune/result',
                          model_base='/local/dev1/chaofeng/llava-v1.5-7b',
                          model_merge_path='/home/chaofeng/workhome/chaofeng/llava_finetune/lora_ckpt')

    
    
    """
    踩坑:
        autocast自动只适用混合精度训练，适用fp32的模型，forward使用fp16，backward使用fp32
        如果一个模型一开始就是float16，则不适用，因为前向传播本身就是半精度的
        模型很大时，直接model.float()会OM

        batchsize很小时候，gradient_accumulation应设置大一些，减小梯度更新的随机性，随机性大很难收敛

    """

    """
    
     https://github.com/Dao-AILab/flash-attention/releases/download/v2.1.1/flash_attn-2.1.1+cu124torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    """