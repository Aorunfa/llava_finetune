import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/dev/shm/chaofeng/llava-v1.6-mistral-7b")
    version: Optional[str] = field(default="mistral_instruct")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14')
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default='/dev/shm/chaofeng/LLaVA-CC3M-Pretrain-595K/chat.json',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/dev/shm/chaofeng/LLaVA-CC3M-Pretrain-595K/dataset')
    image_aspect_ratio: str = 'square'

# transformers.TrainingArguments
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
  
    double_quant: bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # modified
    output_dir: str = field(default='/home/chaofeng/llava_fitune/train/ckpt')
    deepspeed: str = '/home/chaofeng/LLaVA/scripts/zero2.json'
    lora_enable: bool = True
    
    
    bf16: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    evaluation_strategy: str =  "no"
    save_strategy: str = "steps"
    save_steps: str = 50000
    save_total_limit: str = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str =  "cosine"  #####
    #lr_scheduler_kwargs: dict = {}
    logging_steps: int = 1
    tf32: bool = True
    model_max_length: int =  2048
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    

if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print(local_rank)

    from llava_model.llm.llava_mistral import LlavaMistralForCausalLM
    from llava_model.utils import conversation as conversation_lib
    # 模型加载
    attn_implementation = None
    model = LlavaMistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
                )
    model.config.use_cache = False
    # 打开输入梯度，保留projecter梯度信息
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
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # load clip encoder
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # 只训练mm mlp projecter
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    
    # 冻结mlp preojector
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # build trian dataset, collect_fun, val dataset
    from train.dataset import make_supervised_data_module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    from torch.utils.data import DataLoader
    train_loader = DataLoader(data_module['train_dataset'], 
                              num_workers=training_args.dataloader_num_workers, 
                              batch_size=training_args.per_device_train_batch_size, 
                              collate_fn=data_module['data_collator'])
    
    training_args.num_training_steps = training_args.num_train_epochs * len(train_loader)

    def prepare_inputs(inputs: Dict, model):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        inputs_new = {}
        for k, v in inputs.items():
            if v.dtype != torch.float:
                inputs_new[k] = v.to(device)
            else:
                inputs_new[k] = v.to(device).to(dtype)
        return inputs_new


    model.cuda()
    # model.train()
    # for batch in train_loader:  
    #     print(batch.keys())     ## ['input_ids', 'labels', 'attention_mask', 'images']
    #     #label = batch.pop('label')
    #     batch = prepare_inputs(batch, model)
    #     output = model(**batch, return_dict=True)
    #     print(output)
    #     break

    
    

    def build_optimizer_scheduler(training_args, opt_model):
        from transformers import Trainer
        from transformers.trainer import (
                        get_parameter_names,
                        ALL_LAYERNORM_LAYERS
                        )
        from transformers.optimization import get_scheduler
        import math
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if training_args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                    "lr": training_args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": training_args.mm_projector_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        num_warmup_steps = math.ceil(training_args.num_training_steps * training_args.warmup_ratio)

        lr_scheduler = get_scheduler(training_args.lr_scheduler_type,
                                     optimizer=optimizer,
                                    num_warmup_steps=num_warmup_steps,
                                    num_training_steps=training_args.num_training_steps,
                                    #scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )
        return optimizer, lr_scheduler



    def print_loss(step, loss_item, current_lr):
        print("step: {:0>8d}{:>8s} loss: {:.4f} lr: {:.8f}".format(step, '', loss_item, current_lr))


    from torch.amp.autocast_mode import autocast
    from torch.cuda.amp import GradScaler


    import pandas as pd
    def save_metric(data:dict, save_csv):
        with open(save_csv, 'a') as file:
            if isinstance(data, dict):
                data = pd.DataFrame(data, index=[0])
            data.to_csv(file,
                        sep='\t',
                        index=False,
                        header=not os.stat(save_csv).st_size > 0)
        
    def train(model, train_loader, training_args:TrainingArguments):
        model.train()
        device = next(model.parameters()).device
        optimizer, lr_scheduler = build_optimizer_scheduler(training_args, model)
        scaler = GradScaler(enabled=training_args.bf16)
        for epoch in range(training_args.num_train_epochs):
            for step, batch in enumerate(train_loader):
                print('step', step)
                step += 1 + epoch * len(train_loader)
                with autocast('cuda', enabled=training_args.bf16, dtype=torch.bfloat16):
                    batch = prepare_inputs(batch, model)
                    loss = model(**batch, return_dict=True)['loss']
                    loss_item = loss.data.item()

                loss.backward()
                if step % training_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr']
                    print_loss(step, loss_item, current_lr)
                    
                    # save csv
                    data = {'step': step, 'loss_item': loss_item, 'current_lr': current_lr}
                    save_metric(data, '/home/chaofeng/llava_finetune/doc/train_loss.csv')

                lr_scheduler.step()
    


                # if training_args.bf16:
                #     scaler.scale(loss).backward()
                # else:
                #     loss.backward()
                
                # if step % training_args.gradient_accumulation_steps == 0:
                #     if training_args.bf16:
                #         scaler.unscale_(optimizer)
                #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                #         scaler.step(optimizer)
                #         scaler.update()
                #     else:
                #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                #         optimizer.step()
                #     optimizer.zero_grad()

                #     current_lr = optimizer.param_groups[0]['lr']
                #     print_loss(step, loss_item, current_lr)
                
                # lr_scheduler.step()
    
    train(model, train_loader, training_args)
    """
    nohup /var/lib/anaconda3/envs/llava/bin/python /home/chaofeng/llava_finetune/train_torch.py > /home/chaofeng/llava_finetune/doc/log.log 2>&1 &
    """
            
    """
    只训练mm adapter还是不行，损失下降不明显
    
    """

    """
    模型加载
    数据加载
    损失函数: llm自带crocess entropy损失计算


    训练:
        使用transformer自带训练基类进行训练， 可以使用deepspeed
        自己撕一版torch原生的训练方式
    
    怎么以通过transform的方式存储checkpoint?
    """





