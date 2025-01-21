import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from torch.amp.autocast_mode import autocast
from llava.llm.llava_mistral import LlavaMistralForCausalLM
from train.utils import (
                    build_dataloader, 
                    build_optimizer_scheduler, 
                    prepare_inputs, 
                    print_loss, 
                    save_metric)
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/dev/shm/chaofeng/llava-v1.6-mistral-7b")
    version: Optional[str] = field(default="mistral_instruct") # llava_llama_2, llava_mistral, mistral_instruct
    
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')   # clip 图片编码器
    mm_vision_select_layer: Optional[int] = field(default=-2)                          # 选择clip第几层特征进行返回
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)                   
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')                     # mm adapter的类型，这里选择多层
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='spatial_unpad')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default='/dev/shm/chaofeng/LLaVA-CC3M-Pretrain-595K/chat.json',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/dev/shm/chaofeng/LLaVA-CC3M-Pretrain-595K/dataset')
    image_aspect_ratio: str = 'anyres'                                                     # 图片处理方式, anyres划分四个象限和中间 一张图会得到五张图, anyres for v1.6

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    
    # qlora: 4bit, nf4, double quant
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    
    # lora config
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # modified
    output_dir: str = field(default='/home/chaofeng/llava_finetune/lora-checkpoint3')
    metric_csv: str = '/home/chaofeng/llava_finetune/doc/train_loss_qlora_v16.csv'
    
    bf16: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 128 # 128
    evaluation_strategy: str =  "no"
    save_strategy: str = "steps"
    save_steps: str = 50000
    save_total_limit: str = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str =  "cosine"  #####
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

def load_peft_lora_model(lora_path, model_path):
    pass

if __name__ == '__main__':
    """
    使用8/4bit量化微调llava-v1.6-mistral-7b, 只训mm_mlp_adapter和lora线性层
    v1.6使用’image_aspect_ratio=anyres‘ 提取图片上下左右和中间的crop，得到五张图片
    """
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    ## bit config
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
                        device_map={"": training_args.device},
                        #load_in_4bit=training_args.bits == 4,
                        #load_in_8bit=training_args.bits == 8,
                        quantization_config=BitsAndBytesConfig(
                            ## 8bit config
                            # load_in_8bit=training_args.bits == 8,
                            # llm_int8_skip_modules=["mm_projector"], # no need for bit quan
                            # llm_int8_threshold=6.0,                 # find outlier weight keep for fp16 
                            # llm_int8_has_fp16_weight=False,
                            
                            ## 4bit config
                            llm_int8_skip_modules=["mm_projector"],
                            load_in_4bit=training_args.bits == 4,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_use_double_quant=training_args.double_quant,
                            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                        )
                    ))
    
    # model
    attn_implementation = None # or 'flash_attention_2'
    model = LlavaMistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype,
                **bnb_model_from_pretrained_args
                )
    model.config.use_cache = False # for training
    
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token

    # 打开embeding输入的梯度，方便使用梯度检查时，从vison encoder到llm有梯度的传入
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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

    if training_args.bits in [4, 8]:
        # set normal and ouput layer to fp32
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # lora
    if training_args.lora_enable:
        from train.utils import find_all_linear_names
        from peft import LoraConfig, get_peft_model
        target_modules = find_all_linear_names(model)    
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # for name, module in model.named_modules():
    #     try:
    #         print(name, 'dt ', next(module.parameters()).dtype)
    #         print(name, 'gd ', next(module.parameters()).requires_grad)
    #         print('----')
    #     except:
    #         pass


    # reset main module precision
    if training_args.bits in [4, 8]:     
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)

            if 'norm' in name:
                # module = module.to(torch.float32) # need more mem
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # train mm_mlp_adapter
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end     # set image in start and end
    model.config.mm_projector_lr = training_args.mm_projector_lr                                         
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_args.model_version = model_args.version
    data_args.image_grid_pinpoints = model.config.image_grid_pinpoints
    train_loader, _ = build_dataloader(data_args, training_args, tokenizer)
    training_args.num_training_steps = training_args.num_train_epochs * len(train_loader)
    train(model, train_loader, training_args)

    """
    nohup /var/lib/anaconda3/envs/llava/bin/python /home/chaofeng/llava_finetune/train_qlora.py > /home/chaofeng/llava_finetune/doc/log_v16_qlora.log 2>&1 &
    """




