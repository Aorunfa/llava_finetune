import pandas as pd
import torch
import os
from typing import Dict


def prepare_inputs(inputs: Dict, model):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs_new = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            inputs_new[k] = v
            continue
        if v.dtype != torch.float:
            inputs_new[k] = v.to(device)
        else:
            inputs_new[k] = v.to(device).to(dtype)
    return inputs_new 


def save_metric(data:dict, save_csv):
    with open(save_csv, 'a') as file:
        if isinstance(data, dict):
            data = pd.DataFrame(data, index=[0])
        data.to_csv(file,
                    #sep='\t',
                    index=False,
                    header=not os.stat(save_csv).st_size > 0)

def print_loss(step, loss_item, current_lr):
    print("step: {:0>8d}{:>8s} loss: {:.4f} lr: {:.8f}".format(step, '', loss_item, current_lr))


def build_dataloader(data_args, training_args, tokenizer):
    # build trian dataset, collect_fun, val dataset
    from train.dataset import make_supervised_data_module
    from torch.utils.data import DataLoader
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_loader = DataLoader(data_module['train_dataset'], 
                              num_workers=training_args.dataloader_num_workers, 
                              batch_size=training_args.per_device_train_batch_size, 
                              collate_fn=data_module['data_collator'])
    return train_loader


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



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    import logging

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


######### peft lora func ######
def find_all_linear_names(model):
    # find lora trainning linear module
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias): # get lora module state dict
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True): # get no loara state dict
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return