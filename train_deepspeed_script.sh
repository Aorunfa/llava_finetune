PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"

# for accerelate
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export WANDB_DISABLED='true'


deepspeed train_ds.py \
    --deepspeed /home/chaofeng/LLaVA/scripts/zero2.json