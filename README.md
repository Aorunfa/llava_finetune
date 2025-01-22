# 持续整理中，代码源自于原仓库[LLaVA](https://github.com/haotian-liu/LLaVA)
# llava_finetune
一个用于llava-v1.5-7b和llava-v1.6-mistral-7b微调的仓库，主要用途:

00 理解算法设计，vison_encoder, mm_adaper, llm如何组合

01 实战 deepspeed + transformer的分布式训练


02 实战 peft库支持的loar微调、lora模型加载、lora模型合并过程

03 实战peft支持的bit微调

04 实战 fsdp训练过程: 模型分片，模型保存
fsdp分布训练原理，对比ddp
* ddp: 数据并行， 模型、梯度、优化器状态在同一个gpu上，优化器状态分布在主节点更新同步套所有节点。特点速度更快 但单卡峰值的显存依赖更大
* fsdp: 将模型、梯度、优化器进行分片(gups个unit，每个unit含有模型、梯度、优化器的一个module的一个片段)
    - 举例将一个module参数等分为gpus个分片分布在每个fsdpunit中
    - forward fsdp unit 从其他rank中获取层的其他参数，恢复完整后每forward，结果传递给下一个unit
    - backward fsdp unit 从其他rank中获取层的其他参数，恢复完整后每forward，梯度分片保存，梯度函数传递给上一个unit
    - 特点：速度慢一些，但是单卡峰值显存依赖小
    - 当前后分片为多进程的方式工作，即上一节点不用完全等待下一节点处理完，此时可以实现模型(分片)并行的效果
    - * 详细讲解：http://shiyanjun.cn/archives/2292.html
    - * 官方教程：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp

05 实战accelerate分布式训练加速






