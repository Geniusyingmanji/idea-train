#!/bin/bash
# Launch vLLM serving Qwen3-8B-Base on GPU 0.
cd /home/azureuser/workspace-gzy/zyf
mkdir -p idea_train/logs
exec env CUDA_VISIBLE_DEVICES=0 /home/azureuser/.conda/envs/idea/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B-Base \
  --served-model-name qwen3-8b-base \
  --host 127.0.0.1 --port 8801 \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  > /home/azureuser/workspace-gzy/zyf/idea_train/logs/vllm_qwen3-8b-base.log 2>&1
