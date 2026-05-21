#!/bin/bash
# Smoke-eval a fresh RL checkpoint with 5 prompts and report propose rate + R_total.
# Called when a new checkpoint-N appears.
#
# Args: $1 = checkpoint dir (e.g. .../qwen3-8b-agentic-rl/checkpoint-25)

CKPT_DIR=$1
LABEL=$(basename $CKPT_DIR)
LOG=/tmp/rl_ckpt_smoke_${LABEL}.log

if [ ! -f "$CKPT_DIR/adapter_model.safetensors" ]; then
    echo "smoke: $CKPT_DIR not ready"
    exit 0
fi

cd /home/azureuser/workspace-gzy/zyf

echo "=== smoke eval $LABEL @ $(date +%H:%M:%S) ===" | tee $LOG
/home/azureuser/.conda/envs/idea/bin/python -u idea_train/tools/smoke_agentic_rollout.py \
    --student-lora $CKPT_DIR --n-prompts 5 --max-turns 5 --gpu 3 --temperature 0.5 \
    2>&1 | tee -a $LOG | grep -E "Prompt|propose_emitted|REWARD|malformed:|truncated:"
