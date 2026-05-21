#!/bin/bash
# RL training health check — runs once, returns concise status.
# Used by ScheduleWakeup loop every 60-90 min during overnight.

RL_PID=309719
RL_DIR=/home/azureuser/workspace-gzy/zyf/idea_train/train/checkpoints/qwen3-8b-agentic-rl
RL_LOG=/tmp/agentic_rl.log
TRACE=$RL_DIR/trace.jsonl

echo "=== $(date +%H:%M:%S) RL health check ==="

# 1. process alive?
if ps -p $RL_PID > /dev/null; then
    echo "RL alive (PID $RL_PID)"
else
    echo "RL DEAD — last 20 log lines:"
    tail -20 $RL_LOG
    exit 1
fi

# 2. step progress
if [ -f $TRACE ]; then
    n_steps=$(wc -l < $TRACE)
    last_step=$(tail -1 $TRACE | /home/azureuser/.conda/envs/idea/bin/python -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"step={d['step']:4d}  R={d['R_mean']:+.3f}  F={d['R_components_mean']['F']:+.2f}  S={d['R_components_mean']['S']:.2f}  L={d['R_components_mean']['L']:.2f}  loss={d['loss']:+.4f}  sec={d['sec']:.0f}\")
")
    echo "steps written: $n_steps  /  last: $last_step"

    # propose rate over last 10 steps
    propose_rate=$(tail -10 $TRACE | /home/azureuser/.conda/envs/idea/bin/python -c "
import json, sys
n_p, n_t = 0, 0
for line in sys.stdin:
    try:
        d = json.loads(line)
        for s in d['traj_summaries']:
            n_t += 1
            if s['propose_emitted']: n_p += 1
    except: pass
print(f'{n_p}/{n_t} ({100*n_p/max(n_t,1):.0f}%)')
")
    echo "propose rate (last 10 steps): $propose_rate"
fi

# 3. checkpoints
ckpts=$(ls $RL_DIR 2>/dev/null | grep checkpoint | sort | tr '\n' ' ')
if [ -n "$ckpts" ]; then
    echo "checkpoints: $ckpts"
fi

# 4. baseline eval progress
echo "--- baselines ---"
if [ -f /tmp/arena_agentic_sft.log ]; then
    n=$(grep -c "^  \[" /tmp/arena_agentic_sft.log 2>/dev/null || echo "0")
    last=$(tail -1 /tmp/arena_agentic_sft.log 2>/dev/null | head -c 100)
    echo "GENE-Arena baseline: $n progress markers, last: ${last:0:80}"
fi
if [ -f /tmp/sgi_agentic_sft.log ]; then
    n=$(grep -c "^  \[" /tmp/sgi_agentic_sft.log 2>/dev/null || echo "0")
    echo "SGI baseline: $n progress markers"
fi
