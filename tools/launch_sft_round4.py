import os, subprocess
from pathlib import Path

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python", "-u",
    "/home/azureuser/workspace-gzy/zyf/idea_train/tools/generate_sft_round4.py",
    "--n-per-task", "300",
    "--workers", "32",
    "--output", "/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/round4_train.jsonl",
]
log = Path("/home/azureuser/workspace-gzy/zyf/idea_train/logs/sft_round4.log")
log.write_text("")
with log.open("ab") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                          env=os.environ.copy(), cwd="/home/azureuser/workspace-gzy/zyf",
                          start_new_session=True, close_fds=True)
print(f"PID {p.pid} → log: {log}")
