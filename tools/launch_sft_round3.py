import os, subprocess
from pathlib import Path

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python", "-u",
    "/home/azureuser/workspace-gzy/zyf/idea_train/tools/generate_sft_round3.py",
    "--n-per-task", "300",
    "--workers", "24",
    "--output", "/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/round3_train.jsonl",
]
log = Path("/home/azureuser/workspace-gzy/zyf/idea_train/logs/sft_round3.log")
log.write_text("")
with log.open("ab") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy(),
                          cwd="/home/azureuser/workspace-gzy/zyf",
                          start_new_session=True, close_fds=True)
print(f"PID {p.pid} → log: {log}")
