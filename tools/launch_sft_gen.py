"""Launch the full SFT data generation in detached subprocess."""
import os, subprocess
from pathlib import Path

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python",
    "-u",
    "/home/azureuser/workspace-gzy/zyf/idea_train/tools/generate_sft_data.py",
    "--n-gene-card", "3000",
    "--n-idea-gen", "500",
    "--workers", "32",
    "--output", "/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train.jsonl",
]
log = Path("/home/azureuser/workspace-gzy/zyf/idea_train/logs/sft_gen.log")
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("")
with log.open("ab") as f:
    p = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        cwd="/home/azureuser/workspace-gzy/zyf",
        start_new_session=True, close_fds=True,
    )
print(f"PID {p.pid} → log: {log}")
