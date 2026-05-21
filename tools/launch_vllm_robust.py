"""Launch vLLM in a fully-detached subprocess that survives the parent exit.

Uses Popen(start_new_session=True) so the child becomes session leader and is
not killed when the parent (the Bash that invoked us) terminates.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ARGS = [
    "/home/azureuser/.conda/envs/idea/bin/python",
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen3-8B-Base",
    "--served-model-name", "qwen3-8b-base",
    "--host", "127.0.0.1",
    "--port", "8801",
    "--dtype", "bfloat16",
    "--max-model-len", "16384",
    "--gpu-memory-utilization", "0.85",
    "--enforce-eager",
]
LOG = Path("/home/azureuser/workspace-gzy/zyf/idea_train/logs/vllm_qwen3-8b-base.log")
LOG.parent.mkdir(parents=True, exist_ok=True)

# Truncate fresh
LOG.write_text("")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

with LOG.open("ab") as f:
    p = subprocess.Popen(
        ARGS,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        close_fds=True,
    )

print(f"vLLM PID: {p.pid}")
# Don't wait — parent exits, child detached as session leader
