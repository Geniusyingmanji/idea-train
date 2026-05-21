"""Launch SFT training (single-GPU, LoRA) in a detached subprocess."""
import argparse, os, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--data", default="/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train.jsonl")
ap.add_argument("--output-dir", default="/home/azureuser/workspace-gzy/zyf/idea_train/train/checkpoints/qwen3-8b-sft-v1")
ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
ap.add_argument("--lora-r", default="64")
ap.add_argument("--lora-alpha", default="128")
ap.add_argument("--epochs", default="2.0")
ap.add_argument("--lr", default="2e-4")
ap.add_argument("--per-device-batch", default="2")
ap.add_argument("--grad-accum", default="4")
args = ap.parse_args()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
env["TOKENIZERS_PARALLELISM"] = "false"

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
log = Path("/home/azureuser/workspace-gzy/zyf/idea_train/logs/sft_train.log")
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("")

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python", "-u",
    "/home/azureuser/workspace-gzy/zyf/idea_train/train/sft_train.py",
    "--data", args.data,
    "--output-dir", args.output_dir,
    "--epochs", str(args.epochs),
    "--lr", str(args.lr),
    "--per-device-batch", str(args.per_device_batch),
    "--grad-accum", str(args.grad_accum),
    "--lora-r", str(args.lora_r),
    "--lora-alpha", str(args.lora_alpha),
]
with log.open("ab") as f:
    p = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        env=env, cwd="/home/azureuser/workspace-gzy/zyf",
        start_new_session=True, close_fds=True,
    )
print(f"PID {p.pid} → log: {log}")
