"""Launch a baseline GENE-Exam eval in a detached subprocess.

Usage:
  python tools/launch_baseline_eval.py --model Qwen/Qwen3-8B --gpu 2 \
    --output-dir idea_train/eval/results/qwen3-8b_baseline
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--max-new-tokens", type=int, default=2048)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--no-think", action="store_true")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
args = ap.parse_args()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
log = out_dir / "eval.log"
log.write_text("")

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python",
    "-u",  # unbuffered
    "-m", "idea_train.eval.eval_gene_exam_transformers",
    "--model", args.model,
    "--output-dir", str(out_dir),
    "--max-new-tokens", str(args.max_new_tokens),
    "--shard", str(args.shard),
    "--num-shards", str(args.num_shards),
]
if args.limit:
    cmd += ["--limit", str(args.limit)]
if args.no_think:
    cmd += ["--no-think"]

with log.open("ab") as f:
    p = subprocess.Popen(
        cmd,
        stdout=f, stderr=subprocess.STDOUT,
        env=env,
        cwd="/home/azureuser/workspace-gzy/zyf",
        start_new_session=True, close_fds=True,
    )
print(f"PID {p.pid} → log: {log}")
