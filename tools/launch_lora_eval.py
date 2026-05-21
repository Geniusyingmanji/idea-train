"""Launch LoRA-trained-model eval (single shard or sharded) in a detached subprocess."""
import argparse, os, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-8B")
ap.add_argument("--lora", required=True, help="adapter dir")
ap.add_argument("--output-dir", required=True)
ap.add_argument("--max-new-tokens", default="512")
ap.add_argument("--limit", default=None)
ap.add_argument("--shard", default="0")
ap.add_argument("--num-shards", default="1")
ap.add_argument("--no-think", action="store_true")
args = ap.parse_args()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
log = out / "eval.log"
log.write_text("")

cmd = [
    "/home/azureuser/.conda/envs/idea/bin/python", "-u",
    "-m", "idea_train.eval.eval_gene_exam_lora",
    "--base", args.base,
    "--lora", args.lora,
    "--output-dir", str(out),
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
        cmd, stdout=f, stderr=subprocess.STDOUT,
        env=env, cwd="/home/azureuser/workspace-gzy/zyf",
        start_new_session=True, close_fds=True,
    )
print(f"PID {p.pid} → log: {log}")
