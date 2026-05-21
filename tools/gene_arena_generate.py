"""Generate GENE-Arena idea outputs from a local LoRA checkpoint.

Self-contained: doesn't touch arena's adapter framework. For each
(task × setting), uses arena's PromptBuilder to build the prompt and
transformers.generate to produce the JSON idea. Saves outputs in the
schema arena's PES eval expects, so downstream scoring can run on them.

Output layout:
  out_dir/ideas/<trace_id>/<participant>_<setting>.json
  out_dir/manifest.jsonl  (one row per generated idea, for PES eval queue)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving/gene_arena")

from gene_arena.prompt_builder import PromptBuilder, PromptConfig
from gene_arena.arena_config import TASK_DIR

SETTINGS = ("Question", "Library", "Lineage")

SYS_PROMPT = (
    "You are a scientific lineage analyst and idea generator. "
    "Respond with a single JSON object inside ```json ... ``` fences matching the "
    "schema requested in the user prompt. No commentary outside the JSON."
)


def list_tasks(limit: int | None = None) -> list[Path]:
    out = sorted(Path(TASK_DIR).glob("*.json"))
    if limit:
        out = out[:limit]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", default="none",
                    help="LoRA adapter dir, or 'none' for base-only baseline")
    ap.add_argument("--participant",  required=True, help="e.g. 'qwen3-8b-sft-v3'")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-tasks", type=int, default=None, help="None = all 50")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS),
                    help="Which arena settings to run (subset of: Question, Library, Lineage)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ideas_dir = out_dir / "ideas"
    ideas_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("")

    device = f"cuda:{args.gpu}"
    print(f"[1/3] loading {args.student_base} + LoRA {args.student_lora} on {device}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=device,
    )
    if args.student_lora and args.student_lora != "none":
        model = PeftModel.from_pretrained(base, args.student_lora)
    else:
        model = base
    model.eval()
    print(f"  ready")

    tasks = list_tasks(args.n_tasks)
    settings = [s for s in args.settings if s in SETTINGS]
    print(f"[2/3] {len(tasks)} tasks × {len(settings)} settings = "
          f"{len(tasks) * len(settings)} generations")

    print(f"[3/3] generating")
    t0 = time.time()
    n_done = n_err = 0
    for task_path in tasks:
        trace_id = task_path.stem
        task_out_dir = ideas_dir / trace_id
        task_out_dir.mkdir(parents=True, exist_ok=True)
        builder = PromptBuilder(task_path)
        for setting in settings:
            out_path = task_out_dir / f"{args.participant}_{setting}.json"
            if out_path.exists():
                continue                                    # resume-safe
            try:
                user_prompt = builder.build(PromptConfig(setting=setting))
            except Exception as e:
                print(f"  ERR build {trace_id}/{setting}: {e}")
                n_err += 1
                continue
            messages = [{"role": "system", "content": SYS_PROMPT},
                        {"role": "user",   "content": user_prompt}]
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except (TypeError, ValueError):
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=8192).to(device)
            t_gen_start = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out_ids[0, inputs["input_ids"].shape[-1]:]
            completion = tok.decode(gen, skip_special_tokens=True)
            t_gen_ms = (time.time() - t_gen_start) * 1000

            record = {
                "trace_id":          trace_id,
                "task_id":           trace_id,
                "participant_id":    args.participant,
                "participant_type": "llm",
                "provider":          "local_transformers",
                "model":              args.student_base + " + " + args.student_lora,
                "framework":         None,
                "harness":           None,
                "setting":           setting,
                "content":           completion,
                "prompt":            user_prompt,
                "output_schema":     "OUTPUT_JSON_SCHEMA",
                "input_tokens":      int(inputs["input_ids"].shape[-1]),
                "output_tokens":     int(gen.shape[-1]),
                "latency_ms":        t_gen_ms,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": {
                    "system_prompt": SYS_PROMPT,
                    "temperature":   args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            with manifest_path.open("a") as mf:
                mf.write(json.dumps({
                    "trace_id": trace_id, "setting": setting,
                    "participant": args.participant, "path": str(out_path),
                    "output_tokens": int(gen.shape[-1]),
                    "latency_ms": t_gen_ms,
                }) + "\n")
            n_done += 1
            if n_done % 5 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{n_done}] {trace_id}/{setting}  ({el:.1f}min, {t_gen_ms/1000:.1f}s/gen)",
                      flush=True)

    el = (time.time() - t0) / 60
    print(f"\ngenerated {n_done} ideas in {el:.1f} min; {n_err} build errors")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
