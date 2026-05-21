"""GENE-Exam evaluator using plain transformers.generate (no vLLM dependency).

Loads a HF model on N GPUs, iterates over main_challenge profile instances,
and uses idea_train.evo_opd.verifier to score each completion.

Output:
  - per_instance.jsonl    one row per instance with prediction + score
  - summary.json          per-task + per-tier + overall accuracy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.verifier import compute_verifier

REPO = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")

# main_challenge excludes C-01..C-14 calibration tasks
CALIBRATION_TASKS = {
    "C-01_genome_fingerprint", "C-02_fingerprint_resolve", "C-03_genome_field_type_blind",
    "C-04_mech_lim_match", "C-05_delta_lim_match", "C-06_genome_field_assign_2p_6a",
    "C-07_genome_diff_match", "C-08_multihop_diff", "C-09_intruder_basic",
    "C-10_intruder_subfield", "C-11_diff_hallucination", "C-12_counterfactual_removal",
    "C-13_gene_swap", "C-14_citation_lineage",
}

SYSTEM_PROMPT = (
    "You are a scientific lineage analyst. "
    "Answer the question step by step in <think>...</think> tags, then output the final answer "
    "as a single fenced ```json``` code block matching the exact schema requested in the user prompt. "
    "No other text after the JSON."
)


def load_instances(profile: str = "main_challenge", limit: int | None = None,
                    skip_tasks: list[str] | None = None) -> list[dict]:
    insts: list[dict] = []
    skip = set(skip_tasks or [])
    for td in sorted(glob.glob(str(REPO / "gene_exam/Questions/*"))):
        name = Path(td).name
        if profile == "main_challenge" and name in CALIBRATION_TASKS:
            continue
        if name in skip:
            continue
        inst_file = Path(td) / "instances.json"
        if not inst_file.exists():
            continue
        with inst_file.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        for i in data:
            i["_task_dir"] = name
        insts.extend(data)
        if limit and len(insts) >= limit:
            break
    if limit:
        insts = insts[:limit]
    return insts


def build_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def apply_template(tok, messages: list[dict], no_think: bool) -> str:
    """Apply chat template, optionally with Qwen3's enable_thinking=False."""
    try:
        if no_think:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
    except (TypeError, ValueError):
        pass
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--profile", default="main_challenge", choices=["main_challenge", "full"])
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default="/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/qwen3-8b-base_baseline")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index (for multi-GPU split)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards")
    ap.add_argument("--no-think", action="store_true",
                    help="for Qwen3, force enable_thinking=False (much shorter responses)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"per_instance_shard{args.shard:02d}.jsonl"
    summary_path = out_dir / f"summary_shard{args.shard:02d}.json"

    print(f"[1/3] Loading instances (profile={args.profile})", flush=True)
    insts = load_instances(profile=args.profile, limit=args.limit)
    # shard split
    insts = [x for i, x in enumerate(insts) if i % args.num_shards == args.shard]
    print(f"  shard {args.shard}/{args.num_shards}: {len(insts):,} instances", flush=True)

    print(f"[2/3] Loading model {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    has_chat_template = bool(tok.chat_template)
    print(f"  loaded in {time.time()-t0:.1f}s | has_chat_template={has_chat_template}", flush=True)

    print(f"[3/3] Evaluating", flush=True)
    log_fp = log_path.open("w")
    per_task_correct: dict[str, int] = defaultdict(int)
    per_task_total: dict[str, int] = defaultdict(int)
    t_start = time.time()

    # one-by-one for simplicity & memory-safety
    for i, inst in enumerate(insts):
        prompt = inst.get("prompt", "")
        task_type = inst.get("task_type", inst.get("_task_dir", "unknown"))
        gold = inst.get("gold_answer")

        # build input — base model doesn't have a chat template; just concat
        if has_chat_template:
            messages = build_messages(prompt)
            text = apply_template(tok, messages, no_think=args.no_think)
        else:
            # base model: prepend brief instructions
            text = (
                f"{SYSTEM_PROMPT}\n\n"
                f"USER:\n{prompt}\n\n"
                f"ASSISTANT:\n"
            )
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
        completion = tok.decode(gen_tokens, skip_special_tokens=True)

        try:
            score, _pr = compute_verifier(completion, task_type, gold_answer=gold)
            v = score.v
            is_correct = (score.exact_match == 1.0) and (score.schema_valid == 1.0)
        except Exception as e:
            v = 0.0
            is_correct = False
            score = None

        per_task_total[task_type] += 1
        if is_correct:
            per_task_correct[task_type] += 1

        record = {
            "instance_id": inst.get("instance_id"),
            "task_type": task_type,
            "completion": completion,
            "v": v,
            "is_correct": is_correct,
            "verifier": score.to_dict() if score else None,
            "n_input_tokens": int(inputs["input_ids"].shape[-1]),
            "n_output_tokens": int(gen_tokens.shape[-1]),
        }
        log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_fp.flush()

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(insts) - i - 1) / max(rate, 0.001)
            correct_so_far = sum(per_task_correct.values())
            total_so_far = sum(per_task_total.values())
            print(f"  [{i+1:>5}/{len(insts):,}] rate {rate:.2f}/s | ETA {eta/60:.1f}min | "
                  f"acc_so_far {correct_so_far}/{total_so_far} = {100*correct_so_far/max(total_so_far,1):.1f}%",
                  flush=True)

    log_fp.close()

    # summary
    summary = {
        "model": args.model,
        "profile": args.profile,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_instances": sum(per_task_total.values()),
        "n_correct": sum(per_task_correct.values()),
        "macro_accuracy": sum(per_task_correct.values()) / max(sum(per_task_total.values()), 1),
        "per_task_accuracy": {
            t: per_task_correct[t] / per_task_total[t]
            for t in sorted(per_task_total)
        },
        "elapsed_seconds": time.time() - t_start,
    }
    # per-tier
    tiers: dict[str, list[float]] = defaultdict(list)
    for t, acc in summary["per_task_accuracy"].items():
        if t.startswith("T1-"): tiers["T1"].append(acc)
        elif t.startswith("T2-"): tiers["T2"].append(acc)
        elif t.startswith("T3-"): tiers["T3"].append(acc)
        elif t.startswith("T4-"): tiers["T4"].append(acc)
    summary["per_tier_macro"] = {k: sum(v) / len(v) for k, v in tiers.items()}

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"macro_accuracy": summary["macro_accuracy"],
                       "per_tier_macro": summary["per_tier_macro"]}, indent=2))
    print(f"Wrote: {log_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
