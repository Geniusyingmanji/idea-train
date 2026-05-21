"""Same as eval_gene_exam_transformers.py but loads a LoRA adapter on top of base."""
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.verifier import compute_verifier

REPO = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
CAL = {f"C-{i:02d}_{name}" for i, name in enumerate([
    "genome_fingerprint", "fingerprint_resolve", "genome_field_type_blind",
    "mech_lim_match", "delta_lim_match", "genome_field_assign_2p_6a",
    "genome_diff_match", "multihop_diff", "intruder_basic",
    "intruder_subfield", "diff_hallucination", "counterfactual_removal",
    "gene_swap", "citation_lineage",
], 1)}

SYS = (
    "You are a scientific lineage analyst. "
    "Answer step by step in <think>...</think> tags, then output the final answer as a single "
    "fenced ```json``` code block matching the schema. No other text after the JSON."
)


def load_instances(limit=None, shard=0, num_shards=1):
    insts = []
    for td in sorted(glob.glob(str(REPO / "gene_exam/Questions/*"))):
        name = Path(td).name
        if name in CAL:
            continue
        f = Path(td) / "instances.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        if isinstance(data, dict):
            data = list(data.values())
        for d in data:
            d["_task_dir"] = name
        insts.extend(data)
        if limit and len(insts) >= limit:
            break
    if limit:
        insts = insts[:limit]
    if num_shards > 1:
        insts = [x for i, x in enumerate(insts) if i % num_shards == shard]
    return insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora", required=True, help="LoRA adapter dir")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--prompt-suffix-file", default=None,
                     help="If set, contents of this file are appended to every user-prompt "
                          "(diagnostic for testing eval-time prompt augmentation).")
    args = ap.parse_args()
    suffix = ""
    if args.prompt_suffix_file:
        suffix = Path(args.prompt_suffix_file).read_text()
        print(f"[INFO] Appending {len(suffix)} chars to every prompt from {args.prompt_suffix_file}", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = out / f"per_instance_shard{args.shard:02d}.jsonl"
    summary = out / f"summary_shard{args.shard:02d}.json"

    print(f"[1/3] Loading instances", flush=True)
    insts = load_instances(args.limit, args.shard, args.num_shards)
    print(f"  shard {args.shard}/{args.num_shards}: {len(insts)} instances", flush=True)

    print(f"[2/3] Loading {args.base} + LoRA {args.lora}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()
    print(f"  ready", flush=True)

    print(f"[3/3] Eval", flush=True)
    per_task_corr = defaultdict(int)
    per_task_tot = defaultdict(int)
    t0 = time.time()
    with log.open("w") as fp:
        for i, inst in enumerate(insts):
            prompt = inst.get("prompt", "")
            tt = inst.get("task_type", inst.get("_task_dir", "?"))
            gold = inst.get("gold_answer")
            if suffix:
                prompt = prompt.rstrip() + suffix
            messages = [{"role": "system", "content": SYS},
                         {"role": "user", "content": prompt}]
            try:
                if args.no_think:
                    text = tok.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=False)
                else:
                    text = tok.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            except (TypeError, ValueError):
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out_ids[0][inputs["input_ids"].shape[-1]:]
            completion = tok.decode(gen, skip_special_tokens=True)
            try:
                score, _ = compute_verifier(completion, tt, gold_answer=gold)
                v = score.v
                ok = score.schema_valid == 1.0 and score.exact_match == 1.0
            except Exception as e:
                v = 0.0; ok = False; score = None
            per_task_tot[tt] += 1
            if ok:
                per_task_corr[tt] += 1
            fp.write(json.dumps({
                "instance_id": inst.get("instance_id"),
                "task_type": tt, "completion": completion,
                "v": v, "is_correct": ok,
                "verifier": score.to_dict() if score else None,
                "n_input_tokens": int(inputs["input_ids"].shape[-1]),
                "n_output_tokens": int(gen.shape[-1]),
            }, ensure_ascii=False) + "\n")
            fp.flush()
            if (i + 1) % 25 == 0 or i == 0:
                el = time.time() - t0
                rate = (i + 1) / max(el, 1)
                eta = (len(insts) - i - 1) / max(rate, 0.001)
                c = sum(per_task_corr.values()); t = sum(per_task_tot.values())
                print(f"  [{i+1}/{len(insts)}] {rate:.2f}/s | ETA {eta/60:.1f}m | "
                      f"acc {c}/{t}={100*c/max(t,1):.1f}%", flush=True)

    tiers = defaultdict(list)
    for t, n in per_task_tot.items():
        acc = per_task_corr[t] / n
        for k in ("T1", "T2", "T3", "T4"):
            if t.startswith(f"{k}-"):
                tiers[k].append(acc)

    summ = {
        "base": args.base, "lora": args.lora,
        "n_instances": sum(per_task_tot.values()),
        "n_correct": sum(per_task_corr.values()),
        "macro_accuracy": sum(per_task_corr.values()) / max(sum(per_task_tot.values()), 1),
        "per_task_accuracy": {k: per_task_corr[k] / per_task_tot[k] for k in sorted(per_task_tot)},
        "per_tier_macro": {k: sum(v) / len(v) for k, v in tiers.items()},
        "elapsed_seconds": time.time() - t0,
    }
    summary.write_text(json.dumps(summ, indent=2))
    print(json.dumps({"macro_accuracy": summ["macro_accuracy"], "per_tier_macro": summ["per_tier_macro"]}, indent=2))


if __name__ == "__main__":
    main()
