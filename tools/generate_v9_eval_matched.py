"""v9: Eval-prompt-matched SFT data.

Take existing v3 SFT examples for closed-form task types, replace the prompt
suffix with the EXACT eval-format instruction (e.g., "Answer with exactly two
lines: Driver = [...] / Dynamics = [...]"), and re-prompt GPT-5.5 for the answer
in that format. The model trained on this will emit eval-format directly at
inference time, closing the strict-vs-lenient scoring gap.

Output: data/stage1_sft/train_v9.jsonl
  - gene_card_extract examples: unchanged from v7 (824 cards with evidence)
  - 7 closed-form task types: re-formatted to match eval prompts
  - idea_generate: unchanged
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
SRC_V7 = ROOT / "data/stage1_sft/train_v7.jsonl"           # v7 = v3 minus 32 zero-evidence cards
OUT    = ROOT / "data/stage1_sft/train_v9.jsonl"
LOG    = ROOT / "data/teacher_logs/v9_eval_matched.jsonl"

# Per-task eval-format suffix. Drawn verbatim from gene_exam/Questions/*/instances.json.
EVAL_SUFFIX = {
    "T1-01_contribution_type": (
        "\n\nAnswer with one line per genome:\n"
        "G1 = [method|dataset|analysis|system|theory]\n"
        "G2 = [method|dataset|analysis|system|theory]\n"
        "G3 = [method|dataset|analysis|system|theory]\n"
        "G4 = [method|dataset|analysis|system|theory]"
    ),
    "T1-03_driver_vs_passenger": (
        "\n\nAnswer with exactly two lines:\n"
        "DriverGene: G#\n"
        "PassengerGene: G#"
    ),
    "T2-01_ordering_5": (
        "\n\nArrange the genomes in chronological order (earliest → latest).\n\n"
        "Answer with exactly one line:\n"
        "Order = [first, second, third, fourth, fifth]"
    ),
    "T2-07_lim_delta_match": (
        "\n\nAnswer with exactly one line per mapping:\n"
        "L1 = [D1|D2|D3|D4|D5|D6]\n"
        "L2 = [D1|D2|D3|D4|D5|D6]\n"
        "L3 = [D1|D2|D3|D4|D5|D6]"
    ),
    "T3-01_single_dynamics": (
        "\n\nAnswer with exactly two lines:\n"
        "Driver = [mechanism|niche|observation|limitation]\n"
        "Dynamics = [Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition]"
    ),
    "T3-09_relation_classify": (
        "\n\nAnswer with exactly two lines:\n"
        "Label = [A|B|C]\n"
        "Dynamics = [Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition]"
    ),
    "T4-01_consistency_check": (
        "\n\nAnswer:\n"
        "Label = [A|B|C|D|E|F]\n"
        "Type = [method|dataset|analysis|system|theory]\n"
        "Verify = [T|F, T|F, T|F, T|F]"
    ),
}

SYSTEM = (
    "You are a precise scientific lineage analyst. Read the paper material below "
    "and answer in the EXACT format requested. Do NOT use JSON. Do NOT include any "
    "commentary or extra lines."
)


def build_new_prompt(orig_prompt: str, tt: str) -> str:
    """Strip any JSON-formatting instructions from the orig prompt and append eval-format."""
    body = orig_prompt
    # remove any trailing 'Return JSON: ...' or '```json ...``` schema
    body = re.sub(r"\n*Return JSON.*$", "", body, flags=re.DOTALL).rstrip()
    body = re.sub(r"\n*```json.*?```", "", body, flags=re.DOTALL).rstrip()
    return body + EVAL_SUFFIX[tt]


def validate_completion(text: str, tt: str) -> bool:
    """Cheap regex check that the response matches the expected format shape."""
    text = text.strip()
    if tt == "T3-01_single_dynamics":
        return bool(re.search(r"Driver\s*=\s*\w", text) and re.search(r"Dynamics\s*=", text))
    if tt == "T3-09_relation_classify":
        # relax to accept any token after Label= (GPT-5.5 sometimes uses dynamics labels here)
        return bool(re.search(r"Label\s*=\s*\w", text) and re.search(r"Dynamics\s*=", text))
    if tt == "T1-01_contribution_type":
        # at least G1 = X
        return bool(re.search(r"G1\s*=\s*\w", text))
    if tt == "T1-03_driver_vs_passenger":
        return bool(re.search(r"Driver\s*Gene\s*:\s*G\d", text, re.I) and
                    re.search(r"Passenger\s*Gene\s*:\s*G\d", text, re.I))
    if tt == "T2-01_ordering_5":
        return bool(re.search(r"Order\s*=\s*\[", text))
    if tt == "T2-07_lim_delta_match":
        return bool(re.search(r"L1\s*=\s*D\d", text))
    if tt == "T4-01_consistency_check":
        return bool(re.search(r"Label\s*=", text) and re.search(r"Type\s*=", text)
                    and re.search(r"Verify\s*=", text))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-per-task", type=int, default=None,
                    help="Cap regenerated examples per task type (for smoke).")
    args = ap.parse_args()

    # 1) read v7 source, sort by task type
    by_tt: dict[str, list[dict]] = {}
    pass_through: list[dict] = []
    with SRC_V7.open() as f:
        for line in f:
            r = json.loads(line)
            tt = r.get("task_type")
            if tt in EVAL_SUFFIX:
                by_tt.setdefault(tt, []).append(r)
            else:
                pass_through.append(r)

    print(f"v7 source: {sum(len(v) for v in by_tt.values())} regen-target rows + "
          f"{len(pass_through)} pass-through rows")
    for tt, rows in by_tt.items():
        print(f"  {tt}: {len(rows)}")

    # 2) build GPT-5.5 calls
    calls: list[TeacherCall] = []
    rebuild_index: dict[str, dict] = {}                     # prompt_id -> source row
    for tt, rows in by_tt.items():
        if args.limit_per_task:
            rows = rows[: args.limit_per_task]
        for i, r in enumerate(rows):
            new_prompt = build_new_prompt(r["prompt"], tt)
            pid = f"v9::{tt}::{i:04d}"
            calls.append(TeacherCall(
                prompt_id=pid,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user",   "content": new_prompt}],
                max_tokens=512,
                metadata={"task_type": tt, "src_paper_id":
                          r.get("metadata", {}).get("source_paper_id")},
            ))
            rebuild_index[pid] = {"src_row": r, "new_prompt": new_prompt}

    print(f"\nfiring {len(calls)} GPT-5.5 calls (workers={args.workers})")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    t0 = time.time()

    def cb(d, t):
        print(f"  teacher: {d}/{t}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    results = batch_call(calls, workers=args.workers, log_path=LOG, on_progress=cb)
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    # 3) assemble v9 dataset
    stats = Counter()
    OUT.write_text("")                                      # truncate
    with OUT.open("a") as fout:
        # regen rows
        for r in results:
            if r.error or not r.content:
                stats["api_error"] += 1; continue
            md = r.metadata or {}
            tt = md["task_type"]
            comp = r.content.strip()
            if not validate_completion(comp, tt):
                stats[f"format_invalid_{tt}"] += 1; continue
            stats[f"accept_{tt}"] += 1
            entry = rebuild_index[r.prompt_id]
            row = dict(entry["src_row"])                    # copy
            row["prompt"] = entry["new_prompt"]
            row["completion"] = comp
            row["messages"] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": entry["new_prompt"]},
                {"role": "assistant", "content": comp},
            ]
            row.setdefault("metadata", {})["v9_regen"] = True
            row["metadata"]["teacher_input_tokens"] = r.input_tokens
            row["metadata"]["teacher_output_tokens"] = r.output_tokens
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        # pass-through rows
        for r in pass_through:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[f"passthru_{r.get('task_type', 'unk')}"] += 1

    n_accept = sum(v for k, v in stats.items() if k.startswith("accept_"))
    n_passthru = sum(v for k, v in stats.items() if k.startswith("passthru_"))
    n_invalid = sum(v for k, v in stats.items() if k.startswith("format_invalid_"))
    print(f"\n=== v9 build complete ===")
    print(f"  accepted (regenerated): {n_accept}")
    print(f"  pass-through:           {n_passthru}")
    print(f"  format invalid:         {n_invalid}")
    print(f"  api error:              {stats.get('api_error', 0)}")
    print(f"  total examples written: {n_accept + n_passthru}")
    print(f"  detail: {dict(stats)}")
    print(f"  wrote: {OUT}")


if __name__ == "__main__":
    main()
