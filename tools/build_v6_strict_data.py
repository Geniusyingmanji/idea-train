"""Build v6 training data: augment prompts with explicit canonical-key constraints,
and force-rewrite completion JSON keys to canonical schema names.

Why: v3 emits "Driver"/"Dynamics" (capitalized) at eval time, mimicking the eval
prompt's plain-text "Driver = X" phrasing, even though training data used lowercase
keys. The lenient scorer fixes this post-hoc; v6 should fix it at train time so
strict accuracy rises too.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from eval.lenient_rescore import KEY_ALIASES, remap_keys, extract_json

SRC = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train_all.jsonl")
DST = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train_v6.jsonl")

# canonical keys per task type (extracted from GENE-Exam gold_answer)
TASK_CANONICAL = {
    "T1-01_contribution_type": ["multi_contrib_types"],
    "T1-03_driver_vs_passenger": ["driver_gene", "passenger_gene"],
    "T2-01_ordering_5": ["correct_order"],
    "T2-07_lim_delta_match": ["mapping"],
    "T3-01_single_dynamics": ["driver", "dynamics"],
    "T3-09_relation_classify": ["label", "dynamics"],
    "T4-01_consistency_check": ["label", "contribution_type", "verify"],
}

STRICT_SUFFIX = (
    "\n\nIMPORTANT — STRICT JSON FORMAT:\n"
    "Your response MUST be a JSON object inside ```json ... ``` fences with EXACTLY "
    "these lowercase keys (and only these): {keys}.\n"
    "Do NOT capitalize the keys (no \"Driver\", no \"Dynamics\"). "
    "Do NOT use alternative names. Keys are case-sensitive."
)


def rewrite_completion(comp: str, tt: str) -> str:
    """Re-emit the JSON inside the completion with canonical keys."""
    obj = extract_json(comp)
    if obj is None or not isinstance(obj, dict):
        return comp
    fixed = remap_keys(obj, tt)
    return "```json\n" + json.dumps(fixed, ensure_ascii=False) + "\n```"


def main():
    n_in = n_out = n_unchanged_prompt = n_rewrote_completion = 0
    by_tt = {}
    with SRC.open() as f, DST.open("w") as out:
        for line in f:
            r = json.loads(line)
            n_in += 1
            tt = r["task_type"]
            by_tt[tt] = by_tt.get(tt, 0) + 1

            if tt in TASK_CANONICAL:
                keys = TASK_CANONICAL[tt]
                if STRICT_SUFFIX.format(keys=json.dumps(keys)) not in r["prompt"]:
                    r["prompt"] = r["prompt"].rstrip() + STRICT_SUFFIX.format(keys=json.dumps(keys))
                else:
                    n_unchanged_prompt += 1
                new_comp = rewrite_completion(r["completion"], tt)
                if new_comp != r["completion"]:
                    n_rewrote_completion += 1
                    r["completion"] = new_comp
                if r.get("messages"):
                    # also sync messages field
                    for m in r["messages"]:
                        if m.get("role") == "user":
                            m["content"] = r["prompt"]
                        elif m.get("role") == "assistant":
                            m["content"] = r["completion"]

            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"in={n_in}  out={n_out}  prompts_unchanged={n_unchanged_prompt}  completions_rewritten={n_rewrote_completion}")
    print("by task type:")
    for tt, n in sorted(by_tt.items()):
        marker = " [augmented]" if tt in TASK_CANONICAL else ""
        print(f"  {tt}: {n}{marker}")
    print(f"wrote: {DST}")


if __name__ == "__main__":
    main()
