"""Build v7 and v8 SFT data from GeneTrace v0.1 cards.

v7 = v3 data minus the 31 evidence-extraction-failed cards (quality filter only).
v8 = v7 but with gene_card_extract completions enriched with evidence quotes
     (teaches grounded outputs; pure addition to the completion JSON).

Other task types (T1-01, T1-03, T2-01, T2-07, T3-01, T3-09, T4-01, idea_generate)
are passed through unchanged in both v7 and v8.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
SRC_V3 = ROOT / "data/stage1_sft/train_all.jsonl"            # v3 data (2186 examples)
CARDS  = ROOT / "data/genetrace_v0_1/cards.jsonl"            # 855 cards with evidence
OUT_V7 = ROOT / "data/stage1_sft/train_v7.jsonl"
OUT_V8 = ROOT / "data/stage1_sft/train_v8.jsonl"


def load_cards() -> dict[str, dict]:
    """Map paper_id -> card record. Only cards with at least one evidence quote."""
    out = {}
    for line in CARDS.open():
        r = json.loads(line)
        if any(r.get("evidence", {}).values()):
            out[r["paper_id"]] = r
    return out


def parse_completion_json(comp: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comp, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def enrich_gene_card_completion(comp: str, card: dict) -> str:
    """For v8: add an `evidence` field to the gene_card_extract completion.

    Output schema:
      {
        "mechanism_genome":   "<claim>",
        "niche_genome":       "<claim>",
        ...,
        "evidence": {
          "mechanism_genome":   ["<quote>", ...],
          ...
        }
      }
    """
    obj = parse_completion_json(comp)
    if obj is None:
        return comp
    evidence_text = {
        f: [q["quote"] for q in quotes]
        for f, quotes in (card.get("evidence") or {}).items()
        if quotes
    }
    if not evidence_text:
        return comp                                          # nothing to add
    obj["evidence"] = evidence_text
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true",
                    help="Build v8 (with evidence in completions). Default builds v7.")
    args = ap.parse_args()

    cards = load_cards()
    print(f"loaded {len(cards)} evidence-grounded cards")

    n_in = n_out = n_drop = n_enriched = 0
    with SRC_V3.open() as fin, (OUT_V8 if args.enrich else OUT_V7).open("w") as fout:
        for line in fin:
            r = json.loads(line)
            n_in += 1
            tt = r.get("task_type")
            md = r.get("metadata") or {}
            pid = md.get("source_paper_id")

            # quality filter: for gene_card_extract, only keep cards with evidence
            if tt == "gene_card_extract":
                if pid not in cards:
                    n_drop += 1
                    continue
                if args.enrich:
                    new_comp = enrich_gene_card_completion(r["completion"], cards[pid])
                    if new_comp != r["completion"]:
                        r["completion"] = new_comp
                        # also update messages if present
                        if r.get("messages"):
                            for m in r["messages"]:
                                if m.get("role") == "assistant":
                                    m["content"] = new_comp
                        n_enriched += 1

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    out_path = OUT_V8 if args.enrich else OUT_V7
    print(f"in={n_in}  out={n_out}  dropped={n_drop}  enriched={n_enriched}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
