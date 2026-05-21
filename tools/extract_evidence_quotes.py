"""Stage E of GeneTrace v0.1 build: extract per-field evidence quotes from
the source text via GPT-5.5, and patch them into cards.jsonl in-place.

Resumable: cards that already have non-empty evidence are skipped. A
side-log of all teacher responses is written to teacher_logs/ for audit.
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
from evo_opd.schemas import GENE_FIELDS
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

ROOT  = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
CARDS = ROOT / "data/genetrace_v0_1/cards.jsonl"
LOG   = ROOT / "data/teacher_logs/genetrace_v0_1_evidence.jsonl"

SYSTEM = (
    "You are a precise evidence extractor for scientific paper annotations. "
    "Given a paper's source text and six claims derived from it, find supporting "
    "verbatim quotes from the source text for each claim. Quotes MUST be exact "
    "substrings of the source text — character-for-character identical. "
    "Output JSON only, no commentary."
)

USER_TMPL = """[SOURCE TEXT]
{source_text}

[CLAIMS TO GROUND]
mechanism_genome:   {mechanism_genome}
niche_genome:       {niche_genome}
observation_genome: {observation_genome}
limitation_genome:  {limitation_genome}
delta_genome:       {delta_genome}
claim_genome:       {claim_genome}

For each of the 6 fields above, return at most 2 short verbatim quotes (≤ 200 chars each) from the source text that directly support the claim. Empty list `[]` if no support exists OR if the field's claim is empty.

Return ONLY this JSON object inside ```json ... ``` fences:
```json
{{
  "mechanism_genome":   [{{"quote": "..."}}],
  "niche_genome":       [{{"quote": "..."}}],
  "observation_genome": [{{"quote": "..."}}],
  "limitation_genome":  [{{"quote": "..."}}],
  "delta_genome":       [{{"quote": "..."}}],
  "claim_genome":       [{{"quote": "..."}}]
}}
```
"""


def parse_response(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def verify_quotes(parsed: dict, source: str) -> dict:
    """Drop quotes that are not exact substrings; add char_offset for the rest."""
    out = {f: [] for f in GENE_FIELDS}
    for field in GENE_FIELDS:
        for item in (parsed.get(field) or []):
            q = (item or {}).get("quote", "").strip()
            if not q:
                continue
            off = source.find(q)
            if off >= 0:
                out[field].append({"quote": q, "char_offset": off})
    return out


def load_cards() -> list[dict]:
    return [json.loads(l) for l in CARDS.open()]


def write_cards(records: list[dict]) -> None:
    tmp = CARDS.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(CARDS)


def has_evidence(card: dict) -> bool:
    ev = card.get("evidence") or {}
    return any(ev.get(f) for f in GENE_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N cards needing evidence "
                         "(useful for a small smoke run before paying for the full batch).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="Re-extract evidence even for cards that already have it.")
    args = ap.parse_args()

    cards = load_cards()
    print(f"loaded {len(cards)} cards from {CARDS}")

    todo = []
    for i, c in enumerate(cards):
        if not args.force and has_evidence(c):
            continue
        if not c.get("source_text") or not c.get("genome"):
            continue
        todo.append((i, c))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} cards need evidence extraction "
          f"(limit={args.limit}, force={args.force})")
    if not todo:
        print("nothing to do.")
        return

    # build calls
    calls = []
    for i, c in todo:
        gnm = c["genome"]
        user = USER_TMPL.format(
            source_text=c["source_text"][:8000],          # safety cap
            mechanism_genome=gnm.get("mechanism_genome", ""),
            niche_genome=gnm.get("niche_genome", ""),
            observation_genome=gnm.get("observation_genome", ""),
            limitation_genome=gnm.get("limitation_genome", ""),
            delta_genome=gnm.get("delta_genome", ""),
            claim_genome=gnm.get("claim_genome", ""),
        )
        calls.append(TeacherCall(
            prompt_id=f"ev::{c['card_id']}",
            messages=[{"role": "system", "content": SYSTEM},
                       {"role": "user",   "content": user}],
            max_tokens=2048,
            metadata={"card_idx": i, "card_id": c["card_id"]},
        ))

    print(f"firing {len(calls)} GPT-5.5 calls (workers={args.workers})")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    t0 = time.time()

    def cb(d, t):
        print(f"  teacher: {d}/{t}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    results = batch_call(calls, workers=args.workers, log_path=LOG, on_progress=cb)
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    # patch cards
    by_card_id = {(r.metadata or {}).get("card_id"): r for r in results}
    stats = Counter()
    for i, c in todo:
        r = by_card_id.get(c["card_id"])
        if r is None or r.error or not r.content:
            stats["api_error"] += 1
            continue
        parsed = parse_response(r.content)
        if parsed is None:
            stats["parse_error"] += 1
            continue
        verified = verify_quotes(parsed, c["source_text"])
        n_quotes = sum(len(v) for v in verified.values())
        if n_quotes == 0:
            stats["no_quotes_matched"] += 1
        else:
            stats["accept"] += 1
        c["evidence"] = verified
        # bump verifier evidence subscore based on coverage
        n_fields_with_evidence = sum(1 for v in verified.values() if v)
        c.setdefault("verifier", {})["evidence_grounded"] = round(
            n_fields_with_evidence / len(GENE_FIELDS), 3
        )

    write_cards(cards)
    print(f"\npatched cards in place ({CARDS})")
    print(f"stats: {dict(stats)}")
    n_with_ev = sum(1 for c in cards if has_evidence(c))
    print(f"now {n_with_ev}/{len(cards)} cards have at least one evidence quote")


if __name__ == "__main__":
    main()
