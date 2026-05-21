"""Build safe paper pool for SFT teacher generation (avoids denylist).

Selection rules (must satisfy ALL):
  - Paper is in IdeaEvolving/data/paper_db/paper_db.json
  - Has a non-empty abstract OR title at minimum
  - NOT in denylist_v0 (matched by s2_id OR normalized title+year)
  - Year ≤ 2017 (pre-dates the main GENE-bench seed window) OR
    domain NOT in the 15 covered GENE-bench domains (cs, biology, chemistry,
    physics, math, medicine, neuroscience, energy, materials, climate,
    earth_science, agriculture, astronomy, ecology, science)

For Stage 1 SFT data generation we want diverse papers with at least some
content. The output JSONL has one paper per line with fields ready for the
teacher prompts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
DEN = Path("/home/azureuser/workspace-gzy/zyf/idea_train/denylist/denylist_v0.jsonl")
OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/safe_pool/safe_pool_v0.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

COVERED_DOMAINS = {
    "cs", "biology", "chemistry", "physics", "math", "medicine",
    "neuroscience", "energy", "materials", "climate", "earth_science",
    "agriculture", "astronomy", "ecology", "science",
}
SAFE_YEAR_CUTOFF = 2017


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main() -> None:
    print("[1/3] Loading denylist v0 keys")
    deny_s2 = set()
    deny_titles = set()  # (norm_title, year)
    with DEN.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("s2_id"):
                deny_s2.add(r["s2_id"])
            t = norm_title(r.get("title"))
            y = r.get("year")
            if t:
                deny_titles.add((t, y))
    print(f"  denylist: {len(deny_s2):,} s2_ids + {len(deny_titles):,} title+year pairs")

    print("[2/3] Loading paper_db.json")
    with (REPO / "data/paper_db/paper_db.json").open() as f:
        pdb = json.load(f)
    print(f"  paper_db: {len(pdb):,} papers")

    print("[3/3] Filtering safe pool")
    n_total = n_safe = 0
    drop_reasons = Counter()
    safe_records: list[dict] = []
    for pid, rec in pdb.items():
        n_total += 1
        s2 = rec.get("s2_id")
        title = rec.get("title")
        year = rec.get("year")
        domain = rec.get("source") or rec.get("domain") or ""  # paper_db uses "source" as domain-ish

        # denylist hit?
        if s2 and s2 in deny_s2:
            drop_reasons["denylist_s2"] += 1
            continue
        nt = norm_title(title)
        if (nt, year) in deny_titles or (nt and (nt, None) in deny_titles):
            drop_reasons["denylist_title"] += 1
            continue

        # need at least title or abstract
        if not title or not isinstance(title, str) or len(title) < 5:
            drop_reasons["no_title"] += 1
            continue

        # Safety criterion: pre-2017 OR non-covered domain
        try:
            y_int = int(year) if year else None
        except (ValueError, TypeError):
            y_int = None
        domain_norm = domain.lower() if isinstance(domain, str) else ""

        is_safe = False
        if y_int is not None and y_int <= SAFE_YEAR_CUTOFF:
            is_safe = True
            safe_reason = f"pre_{SAFE_YEAR_CUTOFF}"
        elif domain_norm and domain_norm not in COVERED_DOMAINS:
            is_safe = True
            safe_reason = f"domain_{domain_norm}"
        else:
            drop_reasons["covered_recent"] += 1
            continue

        n_safe += 1
        safe_records.append({
            "safe_paper_id": pid,
            "s2_id": s2,
            "title": title,
            "year": year,
            "domain_hint": domain,
            "abstract": rec.get("abstract") or "",
            "key_contribution": rec.get("key_contribution") or "",
            "idea_genome_existing": rec.get("idea_genome"),
            "has_full_text": rec.get("has_full_text", False),
            "safe_reason": safe_reason,
        })

    with OUT.open("w") as f:
        for r in safe_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"=== Safe pool stats ===")
    print(f"  total scanned : {n_total:,}")
    print(f"  safe         : {n_safe:,}  ({100*n_safe/n_total:.1f}%)")
    print(f"  drop reasons :")
    for k, v in drop_reasons.most_common():
        print(f"    {k:<20} {v:>6}")
    # show fraction by safe_reason and presence of abstract
    by_reason = Counter(r["safe_reason"] for r in safe_records)
    has_abs = sum(1 for r in safe_records if r["abstract"])
    has_text = sum(1 for r in safe_records if r["has_full_text"])
    print(f"  by safe_reason: {dict(by_reason)}")
    print(f"  with abstract : {has_abs:,} ({100*has_abs/max(n_safe,1):.1f}%)")
    print(f"  with full text: {has_text:,} ({100*has_text/max(n_safe,1):.1f}%)")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
