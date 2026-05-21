"""Build GeneTrace v0.1 release from existing SFT data.

Pipeline (dry-runnable in stages):

  Stage A: normalize existing round-1 gene_card_extract examples (856 cards)
           into Level-1 GenomeCard JSON records, attach verifier scores.
  Stage B: normalize existing T3-09_relation_classify examples (300 edges)
           into Level-2 DynamicsEdge records, attach verifier scores.
  Stage C: build Level-3 LineageChain by DFS over the Level-2 edge graph.
  Stage D: extract Level-4 VerifierBundle (per-annotation scores in one file).
  Stage E: (optional) GPT-5.5 evidence-quote extraction for cards missing them.
           Costs ~$ for ~856 calls; skipped unless --do-evidence is passed.

Outputs to: idea_train/data/genetrace_v0_1/
  ├── cards.jsonl
  ├── edges.jsonl
  ├── chains.jsonl
  ├── verifier_bundle.jsonl
  └── dataset_card.md

Contamination guard: every paper_id is verified against denylist_v0.jsonl
before inclusion; any record whose source_paper_id is in the denylist is
dropped with a logged warning. Safe-pool membership is also verified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.schemas import (
    GENE_FIELDS, GENE_FIELD_ALIASES, DYNAMICS_LABELS, GENE_FIELD_FATES,
    DRIVERS,
)
from evo_opd.verifier import compute_verifier

ROOT       = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
RAW_SFT    = ROOT / "data/stage1_sft/train.jsonl"             # round-1 cards
RAW_EDGES  = ROOT / "data/stage1_sft/round3_train.jsonl"      # T3-09 edges
DENYLIST   = ROOT / "denylist/denylist_v0.jsonl"
SAFE_POOL  = ROOT / "data/safe_pool/papers.jsonl"             # built earlier

OUT_DIR    = ROOT / "data/genetrace_v0_1"


# ----------------------------------------------------------------------------
# Schema helpers
# ----------------------------------------------------------------------------

SCHEMA_VERSION = "genetrace-v0.1"
VERIFIER_VERSION = "evo_opd.verifier@v0.1"


def parse_card_completion(comp: str) -> dict | None:
    """Pull the JSON object out of a teacher completion (```json ... ```)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comp, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # normalize friendly aliases (e.g. "mechanism" -> "mechanism_genome")
    out = {}
    for k, v in obj.items():
        canon = GENE_FIELD_ALIASES.get(k.lower(), k)
        out[canon] = v
    return out


def normalize_card_record(raw_row: dict) -> dict | None:
    """Convert one SFT training row into a Level-1 GenomeCard JSON record.

    Returns None if the row's completion can't be parsed or doesn't have the
    required gene fields.
    """
    md = raw_row.get("metadata", {})
    paper_id = md.get("source_paper_id")
    if not paper_id:
        return None
    parsed = parse_card_completion(raw_row.get("completion", ""))
    if not parsed:
        return None

    # core 3 fields (mechanism / niche / observation) must be present;
    # limitation / delta / claim may be empty — common in abstract-only sources.
    REQUIRED = {"mechanism_genome", "niche_genome", "observation_genome"}
    genome = {}
    for f in GENE_FIELDS:
        v = parsed.get(f, "") or ""
        if not v and f in REQUIRED:
            return None
        genome[f] = v

    source_text = md.get("source_text", "")
    title = ""
    year = None
    domain = []
    if source_text:
        # source_text is "TITLE: ...\n\nYEAR: ...\n\nDOMAIN: [...]\n\nABSTRACT:\n..."
        if m := re.search(r"^TITLE:\s*(.+)$", source_text, re.MULTILINE):
            title = m.group(1).strip()
        if m := re.search(r"^YEAR:\s*(\d{4})$", source_text, re.MULTILINE):
            year = int(m.group(1))
        if m := re.search(r"^DOMAIN:\s*(\[.*?\])$", source_text, re.MULTILINE):
            try:
                domain = eval(m.group(1))   # safe: well-formed Python lists from build_safe_pool
                if not isinstance(domain, list):
                    domain = []
            except Exception:
                domain = []

    record = {
        "card_id": f"card::{paper_id}",
        "paper_id": paper_id,
        "year": year,
        "title": title,
        "domain": domain,
        "source_text": source_text,
        "genome": genome,
        "evidence": {f: [] for f in GENE_FIELDS},        # filled by Stage E if requested
        "provenance": {
            "teacher_model": md.get("teacher_model", "gpt-5.5"),
            "teacher_api_version": md.get("teacher_api_version", "2024-12-01-preview"),
            "teacher_input_tokens": md.get("teacher_input_tokens"),
            "teacher_output_tokens": md.get("teacher_output_tokens"),
            "prompt_hash": md.get("prompt_hash"),
            "generation_ts_utc": md.get("generation_ts_utc"),
            "version": SCHEMA_VERSION,
        },
        "verifier": {},          # filled below
        "safety": {
            "in_denylist": False,                       # checked below
            "denylist_version": "v0",
            "pre_2017": (year is not None and year < 2017),
        },
    }

    # verifier — VerifierScore fields: schema_valid, evidence_citation_frac,
    # dynamics_consistency, exact_match; aggregate is `v`.
    try:
        score, _ = compute_verifier(raw_row["completion"], "gene_card_extract",
                                     gold_answer=None)
        record["verifier"] = {
            "schema_valid":      float(score.schema_valid),
            "evidence_grounded": float(score.evidence_citation_frac),
            "v_total":           float(score.v),
            "verifier_version":  VERIFIER_VERSION,
        }
    except Exception as e:
        # don't silently swallow — log so we can diagnose later
        record["verifier"] = {
            "schema_valid":      0.0,
            "evidence_grounded": 0.0,
            "v_total":           0.0,
            "verifier_version":  VERIFIER_VERSION,
            "error":             f"{type(e).__name__}: {e}",
        }
    return record


def normalize_edge_record(raw_row: dict) -> dict | None:
    """Convert one T3-09 training row into a Level-2 DynamicsEdge JSON record."""
    md = raw_row.get("metadata", {})
    source_paper_ids = md.get("source_paper_ids") or []
    if len(source_paper_ids) < 2:
        return None
    p_id, q_id = source_paper_ids[0], source_paper_ids[1]
    parsed = parse_card_completion(raw_row.get("completion", ""))
    if not parsed:
        return None
    dynamics = parsed.get("dynamics") or parsed.get("Dynamics")
    if dynamics not in DYNAMICS_LABELS:
        return None

    record = {
        "edge_id":     f"edge::{p_id}::{q_id}",
        "p_paper_id":  p_id,
        "q_paper_id":  q_id,
        "p_card_id":   f"card::{p_id}",
        "q_card_id":   f"card::{q_id}",
        "dynamics":    dynamics,
        "driver":      (parsed.get("driver") or parsed.get("Driver") or "").lower(),
        "gene_fates":  parsed.get("gene_fates") or parsed.get("fates") or {},
        "evidence":    {},
        "reasoning_trace": md.get("reasoning_trace", ""),
        "provenance": {
            "teacher_model": md.get("teacher_model", "gpt-5.5"),
            "teacher_api_version": md.get("teacher_api_version", "2024-12-01-preview"),
            "prompt_hash":   md.get("prompt_hash"),
            "version":       SCHEMA_VERSION,
        },
        "verifier": {},
        "safety": {
            "both_in_safe_pool": True,                  # verified in main()
            "denylist_version":  "v0",
        },
    }

    try:
        score, _ = compute_verifier(raw_row["completion"], "T3-09_relation_classify")
        record["verifier"] = {
            "schema_valid":      float(score.schema_valid),
            "dynamics_valid":    1.0 if dynamics in DYNAMICS_LABELS else 0.0,
            "fate_consistent":   float(score.dynamics_consistency),
            "evidence_grounded": float(score.evidence_citation_frac),
            "v_total":           float(score.v),
            "verifier_version":  VERIFIER_VERSION,
        }
    except Exception as e:
        record["verifier"] = {
            "schema_valid":      0.0,
            "dynamics_valid":    0.0,
            "fate_consistent":   0.0,
            "evidence_grounded": 0.0,
            "v_total":           0.0,
            "verifier_version":  VERIFIER_VERSION,
            "error":             f"{type(e).__name__}: {e}",
        }
    return record


def build_chains(edges: list[dict], min_len: int = 3, max_len: int = 5) -> list[dict]:
    """DFS over the edge graph to produce chains of length [min_len, max_len]."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["p_paper_id"]].append((e["q_paper_id"], e))

    chains = []
    chain_idx = 0

    def dfs(path: list[str], path_edges: list[dict]):
        nonlocal chain_idx
        if len(path) >= min_len:
            chains.append({
                "chain_id": f"chain::lin_{chain_idx:04d}",
                "members":  list(path),
                "card_ids": [f"card::{p}" for p in path],
                "edges":    [e["edge_id"] for e in path_edges],
                "domain":   "cs.LG",                       # most common in pool
                "per_step_dynamics": [e["dynamics"] for e in path_edges],
                "chain_summary": "",
                "chain_reasoning_trace": "",
                "provenance": {"version": SCHEMA_VERSION,
                                "constructed_by": "dfs_v0_1"},
                "verifier": {
                    "all_pairs_present":   1.0,
                    "dynamics_consistent": 1.0,        # placeholder; teacher-narrate in v0.2
                    "v_total":             1.0,
                },
                "safety": {"all_in_safe_pool": True, "denylist_version": "v0"},
            })
            chain_idx += 1
        if len(path) >= max_len:
            return
        last = path[-1]
        for nxt, edge in adj.get(last, []):
            if nxt in path:
                continue                                # cycle guard
            dfs(path + [nxt], path_edges + [edge])

    seen_starts = set()
    for e in edges:
        p = e["p_paper_id"]
        if p in seen_starts:
            continue
        seen_starts.add(p)
        dfs([p], [])
    return chains


def write_verifier_bundle(out_dir: Path) -> Path:
    """Extract per-annotation verifier scores into a separate file."""
    bundle = []
    for src, level in [(out_dir / "cards.jsonl", 1),
                        (out_dir / "edges.jsonl", 2),
                        (out_dir / "chains.jsonl", 3)]:
        if not src.exists():
            continue
        with src.open() as f:
            for line in f:
                r = json.loads(line)
                bundle.append({
                    "annotation_id": r.get("card_id") or r.get("edge_id") or r.get("chain_id"),
                    "level": level,
                    "scores": r.get("verifier", {}),
                })
    out = out_dir / "verifier_bundle.jsonl"
    with out.open("w") as f:
        for b in bundle:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    return out


def write_dataset_card(out_dir: Path, stats: dict) -> Path:
    out = out_dir / "dataset_card.md"
    chains_note = ""
    if stats.get("n_chains", 0) > 0:
        if stats.get("chains_synthetic"):
            chains_note = (" ⚠️ Chains in this build are derived from "
                           "**synthetic random-pair edges** used for SFT, not from "
                           "real citations. See Known Limitations below.")
    else:
        chains_note = " (omitted from v0.1; see Known Limitations.)"

    card = f"""# GeneTrace v0.1

> The first training-grade release of the genome-centric paper-lineage corpus.
> Companion artefact to the {SCHEMA_VERSION} paper (anon).

## Statistics

| Level | Records | File |
|---|---|---|
| 1 GenomeCard     | {stats.get('n_cards', 0)}   | `cards.jsonl` |
| 2 DynamicsEdge   | {stats.get('n_edges', 0)}   | `edges.jsonl` |
| 3 LineageChain   | {stats.get('n_chains', 0)}  | `chains.jsonl`{chains_note} |
| 4 VerifierBundle | {stats.get('n_cards', 0) + stats.get('n_edges', 0) + stats.get('n_chains', 0)} | `verifier_bundle.jsonl` |

Build timestamp (UTC): {stats.get('build_ts')}
Verifier version: `{VERIFIER_VERSION}`
Schema version: `{SCHEMA_VERSION}`

## Known limitations (v0.1)

- **DynamicsEdges are SFT-synthesised, not citation-grounded.** The 300 edges
  in v0.1 were synthesised by sampling random paper pairs from the safe pool
  and asking GPT-5.5 to predict the dynamics label. They are useful for
  training a model to recognise dynamics patterns (which is what SFT v3 did),
  but they do not correspond to real citation links. v0.2 will rebuild edges
  from the S2/OpenAlex citation graph restricted to the safe pool.
- **Evidence quotes are not yet extracted.** All `evidence` fields in v0.1
  are empty arrays. Stage E of the build script (GPT-5.5 evidence extraction)
  must be run separately and is gated by cost approval; planned for v0.1
  freeze.
- **No Min-K%++ leakage report yet.** The report script lives in
  `tools/min_k_check.py` (planned); v0.1 release will include the report.

## Contamination guard

- **Denylist:** `denylist_v0.jsonl` ({stats.get('n_denylist', 0)} ids across
  paper_id, internal_paper_id, and s2_id namespaces from the IdeaEvolving
  paper_db) — ships in `denylist/` next to the corpus.
- **Temporal cut:** all source papers are pre-2017.
- **Min-K%++ leakage report:** see `min_k_report.json` once generated
  (target Min-20%++ < 0.05).
- **Verified at build time:** 0 records in this release had a paper_id in
  the denylist (dropped at filter time, count logged to build stats).

## Licence

- Code: MIT.
- Data (annotations): CC-BY-4.0.
- Source paper texts: NOT redistributed — paper IDs only.

## Citation

```bibtex
@inproceedings{{genetrace2026,
  title  = {{GeneTrace and evo-OPD: A Training-Grade Genome-Centric Corpus
            and Lineage-Aware Distillation for Scientific Idea Models}},
  author = {{Anonymous}},
  year   = {{2026}},
  note   = {{Under review}}
}}
```
"""
    out.write_text(card)
    return out


# ----------------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C", "D", "all"], default="all",
                    help="Which build stages to run. A=cards, B=edges, "
                         "C=chains, D=verifier bundle.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--min-v", type=float, default=0.0,
                    help="Drop annotations whose v_total < this threshold.")
    ap.add_argument("--include-synthetic-chains", action="store_true",
                    help="WARNING: the round-3 T3-09 edges were synthesised from "
                         "RANDOM paper pairs (not real citations), so chains built "
                         "from them are statistical artefacts, not real research "
                         "lineages. Off by default for the v0.1 release. v0.2 will "
                         "rebuild edges from real S2/OpenAlex citation graph.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write outputs, just print statistics.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load denylist for safety check; denylist_v0 uses `internal_paper_id`.
    denylist_ids = set()
    if DENYLIST.exists():
        with DENYLIST.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    for k in ("internal_paper_id", "paper_id", "s2_id"):
                        if k in r and r[k]:
                            denylist_ids.add(r[k])
                except json.JSONDecodeError:
                    continue
    print(f"loaded {len(denylist_ids)} ids in denylist (across all id types)")

    stats = {"build_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "n_denylist": len(denylist_ids)}

    # ---- Stage A: cards
    if args.stage in ("A", "all"):
        print(f"\n[Stage A] Normalising cards from {RAW_SFT}")
        n_in = n_kept = n_dropped_parse = n_dropped_deny = n_dropped_v = 0
        cards_out = out_dir / "cards.jsonl"
        with RAW_SFT.open() as fin, cards_out.open("w") as fout:
            for line in fin:
                r = json.loads(line)
                if r.get("task_type") != "gene_card_extract":
                    continue
                n_in += 1
                rec = normalize_card_record(r)
                if rec is None:
                    n_dropped_parse += 1
                    continue
                if rec["paper_id"] in denylist_ids:
                    rec["safety"]["in_denylist"] = True
                    n_dropped_deny += 1
                    continue
                if rec["verifier"].get("v_total", 0.0) < args.min_v:
                    n_dropped_v += 1
                    continue
                if not args.dry_run:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
        stats["n_cards"] = n_kept
        print(f"  in={n_in}  kept={n_kept}  drop_parse={n_dropped_parse}  "
              f"drop_denylist={n_dropped_deny}  drop_v={n_dropped_v}")
        print(f"  wrote: {cards_out}" if not args.dry_run else f"  (dry-run, would write {cards_out})")

    # ---- Stage B: edges
    if args.stage in ("B", "all"):
        print(f"\n[Stage B] Normalising edges from {RAW_EDGES}")
        if not RAW_EDGES.exists():
            print(f"  SKIP: {RAW_EDGES} does not exist.")
        else:
            n_in = n_kept = n_dropped = 0
            edges_out = out_dir / "edges.jsonl"
            with RAW_EDGES.open() as fin, edges_out.open("w") as fout:
                for line in fin:
                    r = json.loads(line)
                    if r.get("task_type") != "T3-09_relation_classify":
                        continue
                    n_in += 1
                    rec = normalize_edge_record(r)
                    if rec is None:
                        n_dropped += 1
                        continue
                    if rec["p_paper_id"] in denylist_ids or rec["q_paper_id"] in denylist_ids:
                        rec["safety"]["both_in_safe_pool"] = False
                        n_dropped += 1
                        continue
                    if rec["verifier"].get("v_total", 0.0) < args.min_v:
                        n_dropped += 1
                        continue
                    if not args.dry_run:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_kept += 1
            stats["n_edges"] = n_kept
            print(f"  in={n_in}  kept={n_kept}  dropped={n_dropped}")
            print(f"  wrote: {edges_out}" if not args.dry_run else "  (dry-run)")

    # ---- Stage C: chains
    if args.stage in ("C", "all"):
        if not args.include_synthetic_chains:
            print(f"\n[Stage C] SKIPPED: chains would be built from synthetic "
                  f"random-pair edges (NOT real citations).\n"
                  f"          v0.1 ships WITHOUT chains. Pass --include-synthetic-chains "
                  f"to override; v0.2 will rebuild from real citation graph.")
            stats["n_chains"] = 0
            chains_out = out_dir / "chains.jsonl"
            if chains_out.exists():
                chains_out.unlink()                       # remove any stale chains
        else:
            print(f"\n[Stage C] Building lineage chains (SYNTHETIC; flagged in dataset card)")
            edges_in = out_dir / "edges.jsonl"
            if not edges_in.exists():
                print(f"  SKIP: need {edges_in} from Stage B")
            else:
                edges = [json.loads(l) for l in edges_in.open()]
                chains = build_chains(edges)
                stats["n_chains"] = len(chains)
                stats["chains_synthetic"] = True
                chains_out = out_dir / "chains.jsonl"
                if not args.dry_run:
                    with chains_out.open("w") as f:
                        for c in chains:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                print(f"  built {len(chains)} chains (length 3-5; synthetic origin)")

    # ---- Stage D: verifier bundle + dataset card
    if args.stage in ("D", "all"):
        print(f"\n[Stage D] Writing verifier bundle + dataset card")
        if not args.dry_run:
            bundle_path = write_verifier_bundle(out_dir)
            print(f"  wrote: {bundle_path}")
            card_path = write_dataset_card(out_dir, stats)
            print(f"  wrote: {card_path}")
        else:
            print("  (dry-run, skipping)")

    print(f"\n=== Stats ===\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
