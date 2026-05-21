"""Generate Stage 1 SFT data using GPT-5.5 over the safe paper pool.

For each safe paper with an abstract (or full text), run one or more teacher
templates and filter via the verifier. Save to data/stage1_sft/train.jsonl.

Target: ~5000 examples balanced across the 5 SFT task types.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.teachers.prompts import build_messages
from evo_opd.verifier import compute_verifier

SAFE_POOL = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/safe_pool/safe_pool_v0.jsonl")
OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_safe_papers(min_chars: int = 200) -> list[dict]:
    """Load papers that have either abstract or full text long enough to be useful."""
    papers: list[dict] = []
    with SAFE_POOL.open() as f:
        for line in f:
            r = json.loads(line)
            text_len = len(r.get("abstract") or "") + (5000 if r.get("has_full_text") else 0)
            if text_len >= min_chars:
                papers.append(r)
    return papers


def paper_text(p: dict) -> str:
    """Build the paper text we'll feed to teacher."""
    bits = []
    if p.get("title"):
        bits.append(f"TITLE: {p['title']}")
    if p.get("year"):
        bits.append(f"YEAR: {p['year']}")
    if p.get("domain_hint"):
        bits.append(f"DOMAIN: {p['domain_hint']}")
    if p.get("abstract"):
        bits.append(f"ABSTRACT:\n{p['abstract']}")
    if p.get("key_contribution"):
        bits.append(f"KEY CONTRIBUTION: {p['key_contribution']}")
    if not p.get("abstract") and p.get("idea_genome_existing"):
        # use existing gene fields as proxy abstract content
        ig = p["idea_genome_existing"]
        if isinstance(ig, dict):
            for k in ("mechanism_genome", "niche_genome", "observation_genome",
                      "limitation_genome", "claim_genome"):
                v = ig.get(k)
                if v:
                    bits.append(f"{k.upper()}: {v}")
    return "\n\n".join(bits)


def build_calls_gene_card(papers: list[dict], n: int) -> list[TeacherCall]:
    calls: list[TeacherCall] = []
    for p in random.sample(papers, min(n, len(papers))):
        text = paper_text(p)
        if len(text) < 200:
            continue
        msgs = build_messages("gene_card_extract", paper_text=text[:6000])
        calls.append(TeacherCall(
            prompt_id=f"gene_card::{p['safe_paper_id']}",
            messages=msgs,
            max_tokens=2000,
            metadata={
                "task_type": "gene_card_extract",
                "source_paper_id": p["safe_paper_id"],
                "source_text": text[:6000],  # for verifier evidence check
                "domain": p.get("domain_hint", ""),
                "year": p.get("year"),
            },
        ))
    return calls


def build_calls_idea_generate(papers: list[dict], n: int) -> list[TeacherCall]:
    """For idea_generate, we group 3-5 papers from same domain into a 'lineage'."""
    by_domain: dict[str, list[dict]] = {}
    for p in papers:
        if p.get("abstract"):
            dh = p.get("domain_hint")
            if isinstance(dh, list):
                dh = dh[0] if dh else "misc"
            elif not isinstance(dh, str):
                dh = "misc"
            by_domain.setdefault(dh, []).append(p)
    calls: list[TeacherCall] = []
    domains = [d for d, ps in by_domain.items() if len(ps) >= 3]
    random.shuffle(domains)
    for d in domains:
        if len(calls) >= n:
            break
        for _ in range(min(3, max(1, n // max(len(domains), 1)))):
            chosen = random.sample(by_domain[d], min(4, len(by_domain[d])))
            cards_text = "\n\n".join(
                f"PAPER {i+1}: {p['title']} ({p.get('year','?')})\n"
                f"  Mechanism: {(p.get('idea_genome_existing') or {}).get('mechanism_genome') or p.get('abstract','')[:200]}\n"
                for i, p in enumerate(chosen)
            )
            msgs = build_messages("idea_generate", lineage_text=cards_text,
                                  open_question=f"What is the next frontier idea in {d}?")
            calls.append(TeacherCall(
                prompt_id=f"idea_gen::{d}::{chosen[0]['safe_paper_id']}",
                messages=msgs,
                max_tokens=1500,
                metadata={"task_type": "idea_generate", "domain": d,
                          "source_paper_ids": [c["safe_paper_id"] for c in chosen]},
            ))
    return calls[:n]


def verify_and_save(results, out_path: Path) -> dict:
    """Filter results by verifier and write the survivors as SFT examples."""
    out_fp = out_path.open("a")
    stats = Counter()
    for r in results:
        if r.error or not r.content:
            stats["api_error"] += 1
            continue
        md = r.metadata or {}
        task_type = md.get("task_type", "unknown")
        gold = md.get("gold_answer")  # None for free-form tasks
        source_text = md.get("source_text")
        score, _ = compute_verifier(r.content, task_type,
                                     source_text=source_text, gold_answer=gold)
        # Accept threshold: schema must be valid + (if applicable) evidence ≥ 0.5
        accept = score.schema_valid >= 1.0
        if score.evidence_citation_frac and score.evidence_citation_frac < 0.5:
            accept = False
        if not accept:
            stats[f"reject_{task_type}"] += 1
            continue
        stats[f"accept_{task_type}"] += 1
        record = {
            "instance_id": r.prompt_id,
            "task_type": task_type,
            "prompt": r.metadata.get("source_text", "")[:200] if task_type == "gene_card_extract" else "",
            "messages": [],  # populated below
            "completion": r.content,
            "metadata": {
                **md,
                "teacher_model": "gpt-5.5",
                "teacher_input_tokens": r.input_tokens,
                "teacher_output_tokens": r.output_tokens,
                "teacher_latency_ms": r.latency_ms,
                "verifier_score": score.to_dict(),
            },
        }
        # Reconstruct messages from prompt template for training
        # (the actual prompt sent to teacher is what we'd use as 'prompt' field)
        out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_fp.close()
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gene-card", type=int, default=2500, help="number of gene_card_extract calls")
    ap.add_argument("--n-idea-gen", type=int, default=500, help="number of idea_generate calls")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output) if args.output else OUT_DIR / "train.jsonl"
    teacher_log_path = OUT_DIR.parent / "teacher_logs" / "gpt55_sft_gen.jsonl"

    print(f"[1/4] Loading safe papers")
    papers = load_safe_papers()
    print(f"  {len(papers):,} papers eligible (have abstract or full text)")

    print(f"[2/4] Building call list")
    calls: list[TeacherCall] = []
    if args.n_gene_card > 0:
        calls += build_calls_gene_card(papers, args.n_gene_card)
        print(f"  +{args.n_gene_card} gene_card_extract calls")
    if args.n_idea_gen > 0:
        calls += build_calls_idea_generate(papers, args.n_idea_gen)
        print(f"  +{args.n_idea_gen} idea_generate calls")
    print(f"  total: {len(calls):,} teacher calls")

    print(f"[3/4] Running teacher (workers={args.workers})")
    def progress(d, t):
        print(f"  teacher progress: {d}/{t}", flush=True)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers, log_path=teacher_log_path,
                         on_progress=progress)
    el = time.time() - t0
    n_ok = sum(1 for r in results if r.content and not r.error)
    n_err = sum(1 for r in results if r.error)
    print(f"  done in {el/60:.1f}min: {n_ok} OK / {n_err} errors")

    print(f"[4/4] Verifying + writing SFT examples")
    out_path.write_text("")  # truncate
    stats = verify_and_save(results, out_path)
    n_total = sum(stats.values())
    n_accept = sum(v for k, v in stats.items() if k.startswith("accept_"))
    print(f"  acceptance rate: {n_accept}/{n_total} ({100*n_accept/max(n_total,1):.1f}%)")
    print(f"  detail: {stats}")
    print(f"Wrote: {out_path}  ({n_accept} examples)")

    # save stats
    stats_path = OUT_DIR / "sft_gen_stats.json"
    with stats_path.open("w") as f:
        json.dump({
            "n_calls": len(calls),
            "n_api_ok": n_ok,
            "n_api_err": n_err,
            "n_accepted": n_accept,
            "acceptance_rate": n_accept / max(n_total, 1),
            "elapsed_seconds": el,
            "detail": dict(stats),
        }, f, indent=2)
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
