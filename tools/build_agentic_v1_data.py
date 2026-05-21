"""Build agentic-OPD training data from GeneTrace v0.1.

For each (parent → child) edge in edges.jsonl, construct a training prompt:

  topic   = derived from child's niche_genome (problem domain, no mechanism leak)
  target  = the child paper itself (kept as "intended follow-up" reference;
            not shown to the agent during rollout)
  lineage = ancestors-closure of the child (multi-hop walk over edges)
  gold_proposal = child's gene_genome (used as reference for R_struct similarity)

Output files:
  data/agentic_v1/
    prompts.jsonl       — one prompt per row, see schema below
    tool_corpus.jsonl   — 855 cards reformatted for the `read` tool
    bm25_corpus.jsonl   — 855 cards with searchable text for the `search` tool
    stats.json          — summary statistics

Prompt schema:
{
  "prompt_id": "agentic_v1::p_0042",
  "topic": "...",                    // shown to agent
  "discipline": "cs",                // hint for tool wrappers
  "year_min_hint": 2018,             // search-window hint
  "target_paper_id": "paper:foo:2024",  // NOT shown to agent; for gold_lineage scoring
  "gold_lineage": ["paper:bar:2019", ...],  // ancestor closure of target, used for R_lineage
  "gold_proposal": { ...gene_genome... },   // used for R_struct comparison
  "parent_card_compressed": { title, niche, abstract_excerpt }  // for fallback when agent doesn't search
}
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

GENETRACE_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/genetrace_v0_1")
OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v1")


def load_cards() -> dict[str, dict]:
    """paper_id -> card dict"""
    cards = {}
    with (GENETRACE_DIR / "cards.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            cards[c["paper_id"]] = c
    return cards


def load_edges() -> list[dict]:
    edges = []
    with (GENETRACE_DIR / "edges.jsonl").open() as f:
        for line in f:
            edges.append(json.loads(line))
    return edges


def build_ancestor_closure(edges: list[dict], max_depth: int = 4) -> dict[str, list[str]]:
    """For each paper, walk backwards along (p → q) edges to collect ancestors."""
    parents_of: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        parents_of[e["q_paper_id"]].add(e["p_paper_id"])

    ancestors: dict[str, list[str]] = {}
    for q in parents_of:
        seen: list[str] = []
        seen_set: set[str] = set()
        frontier = deque([(q, 0)])
        while frontier:
            cur, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for p in parents_of.get(cur, []):
                if p in seen_set:
                    continue
                seen.append(p)
                seen_set.add(p)
                frontier.append((p, depth + 1))
        ancestors[q] = seen
    return ancestors


def infer_discipline(card: dict) -> str:
    """Coarse domain from card.domain field + title heuristics."""
    dom = card.get("domain") or []
    if not dom:
        return "general"
    if dom[0] == "corpus_recovered":
        title = (card.get("title") or "").lower()
        # quick heuristic mapping for "corpus_recovered" cards
        for kw, disc in [
            ("rna|protein|cell|gene|molecul", "biology"),
            ("brain|neur|cortex|synap", "neuroscience"),
            ("diffusion|gan|transformer|llm|vision|speech|reinforce|robot|agent|reasoning", "cs"),
            ("crystal|catalys|battery|solar|material", "materials"),
            ("climate|atmosp|ocean|earthquake|geolog", "earth_science"),
            ("quantum|particle|physics|cosmol|astro", "physics"),
            ("chem|reaction|molecul|drug", "chemistry"),
            ("math|theorem|optim|combinator", "mathematics"),
        ]:
            if re.search(kw, title):
                return disc
        return "cs"  # default to CS for ambiguous corpus_recovered
    return str(dom[0]).lower()


def make_topic(card: dict) -> str:
    """Build a `topic` string that gives the agent enough to search, without
    leaking the mechanism or the specific paper."""
    niche = (card.get("genome") or {}).get("niche_genome", "")
    domain = infer_discipline(card)
    if niche:
        return f"[{domain}] {niche.strip()}"
    return f"[{domain}] (no niche specified) — {card.get('title', '')[:120]}"


def make_compressed_parent_card(card: dict) -> dict:
    """Lightweight subset of parent card used as fallback context."""
    genome = card.get("genome") or {}
    return {
        "paper_id": card["paper_id"],
        "title": card.get("title", ""),
        "year": card.get("year"),
        "niche": genome.get("niche_genome", "")[:300],
        "abstract_excerpt": (card.get("source_text") or "")[200:600],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ancestors", type=int, default=1,
                    help="prompts must have at least this many ancestors in lineage closure")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--max-ancestor-depth", type=int, default=4)
    args = ap.parse_args()

    print(f"[1/4] loading GeneTrace v0.1 from {GENETRACE_DIR}")
    cards = load_cards()
    edges = load_edges()
    print(f"  {len(cards)} cards, {len(edges)} edges")

    print(f"[2/4] computing ancestor closure (max_depth={args.max_ancestor_depth})")
    ancestors = build_ancestor_closure(edges, max_depth=args.max_ancestor_depth)
    n_with_ancestors = sum(1 for v in ancestors.values() if len(v) >= args.min_ancestors)
    depth_hist = defaultdict(int)
    for v in ancestors.values():
        depth_hist[len(v)] += 1
    print(f"  {n_with_ancestors} papers with ≥{args.min_ancestors} ancestor(s)")
    print(f"  ancestry length histogram (top 10): "
          f"{dict(sorted(depth_hist.items())[:10])}")

    print(f"[3/4] building prompts")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path = OUT_DIR / "prompts.jsonl"
    n_prompts = 0
    discipline_counts = defaultdict(int)
    with prompts_path.open("w") as f:
        for paper_id, anc_list in ancestors.items():
            if len(anc_list) < args.min_ancestors:
                continue
            if paper_id not in cards:
                continue
            target_card = cards[paper_id]
            topic = make_topic(target_card)
            disc = infer_discipline(target_card)
            discipline_counts[disc] += 1
            # parent is direct ancestor (first hop); used for fallback context only
            direct_parent_id = anc_list[0]
            parent_card_compressed = (
                make_compressed_parent_card(cards[direct_parent_id])
                if direct_parent_id in cards else None
            )
            prompt = {
                "prompt_id": f"agentic_v1::p_{n_prompts:04d}",
                "topic": topic,
                "discipline": disc,
                "year_min_hint": max(2015, (target_card.get("year") or 2020) - 7),
                "year_max_hint": (target_card.get("year") or 2024) - 1,
                "target_paper_id": paper_id,
                "gold_lineage": anc_list,
                "gold_proposal": target_card.get("genome", {}),
                "parent_card_compressed": parent_card_compressed,
            }
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            n_prompts += 1
            if args.max_prompts and n_prompts >= args.max_prompts:
                break
    print(f"  wrote {n_prompts} prompts → {prompts_path}")
    print(f"  discipline split: {dict(discipline_counts)}")

    print(f"[4/4] writing tool corpora")
    # tool_corpus: cards in the format the `read` tool returns
    tool_corpus_path = OUT_DIR / "tool_corpus.jsonl"
    bm25_corpus_path = OUT_DIR / "bm25_corpus.jsonl"
    with tool_corpus_path.open("w") as tc, bm25_corpus_path.open("w") as bc:
        for pid, card in sorted(cards.items()):
            genome = card.get("genome") or {}
            # text fed to BM25 search
            searchable = " ".join([
                card.get("title", ""),
                genome.get("niche_genome", ""),
                genome.get("mechanism_genome", ""),
                genome.get("delta_genome", ""),
                (card.get("source_text") or "")[:600],
            ])
            bc.write(json.dumps({
                "paper_id": pid,
                "title": card.get("title", ""),
                "year": card.get("year"),
                "discipline": infer_discipline(card),
                "searchable_text": searchable[:2000],
            }, ensure_ascii=False) + "\n")
            # full card text for read
            tc.write(json.dumps({
                "paper_id": pid,
                "title": card.get("title", ""),
                "year": card.get("year"),
                "abstract": (card.get("source_text") or "")[:1500],
                "genome": genome,
            }, ensure_ascii=False) + "\n")
    print(f"  wrote tool_corpus → {tool_corpus_path}")
    print(f"  wrote bm25_corpus → {bm25_corpus_path}")

    # stats
    stats = {
        "n_cards": len(cards),
        "n_edges": len(edges),
        "n_prompts": n_prompts,
        "ancestry_length_hist": dict(depth_hist),
        "discipline_counts": dict(discipline_counts),
        "max_ancestor_depth": args.max_ancestor_depth,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  stats → {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
