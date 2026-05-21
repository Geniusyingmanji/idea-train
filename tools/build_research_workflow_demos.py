"""Build agentic_v3 — multi-archetype research workflow demos.

10 archetypes (W1..W10, see research_workflows.md). For each (prompt,
archetype) pair, GPT-5.5 generates a complete agentic trajectory that follows
that archetype's tool-sequence template.

Goal: ~1000-1200 high-quality demos covering 8 research skills × 10 disciplines
× 10 archetypes.

Output: data/agentic_v3/sft_demos.jsonl (+ skill_breakdown.json)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.tools.web_search import WebSearchTool


OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"
SKILLS_OUT = OUT_DIR / "skill_breakdown.json"


# ─────────────────────────────────────────────────────────────────────────────
# Archetype specifications
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPES = {
    "W1_pure_discovery": {
        "description": "An open, somewhat vague research question. The agent must search the literature broadly, read a few high-relevance papers, synthesize, and propose a concrete research direction grounded in 2-4 papers it actually read.",
        "tool_pattern": "3-5 searches with progressively narrower queries → 2-3 reads → propose",
        "skills": ["problem_clarification", "lineage_construction", "multi_source_synthesis"],
        "typical_length": "5-7 actions",
    },
    "W2_single_paper_extension": {
        "description": "Given one anchor paper, the agent extracts its structured genome, scans for related work, then proposes a follow-up that specifically addresses the anchor's stated limitation.",
        "tool_pattern": "extract_genome(anchor) → search(related) → 1-2 reads → genome_diff(anchor, proposed) → propose (possibly revised)",
        "skills": ["lineage_construction", "gap_identification", "reflection_revision"],
        "typical_length": "5-6 actions",
    },
    "W3_multi_paper_synthesis": {
        "description": "Given 2-4 anchor papers in related areas, the agent extracts each genome, runs pairwise genome_diff to find the shared/divergent mechanism dimensions, then proposes a unifying framework that subsumes them.",
        "tool_pattern": "extract × N → genome_diff × pairs → search bridge concept → propose unified",
        "skills": ["multi_source_synthesis", "method_transfer"],
        "typical_length": "7-9 actions",
    },
    "W4_literature_review": {
        "description": "The agent surveys a sub-field by issuing multiple search queries (different angles), reads 5-7 papers, then proposes a structured mini-review in the 6-field genome format (where mechanism_genome describes the dominant approach, niche the problem domain, etc.).",
        "tool_pattern": "search × 3-4 (varied keywords) → read × 5-6 → optional extract on representative → propose (as structured review)",
        "skills": ["multi_source_synthesis", "problem_clarification"],
        "typical_length": "8-10 actions",
    },
    "W5_critical_analysis": {
        "description": "Given an anchor paper, the agent reads it, extracts its genome, runs novelty_check, then searches for limitations/replications/critiques. After 2-3 reads, proposes a validation or fixed-method experiment that addresses the identified weakness.",
        "tool_pattern": "read(anchor) → extract → novelty_check → search('limitations of X') → read 2-3 → propose validation experiment",
        "skills": ["critical_evaluation", "hypothesis_specification"],
        "typical_length": "6-8 actions",
    },
    "W6_cross_domain_bridge": {
        "description": "Given a method from field A (e.g. transformers in NLP) and a problem from field B (e.g. protein folding), the agent extracts both, runs genome_diff to spot transferable structure, then proposes how to port A's method to solve B's problem.",
        "tool_pattern": "search(A method) → read → extract → search(B problem) → read → extract → genome_diff → propose transfer",
        "skills": ["method_transfer", "multi_source_synthesis"],
        "typical_length": "7-8 actions",
    },
    "W7_hypothesis_refinement": {
        "description": "The agent has a rough initial idea. It first calls propose with the rough version, then runs genome_diff and novelty_check to self-assess, identifies missing components, searches and reads to fill them, then re-proposes the refined version.",
        "tool_pattern": "(rough rationale) → search → propose v1 → genome_diff(self) → novelty_check → search missing → read → propose v2 (refined)",
        "skills": ["reflection_revision", "hypothesis_specification"],
        "typical_length": "6-8 actions (with one propose mid-trajectory)",
    },
    "W8_reproduction_doubt": {
        "description": "The agent reads an anchor paper whose claim seems strong, then searches for replications, criticisms, and adjacent failures. After reading 2-3 such sources, proposes a validation experiment that would settle the doubt.",
        "tool_pattern": "read(anchor) → search('replication of X') OR search('criticism of X') → read 2-3 → propose validation",
        "skills": ["critical_evaluation", "hypothesis_specification"],
        "typical_length": "5-7 actions",
    },
    "W9_tool_heavy_comparison": {
        "description": "Given 3+ competing methods, the agent extracts each, runs genome_diff between every pair, runs novelty_check on the most promising candidate, and proposes a unified comparison or hybrid.",
        "tool_pattern": "search → read × 3+ → extract × 3+ → genome_diff × multiple pairs → novelty_check → propose comparison/hybrid",
        "skills": ["multi_source_synthesis", "critical_evaluation"],
        "typical_length": "8-10 actions",
    },
    "W10_vague_to_concrete": {
        "description": "Starts with an extremely vague question (e.g. 'how can we make scientific discovery more efficient'). Through multiple clarifying searches, the agent narrows the problem to a specific sub-domain, then proposes a concrete plan in that narrower scope.",
        "tool_pattern": "search(very broad) → reflect → search(narrower) → search(narrower still) → read 1-2 → propose specific plan",
        "skills": ["problem_clarification", "lineage_construction"],
        "typical_length": "5-7 actions",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Build prompt pool — diverse disciplines + archetypes
# ─────────────────────────────────────────────────────────────────────────────

DISCIPLINES = [
    ("astronomy", ["exoplanet detection", "gravitational waves", "cosmological inflation",
                   "galactic dynamics", "stellar nucleosynthesis"]),
    ("chemistry", ["catalyst design", "drug discovery", "retrosynthesis",
                   "molecular dynamics", "reaction mechanism"]),
    ("biology", ["protein design", "single-cell RNA-seq", "gene regulation",
                 "antibiotic resistance", "neural development"]),
    ("materials", ["perovskite solar cells", "battery electrolytes", "topological insulators",
                   "MOF catalysis", "high-entropy alloys"]),
    ("computer_science", ["large language model reasoning", "diffusion models",
                          "neural architecture search", "federated learning",
                          "graph neural networks"]),
    ("physics", ["quantum error correction", "fusion plasma confinement",
                 "neutrino oscillation", "ultracold atoms", "Higgs boson rare decay"]),
    ("earth_science", ["climate tipping points", "earthquake prediction",
                       "ice sheet dynamics", "atmospheric chemistry",
                       "remote sensing foundation models"]),
    ("energy", ["small modular nuclear", "grid storage", "carbon capture",
                "hydrogen electrolysis", "perovskite tandem cells"]),
    ("mathematics", ["automated theorem proving", "combinatorial optimization",
                     "geometric deep learning", "algebraic topology",
                     "spectral graph theory"]),
    ("neuroscience", ["connectomics", "brain-computer interface", "visual cortex coding",
                      "neural manifolds", "memory consolidation"]),
]


VAGUE_QUESTIONS_PROMPT = """\
Generate {n} VAGUE, OPEN research questions in the field of {discipline}, suitable for the "{archetype_short}" workflow archetype. {style_hint}

Each question should be:
- 1-2 sentences, intentionally open / under-specified for the archetype
- not name-dropping specific paper or method
- distinct from the others

Output a JSON array of strings inside ```json ... ``` fences."""


# style hints customize the prompt for each archetype
ARCHETYPE_PROMPT_STYLE = {
    "W1_pure_discovery": "These should be broad, exploratory questions a researcher would ask early in a project (e.g. 'What are the most promising approaches to X?').",
    "W2_single_paper_extension": "Frame each as 'given a specific 2022-2024 paper introducing technique X for problem Y, what's the natural follow-up?' Include the imagined paper's gist.",
    "W3_multi_paper_synthesis": "Frame each as 'methods A, B, and C all attack different aspects of X; how can they be combined?' Reference 2-3 abstract method types.",
    "W4_literature_review": "Frame each as a request for a structured survey of a sub-area (e.g. 'survey of techniques for reducing hallucination in LLMs since 2022').",
    "W5_critical_analysis": "Frame each as questioning a claim or methodology (e.g. 'is method X really better than baseline Y, given limitations Z?').",
    "W6_cross_domain_bridge": "Frame each as porting a method from field A to field B (cross both disciplines deliberately).",
    "W7_hypothesis_refinement": "State a rough initial idea that has obvious holes; the researcher needs to refine it.",
    "W8_reproduction_doubt": "Express doubt about a recent strong claim (e.g. 'paper claims X but I'm not convinced it generalizes — how to verify?').",
    "W9_tool_heavy_comparison": "Frame as 'compare methods A vs B vs C for problem X' — explicitly multi-method.",
    "W10_vague_to_concrete": "Make these EXTREMELY vague (e.g. 'how can we make protein design more controllable?'). One sentence max, no specifics.",
}


def build_synthetic_prompts(n_per_combo: int = 5, client=None,
                             workers: int = 12,
                             ) -> list[dict]:
    """Synthesize prompts: 10 disciplines × 10 archetypes × n_per_combo."""
    if client is None:
        client = build_client()
    calls = []
    for disc, _ in DISCIPLINES:
        for arch_id, arch_spec in ARCHETYPES.items():
            style = ARCHETYPE_PROMPT_STYLE[arch_id]
            arch_short = arch_id.split("_", 1)[1]
            calls.append(TeacherCall(
                prompt_id=f"v3::{disc}::{arch_id}",
                messages=[{
                    "role": "user",
                    "content": VAGUE_QUESTIONS_PROMPT.format(
                        n=n_per_combo, discipline=disc,
                        archetype_short=arch_short, style_hint=style,
                    ),
                }],
                max_tokens=900,
                temperature=0.75,
                metadata={"discipline": disc, "archetype": arch_id},
            ))
    print(f"  dispatching {len(calls)} GPT-5.5 calls (synthetic prompt gen)")
    t0 = time.time()
    results = batch_call(calls, workers=workers)
    print(f"  done in {time.time() - t0:.1f}s")

    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    for r in results:
        disc = r.metadata["discipline"]
        arch = r.metadata["archetype"]
        if r.error or not r.content:
            continue
        m = fence_re.search(r.content)
        try:
            arr = json.loads(m.group(1) if m else r.content)
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        for i, q in enumerate(arr[:n_per_combo]):
            if not isinstance(q, str) or len(q) < 30:
                continue
            out.append({
                "prompt_id": f"v3::{disc}::{arch}::{i:02d}",
                "source": "synthetic_v3",
                "topic": f"[{disc}/{arch.split('_', 1)[1]}] {q[:160]}",
                "discipline": disc,
                "archetype": arch,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": q.strip(),
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Demo generation
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SYS_TEMPLATE = """You are demonstrating an agentic research trajectory in the {archetype_id} style ({description_short}).

Tool pattern: {tool_pattern}.

Format: for each step, write a 1-2 sentence rationale, then a ```action ... ``` block with one tool call, then a [result]: line with a 2-4 sentence simulated tool result. Repeat until you call `propose`.

Tools:
  search: {{"tool": "search", "query": "...", "year_min": ..., "year_max": ..., "k": 5}}
  read: {{"tool": "read", "paper_id": "..."}}
  extract_genome: {{"tool": "extract_genome", "paper_id": "..."}}
  genome_diff: {{"tool": "genome_diff", "parent_id": "...", "proposed_genome": {{...6 fields...}}}}
  novelty_check: {{"tool": "novelty_check", "mechanism": "...", "year_min": ..., "year_max": ...}}
  propose: {{"tool": "propose", "gene_genome": {{mechanism_genome, niche_genome, observation_genome, limitation_genome, delta_genome, claim_genome}}}}

Use 4-8 total actions, ending with propose. The final propose must fill all 6 fields. Use the real OpenAlex paper IDs provided. Keep total output under 1800 words."""


def build_demo_user(prompt: dict, candidates: list[dict]) -> str:
    cand_blob = ""
    if candidates:
        cand_blob = "\n\nReal OpenAlex candidates already retrieved for this topic (use these in your simulated tool results):\n"
        for i, c in enumerate(candidates[:5]):
            cand_blob += (
                f"\n  [{i+1}] paper_id={c['paper_id']}, year={c.get('year', '?')}\n"
                f"      title: {c.get('title', '')[:120]}\n"
                f"      snippet: {c.get('snippet', '')[:200]}\n"
            )
    arch_id = prompt.get("archetype", "W1_pure_discovery")
    return f"""TOPIC / FULL PROMPT TO THE AGENT:

{prompt['full_prompt'][:3000]}

Discipline: {prompt.get('discipline', 'general')}
Year window: {prompt.get('year_min_hint', '2018')}..{prompt.get('year_max_hint', '2025')}
Workflow archetype: {arch_id}
{cand_blob}

Now write the complete agent trajectory in the {arch_id} style."""


def prefetch_candidates(prompt: dict, search_tool: WebSearchTool, k: int = 5) -> list[dict]:
    topic = prompt.get("topic", "") or prompt.get("full_prompt", "")[:200]
    q = topic
    if q.startswith("[") and "]" in q:
        q = q.split("]", 1)[1].strip()
    q = q[:200]
    try:
        results = search_tool.search(
            q, k=k,
            year_min=prompt.get("year_min_hint"),
            year_max=prompt.get("year_max_hint"),
        )
        return [r.to_dict() for r in results]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=5,
                    help="prompts per (discipline × archetype) — default 5 → 500 synthetic")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--include-v2", action="store_true",
                    help="also include data/agentic_v2/prompts.jsonl (478 prompts)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-prompt-gen", action="store_true",
                    help="re-use existing v3 prompts.jsonl")
    args = ap.parse_args()

    # ─── Phase 1: build prompt pool ───────────────────────────────────────
    prompts = []
    if not args.skip_prompt_gen:
        print("[A1/3] generating diverse prompts")
        synth = build_synthetic_prompts(n_per_combo=args.n_per_combo,
                                         workers=args.workers)
        prompts.extend(synth)
        print(f"  synthetic: {len(synth)}")

        if args.include_v2:
            print("[A2/3] adding v2 prompts with archetype tagging")
            with Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v2/prompts.jsonl").open() as f:
                rng = random.Random(7)
                arch_ids = list(ARCHETYPES.keys())
                for line in f:
                    p = json.loads(line)
                    # tag v2 prompts with a sensible archetype based on source
                    if p["source"] == "gene_arena":
                        setting = p.get("setting", "Question")
                        if setting == "Library":
                            p["archetype"] = "W4_literature_review"
                        elif setting == "Lineage":
                            p["archetype"] = "W2_single_paper_extension"
                        else:
                            p["archetype"] = "W1_pure_discovery"
                    elif p["source"] == "sgi_bench":
                        # SGI questions are often vague + open
                        p["archetype"] = rng.choice([
                            "W1_pure_discovery", "W10_vague_to_concrete",
                            "W4_literature_review",
                        ])
                    elif p["source"] == "synthetic":
                        p["archetype"] = rng.choice(arch_ids)
                    else:
                        p["archetype"] = "W1_pure_discovery"
                    p["prompt_id"] = "v3-from-v2::" + p["prompt_id"]
                    prompts.append(p)
            print(f"  + v2: {len(prompts) - len(synth)}")

        with PROMPTS_OUT.open("w") as f:
            for p in prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  wrote {len(prompts)} prompts → {PROMPTS_OUT}")
    else:
        with PROMPTS_OUT.open() as f:
            for line in f:
                prompts.append(json.loads(line))
        print(f"[A1/3] using existing {len(prompts)} prompts")

    # ─── Phase 2: pre-fetch OpenAlex candidates for each prompt ───────────
    print(f"[A2/3] pre-fetching OpenAlex candidates ({len(prompts)} prompts)")
    search_tool = WebSearchTool()
    t0 = time.time()
    prefetched: dict[str, list[dict]] = {}
    for i, p in enumerate(prompts):
        cands = prefetch_candidates(p, search_tool, k=5)
        prefetched[p["prompt_id"]] = cands
        if (i + 1) % 100 == 0:
            print(f"  prefetch {i+1}/{len(prompts)} ({time.time() - t0:.0f}s)")
    print(f"  done in {time.time() - t0:.1f}s")

    # ─── Phase 3: generate demos ──────────────────────────────────────────
    done_ids: set[str] = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try:
                    d = json.loads(line); done_ids.add(d["prompt_id"])
                except Exception:
                    pass
        print(f"  resume: {len(done_ids)} demos already generated")
        prompts = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"  {len(prompts)} remaining")

    print(f"[A3/3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        arch_id = p.get("archetype", "W1_pure_discovery")
        if arch_id not in ARCHETYPES:
            arch_id = "W1_pure_discovery"
        spec = ARCHETYPES[arch_id]
        # condensed description for the simpler template
        desc_short = spec["description"][:200]
        sys_msg = DEMO_SYS_TEMPLATE.format(
            archetype_id=arch_id,
            description_short=desc_short,
            tool_pattern=spec["tool_pattern"][:200],
        )
        user_msg = build_demo_user(p, prefetched.get(p["prompt_id"], []))
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4000,
            temperature=0.45,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], [])},
        ))
    print(f"  dispatching {len(calls)} GPT-5.5 calls")

    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda done, total: print(
            f"  [{done}/{total}] {time.time() - t0:.0f}s elapsed", flush=True
        ),
    )

    n_valid = 0
    n_invalid = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 300:
                n_invalid += 1
                continue
            # Relaxed gate: at least 1 action AND a propose somewhere
            if r.content.count("```action") < 1 or '"propose"' not in r.content:
                n_invalid += 1
                continue
            demo = {
                "prompt_id": p["prompt_id"],
                "source": p.get("source"),
                "discipline": p.get("discipline"),
                "archetype": p.get("archetype"),
                "topic": p.get("topic", "")[:200],
                "full_prompt": p.get("full_prompt", ""),
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "skills_inferred": ARCHETYPES[p.get("archetype", "W1_pure_discovery")]["skills"],
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
            n_valid += 1

    print(f"\nDone. valid={n_valid}, invalid={n_invalid}, total={len(results)}")
    print(f"saved → {DEMOS_OUT}")
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
