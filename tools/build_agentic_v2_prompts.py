"""Build agentic_v2 prompt pool (~500 prompts).

Sources:
  1) GENE-Arena: 50 tasks × 3 settings = 150 prompts (Library / Lineage / Question)
     - Re-use IdeaEvolving's PromptBuilder to construct
  2) SGI-Bench task_2: 200 (sample from 315) — stratified across 10 disciplines
  3) Synthetic: 150 — GPT-5.5 generates 15 topics per discipline × 10 disciplines

Each prompt has:
  - prompt_id, source, topic (text shown to agent), discipline,
    year_min_hint, year_max_hint, full_prompt (the actual user message)

The `full_prompt` is what the agent's user-turn will contain; `topic` is a
short summary (for logs / system prompts that reference the goal).

Output: data/agentic_v2/prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# IdeaEvolving for arena builder
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving/gene_arena")
os.environ.setdefault("no_proxy", "*"); os.environ.setdefault("NO_PROXY", "*")

from gene_arena.arena_config import TASK_DIR
from gene_arena.prompt_builder import PromptBuilder, PromptConfig

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- 1) GENE-Arena prompts ----------------------------------------------

def build_arena_prompts() -> list[dict]:
    """50 tasks × 3 settings = 150 prompts."""
    out = []
    task_paths = sorted(Path(TASK_DIR).glob("*.json"))
    for task_path in task_paths[:50]:
        trace_id = task_path.stem
        builder = PromptBuilder(task_path)
        for setting in ["Library", "Lineage", "Question"]:
            try:
                full_prompt = builder.build(PromptConfig(setting=setting))
            except Exception as e:
                print(f"  arena build err {trace_id}/{setting}: {e}")
                continue
            disc = trace_id.split("_")[0] if "_" in trace_id else "general"
            topic = f"[{disc}/{setting}] {trace_id}"
            out.append({
                "prompt_id": f"v2::arena::{trace_id}::{setting}",
                "source": "gene_arena",
                "trace_id": trace_id,
                "setting": setting,
                "topic": topic,
                "discipline": disc,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": full_prompt,
            })
    return out


# ---- 2) SGI-Bench prompts ------------------------------------------------

def build_sgi_prompts(n: int = 200) -> list[dict]:
    """Sample 200 from SGI-IdeaGeneration (stratified across 10 disciplines)."""
    from datasets import load_dataset
    ds = load_dataset("InternScience/SGI-IdeaGeneration", split="test")
    by_disc: dict[str, list[dict]] = {}
    for q in ds:
        by_disc.setdefault(q["discipline"], []).append(q)
    per = max(1, n // max(len(by_disc), 1))
    rng = random.Random(42)
    out = []
    for disc, items in sorted(by_disc.items()):
        rng.shuffle(items)
        for q in items[:per]:
            full_prompt = q["question"]
            topic = f"[{disc}] {full_prompt[:120]}"
            out.append({
                "prompt_id": f"v2::sgi::{disc}::{str(q['idx'])}",
                "source": "sgi_bench",
                "topic": topic,
                "discipline": disc,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": full_prompt,
                "sgi_orig": q,           # carry through for downstream scoring
            })
    return out[:n]


# ---- 3) Synthetic prompts (GPT-5.5) -------------------------------------

DISCIPLINES = [
    "astronomy", "chemistry", "earth_science", "energy", "computer_science",
    "biology", "materials", "neuroscience", "physics", "mathematics",
]

SYNTHETIC_GEN_PROMPT = """\
Generate {n_topics} diverse RESEARCH TOPICS in the field of {discipline}. Each \
topic should be:
- 1-3 sentences
- specific enough to seed a follow-up research idea (mentions a sub-area, problem, \
  or recent technique)
- not a question, but a topic / problem statement
- distinct from the others

Output JSON array of strings ONLY, inside ```json ... ``` fences. Example:
```json
[
  "Develop fault-tolerant superconducting qubit architectures that maintain coherence ...",
  "Apply continual-learning methods to ...",
  ...
]
```"""


def build_synthetic_prompts(n_per_discipline: int = 15) -> list[dict]:
    """For each of 10 disciplines, ask GPT-5.5 for 15 diverse research topics."""
    client = build_client()
    calls = []
    for disc in DISCIPLINES:
        calls.append(TeacherCall(
            prompt_id=f"syn::{disc}",
            messages=[{
                "role": "user",
                "content": SYNTHETIC_GEN_PROMPT.format(
                    discipline=disc, n_topics=n_per_discipline,
                ),
            }],
            max_tokens=1500,
            temperature=0.7,
            metadata={"discipline": disc},
        ))
    print(f"  dispatching {len(calls)} GPT-5.5 calls (synthetic topic gen)")
    t0 = time.time()
    results = batch_call(calls, workers=10)
    print(f"  done in {time.time() - t0:.1f}s")

    import re
    out = []
    for r in results:
        disc = r.metadata["discipline"]
        if r.error:
            print(f"  err {disc}: {r.error}")
            continue
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", r.content, re.DOTALL)
        try:
            arr = json.loads(m.group(1) if m else r.content)
        except Exception:
            print(f"  parse err for {disc}; trying lenient")
            arr = []
        if isinstance(arr, list):
            for i, topic in enumerate(arr[:n_per_discipline]):
                if not isinstance(topic, str) or len(topic) < 30:
                    continue
                out.append({
                    "prompt_id": f"v2::syn::{disc}::{i:02d}",
                    "source": "synthetic",
                    "topic": f"[{disc}] {topic[:160]}",
                    "discipline": disc,
                    "year_min_hint": 2018,
                    "year_max_hint": 2025,
                    "full_prompt": (
                        f"Research topic: {topic}\n\n"
                        f"Discipline: {disc}\n\n"
                        "Propose a novel follow-up research idea building on relevant prior work."
                    ),
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sgi", type=int, default=200)
    ap.add_argument("--n-synth-per-disc", type=int, default=15)
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total prompts (for smoke testing)")
    args = ap.parse_args()

    print("[1/3] building GENE-Arena prompts (50 tasks × 3 settings)")
    arena = build_arena_prompts()
    print(f"  {len(arena)} arena prompts")

    print(f"[2/3] building SGI-Bench prompts (target n={args.n_sgi})")
    sgi = build_sgi_prompts(args.n_sgi)
    print(f"  {len(sgi)} SGI prompts")

    if args.skip_synth:
        synth = []
        print("[3/3] skipping synthetic (--skip-synth)")
    else:
        print(f"[3/3] building synthetic prompts ({args.n_synth_per_disc} × {len(DISCIPLINES)})")
        synth = build_synthetic_prompts(args.n_synth_per_disc)
        print(f"  {len(synth)} synthetic prompts")

    all_prompts = arena + sgi + synth
    if args.limit:
        random.Random(42).shuffle(all_prompts)
        all_prompts = all_prompts[:args.limit]
    print(f"\nTOTAL: {len(all_prompts)} prompts")

    out_path = OUT_DIR / "prompts.jsonl"
    with out_path.open("w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote → {out_path}")

    # discipline distribution
    by_disc = {}
    by_source = {}
    for p in all_prompts:
        by_disc[p["discipline"]] = by_disc.get(p["discipline"], 0) + 1
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    print(f"by discipline: {dict(sorted(by_disc.items(), key=lambda x: -x[1]))}")
    print(f"by source:     {by_source}")


if __name__ == "__main__":
    main()
