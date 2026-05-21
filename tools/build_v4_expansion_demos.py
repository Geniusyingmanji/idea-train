"""v4 expansion: add diversity along 3 axes.

Goal: ~1200 new demos covering:
  1) LENGTH DIVERSITY  — short (1-3 tools), medium, long, varied
  2) DISCIPLINE EXPANSION — 10 new domains + interdisciplinary
  3) PROMPT STYLE — peer-review, learner Q, industry, policy, failure-recovery

Together with v3's 1032, total ~2200 demos. Critically: short demos help the
model learn to propose QUICKLY when extended deliberation isn't needed.

Output appended to data/agentic_v3/sft_demos.jsonl (so SFT can train on union).
Optionally also writes data/agentic_v4/sft_demos.jsonl separately.
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

from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT_V2
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Length tiers — explicit instruction to GPT-5.5 about how short/long to be
# ─────────────────────────────────────────────────────────────────────────────

LENGTH_TIERS = {
    "short": {
        "n_actions": "2-3 (1 search + 1 read + propose, OR just search + propose, OR direct propose)",
        "tools_to_use": "search + propose, optionally one read",
        "guidance": "Be DECISIVE — when the topic is clear enough, search once, optionally read 1 paper, then propose directly. Don't over-deliberate.",
    },
    "medium": {
        "n_actions": "4-6 (search + read + maybe extract or novelty + propose)",
        "tools_to_use": "any mix of 3-5 tools then propose",
        "guidance": "Moderate depth: search, read 1-2 papers, optionally extract or novelty_check, then propose.",
    },
    "long": {
        "n_actions": "7-9 (rich multi-turn with many tools)",
        "tools_to_use": "all 6 tools possibly multiple times",
        "guidance": "Full agentic chain — search multiple times, read several papers, extract their genomes, optionally diff/novelty check, then propose.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# New disciplines + cross-discipline pairs
# ─────────────────────────────────────────────────────────────────────────────

NEW_DISCIPLINES = [
    ("clinical_medicine", [
        "clinical trial design", "drug-drug interaction prediction",
        "personalized therapy", "diagnostic imaging AI",
        "EHR-based outcome prediction"
    ]),
    ("pharmacology", [
        "ADMET prediction", "polypharmacology",
        "natural product drug leads", "antibiotic discovery"
    ]),
    ("robotics_control", [
        "dexterous manipulation", "legged locomotion",
        "sim-to-real transfer", "model predictive control",
        "humanoid whole-body control"
    ]),
    ("economics_finance", [
        "financial time series forecasting", "macro causal modeling",
        "auction design", "asset pricing anomalies",
        "central bank policy NLP"
    ]),
    ("sociology", [
        "online radicalization dynamics", "social mobility measurement",
        "content moderation effectiveness", "misinformation spread"
    ]),
    ("agriculture_food", [
        "precision agriculture", "vertical farming optimization",
        "crop disease forecasting", "soil microbiome"
    ]),
    ("urban_planning", [
        "traffic flow optimization", "renewable grid integration",
        "smart city sensor networks", "urban heat islands"
    ]),
    ("cognitive_science", [
        "human attention modeling", "working memory capacity",
        "creativity in problem solving", "language acquisition"
    ]),
    ("philosophy_ethics", [
        "AI alignment foundations", "experimental philosophy",
        "moral uncertainty quantification", "value pluralism"
    ]),
    ("interdisciplinary", [
        "physics-informed ML for fluid dynamics",
        "economic field experiments × causal ML",
        "neuroscience × deep learning for representations",
        "evolution × multi-agent RL",
        "philosophy of AI × AI safety benchmark design",
        "social network × epidemic modeling",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt styles
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_STYLES = {
    "academic_question": "phrase as an open research question from the perspective of an academic researcher",
    "conference_cfp": "phrase as a Call-For-Papers excerpt soliciting submissions on a sub-topic",
    "peer_review": "phrase as a peer-review request: critique a 'proposed work' described in 2-3 sentences and suggest improvements",
    "learner_question": "phrase as a curious newcomer asking what they should research, vague but earnest",
    "industry_problem": "phrase as an industry practitioner asking how to solve a concrete real-world problem",
    "policy_question": "phrase as a policy-maker or funder asking what evidence-based research would address a societal issue",
    "failure_recovery": "phrase as a scenario where initial search returns 0 results or wrong domain — the agent must recover by reformulating the query",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt synthesis
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_GEN_TEMPLATE = """\
Generate {n} diverse research-related prompts in the area of {discipline}, in the "{style_name}" style. {style_hint}

Each prompt should be:
- 1-3 sentences
- distinct from the others
- realistic — would actually come from a researcher, practitioner, or learner

Output JSON array of strings inside ```json ... ``` fences only."""


def build_synthetic_prompts(n_per_combo: int = 3, client=None,
                             workers: int = 12,
                             ) -> list[dict]:
    """Synthesize prompts: (new_disciplines + interdisciplinary) × prompt_styles × n_per_combo."""
    if client is None:
        client = build_client()
    calls = []
    for disc, _ in NEW_DISCIPLINES:
        for style_name, style_hint in PROMPT_STYLES.items():
            calls.append(TeacherCall(
                prompt_id=f"v4::{disc}::{style_name}",
                messages=[{
                    "role": "user",
                    "content": PROMPT_GEN_TEMPLATE.format(
                        n=n_per_combo, discipline=disc,
                        style_name=style_name, style_hint=style_hint,
                    ),
                }],
                max_tokens=900,
                temperature=0.8,
                metadata={"discipline": disc, "style": style_name},
            ))
    print(f"  dispatching {len(calls)} prompt-gen calls "
          f"({len(NEW_DISCIPLINES)} disc × {len(PROMPT_STYLES)} styles × {n_per_combo} each)")
    t0 = time.time()
    results = batch_call(calls, workers=workers)
    print(f"  done in {time.time() - t0:.1f}s")

    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try:
            arr = json.loads(m.group(1) if m else r.content)
        except Exception:
            continue
        if not isinstance(arr, list): continue
        disc = r.metadata["discipline"]
        style = r.metadata["style"]
        for i, q in enumerate(arr[:n_per_combo]):
            if not isinstance(q, str) or len(q) < 30: continue
            # Randomly assign length tier — short more frequent (we need them)
            tier = random.Random(hash(f"{disc}{style}{i}")).choices(
                ["short", "medium", "long"], weights=[0.45, 0.35, 0.20]
            )[0]
            out.append({
                "prompt_id": f"v4::{disc}::{style}::{i:02d}",
                "source": "synthetic_v4",
                "topic": f"[{disc}/{style}] {q[:160]}",
                "discipline": disc,
                "style": style,
                "length_tier": tier,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": q.strip(),
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Demo generation
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SYS_TEMPLATE = """You are demonstrating an agentic research trajectory.

Length tier: {tier_name}
Tools to use: {tools_guidance}
Pattern: {n_actions}

{tier_guidance}

Format: rationale (1-2 sentences) + ```action ... ``` block with one tool call + [result]: 2-4 sentence simulated tool result. Repeat until propose.

Tools available:
  search: {{"tool": "search", "query": "...", "year_min": ..., "year_max": ..., "k": 5}}
  read: {{"tool": "read", "paper_id": "..."}}
  extract_genome: {{"tool": "extract_genome", "paper_id": "..."}}
  genome_diff: {{"tool": "genome_diff", "parent_id": "...", "proposed_genome": {{6 fields}}}}
  novelty_check: {{"tool": "novelty_check", "mechanism": "...", "year_min": ..., "year_max": ...}}
  propose: {{"tool": "propose", "gene_genome": {{mechanism_genome, niche_genome, observation_genome, limitation_genome, delta_genome, claim_genome}}}}

Use the real OpenAlex paper IDs from the candidates below. Keep total length under 1500 words for short, 2500 for medium, 3500 for long. Final propose MUST fill all 6 gene_genome fields."""


def build_demo_user(prompt: dict, candidates: list[dict]) -> str:
    cand_blob = ""
    if candidates:
        cand_blob = "\n\nReal OpenAlex candidates already retrieved:\n"
        for i, c in enumerate(candidates[:5]):
            cand_blob += (
                f"\n  [{i+1}] paper_id={c['paper_id']}, year={c.get('year', '?')}\n"
                f"      title: {c.get('title', '')[:120]}\n"
                f"      snippet: {c.get('snippet', '')[:200]}\n"
            )
    return f"""TOPIC / PROMPT TO THE AGENT:

{prompt['full_prompt'][:3000]}

Discipline: {prompt.get('discipline', 'general')}
Style: {prompt.get('style', 'academic')}
Length tier: {prompt.get('length_tier', 'medium')}
Year window: {prompt.get('year_min_hint', '2018')}..{prompt.get('year_max_hint', '2025')}
{cand_blob}

Write the complete agent trajectory in the {prompt.get('length_tier','medium')} length style."""


def prefetch_candidates(prompt: dict, search_tool: WebSearchTool, k: int = 5) -> list[dict]:
    topic = prompt.get("full_prompt", "")[:200]
    try:
        results = search_tool.search(
            topic, k=k,
            year_min=prompt.get("year_min_hint"),
            year_max=prompt.get("year_max_hint"),
        )
        return [r.to_dict() for r in results]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=3,
                    help="prompts per (discipline × style). 3 → 10 disc × 7 styles × 3 = 210 prompts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # ─── Phase 1: build prompt pool ───────────────────────────────────────
    print("[A1/3] generating diverse prompts (new disciplines × styles × length tiers)")
    prompts = build_synthetic_prompts(n_per_combo=args.n_per_combo, workers=args.workers)
    with PROMPTS_OUT.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {len(prompts)} prompts → {PROMPTS_OUT}")

    # length tier distribution
    from collections import Counter
    tier_counts = Counter(p["length_tier"] for p in prompts)
    style_counts = Counter(p["style"] for p in prompts)
    print(f"  length tiers: {dict(tier_counts)}")
    print(f"  styles: {dict(style_counts)}")

    # resume
    done_ids = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass
        print(f"  resume: {len(done_ids)} already done")
        prompts = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"  {len(prompts)} remaining")

    # ─── Phase 2: pre-fetch candidates ─────────────────────────────────────
    print(f"[A2/3] pre-fetching OpenAlex candidates ({len(prompts)})")
    search_tool = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch_candidates(p, search_tool)
        if (i + 1) % 50 == 0:
            print(f"  prefetch {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    # ─── Phase 3: generate demos ──────────────────────────────────────────
    print(f"[A3/3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        tier = LENGTH_TIERS[p.get("length_tier", "medium")]
        sys_msg = DEMO_SYS_TEMPLATE.format(
            tier_name=p.get("length_tier", "medium"),
            tools_guidance=tier["tools_to_use"],
            n_actions=tier["n_actions"],
            tier_guidance=tier["guidance"],
        )
        user_msg = build_demo_user(p, prefetched.get(p["prompt_id"], []))
        # adjust max_tokens by tier
        mt = {"short": 2500, "medium": 3500, "long": 4500}[p.get("length_tier", "medium")]
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=mt,
            temperature=0.5,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], [])},
        ))

    print(f"  dispatching {len(calls)} GPT-5.5 calls")
    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = 0
    n_invalid = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 300:
                n_invalid += 1; continue
            if r.content.count("```action") < 1 or '"propose"' not in r.content:
                n_invalid += 1; continue
            demo = {
                "prompt_id": p["prompt_id"],
                "source": p.get("source"),
                "discipline": p.get("discipline"),
                "style": p.get("style"),
                "length_tier": p.get("length_tier"),
                "archetype": "v4_mixed",  # not strictly tied to W1-W10
                "topic": p.get("topic", "")[:200],
                "full_prompt": p.get("full_prompt", ""),
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
            n_valid += 1

    print(f"\nDone. valid={n_valid} / total={len(results)} (rate={n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")
    print(f"elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
