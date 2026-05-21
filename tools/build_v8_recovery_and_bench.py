"""v8: failure-recovery + bench-specific specialized demos.

Two sub-buckets:
  A) FAILURE_RECOVERY (~200 demos):
     - search returns 0 results → reformulate query and try again
     - read returns "PDF not found" → fall back to abstract from search
     - extract_genome fails → propose without parent_id
     - tool returns malformed → agent notices and re-tries
     Teaches resilient agent behavior.

  B) BENCH_SPECIFIC (~200 demos):
     - SGI-Bench task_2 schema: explicit idea_plan with full impl steps,
       data, metrics, expected outcome — graph-similarity friendly
     - ArenaRL Open-Travel style: Chinese travel itinerary planning with
       constraints (budget, days, interests)
     - GENE-Arena boundary: cross-cutting topics that span 2+ disciplines

Output: data/agentic_v8/sft_demos.jsonl (~400 demos)
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

from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v8")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─── A) Failure-recovery scenarios ─────────────────────────────────────────

RECOVERY_SCENARIOS = {
    "zero_results_then_reformulate": {
        "instruction": "First search returns 0 results (jargon too narrow). Agent reformulates with broader terms and succeeds on second search. Then read + propose.",
        "n_actions": "4 (search → search → read → propose)",
        "weight": 0.25,
    },
    "wrong_domain_then_pivot": {
        "instruction": "First search returns papers from a wrong sub-field. Agent notices and pivots query. Second search succeeds. Then propose.",
        "n_actions": "3-4 (search → search → propose, optionally read)",
        "weight": 0.20,
    },
    "read_fails_use_abstract": {
        "instruction": "Read attempt returns no full text. Agent falls back to the abstract/snippet from the search result and proceeds. Total 3 actions.",
        "n_actions": "3 (search → read [fail] → propose using snippet)",
        "weight": 0.15,
    },
    "extract_fails_propose_anyway": {
        "instruction": "extract_genome fails (paper text too short / error). Agent skips it and proposes from its own analysis. 3 actions.",
        "n_actions": "3-4",
        "weight": 0.15,
    },
    "novelty_says_not_novel_revise": {
        "instruction": "Agent proposes mechanism, novelty_check returns 'too similar to existing work', agent revises with a sharper delta and proposes again.",
        "n_actions": "4-5 (search → propose v1 → novelty → propose v2)",
        "weight": 0.15,
    },
    "long_query_search_no_match_break_up": {
        "instruction": "First search uses a long compound query, returns weak results. Agent breaks into 2 narrower queries, joins findings, then proposes.",
        "n_actions": "4-5",
        "weight": 0.10,
    },
}


RECOVERY_DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "energy", "robotics_control", "economics_finance",
    "clinical_medicine", "interdisciplinary",
]

RECOVERY_PROMPT_TMPL = """Generate {n} diverse research prompts in {discipline} that would naturally surface the failure mode "{scenario}". Each prompt 1-3 sentences. Output JSON array inside ```json ... ``` fences."""


# ─── B) Bench-specific scenarios ────────────────────────────────────────────

BENCH_SPECS = {
    "sgi_idea_plan_strict": {
        "system_extra": (
            "STRICT SGI-Bench format. The `propose` action MUST emit:\n"
            '{"tool": "propose", "idea_plan": {\n'
            '  "Idea": "<one-paragraph idea statement>",\n'
            '  "ImplementationSteps": {"1": "...", "2": "...", "3": "...", "4": "..."} (4-6 steps),\n'
            '  "ImplementationOrder": ["1-2", "2-3", "3-4"] (sequential dependencies),\n'
            '  "Dataset": "<concrete benchmark name>",\n'
            '  "EvaluationMetrics": {"primary": "...", "secondary": "..."} (2-4 metrics),\n'
            '  "ExpectedOutcome": "<concrete numeric or qualitative target>"\n'
            "}}\n\n"
            "All fields must be filled with substance, no placeholders. Graph-similarity-friendly: use precise method names, dataset names, metric names — these become graph nodes."
        ),
        "user_topic": "SGI-Bench task_2 style: propose a concrete research idea + implementation plan",
        "n_actions": "3-4 (search + read + propose, OR search + propose)",
        "weight": 0.4,
        "disciplines": ["computer_science", "biology", "chemistry", "materials",
                        "neuroscience", "robotics_control", "clinical_medicine"],
    },
    "arena_travel_zh": {
        "system_extra": (
            "ArenaRL Open-Travel 风格。最终 propose 字段使用 'answer' 给出中文行程方案：\n"
            '{"tool": "propose", "answer": "<3-5段中文行程，包括每日安排、预算估计、推荐理由>"}\n\n'
            "搜索时使用旅游/景点相关查询，可以用 search 工具检索相关攻略文献或案例。注意按用户约束（天数、预算、偏好）来组织答案。"
        ),
        "user_topic": "ArenaRL 开放旅行：根据约束给出旅行行程方案",
        "n_actions": "2-4 (用1-2次search可选read，然后propose)",
        "weight": 0.3,
        "disciplines": ["travel_planning"],
        "lang": "zh",
    },
    "gene_arena_crosscut": {
        "system_extra": (
            "GENE-Arena 风格 cross-cutting topic — propose a gene_genome where mechanism comes from one discipline but is applied to another (e.g., physics-informed ML in fluids, evolution × multi-agent RL).\n"
            "Final propose:\n"
            '{"tool": "propose", "gene_genome": {"mechanism_genome": "...", "niche_genome": "...", "observation_genome": "...", "limitation_genome": "...", "delta_genome": "...", "claim_genome": "..."}}\n\n'
            "Each field one sentence, specific. The mechanism_genome should explicitly name the borrowed concept."
        ),
        "user_topic": "GENE-Arena cross-disciplinary: propose a gene_genome where the mechanism crosses fields",
        "n_actions": "4-6",
        "weight": 0.3,
        "disciplines": ["interdisciplinary"],
    },
}

BENCH_PROMPT_TMPL = """Generate {n} diverse research prompts that fit "{bench_type}" style: {user_topic}. Discipline focus: {disc}. Each prompt 1-3 sentences. Output JSON array inside ```json ... ``` fences. {lang_hint}"""


# ─── Demo system templates ─────────────────────────────────────────────────

RECOVERY_SYS_TMPL = """You are demonstrating an agentic research trajectory with a SPECIFIC FAILURE-RECOVERY pattern.

PATTERN: {scenario}
INSTRUCTION: {instruction}
ACTIONS: {n_actions}

Format: 1-2 sentence rationale (explicitly call out what went wrong and how you'll recover) + ```action ... ``` JSON + [result]: simulated tool result that reflects the failure or recovery state. End with `propose` using gene_genome.

Tools: search, read, extract_genome, genome_diff, novelty_check, propose.
Final propose schema: {{"tool": "propose", "gene_genome": {{6 fields}}}}.

The trajectory must SHOW the failure and the recovery, not just describe it."""


BENCH_SYS_TMPL = """You are demonstrating an agentic research trajectory for a specific BENCHMARK STYLE.

BENCHMARK: {bench_type}
TOPIC: {user_topic}
ACTIONS: {n_actions}

{system_extra}

Format: rationale (1-2 sentences) + ```action ... ``` JSON + [result]: simulated tool result. End with `propose` in the schema specified above.

Tools: search, read, extract_genome, genome_diff, novelty_check, propose."""


# ─── Prompt synthesis ──────────────────────────────────────────────────────

def gen_recovery_prompts(n_per_combo: int, workers: int) -> list[dict]:
    calls = []
    for scenario in RECOVERY_SCENARIOS:
        for disc in RECOVERY_DISCIPLINES:
            calls.append(TeacherCall(
                prompt_id=f"v8::recovery::{scenario}::{disc}",
                messages=[{
                    "role": "user",
                    "content": RECOVERY_PROMPT_TMPL.format(
                        n=n_per_combo, discipline=disc, scenario=scenario,
                    ),
                }],
                max_tokens=1200, temperature=0.85,
                metadata={"kind": "recovery", "scenario": scenario, "disc": disc},
            ))
    return calls


def gen_bench_prompts(n_per_combo: int, workers: int) -> list[dict]:
    calls = []
    for bench_type, info in BENCH_SPECS.items():
        for disc in info["disciplines"]:
            lang = info.get("lang", "en")
            lang_hint = "Output in Chinese." if lang == "zh" else ""
            calls.append(TeacherCall(
                prompt_id=f"v8::bench::{bench_type}::{disc}",
                messages=[{
                    "role": "user",
                    "content": BENCH_PROMPT_TMPL.format(
                        n=n_per_combo, bench_type=bench_type,
                        user_topic=info["user_topic"], disc=disc,
                        lang_hint=lang_hint,
                    ),
                }],
                max_tokens=1200, temperature=0.85,
                metadata={"kind": "bench", "bench_type": bench_type, "disc": disc, "lang": lang},
            ))
    return calls


def parse_prompts(results, n_per_combo: int) -> list[dict]:
    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2027)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        for i, q in enumerate(arr[:n_per_combo]):
            if not isinstance(q, str) or len(q) < 25: continue
            md = r.metadata
            if md["kind"] == "recovery":
                out.append({
                    "prompt_id": f"v8::recovery::{md['scenario']}::{md['disc']}::{i:02d}",
                    "source": "synthetic_v8",
                    "kind": "recovery",
                    "scenario": md["scenario"],
                    "discipline": md["disc"],
                    "lang": "en",
                    "full_prompt": q.strip(),
                })
            else:
                out.append({
                    "prompt_id": f"v8::bench::{md['bench_type']}::{md['disc']}::{i:02d}",
                    "source": "synthetic_v8",
                    "kind": "bench",
                    "bench_type": md["bench_type"],
                    "discipline": md["disc"],
                    "lang": md.get("lang", "en"),
                    "full_prompt": q.strip(),
                })
    return out


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=5, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=4,
                    help="recovery: 6 × 11 × 4 = 264; bench: ~11 × 4 = 44 prompts")
    ap.add_argument("--bench-per-combo", type=int, default=10,
                    help="bench prompts per (bench_type × disc); 7+1+1 = 9 disc × 10 = 90")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v8/A1] generating prompt-gen calls")
    rec_calls = gen_recovery_prompts(args.n_per_combo, args.workers)
    # bench: use larger n_per_combo since fewer disciplines
    bench_calls = []
    for bench_type, info in BENCH_SPECS.items():
        for disc in info["disciplines"]:
            lang = info.get("lang", "en")
            lang_hint = "Output in Chinese." if lang == "zh" else ""
            bench_calls.append(TeacherCall(
                prompt_id=f"v8::bench::{bench_type}::{disc}",
                messages=[{
                    "role": "user",
                    "content": BENCH_PROMPT_TMPL.format(
                        n=args.bench_per_combo, bench_type=bench_type,
                        user_topic=info["user_topic"], disc=disc,
                        lang_hint=lang_hint,
                    ),
                }],
                max_tokens=1500, temperature=0.85,
                metadata={"kind": "bench", "bench_type": bench_type, "disc": disc, "lang": lang},
            ))

    all_calls = rec_calls + bench_calls
    print(f"  dispatching {len(all_calls)} ({len(rec_calls)} recovery + {len(bench_calls)} bench)")
    t0 = time.time()
    results = batch_call(all_calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    # parse
    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        md = r.metadata
        # recovery uses smaller n_per_combo; bench uses bench_per_combo
        nlim = args.n_per_combo if md["kind"] == "recovery" else args.bench_per_combo
        for i, q in enumerate(arr[:nlim]):
            if not isinstance(q, str) or len(q) < 25: continue
            if md["kind"] == "recovery":
                prompts.append({
                    "prompt_id": f"v8::recovery::{md['scenario']}::{md['disc']}::{i:02d}",
                    "source": "synthetic_v8", "kind": "recovery",
                    "scenario": md["scenario"], "discipline": md["disc"],
                    "lang": "en", "full_prompt": q.strip(),
                })
            else:
                prompts.append({
                    "prompt_id": f"v8::bench::{md['bench_type']}::{md['disc']}::{i:02d}",
                    "source": "synthetic_v8", "kind": "bench",
                    "bench_type": md["bench_type"], "discipline": md["disc"],
                    "lang": md.get("lang", "en"), "full_prompt": q.strip(),
                })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; kinds={dict(Counter(p['kind'] for p in prompts))}")

    done = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done]

    print("[v8/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch(p, st)
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v8/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        if p["kind"] == "recovery":
            scn = RECOVERY_SCENARIOS[p["scenario"]]
            sys_msg = RECOVERY_SYS_TMPL.format(
                scenario=p["scenario"],
                instruction=scn["instruction"],
                n_actions=scn["n_actions"],
            )
            mt = 2500
        else:
            info = BENCH_SPECS[p["bench_type"]]
            sys_msg = BENCH_SYS_TMPL.format(
                bench_type=p["bench_type"],
                user_topic=info["user_topic"],
                n_actions=info["n_actions"],
                system_extra=info["system_extra"],
            )
            mt = 3500 if p["bench_type"] == "sgi_idea_plan_strict" else 2500
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nCandidates:\n" if p["lang"] == "en" else "\n\n候选：\n"
            for i, c in enumerate(cands[:5]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2500]}\nDiscipline: {p['discipline']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=mt, temperature=0.5,
            metadata={"prompt": p, "candidates": cands},
        ))

    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 250: continue
            if r.content.count("```action") < 1 or '"propose"' not in r.content: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "kind": p["kind"],
                "scenario": p.get("scenario"),
                "bench_type": p.get("bench_type"),
                "discipline": p["discipline"], "lang": p["lang"],
                "archetype": f"v8_{p['kind']}",
                "topic": f"[v8/{p['kind']}/{p.get('scenario') or p.get('bench_type')}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v8. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
