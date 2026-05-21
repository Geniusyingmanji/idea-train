"""v9: multi-turn refinement + ArenaRL Chinese expansion + GENE-Arena cross-cut.

Three buckets:
  A) MULTI_TURN_REFINEMENT (~200 demos): 2-turn dialogues.
     Turn 1: agent proposes (gene_genome).
     Turn 2: user feedback: "the delta is too vague" / "mechanism overlaps with X" / "no concrete experiment"
     Turn 3: agent revises and re-proposes.
     This teaches the model to incorporate feedback, key for RL with reward signals.

  B) ARENA_TRAVEL_ZH boost (~100 demos): Chinese travel-itinerary planning,
     ArenaRL Open-Travel style. Diverse cities, durations, budgets, themes.

  C) GENE_ARENA_CROSSCUT boost (~100 demos): explicitly cross-disciplinary
     mechanism borrowing (physics→ML, evolution→RL, etc).

Output: data/agentic_v9/sft_demos.jsonl  (~400 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v9")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─── A) Multi-turn refinement ──────────────────────────────────────────────

FEEDBACK_PATTERNS = [
    "the delta_genome is too vague — quantify the improvement (e.g., '15% better F1' or 'reduces compute by 3x') and name the baseline",
    "the mechanism_genome overlaps with an existing work — sharpen it to highlight what's NEW",
    "the limitation_genome doesn't suggest how to overcome the limitation — add a specific path",
    "no concrete experiment is specified — add a measurable validation plan in claim_genome",
    "the niche_genome is too broad — narrow to a specific population/dataset/regime",
    "observation_genome reads like a summary, not an observation — give one concrete data point",
    "delta_genome contradicts mechanism_genome — make them consistent",
    "the proposal lacks novelty — pivot to a sharper angle while staying in the same niche",
]

MULTITURN_DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "robotics_control", "energy", "clinical_medicine",
    "interdisciplinary",
]

MULTITURN_PROMPT_GEN = """Generate {n} diverse research prompts in {discipline}. Each 1-3 sentences, suitable for a researcher asking for a gene_genome-style idea proposal. Output JSON array inside ```json ... ``` fences."""

MULTITURN_DEMO_SYS = """You are demonstrating a TWO-TURN refinement trajectory.

TURN 1: The agent proposes a gene_genome using 2-4 tools (search, read, propose).
TURN 2 (USER FEEDBACK): The user replies with a SPECIFIC critique: "{feedback}"
TURN 3: The agent acknowledges the feedback, optionally does 1 more tool call (search or novelty_check), then re-proposes a REVISED gene_genome that directly addresses the critique.

Format:
  [Turn 1 - Agent]
  <rationale> + ```action ... ``` + [result], ending with a `propose` action.
  [Turn 2 - User]
  <one-line user critique using EXACTLY the feedback above>
  [Turn 3 - Agent]
  <acknowledgment of feedback> + optional 1 tool call + revised `propose`.

The revised gene_genome must visibly fix the criticized field while keeping the rest coherent. Tools: search, read, extract_genome, genome_diff, novelty_check, propose."""


# ─── B) Arena travel ZH ─────────────────────────────────────────────────────

TRAVEL_PROMPT_GEN = """生成 {n} 个中文旅行行程规划提示词。每个 1-3 句，包含明确的约束（天数、预算、出发地、目的地、偏好如美食/历史/亲子）。多样化：日本、欧洲、东南亚、国内、北美等。输出 JSON 字符串数组，放在 ```json ... ``` 代码块内。"""

TRAVEL_DEMO_SYS = """你在演示一个 agentic 旅行规划轨迹（ArenaRL Open-Travel 风格）。

长度：2-4 步（可选 1-2 次 search 检索相关攻略/地点，然后 propose）。

格式：每一步 1-2 句中文 rationale + ```action ... ``` JSON 工具调用 + [result]: 模拟工具结果。最终 propose：
{"tool": "propose", "answer": "<3-5段中文行程，包括逐日安排、预算估计、餐饮/交通推荐、注意事项>"}

工具：search, read, propose（其他工具一般不用）。如使用 search，用中文或英文检索旅游攻略关键词。"""


# ─── C) GENE-Arena cross-cutting ────────────────────────────────────────────

CROSSCUT_PAIRS = [
    ("physics", "machine_learning"),
    ("biology", "robotics"),
    ("evolution", "multi_agent_systems"),
    ("neuroscience", "deep_learning"),
    ("chemistry", "materials_discovery"),
    ("economics", "social_simulation"),
    ("information_theory", "biology"),
    ("statistical_mechanics", "neural_networks"),
    ("game_theory", "communication_systems"),
    ("topology", "computer_vision"),
    ("category_theory", "programming_languages"),
    ("astronomy", "data_science"),
]

CROSSCUT_PROMPT_GEN = """Generate {n} research prompts that explicitly bridge {field_a} concepts with {field_b} applications. Each prompt 1-3 sentences, asking for a research idea where a mechanism from {field_a} is applied to a {field_b} problem. Output JSON array inside ```json ... ``` fences."""

CROSSCUT_DEMO_SYS = """You are demonstrating an agentic research trajectory for a CROSS-DISCIPLINARY proposal.

Length: 4-6 actions. The trajectory should explicitly search BOTH fields' literature, then synthesize.

Final propose schema:
{"tool": "propose", "gene_genome": {"mechanism_genome": "<from field A — name the borrowed concept>", "niche_genome": "<the field B problem>", "observation_genome": "<the empirical surprise or gap>", "limitation_genome": "<what makes this hard>", "delta_genome": "<the specific cross-domain contribution>", "claim_genome": "<measurable claim>"}}

Format: rationale + ```action ... ``` + [result]. Tools: search, read, extract_genome, genome_diff, novelty_check, propose. Use search to query both fields' literature when relevant."""


# ─── Prompt synthesis ──────────────────────────────────────────────────────

def gen_multiturn_prompts(n_per_disc, workers):
    calls = [TeacherCall(
        prompt_id=f"v9::mt::{disc}",
        messages=[{"role": "user", "content": MULTITURN_PROMPT_GEN.format(n=n_per_disc, discipline=disc)}],
        max_tokens=1200, temperature=0.85,
        metadata={"kind": "multiturn", "discipline": disc},
    ) for disc in MULTITURN_DISCIPLINES]
    return calls


def gen_travel_prompts(n_total, workers):
    # 10 batches of n_total/10
    n_per_batch = max(5, n_total // 10)
    return [TeacherCall(
        prompt_id=f"v9::travel::{i}",
        messages=[{"role": "user", "content": TRAVEL_PROMPT_GEN.format(n=n_per_batch)}],
        max_tokens=1200, temperature=0.9,
        metadata={"kind": "travel", "batch": i},
    ) for i in range(10)]


def gen_crosscut_prompts(n_per_pair, workers):
    return [TeacherCall(
        prompt_id=f"v9::cc::{a}__{b}",
        messages=[{"role": "user", "content": CROSSCUT_PROMPT_GEN.format(n=n_per_pair, field_a=a, field_b=b)}],
        max_tokens=1200, temperature=0.85,
        metadata={"kind": "crosscut", "field_a": a, "field_b": b},
    ) for a, b in CROSSCUT_PAIRS]


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=5, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt-per-disc", type=int, default=20)  # 10 disc × 20 = 200 mt prompts
    ap.add_argument("--travel-total", type=int, default=100)
    ap.add_argument("--cc-per-pair", type=int, default=10)  # 12 pairs × 10 = 120 cc prompts
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v9/A1] generating prompt-gen calls")
    mt_calls = gen_multiturn_prompts(args.mt_per_disc, args.workers)
    travel_calls = gen_travel_prompts(args.travel_total, args.workers)
    cc_calls = gen_crosscut_prompts(args.cc_per_pair, args.workers)
    all_calls = mt_calls + travel_calls + cc_calls
    print(f"  dispatching {len(all_calls)} ({len(mt_calls)} mt + {len(travel_calls)} travel + {len(cc_calls)} cc)")
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
        kind = md["kind"]
        for i, q in enumerate(arr):
            if not isinstance(q, str) or len(q) < 20: continue
            base = {"source": "synthetic_v9", "kind": kind, "lang": "zh" if kind == "travel" else "en",
                    "year_min_hint": 2018, "year_max_hint": 2025, "full_prompt": q.strip()}
            if kind == "multiturn":
                if len(prompts) - sum(1 for p in prompts if p["kind"] != "multiturn") >= args.mt_per_disc * len(MULTITURN_DISCIPLINES):
                    break
                fb = random.Random(hash(f"{md['discipline']}{i}")).choice(FEEDBACK_PATTERNS)
                base.update({"prompt_id": f"v9::mt::{md['discipline']}::{i:02d}",
                             "discipline": md["discipline"], "feedback": fb})
            elif kind == "travel":
                base.update({"prompt_id": f"v9::travel::{md['batch']}::{i:02d}", "discipline": "travel_planning"})
            else:  # crosscut
                base.update({"prompt_id": f"v9::cc::{md['field_a']}_x_{md['field_b']}::{i:02d}",
                             "discipline": f"{md['field_a']}__{md['field_b']}",
                             "field_a": md["field_a"], "field_b": md["field_b"]})
            prompts.append(base)

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

    print("[v9/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch(p, st)
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v9/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        kind = p["kind"]
        if kind == "multiturn":
            sys_msg = MULTITURN_DEMO_SYS.format(feedback=p["feedback"])
            mt = 3500
        elif kind == "travel":
            sys_msg = TRAVEL_DEMO_SYS
            mt = 2800
        else:
            sys_msg = CROSSCUT_DEMO_SYS
            mt = 3000
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
            max_tokens=mt, temperature=0.55,
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
                "prompt_id": p["prompt_id"], "source": p["source"], "kind": p["kind"],
                "discipline": p.get("discipline"), "lang": p["lang"],
                "feedback": p.get("feedback"),
                "field_a": p.get("field_a"), "field_b": p.get("field_b"),
                "archetype": f"v9_{p['kind']}",
                "topic": f"[v9/{p['kind']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v9. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
