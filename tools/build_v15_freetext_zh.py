"""v15: free_text_answer focus + ZH-emphasis.

Across combined v3-v14, free_text_answer is the rarest schema and ZH is
under-represented (15%). v15 generates ~300 demos heavily weighted toward
both, mostly short.

Output: data/agentic_v15/sft_demos.jsonl
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v15")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


CONFIGS = [
    # (lang, schema, weight)
    ("zh", "free_text_answer", 0.30),
    ("en", "free_text_answer", 0.20),
    ("zh", "gene_genome", 0.20),
    ("zh", "idea_plan", 0.15),
    ("en", "idea_plan", 0.15),
]

DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials",
    "neuroscience", "robotics_control", "energy", "clinical_medicine",
    "economics_finance", "interdisciplinary",
]


PROMPT_GEN_EN = """Generate {n} research prompts in {discipline}. Mix of styles: some explicitly asking for a free-text research proposal, others implicit. Each 1-3 sentences. Output JSON array inside ```json ... ``` fences."""
PROMPT_GEN_ZH = """请生成 {n} 个 {discipline} 领域的中文科研提示词。一部分明确要求"自由文本研究提案"，一部分则隐式。每个 1-3 句。输出 JSON 字符串数组放在 ```json ... ``` 代码块内。"""

SCHEMA_INSTRUCTIONS = {
    ("zh", "free_text_answer"): '最终 propose: {"tool":"propose","answer":"<3-5段中文自由文本研究提案>"}',
    ("en", "free_text_answer"): 'Final propose: {"tool":"propose","answer":"<3-5 paragraph free-text research proposal>"}',
    ("zh", "gene_genome"): '最终 propose: {"tool":"propose","gene_genome":{"mechanism_genome":"...","niche_genome":"...","observation_genome":"...","limitation_genome":"...","delta_genome":"...","claim_genome":"..."}}',
    ("zh", "idea_plan"): '最终 propose: {"tool":"propose","idea_plan":{"Idea":"...","ImplementationSteps":{"1":"..."},"ImplementationOrder":["1-2"],"Dataset":"...","EvaluationMetrics":{"metric":"desc"},"ExpectedOutcome":"..."}}',
    ("en", "idea_plan"): 'Final propose: {"tool":"propose","idea_plan":{"Idea":"...","ImplementationSteps":{"1":"..."},"ImplementationOrder":["1-2"],"Dataset":"...","EvaluationMetrics":{"metric":"desc"},"ExpectedOutcome":"..."}}',
}

DEMO_SYS_EN = """You are demonstrating a short agentic research trajectory (2-3 actions: 1 search, optionally 1 read, then propose).

{schema_inst}

Format: rationale (1-2 sentences) + ```action ... ``` JSON tool call + [result]: simulated tool result. End with propose. Tools: search, read, propose."""

DEMO_SYS_ZH = """你在演示一个短 agentic 科研轨迹（2-3 步：1 次 search，可选 1 次 read，然后 propose）。

{schema_inst}

格式：1-2 句中文 rationale + ```action ... ``` JSON 工具调用 + [result]: 模拟工具结果。最后以 propose 结束。工具：search, read, propose。"""


def gen_prompts(n_per_combo, workers):
    calls = []
    for lang, schema, _ in CONFIGS:
        for disc in DISCIPLINES:
            tmpl = PROMPT_GEN_ZH if lang == "zh" else PROMPT_GEN_EN
            calls.append(TeacherCall(
                prompt_id=f"v15::{lang}::{schema}::{disc}",
                messages=[{"role": "user", "content": tmpl.format(n=n_per_combo, discipline=disc)}],
                max_tokens=1500, temperature=0.85,
                metadata={"lang": lang, "schema": schema, "disc": disc},
            ))
    return calls


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=3, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=8,
                    help="5 configs × 10 disc × 8 = 400 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v15/A1] generating prompts")
    calls = gen_prompts(args.n_per_combo, args.workers)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        md = r.metadata
        for i, q in enumerate(arr[:args.n_per_combo]):
            if not isinstance(q, str) or len(q) < 20: continue
            prompts.append({
                "prompt_id": f"v15::{md['lang']}::{md['schema']}::{md['disc']}::{i:02d}",
                "source": "synthetic_v15",
                "lang": md["lang"], "schema": md["schema"], "discipline": md["disc"],
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; lang={dict(Counter(p['lang'] for p in prompts))} schema={dict(Counter(p['schema'] for p in prompts))}")

    print("[v15/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch(p, st) for p in prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v15/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        sys_tmpl = DEMO_SYS_ZH if p["lang"] == "zh" else DEMO_SYS_EN
        sys_msg = sys_tmpl.format(schema_inst=SCHEMA_INSTRUCTIONS[(p["lang"], p["schema"])])
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nCandidates:\n" if p["lang"] == "en" else "\n\n候选：\n"
            for i, c in enumerate(cands[:3]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2000]}\nDiscipline: {p['discipline']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=2400, temperature=0.55,
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
            if r.error or not r.content or len(r.content) < 200: continue
            if r.content.count("```action") < 1 or '"propose"' not in r.content: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "lang": p["lang"], "schema": p["schema"],
                "discipline": p["discipline"], "archetype": "v15_freetext_zh",
                "topic": f"[v15/{p['lang']}/{p['schema']}/{p['discipline']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v15. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
