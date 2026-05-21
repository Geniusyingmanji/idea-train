"""v12: pure Chinese (zh) broad-discipline expansion.

Current Chinese demos in combined corpus: 279 (8% of 3280), mostly v6 + v9-travel.
This adds 300+ Chinese demos across 15 disciplines, short-biased, mixed schemas.

Output: data/agentic_v12/sft_demos.jsonl  (~300 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v12")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


DISCIPLINES_ZH = [
    ("计算机科学", "computer_science"),
    ("生物学", "biology"),
    ("化学", "chemistry"),
    ("材料科学", "materials"),
    ("物理学", "physics"),
    ("神经科学", "neuroscience"),
    ("机器人控制", "robotics_control"),
    ("能源", "energy"),
    ("临床医学", "clinical_medicine"),
    ("药理学", "pharmacology"),
    ("农业与食品", "agriculture_food"),
    ("城市规划", "urban_planning"),
    ("经济金融", "economics_finance"),
    ("认知科学", "cognitive_science"),
    ("交叉学科", "interdisciplinary"),
]


PROMPT_GEN_ZH = """请生成 {n} 个 {disc} 领域的中文科研提示词。每个 1-3 句，多样化。
有些应该显式要求 gene_genome 格式输出（6 字段结构），有些要求 idea_plan，有些要求自由文本。
输出 JSON 字符串数组，放在 ```json ... ``` 代码块内。"""


LENGTH_TIERS = [
    ("very_short", 0.30, 1800),
    ("short", 0.35, 2200),
    ("medium", 0.25, 2800),
    ("long", 0.10, 3500),
]

LENGTH_GUIDE_ZH = {
    "very_short": "1-2 步（直接 propose 或单次 search 后 propose）",
    "short": "2-3 步（search + 可选 read + propose）",
    "medium": "4-5 步（多次 search + read + propose）",
    "long": "6-8 步（完整链路：多次工具调用）",
}

SCHEMAS = ["gene_genome", "idea_plan", "free_text_answer"]
SCHEMA_HINTS_ZH = {
    "gene_genome": '最终 propose: {"tool":"propose","gene_genome":{"mechanism_genome":"...","niche_genome":"...","observation_genome":"...","limitation_genome":"...","delta_genome":"...","claim_genome":"..."}}',
    "idea_plan": '最终 propose: {"tool":"propose","idea_plan":{"Idea":"...","ImplementationSteps":{"1":"...","2":"..."},"ImplementationOrder":["1-2","2-3"],"Dataset":"...","EvaluationMetrics":{"metric":"desc"},"ExpectedOutcome":"..."}}',
    "free_text_answer": '最终 propose: {"tool":"propose","answer":"<3-5段中文自然语言研究提案>"}',
}


DEMO_SYS_ZH = """你在演示一个 agentic 科研轨迹（中文）。

长度档：{tier_name} ({tier_guide})
输出格式：{schema_name}
{schema_hint}

格式：每一步 1-2 句中文 rationale + ```action ... ``` JSON 工具调用 + [result]: 1-3 句模拟工具结果。最后以 propose 结束。

工具：search, read, extract_genome, genome_diff, novelty_check, propose. JSON key 仍用英文，但 rationale 和分析内容用中文。如有候选论文，请使用真实的 OpenAlex paper_id。"""


def gen_prompts(n_per_disc, workers):
    return [TeacherCall(
        prompt_id=f"v12::{en_disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_ZH.format(n=n_per_disc, disc=zh_disc)}],
        max_tokens=2500, temperature=0.9,
        metadata={"disc_zh": zh_disc, "disc_en": en_disc},
    ) for zh_disc, en_disc in DISCIPLINES_ZH]


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=5, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=25,
                    help="15 disc × 25 = 375 prompts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v12/A1] generating Chinese prompts across 15 disciplines")
    calls = gen_prompts(args.n_per_disc, args.workers)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2030)
    tier_names = [t[0] for t in LENGTH_TIERS]
    tier_weights = [t[1] for t in LENGTH_TIERS]
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        for i, q in enumerate(arr[:args.n_per_disc]):
            if not isinstance(q, str) or len(q) < 15: continue
            tier = rng.choices(tier_names, weights=tier_weights)[0]
            schema = rng.choice(SCHEMAS)
            prompts.append({
                "prompt_id": f"v12::{r.metadata['disc_en']}::{i:02d}::{tier}::{schema}",
                "source": "synthetic_v12", "discipline": r.metadata["disc_en"],
                "discipline_zh": r.metadata["disc_zh"],
                "lang": "zh", "length_tier": tier, "schema": schema,
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; tiers={dict(Counter(p['length_tier'] for p in prompts))} schemas={dict(Counter(p['schema'] for p in prompts))}")

    done = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done]

    print("[v12/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch(p, st)
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v12/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        tier_info = next(t for t in LENGTH_TIERS if t[0] == p["length_tier"])
        sys_msg = DEMO_SYS_ZH.format(
            tier_name=p["length_tier"],
            tier_guide=LENGTH_GUIDE_ZH[p["length_tier"]],
            schema_name=p["schema"],
            schema_hint=SCHEMA_HINTS_ZH[p["schema"]],
        )
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\n候选论文：\n"
            for i, c in enumerate(cands[:5]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"主题：{p['full_prompt'][:2500]}\n学科：{p['discipline_zh']}\n长度档：{p['length_tier']}\n输出格式：{p['schema']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=tier_info[2], temperature=0.5,
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
                "discipline": p["discipline"], "discipline_zh": p["discipline_zh"],
                "lang": "zh", "length_tier": p["length_tier"], "schema": p["schema"],
                "archetype": "v12_chinese_broad",
                "topic": f"[v12/zh/{p['discipline']}/{p['schema']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v12. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
