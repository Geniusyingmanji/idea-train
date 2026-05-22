"""v21: ZH demos with varied prompt styles (industry/policy/learner/peer-review).

Existing ZH (782 demos) is mostly research-question style. v21 adds:
  - 工业实践 (industry practitioner asking about a real problem)
  - 政策咨询 (policy maker asking for evidence-based research)
  - 学者发问 (curious learner asking for direction)
  - 同行评议 (peer-review style critique)

Output: data/agentic_v21/sft_demos.jsonl  (~200 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v21")
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT / "prompts.jsonl"
DEMOS_OUT = OUT / "sft_demos.jsonl"


STYLES_ZH = [
    ("工业实践", "以工业实践者的口吻提出真实业务问题"),
    ("政策咨询", "以政策制定者的口吻寻求科研证据"),
    ("学者发问", "以好奇的初学者口吻，请求研究方向建议"),
    ("同行评议", "请就一段 2-3 句的提议给出同行评议风格的批评"),
    ("会议征稿", "以会议征稿(CFP)风格描述子课题"),
]

DISCIPLINES_ZH = [
    ("计算机科学", "computer_science"),
    ("生物学", "biology"),
    ("化学", "chemistry"),
    ("材料科学", "materials"),
    ("神经科学", "neuroscience"),
    ("临床医学", "clinical_medicine"),
    ("机器人控制", "robotics_control"),
    ("经济金融", "economics_finance"),
    ("能源", "energy"),
    ("交叉学科", "interdisciplinary"),
]


PROMPT_GEN_TMPL = """请生成 {n} 个 {disc} 领域的中文科研提示词，{style_hint}。每个 1-3 句中文，自然真实。输出 JSON 字符串数组放在 ```json ... ``` 代码块内。"""


DEMO_SYS = """你在演示一个短 agentic 科研轨迹（中文，2-3 步：1 次 search、可选 1 次 read、然后 propose）。

输出格式：propose 中的 gene_genome (6 字段) 或 idea_plan，按提示词自然选择。

格式：1-2 句中文 rationale + ```action ... ``` JSON 工具调用 + [result]: 1-3 句中文模拟工具结果。最后以 propose 结束。

工具：search, read, propose。最终 propose 示例：
{"tool":"propose","gene_genome":{"mechanism_genome":"...","niche_genome":"...","observation_genome":"...","limitation_genome":"...","delta_genome":"...","claim_genome":"..."}}"""


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=3, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=5,
                    help="5 styles × 10 disc × 5 = 250 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v21/A1] generating ZH style-varied prompts")
    calls = [TeacherCall(
        prompt_id=f"v21::{en_disc}::{style}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(
            n=args.n_per_combo, disc=zh_disc, style_hint=style_hint,
        )}],
        max_tokens=1500, temperature=0.9,
        metadata={"zh_disc": zh_disc, "en_disc": en_disc, "style": style},
    ) for style, style_hint in STYLES_ZH for zh_disc, en_disc in DISCIPLINES_ZH]
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
            if not isinstance(q, str) or len(q) < 15: continue
            prompts.append({
                "prompt_id": f"v21::{md['en_disc']}::{md['style']}::{i:02d}",
                "source": "synthetic_v21", "lang": "zh",
                "discipline": md["en_disc"], "discipline_zh": md["zh_disc"],
                "style": md["style"], "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; styles={dict(Counter(p['style'] for p in prompts))}")

    print("[v21/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch(p, st) for p in prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v21/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\n候选：\n"
            for i, c in enumerate(cands[:3]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"主题（{p['style']}）：{p['full_prompt'][:2000]}\n学科：{p['discipline_zh']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": DEMO_SYS},
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
                "lang": "zh", "discipline": p["discipline"],
                "discipline_zh": p["discipline_zh"], "style": p["style"],
                "archetype": "v21_zh_style_variety",
                "topic": f"[v21/zh/{p['discipline']}/{p['style']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v21. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
