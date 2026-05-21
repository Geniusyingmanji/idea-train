"""v6: multilingual + multi-schema robustness expansion.

Goals:
  1) BILINGUAL: 150 Chinese-language demos (for ArenaRL Open-DeepResearch bridge)
  2) SCHEMA DIVERSITY: explicit demos for each of the 3 output schemas the
     model needs at eval time (gene_genome, idea_plan, free_text_answer) so
     it doesn't collapse into one schema
  3) CONSTRAINED PROMPTS: demos where the prompt explicitly says "respond in
     <schema> only — no extra prose" — train the schema-following habit

Output: data/agentic_v6/sft_demos.jsonl  (~400 demos)
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

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v6")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─── 4 language × schema variants ───────────────────────────────────────────

VARIANTS = [
    # (language, schema, weight)
    ("en", "gene_genome", 0.25),
    ("en", "idea_plan", 0.25),
    ("en", "free_text_answer", 0.10),
    ("zh", "gene_genome", 0.15),
    ("zh", "idea_plan", 0.15),
    ("zh", "free_text_answer", 0.10),
]

SCHEMA_INFO = {
    "gene_genome": {
        "name_en": "gene_genome (6-field structured)",
        "name_zh": "gene_genome（6字段结构）",
        "fields_en": "mechanism_genome, niche_genome, observation_genome, limitation_genome, delta_genome, claim_genome",
        "fields_zh": "mechanism_genome（机制）, niche_genome（场景）, observation_genome（观察）, limitation_genome（局限）, delta_genome（增量）, claim_genome（主张）",
        "format": '{"tool": "propose", "gene_genome": {"mechanism_genome": "...", "niche_genome": "...", "observation_genome": "...", "limitation_genome": "...", "delta_genome": "...", "claim_genome": "..."}}',
    },
    "idea_plan": {
        "name_en": "idea_plan (SGI-Bench style)",
        "name_zh": "idea_plan（SGI-Bench 格式）",
        "fields_en": "Idea, ImplementationSteps (dict), ImplementationOrder (list), Dataset, EvaluationMetrics (dict), ExpectedOutcome",
        "fields_zh": "Idea（想法）, ImplementationSteps（步骤dict）, ImplementationOrder（顺序list）, Dataset（数据集）, EvaluationMetrics（评估指标dict）, ExpectedOutcome（预期结果）",
        "format": '{"tool": "propose", "idea_plan": {"Idea": "...", "ImplementationSteps": {"1": "...", "2": "..."}, "ImplementationOrder": ["1-2", "2-3"], "Dataset": "...", "EvaluationMetrics": {"metric": "desc"}, "ExpectedOutcome": "..."}}',
    },
    "free_text_answer": {
        "name_en": "free-text answer",
        "name_zh": "自由文本回答",
        "fields_en": "answer (3-5 paragraphs of natural language research proposal)",
        "fields_zh": "answer（3-5段自然语言研究提案）",
        "format": '{"tool": "propose", "answer": "<3-5 paragraph natural language proposal>"}',
    },
}


# ─── Disciplines (subset, mixed) ────────────────────────────────────────────

DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "clinical_medicine", "robotics_control",
    "economics_finance", "energy", "interdisciplinary",
]


PROMPT_GEN_TEMPLATE_EN = """Generate {n} diverse research prompts in the area of {discipline}. The agent will eventually output an answer in {schema_name} schema. Make some prompts EXPLICITLY mention the required output schema (e.g., "respond only as gene_genome"), and others leave it implicit. Each prompt 1-3 sentences. Output a JSON array of strings inside ```json ... ``` fences."""

PROMPT_GEN_TEMPLATE_ZH = """请生成 {n} 个 {discipline} 领域的科研提示词。代理将以 {schema_name} 格式给出最终答案。要求其中一部分提示词显式说明输出格式（如"请仅以 gene_genome 格式回答"），另一部分则隐式不提。每个提示词 1-3 句，使用中文。输出 JSON 字符串数组，放入 ```json ... ``` 代码块内。"""


def build_synthetic_prompts(n_per_combo: int, workers: int) -> list[dict]:
    client = build_client()
    calls = []
    for lang, schema, _ in VARIANTS:
        for disc in DISCIPLINES:
            tmpl = PROMPT_GEN_TEMPLATE_ZH if lang == "zh" else PROMPT_GEN_TEMPLATE_EN
            schema_name = SCHEMA_INFO[schema][f"name_{lang}"]
            calls.append(TeacherCall(
                prompt_id=f"v6::{lang}::{schema}::{disc}",
                messages=[{
                    "role": "user",
                    "content": tmpl.format(n=n_per_combo, discipline=disc, schema_name=schema_name),
                }],
                max_tokens=1200,
                temperature=0.85,
                metadata={"lang": lang, "schema": schema, "discipline": disc},
            ))
    print(f"  dispatching {len(calls)} prompt-gen calls ({len(VARIANTS)} variants × {len(DISCIPLINES)} disc)")
    t0 = time.time()
    results = batch_call(calls, workers=workers)
    print(f"  done in {time.time()-t0:.1f}s")

    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2026)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try:
            arr = json.loads(m.group(1) if m else r.content)
        except Exception:
            continue
        if not isinstance(arr, list): continue
        lang, schema, disc = r.metadata["lang"], r.metadata["schema"], r.metadata["discipline"]
        for i, q in enumerate(arr[:n_per_combo]):
            if not isinstance(q, str) or len(q) < 20: continue
            # bias toward short/medium (model needs decisive trajectories)
            tier = rng.choices(
                ["very_short", "short", "medium", "long"],
                weights=[0.2, 0.4, 0.3, 0.1],
            )[0]
            out.append({
                "prompt_id": f"v6::{lang}::{schema}::{disc}::{i:02d}::{tier}",
                "source": "synthetic_v6",
                "lang": lang,
                "schema": schema,
                "discipline": disc,
                "length_tier": tier,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": q.strip(),
            })
    return out


LEN_HINTS = {
    "very_short": ("1-2 actions", 1800),
    "short": ("2-3 actions", 2200),
    "medium": ("4-5 actions", 2800),
    "long": ("6-8 actions", 3800),
}


DEMO_SYS_EN = """You are demonstrating an agentic research trajectory.

Length: {len_hint}.  Output schema: {schema_name}
Final propose MUST emit exactly: {schema_format}
Fields: {fields}

Format each step as: brief rationale + ```action ... ``` block + [result]: simulated tool result. End with `propose`.

Tools: search, read, extract_genome, genome_diff, novelty_check, propose. Use the real OpenAlex paper IDs in candidates when available."""

DEMO_SYS_ZH = """请演示一个agentic科研轨迹。

长度：{len_hint}。输出格式：{schema_name}
最后的 propose 必须严格使用：{schema_format}
字段：{fields}

每个步骤格式：1-2句中文 rationale + ```action ... ``` JSON 工具调用 + [result]: 模拟的工具结果（1-3句）。最后必须以 propose 结束。

可用工具：search, read, extract_genome, genome_diff, novelty_check, propose. 使用候选列表中真实的 OpenAlex paper_id（如有）。工具调用内部仍用英文 key，但 rationale/分析内容用中文。"""


def build_demo_user(prompt: dict, candidates: list[dict]) -> str:
    cand_blob = ""
    if candidates:
        cand_blob = "\n\nCandidates:\n" if prompt["lang"] == "en" else "\n\n候选论文：\n"
        for i, c in enumerate(candidates[:5]):
            cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
    if prompt["lang"] == "en":
        return f"TOPIC: {prompt['full_prompt'][:3000]}\n\nDiscipline: {prompt['discipline']}\nLength tier: {prompt['length_tier']}\nSchema: {prompt['schema']}{cand_blob}"
    return f"主题: {prompt['full_prompt'][:3000]}\n\n学科：{prompt['discipline']}\n长度档：{prompt['length_tier']}\n格式：{prompt['schema']}{cand_blob}"


def prefetch_candidates(prompt: dict, search_tool: WebSearchTool) -> list[dict]:
    try:
        results = search_tool.search(
            prompt["full_prompt"][:200], k=5,
            year_min=prompt.get("year_min_hint"),
            year_max=prompt.get("year_max_hint"),
        )
        return [r.to_dict() for r in results]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=6,
                    help="prompts per (variant × disc); 6 variants × 11 disc × 6 = 396 prompts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v6/A1] synthesizing bilingual + multi-schema prompts")
    prompts = build_synthetic_prompts(args.n_per_combo, args.workers)
    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"  {len(prompts)} prompts; "
          f"lang={dict(Counter(p['lang'] for p in prompts))} "
          f"schema={dict(Counter(p['schema'] for p in prompts))} "
          f"tier={dict(Counter(p['length_tier'] for p in prompts))}")

    done = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done]
        print(f"  resume: {len(prompts)} remaining")

    print(f"[v6/A2] prefetching OpenAlex candidates")
    search_tool = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch_candidates(p, search_tool)
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v6/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        len_hint, mt = LEN_HINTS[p["length_tier"]]
        si = SCHEMA_INFO[p["schema"]]
        sys_tmpl = DEMO_SYS_ZH if p["lang"] == "zh" else DEMO_SYS_EN
        sys_msg = sys_tmpl.format(
            len_hint=len_hint,
            schema_name=si[f"name_{p['lang']}"],
            schema_format=si["format"],
            fields=si[f"fields_{p['lang']}"],
        )
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": build_demo_user(p, prefetched.get(p["prompt_id"], []))},
            ],
            max_tokens=mt,
            temperature=0.55,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], [])},
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
                "lang": p["lang"], "schema": p["schema"],
                "discipline": p["discipline"], "length_tier": p["length_tier"],
                "archetype": "v6_multilingual",
                "topic": f"[{p['lang']}/{p['schema']}/{p['discipline']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v6. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
