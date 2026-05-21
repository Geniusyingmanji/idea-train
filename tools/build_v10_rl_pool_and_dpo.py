"""v10: pure RL prompt pool + extended DPO pairs.

Two outputs:
  A) data/agentic_v10/rl_prompts.jsonl — ~1200 diverse prompts (NO demos)
     drawn from disciplines × length-hints × schemas. For future RL rollouts.

  B) data/agentic_v10/preferences.jsonl — ~400 more DPO pairs focused on the
     two HIGHEST-yield rejection modes from v7 (wrong_schema, premature_propose),
     since v7's complex modes (rambly_overlong, made_up_papers) gave low pair rates.
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v10")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RL_OUT = OUT_DIR / "rl_prompts.jsonl"
DPO_OUT = OUT_DIR / "preferences.jsonl"


# ─── A) RL prompt pool ───────────────────────────────────────────────────────

RL_DISCIPLINES = [
    "computer_science", "physics", "chemistry", "biology", "materials",
    "mathematics", "neuroscience", "astronomy", "earth_science", "energy",
    "clinical_medicine", "pharmacology", "robotics_control", "economics_finance",
    "sociology", "agriculture_food", "urban_planning", "cognitive_science",
    "philosophy_ethics", "interdisciplinary",
]

RL_SCHEMAS = ["gene_genome", "idea_plan", "free_text_answer"]
RL_LANGS = ["en", "zh"]

RL_GEN_TEMPLATE_EN = """Generate {n} diverse research prompts in {discipline}. Each 1-3 sentences. {schema_hint} Output JSON array inside ```json ... ``` fences."""
RL_GEN_TEMPLATE_ZH = """生成 {n} 个 {discipline} 领域的科研提示词。每个 1-3 句中文。{schema_hint} 输出 JSON 字符串数组放在 ```json ... ``` 代码块内。"""

SCHEMA_HINTS = {
    "gene_genome": "Make some prompts implicitly invite a gene_genome-style structured idea proposal.",
    "idea_plan": "Make some prompts explicitly ask for an idea + implementation plan (steps, dataset, metrics, expected outcome).",
    "free_text_answer": "Make some prompts open-ended, inviting a 3-5 paragraph natural-language proposal.",
}
SCHEMA_HINTS_ZH = {
    "gene_genome": "部分提示词隐式要求 gene_genome 结构。",
    "idea_plan": "部分提示词显式要求给出 idea + 实现计划（步骤、数据集、指标、预期结果）。",
    "free_text_answer": "部分提示词开放式，邀请 3-5 段自由文本提案。",
}


# ─── B) DPO pairs (focused) ─────────────────────────────────────────────────

DPO_REJECTION_MODES = {
    "premature_propose": {
        "instruction": "Skip searching/reading entirely. Jump directly to `propose` with a vague, generic gene_genome that has no evidence basis. 6 fields each one short generic sentence.",
        "max_tokens": 1500,
    },
    "wrong_schema": {
        "instruction": "Use gene_genome schema when prompt explicitly asks for idea_plan, OR idea_plan when prompt asks for gene_genome. Trajectory itself is sound — just the final schema is wrong.",
        "max_tokens": 2200,
    },
}

DPO_DISCIPLINES = ["computer_science", "biology", "chemistry", "materials",
                   "physics", "neuroscience", "robotics_control",
                   "economics_finance", "interdisciplinary"]

DPO_PROMPT_GEN = """Generate {n} research prompts in {discipline}. Half should explicitly ask for gene_genome output, half should explicitly ask for idea_plan output. Each 1-3 sentences. Output JSON array inside ```json ... ``` fences."""

SCHEMA_GUIDE_DPO = """\
gene_genome: 6-field structure [mechanism_genome, niche_genome, observation_genome, limitation_genome, delta_genome, claim_genome]
idea_plan: [Idea, ImplementationSteps (dict), ImplementationOrder (list), Dataset, EvaluationMetrics (dict), ExpectedOutcome]
"""

CHOSEN_SYS_V10 = f"""You are demonstrating a HIGH-QUALITY agentic research trajectory.

Length: 3-5 actions (search → maybe read → maybe novelty_check → propose).
Be decisive, evidence-grounded, schema-correct. Follow the schema the prompt requests.

{SCHEMA_GUIDE_DPO}

Format each step as: rationale + ```action ... ``` JSON tool call + [result]: 1-3 sentence simulated tool result. End with `propose`.

Tools: search, read, extract_genome, genome_diff, novelty_check, propose. Use real paper_ids from candidates."""

REJECTED_SYS_V10_TMPL = f"""You are demonstrating a LOW-QUALITY agentic trajectory exhibiting a specific failure.

FAILURE MODE: {{rejection_type}}
INSTRUCTION: {{rejection_instr}}

The output should still LOOK like agentic JSON-formatted ReAct attempts, ending in `propose`. But it must exhibit the failure above.

{SCHEMA_GUIDE_DPO}

Format: rationale + ```action ... ``` + [result], ending with `propose`."""


def gen_rl_prompts(n_per_combo, workers):
    """20 disc × 3 schemas × 2 langs = 120 combos × n_per_combo."""
    calls = []
    for disc in RL_DISCIPLINES:
        for schema in RL_SCHEMAS:
            for lang in RL_LANGS:
                tmpl = RL_GEN_TEMPLATE_ZH if lang == "zh" else RL_GEN_TEMPLATE_EN
                hint = (SCHEMA_HINTS_ZH if lang == "zh" else SCHEMA_HINTS)[schema]
                calls.append(TeacherCall(
                    prompt_id=f"v10rl::{lang}::{disc}::{schema}",
                    messages=[{"role": "user", "content": tmpl.format(n=n_per_combo, discipline=disc, schema_hint=hint)}],
                    max_tokens=1200, temperature=0.9,
                    metadata={"kind": "rl_prompt", "lang": lang, "disc": disc, "schema": schema},
                ))
    return calls


def gen_dpo_prompts(n_per_disc, workers):
    return [TeacherCall(
        prompt_id=f"v10dpo::{disc}",
        messages=[{"role": "user", "content": DPO_PROMPT_GEN.format(n=n_per_disc, discipline=disc)}],
        max_tokens=1500, temperature=0.85,
        metadata={"kind": "dpo_prompt", "disc": disc},
    ) for disc in DPO_DISCIPLINES]


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=5, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-per-combo", type=int, default=10,
                    help="20 disc × 3 schemas × 2 langs × 10 = 1200 prompts")
    ap.add_argument("--dpo-per-disc", type=int, default=40,
                    help="9 disc × 40 = 360 prompts → ~360 pairs")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v10/A1] generating prompt-gen calls (RL pool + DPO seeds)")
    rl_calls = gen_rl_prompts(args.rl_per_combo, args.workers)
    dpo_calls = gen_dpo_prompts(args.dpo_per_disc, args.workers)
    all_calls = rl_calls + dpo_calls
    print(f"  {len(all_calls)} ({len(rl_calls)} RL + {len(dpo_calls)} DPO)")
    t0 = time.time()
    results = batch_call(all_calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    rl_prompts, dpo_prompts = [], []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2028)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        md = r.metadata
        if md["kind"] == "rl_prompt":
            for i, q in enumerate(arr[:args.rl_per_combo]):
                if not isinstance(q, str) or len(q) < 20: continue
                rl_prompts.append({
                    "prompt_id": f"v10rl::{md['lang']}::{md['disc']}::{md['schema']}::{i:02d}",
                    "source": "v10_rl_pool", "lang": md["lang"],
                    "discipline": md["disc"], "schema": md["schema"],
                    "full_prompt": q.strip(),
                })
        else:
            rejection_modes = list(DPO_REJECTION_MODES.keys())
            for i, q in enumerate(arr[:args.dpo_per_disc]):
                if not isinstance(q, str) or len(q) < 25: continue
                mode = rng.choice(rejection_modes)
                dpo_prompts.append({
                    "prompt_id": f"v10dpo::{md['disc']}::{i:02d}::{mode}",
                    "source": "v10_dpo", "discipline": md["disc"],
                    "rejection_mode": mode, "full_prompt": q.strip(),
                })

    with RL_OUT.open("w") as f:
        for p in rl_prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  RL pool: {len(rl_prompts)} prompts saved → {RL_OUT}")

    from collections import Counter
    print(f"  RL by schema: {dict(Counter(p['schema'] for p in rl_prompts))}")
    print(f"  RL by lang: {dict(Counter(p['lang'] for p in rl_prompts))}")
    print(f"  DPO prompts: {len(dpo_prompts)}, modes: {dict(Counter(p['rejection_mode'] for p in dpo_prompts))}")

    # ─── DPO generation phase ────────────────────────────────────────────────
    print(f"[v10/A2] prefetching DPO candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch(p, st) for p in dpo_prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v10/A3] generating DPO chosen+rejected pairs (workers={args.workers})")
    chosen_calls, rejected_calls = [], []
    for p in dpo_prompts:
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nCandidates:\n"
            for i, c in enumerate(cands[:5]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2500]}\nDiscipline: {p['discipline']}{cand_blob}"
        chosen_calls.append(TeacherCall(
            prompt_id=f"{p['prompt_id']}::chosen",
            messages=[{"role": "system", "content": CHOSEN_SYS_V10},
                      {"role": "user", "content": user_msg}],
            max_tokens=2800, temperature=0.35,
            metadata={"prompt": p, "candidates": cands, "kind": "chosen"},
        ))
        rej = DPO_REJECTION_MODES[p["rejection_mode"]]
        rej_sys = REJECTED_SYS_V10_TMPL.format(
            rejection_type=p["rejection_mode"],
            rejection_instr=rej["instruction"],
        )
        rejected_calls.append(TeacherCall(
            prompt_id=f"{p['prompt_id']}::rejected",
            messages=[{"role": "system", "content": rej_sys},
                      {"role": "user", "content": user_msg}],
            max_tokens=rej["max_tokens"], temperature=0.6,
            metadata={"prompt": p, "candidates": cands, "kind": "rejected"},
        ))

    all_calls = chosen_calls + rejected_calls
    print(f"  {len(all_calls)} calls ({len(chosen_calls)} chosen + {len(rejected_calls)} rejected)")
    raw_log = DPO_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        all_calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    by_kind = {"chosen": {}, "rejected": {}}
    for r in results:
        if r.error or not r.content or len(r.content) < 200: continue
        kind = r.metadata["kind"]; p = r.metadata["prompt"]
        by_kind[kind][p["prompt_id"]] = (r, p)

    n_pairs = 0
    with DPO_OUT.open("w") as f:
        for pid, (cr, p) in by_kind["chosen"].items():
            if pid not in by_kind["rejected"]: continue
            rr, _ = by_kind["rejected"][pid]
            if cr.content.count("```action") < 1 or '"propose"' not in cr.content: continue
            f.write(json.dumps({
                "prompt_id": pid, "discipline": p["discipline"],
                "rejection_mode": p["rejection_mode"], "full_prompt": p["full_prompt"],
                "candidates": cr.metadata.get("candidates", []),
                "chosen": cr.content, "rejected": rr.content,
                "chosen_output_tokens": cr.output_tokens,
                "rejected_output_tokens": rr.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_pairs += 1
    print(f"\nDone v10. RL prompts={len(rl_prompts)}, DPO pairs={n_pairs}/{len(dpo_prompts)}")
    print(f"  → {RL_OUT}")
    print(f"  → {DPO_OUT}")


if __name__ == "__main__":
    main()
