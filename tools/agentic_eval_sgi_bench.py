"""Agentic-OPD inference adapter for SGI-Bench task_2 (idea generation).

The SGI-Bench expects per-question outputs in this schema:
  {
    "Idea": str,
    "ImplementationSteps": {"1": str, "2": str, ...},
    "ImplementationOrder": ["1-2", "2-3", ...],
    "Dataset": str,
    "EvaluationMetrics": {name: desc, ...},
    "ExpectedOutcome": str
  }

We swap our agent's `propose` tool to emit THIS schema (via system prompt),
then run rollouts on the SGI dataset. The model still searches our 855-card
corpus (which may be discipline-irrelevant — that's OK, the agent will still
produce a structured plan).

Output format matches SGI's step_1: a list of `ques_dict` objects, each with
`generated_idea_text` (raw assistant trace) and `generated_data` (parsed
schema dict). This file can then be fed to SGI's step_2_score.py.

Usage:
  python tools/agentic_eval_sgi_bench.py \\
      --student-lora .../qwen3-8b-agentic-rl/final \\
      --output-json ./eval/results/sgi_bench/agentic_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.agentic.rollout import _ACTION_BLOCK_RE, run_rollout
from evo_opd.tools.read import ReadTool
from evo_opd.tools.search import SearchTool
from evo_opd.tools.web_search import WebSearchTool
from evo_opd.tools.web_read import HybridReadTool


SGI_SYS_PROMPT = """You are a scientific research agent. Given a research question, you must:
  1) Search the local paper corpus for relevant prior work.
  2) Read one or more promising papers (or skip if none are relevant).
  3) Propose a complete research idea with implementation plan.

You have three tools. Each tool call is ONE JSON object inside a fenced ```action ... ``` block. Only one tool call per turn. After each action, you will see a tool result.

Tools:
  search  → ```action
  {"tool": "search", "query": "<short keyword phrase>", "year_min": null, "year_max": null, "k": 5}
  ```
  read    → ```action
  {"tool": "read", "paper_id": "<paper_id_from_search>"}
  ```
  propose → ```action
  {"tool": "propose", "idea_plan": {
    "Idea": "<concrete research idea, 2-4 sentences>",
    "ImplementationSteps": {
      "1": "<step 1 description>",
      "2": "<step 2 description>",
      "3": "<step 3 description>",
      "4": "<step 4 description>",
      "5": "<step 5 description>"
    },
    "ImplementationOrder": ["1-2", "2-3", "3-4", "4-5"],
    "Dataset": "<datasets used / collected>",
    "EvaluationMetrics": {
      "metric_name_1": "<how computed>",
      "metric_name_2": "<how computed>"
    },
    "ExpectedOutcome": "<expected empirical result and significance>"
  }}
  ```

Strict rules:
  - Use AT MOST 4 tool calls total.
  - You MUST call `propose` exactly once, and it MUST be your LAST action.
  - Write a brief 1-2 sentence rationale before each action.
"""


def parse_sgi_propose(raw_action_text: str) -> dict | None:
    """Find the propose action and extract idea_plan in SGI schema."""
    m = _ACTION_BLOCK_RE.search(raw_action_text)
    if not m:
        return None
    try:
        args = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(args, dict):
        return None
    if args.get("tool", "").strip().lower() != "propose":
        return None
    plan = args.get("idea_plan") or args.get("gene_genome") or {}
    if not isinstance(plan, dict):
        return None
    return plan


def load_sgi_dataset(discipline_filter=None):
    """Load SGI-IdeaGeneration test split from HF."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("install `datasets` package")
    ds = load_dataset("InternScience/SGI-IdeaGeneration", split="test")
    if discipline_filter:
        ds = ds.filter(lambda q: q["discipline"] in discipline_filter)
    return ds


def run_sgi_eval(
    model, tokenizer, device: str,
    search_tool, read_tool,
    questions: list[dict],
    *, max_turns: int = 4, temperature: float = 0.5,
    max_new_tokens_per_turn: int = 400,
    extract_tool=None, diff_tool=None, novelty_tool=None,
    workers: int = 1,
) -> list[dict]:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # serialize model.generate() so multiple worker threads can share single GPU
    _orig_generate = model.generate
    _gen_lock = threading.Lock()
    def _locked_generate(*a, **kw):
        with _gen_lock:
            return _orig_generate(*a, **kw)
    model.generate = _locked_generate

    results = [None] * len(questions)
    t0 = time.time()
    n_done = [0]
    n_done_lock = threading.Lock()

    def _run_one(i, q):
        prompt = {
            "prompt_id": f"sgi::{q.get('discipline', '?')}::{i:04d}",
            "topic": q.get("question", ""),
            "discipline": q.get("discipline", ""),
            "year_min_hint": 2018,
            "year_max_hint": 2024,
            "target_paper_id": None,
        }
        traj = run_rollout(
            model=model, tokenizer=tokenizer, device=device,
            prompt=prompt,
            search_tool=search_tool, read_tool=read_tool,
            max_turns=max_turns,
            max_new_tokens_per_turn=max_new_tokens_per_turn,
            temperature=temperature,
            exclude_target_from_search=False,
            extract_tool=extract_tool,
            diff_tool=diff_tool,
            novelty_tool=novelty_tool,
            system_prompt=SGI_SYS_PROMPT,
        )
        parsed = traj.final_proposal or {}
        sgi_out = {
            "Idea":               parsed.get("Idea", parsed.get("idea", "")),
            "ImplementationSteps": parsed.get("ImplementationSteps", parsed.get("implementation_steps", {})),
            "ImplementationOrder": parsed.get("ImplementationOrder", parsed.get("implementation_order", [])),
            "Dataset":            parsed.get("Dataset", parsed.get("dataset", "")),
            "EvaluationMetrics":  parsed.get("EvaluationMetrics", parsed.get("evaluation_metrics", {})),
            "ExpectedOutcome":    parsed.get("ExpectedOutcome", parsed.get("expected_outcome", "")),
        }
        out = dict(q)
        out["generated_idea_text"] = traj.raw_text
        out["generated_data"] = sgi_out
        out["agentic_meta"] = {
            "n_actions": len(traj.actions),
            "search_count": traj.search_count,
            "read_count": traj.read_count,
            "extract_count": getattr(traj, "extract_count", 0),
            "diff_count": getattr(traj, "diff_count", 0),
            "novelty_count": getattr(traj, "novelty_count", 0),
            "read_paper_ids": traj.read_paper_ids,
            "propose_emitted": traj.propose_emitted,
            "malformed_count": traj.malformed_count,
            "wall_time_s": traj.wall_time_s,
        }
        results[i] = out
        with n_done_lock:
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == 1:
                el = time.time() - t0
                avg = el / n_done[0]
                eta = avg * (len(questions) - n_done[0]) / 60
                print(f"  [{n_done[0]}/{len(questions)}] avg={avg:.1f}s/q ETA={eta:.1f}min", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_one, i, q) for i, q in enumerate(questions)]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"  ERR rollout: {type(e).__name__}: {e}", flush=True)

    # drop Nones (errors)
    return [r for r in results if r is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--discipline", nargs="+", default=None,
                    help="filter discipline (e.g. cs); default = all")
    ap.add_argument("--n-questions", type=int, default=None,
                    help="cap dataset size (for smoke); default = all 1000+")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens-per-turn", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--search-backend", choices=["local", "web"], default="local",
                    help="local=BM25 over 855 cards; web=OpenAlex (recommended for OOD prompts)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"[1/4] loading SGI-IdeaGeneration from HF")
    ds = load_sgi_dataset(args.discipline)
    questions = list(ds)
    if args.n_questions:
        questions = questions[:args.n_questions]
    print(f"  {len(questions)} questions"
          + (f" (disciplines={args.discipline})" if args.discipline else ""))

    print(f"[2/4] loading {args.student_base} + LoRA on {device}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, args.student_lora)
    model.eval()

    print(f"[3/4] loading tools (search backend: {args.search_backend})")
    if args.search_backend == "web":
        search_tool = WebSearchTool()
        read_tool = HybridReadTool()
    else:
        search_tool = SearchTool()
        read_tool = ReadTool()
    from evo_opd.tools.genome_tool import GenomeExtractTool
    from evo_opd.tools.diff_tool import GenomeDiffTool
    from evo_opd.tools.novelty_tool import NoveltyCheckTool
    extract_tool = GenomeExtractTool()
    diff_tool = GenomeDiffTool(extract_tool=extract_tool)
    novelty_tool = NoveltyCheckTool()

    print(f"[4/4] running SGI eval (max_turns={args.max_turns}, temp={args.temperature})")
    t0 = time.time()
    results = run_sgi_eval(
        model, tok, device, search_tool, read_tool, questions,
        max_turns=args.max_turns, temperature=args.temperature,
        max_new_tokens_per_turn=args.max_new_tokens_per_turn,
        extract_tool=extract_tool, diff_tool=diff_tool, novelty_tool=novelty_tool,
        workers=args.workers,
    )
    el = (time.time() - t0) / 60
    print(f"  done in {el:.1f} min")

    # save
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ wrote {len(results)} results → {out_path}")

    # diagnostics summary
    n_propose = sum(1 for r in results if r["agentic_meta"]["propose_emitted"])
    n_read = sum(r["agentic_meta"]["read_count"] for r in results)
    avg_actions = sum(r["agentic_meta"]["n_actions"] for r in results) / max(len(results), 1)
    print(f"  propose_emitted: {n_propose}/{len(results)} ({n_propose/len(results)*100:.1f}%)")
    print(f"  avg n_actions: {avg_actions:.2f}")
    print(f"  total reads: {n_read}")


if __name__ == "__main__":
    main()
