"""Agentic-OPD inference adapter for GENE-Arena PES eval.

GENE-Arena uses 50 tasks × 3 settings (Library / Lineage / Question). For each
(task, setting), it constructs a prompt via gene_arena.prompt_builder.PromptBuilder
and expects the model to emit an idea_genome JSON.

With agentic mode, we use the same arena prompt as the `topic` for our rollout
loop. The agent may search our local 855-card corpus for related work, then
emit the final gene_genome via the `propose` tool. Output is saved in the
exact format arena's population_eval expects (same as gene_arena_generate.py).

Then run `tools/gene_arena_pes_eval.py` to score (already exists from v6 work).
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
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving/gene_arena")

import os
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from gene_arena.arena_config import TASK_DIR
from gene_arena.prompt_builder import PromptBuilder, PromptConfig

from evo_opd.agentic.rollout import _ACTION_BLOCK_RE, run_rollout
from evo_opd.tools.read import ReadTool
from evo_opd.tools.search import SearchTool
from evo_opd.tools.web_search import WebSearchTool
from evo_opd.tools.web_read import HybridReadTool


def list_tasks(limit=None):
    out = sorted(Path(TASK_DIR).glob("*.json"))
    if limit:
        out = out[:limit]
    return out


def extract_arena_propose(traj) -> dict | None:
    """Find the propose action's gene_genome in our agentic format."""
    if traj.final_proposal:
        return traj.final_proposal
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True)
    ap.add_argument("--participant", required=True,
                    help="e.g. 'qwen3-8b-agentic-rl'")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="None = all 50")
    ap.add_argument("--settings", nargs="+",
                    default=["Library", "Lineage", "Question"])
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens-per-turn", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--search-backend", choices=["local", "web"], default="local",
                    help="local=BM25 over 855 cards; web=OpenAlex (recommended for OOD arena tasks)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel rollouts (model serialized via lock; tools run free)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ideas_dir = out_dir / "ideas"
    ideas_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("")

    device = f"cuda:{args.gpu}"
    print(f"[1/4] loading model on {device}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, args.student_lora)
    model.eval()

    print(f"[2/4] loading tools (search backend: {args.search_backend})")
    if args.search_backend == "web":
        search_tool = WebSearchTool()
        read_tool = HybridReadTool()
    else:
        search_tool = SearchTool()
        read_tool = ReadTool()
    # v2 tools for richer rollouts
    from evo_opd.tools.genome_tool import GenomeExtractTool
    from evo_opd.tools.diff_tool import GenomeDiffTool
    from evo_opd.tools.novelty_tool import NoveltyCheckTool
    from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT_V2
    extract_tool = GenomeExtractTool()
    diff_tool = GenomeDiffTool(extract_tool=extract_tool)
    novelty_tool = NoveltyCheckTool()

    print(f"[3/4] enumerating arena tasks")
    tasks = list_tasks(args.n_tasks)
    settings = [s for s in args.settings if s in ("Library", "Lineage", "Question")]
    print(f"  {len(tasks)} tasks × {len(settings)} settings = "
          f"{len(tasks) * len(settings)} rollouts")

    print(f"[4/4] generating (workers={args.workers})")
    # Enumerate all (task, setting) pairs to process
    jobs = []
    for task_path in tasks:
        trace_id = task_path.stem
        task_out_dir = ideas_dir / trace_id
        task_out_dir.mkdir(parents=True, exist_ok=True)
        builder = PromptBuilder(task_path)
        for setting in settings:
            out_path = task_out_dir / f"{args.participant}_{setting}.json"
            if out_path.exists():
                continue
            try:
                user_prompt = builder.build(PromptConfig(setting=setting))
            except Exception as e:
                print(f"  ERR build {trace_id}/{setting}: {e}")
                continue
            jobs.append((trace_id, setting, user_prompt, out_path))
    print(f"  {len(jobs)} jobs to run")

    # Model is single-GPU — serialize generate() with a lock.
    # Tool calls (OpenAlex, GPT-5.5) run concurrently inside the rollout closure.
    import threading
    model_lock = threading.Lock()

    # Monkey-patch run_rollout to serialize the model.generate() at the call
    # site? Cleaner: wrap the model itself with a lock.
    _orig_generate = model.generate
    def _locked_generate(*a, **kw):
        with model_lock:
            return _orig_generate(*a, **kw)
    model.generate = _locked_generate
    # Also lock forward pass for grad-free inference
    _orig_call = model.__call__
    def _locked_call(*a, **kw):
        with model_lock:
            return _orig_call(*a, **kw)
    # Note: only generate is used in eval rollouts, no need to lock __call__

    t0 = time.time()
    n_done = n_err = 0
    n_done_lock = threading.Lock()

    def process_job(job):
        nonlocal n_done, n_err
        trace_id, setting, user_prompt, out_path = job
        agent_prompt = {
            "prompt_id": f"arena::{trace_id}::{setting}",
            "topic": user_prompt[:3000],
            "discipline": "general",
            "year_min_hint": 2015,
            "year_max_hint": 2025,
            "target_paper_id": None,
        }
        t_gen_start = time.time()
        try:
            traj = run_rollout(
                model=model, tokenizer=tok, device=device,
                prompt=agent_prompt,
                search_tool=search_tool, read_tool=read_tool,
                max_turns=args.max_turns,
                max_new_tokens_per_turn=args.max_new_tokens_per_turn,
                temperature=args.temperature,
                exclude_target_from_search=False,
                extract_tool=extract_tool,
                diff_tool=diff_tool,
                novelty_tool=novelty_tool,
                system_prompt=ROLLOUT_SYS_PROMPT_V2,
            )
        except Exception as e:
            print(f"  ERR rollout {trace_id}/{setting}: {e}", flush=True)
            with n_done_lock:
                n_err += 1
            return
        t_gen_ms = (time.time() - t_gen_start) * 1000

        gene_genome = extract_arena_propose(traj) or {}
        content = "```json\n" + json.dumps(gene_genome, ensure_ascii=False, indent=2) + "\n```"
        record = {
            "trace_id": trace_id, "task_id": trace_id,
            "participant_id": args.participant,
            "participant_type": "llm",
            "provider": "local_transformers_agentic",
            "model": args.student_base + " + " + args.student_lora,
            "framework": "agentic-opd-v2",
            "harness": None,
            "setting": setting, "content": content, "prompt": user_prompt,
            "output_schema": "OUTPUT_JSON_SCHEMA",
            "input_tokens": 0, "output_tokens": traj.n_generated_tokens,
            "latency_ms": t_gen_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "agentic_mode": True,
                "n_actions": len(traj.actions),
                "search_count": traj.search_count,
                "read_count": traj.read_count,
                "extract_count": getattr(traj, "extract_count", 0),
                "diff_count": getattr(traj, "diff_count", 0),
                "novelty_count": getattr(traj, "novelty_count", 0),
                "read_paper_ids": traj.read_paper_ids,
                "propose_emitted": traj.propose_emitted,
                "malformed_count": traj.malformed_count,
                "temperature": args.temperature,
            },
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        with n_done_lock:
            with manifest.open("a") as mf:
                mf.write(json.dumps({
                    "trace_id": trace_id, "setting": setting,
                    "participant": args.participant, "path": str(out_path),
                    "output_tokens": traj.n_generated_tokens,
                    "latency_ms": t_gen_ms,
                    "propose_emitted": traj.propose_emitted,
                }) + "\n")
            n_done += 1
            if n_done % 5 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{n_done}/{len(jobs)}] {trace_id}/{setting} "
                      f"({el:.1f}min, {t_gen_ms/1000:.1f}s/gen, "
                      f"propose={traj.propose_emitted})", flush=True)

    # Dispatch jobs across worker threads
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_job, j) for j in jobs]
        for f in futures:
            f.result()

    el = (time.time() - t0) / 60
    print(f"\ngenerated {n_done} ideas in {el:.1f} min; {n_err} errors")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
