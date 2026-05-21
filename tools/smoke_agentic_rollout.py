"""Smoke test: load v3 LoRA + run ONE agentic rollout end-to-end + score it.

Validates:
  - chat template / generation works with the tool-call protocol
  - model emits parseable ```action ... ``` blocks
  - search/read/propose dispatch works
  - trajectory tokens are tagged correctly
  - reward composition produces sensible numbers

This catches integration bugs BEFORE we invest in SFT/RL training.
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

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf")
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.agentic.rewards import (
    AgenticRewardConfig, compute_trajectory_reward,
)
from evo_opd.agentic.rollout import run_rollout
from evo_opd.tools.read import ReadTool
from evo_opd.tools.search import SearchTool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora",
                    default="/home/azureuser/workspace-gzy/zyf/idea_train/train/checkpoints/qwen3-8b-sft-v3/final")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-prompts", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens-per-turn", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"[1/4] loading {args.student_base} + LoRA on {device}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, args.student_lora)
    model.eval()

    print(f"[2/4] loading tools")
    search_tool = SearchTool()
    read_tool = ReadTool()
    print(f"  search corpus: {len(search_tool.docs)} docs")
    print(f"  read corpus:   {len(read_tool.cards)} cards")

    print(f"[3/4] loading prompts")
    prompts_path = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v1/prompts.jsonl")
    prompts = []
    with prompts_path.open() as f:
        for line in f:
            prompts.append(json.loads(line))
            if len(prompts) >= args.n_prompts:
                break
    print(f"  {len(prompts)} prompts loaded")

    print(f"[4/4] running rollouts")
    cfg = AgenticRewardConfig()
    for i, p in enumerate(prompts):
        print(f"\n===== Prompt {i}: {p['prompt_id']} =====")
        print(f"  topic: {p['topic']}")
        print(f"  gold_lineage: {p['gold_lineage']}")
        t0 = time.time()
        traj = run_rollout(
            model=model, tokenizer=tok, device=device,
            prompt=p,
            search_tool=search_tool, read_tool=read_tool,
            max_turns=args.max_turns,
            max_new_tokens_per_turn=args.max_new_tokens_per_turn,
            temperature=args.temperature,
        )
        print(f"\nGENERATION ({traj.wall_time_s:.1f}s, {traj.n_generated_tokens} toks):")
        print("-" * 60)
        print(traj.raw_text)
        print("-" * 60)
        print(f"Actions: {len(traj.actions)} turns")
        for a in traj.actions:
            print(f"  turn {a.turn}: tool={a.tool}  args_keys={list(a.action_args.keys())}")
            print(f"      obs[:100]: {a.observation_text[:120]!r}")
        print(f"  read_paper_ids: {traj.read_paper_ids}")
        print(f"  propose_emitted: {traj.propose_emitted}")
        print(f"  malformed: {traj.malformed_count}")
        print(f"  truncated: {traj.truncated}")
        print(f"  final_proposal: {bool(traj.final_proposal)}")

        # parent card from the prompt's compressed parent for struct
        parent_card = p.get("parent_card_compressed") or {}
        r = compute_trajectory_reward(
            traj, gold_lineage=p["gold_lineage"],
            parent_card=parent_card, config=cfg,
        )
        print(f"\n  REWARD: R_total={r.R_total:+.3f}  "
              f"L={r.R_lineage:.2f} S={r.R_struct:.2f} F={r.R_format:+.2f} E={r.R_efficiency:+.2f}")


if __name__ == "__main__":
    main()
