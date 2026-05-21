"""GRPO training loop for agentic-OPD.

For each step:
  1. Sample one prompt from data/agentic_v1/prompts.jsonl
  2. Run K trajectory rollouts (search → read → propose, with tools)
  3. Compute per-trajectory reward via rewards_agentic
  4. (Optional) tournament over the K final proposals → arena rank advantage
  5. Group z-normalize R_total → trajectory advantage
  6. PG update on generated tokens (gen_mask), with reference-policy KL

Key differences vs v6 single-turn loop:
  - Generation is multi-turn with tool dispatch (handled by rollout.py)
  - Token-level mask separates model tokens from observation tokens
  - Reward is per-trajectory (not per-rollout-token-broadcast)
  - Group size K=4 (smaller because rollouts are 4-6× longer)
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

from evo_opd.agentic.rewards import (
    AgenticRewardConfig, compute_trajectory_reward,
)
from evo_opd.agentic.rollout import run_rollout
from evo_opd.judges.pairwise_pes import build_client as build_judge_client
from evo_opd.tools.read import ReadTool
from evo_opd.tools.search import SearchTool
from evo_opd.tools.web_search import WebSearchTool
from evo_opd.tools.web_read import HybridReadTool
from evo_opd.tools.genome_tool import GenomeExtractTool
from evo_opd.tools.diff_tool import GenomeDiffTool
from evo_opd.tools.novelty_tool import NoveltyCheckTool
from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT_V2
from evo_opd.trainer.evo_opd_loop_v3 import make_ref_lp_fn
from evo_opd.trainer.tournament import run_tournament


def trajectory_pg_loss(
    *,
    model, full_ids: list[int], gen_mask: list[bool],
    advantage: float, beta_kl_ref: float, ref_lp_fn,
    device: str,
) -> tuple[torch.Tensor, float, int]:
    """Standard GRPO loss on generated tokens of one trajectory.

    Returns (loss_tensor, mean_kl_ref, n_grad_tokens).
    """
    import torch.nn.functional as F

    full_t = torch.tensor([full_ids], device=device)
    out = model(full_t)
    logits = out.logits[0]                   # [T, V]
    # mask: predict-token at position t is generated iff gen_mask[t+1] is True
    mask_next = torch.tensor(gen_mask[1:], device=device, dtype=torch.float32)
    n_grad_tokens = int(mask_next.sum().item())
    if n_grad_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    # MEMORY-EFFICIENT log P_θ(full_ids[t+1] | full_ids[:t+1]):
    # F.cross_entropy on bf16 logits — no .float() cast (saves ~3GB on long seq).
    # PyTorch ≥2.0 handles bf16 CE numerics fine for our scale.
    shift_logits = logits[:-1]                                  # [T-1, V] bf16
    next_ids = torch.tensor(full_ids[1:], device=device)
    stu_lp = -F.cross_entropy(
        shift_logits, next_ids, reduction="none",
    ).float()                                                    # [T-1] cast result only

    # reference policy log-probs (frozen reference — already detached on the way out)
    ref_lp = ref_lp_fn(full_t)
    kl_ref = (stu_lp - ref_lp.to(device)).detach()              # [T-1]

    # per-token reward = advantage − β·kl_to_ref, masked to generated tokens
    r_per_tok = (advantage - beta_kl_ref * kl_ref) * mask_next
    pg = -(r_per_tok * stu_lp).sum() / max(n_grad_tokens, 1)
    return pg, float((kl_ref * mask_next).sum().item() / max(n_grad_tokens, 1)), n_grad_tokens


@torch.no_grad()
def make_full_ref_lp_fn(ref_model, ref_device: str, student_device: str):
    """Closure returning per-token reference log-probs for a full sequence.
    Uses F.cross_entropy for memory efficiency (no full softmax tensor)."""
    import torch.nn.functional as F
    def _fn(full_ids_t: torch.Tensor) -> torch.Tensor:
        full_t = full_ids_t.to(ref_device)
        out = ref_model(full_t)
        logits = out.logits[0]
        shift = logits[:-1]
        next_ids = full_t[0, 1:]
        # bf16 CE for memory; cast result to fp32 for downstream subtraction
        lp = -F.cross_entropy(shift, next_ids, reduction="none").float()
        return lp.to(student_device)
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True,
                    help="path to v3 LoRA OR sft-agentic-warmstart checkpoint")
    ap.add_argument("--student-gpu", default="0")
    ap.add_argument("--ref-gpu", default="2")
    ap.add_argument("--prompts",
                    default="/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v1/prompts.jsonl")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--K", type=int, default=4, help="group size")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens-per-turn", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--beta-kl-ref", type=float, default=0.01)
    ap.add_argument("--alpha-lineage", type=float, default=0.3)
    ap.add_argument("--alpha-struct", type=float, default=0.5)
    ap.add_argument("--alpha-arena", type=float, default=0.0,
                    help="set >0 to enable tournament over final proposals")
    ap.add_argument("--alpha-efficiency", type=float, default=0.1)
    ap.add_argument("--alpha-format", type=float, default=0.3)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--output-dir", required=True)
    # v2: tool selection
    ap.add_argument("--use-web-search", action="store_true",
                    help="use OpenAlex web search instead of local BM25")
    ap.add_argument("--use-v2-tools", action="store_true",
                    help="enable extract_genome / genome_diff / novelty_check")
    ap.add_argument("--system-prompt", default="v1",
                    choices=["v1", "v2"],
                    help="v1=3-tool prompt; v2=6-tool prompt")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "train.log").open("w")
    jsonl_fp = (out_dir / "trace.jsonl").open("w")
    def log(s): print(s, flush=True); log_fp.write(s + "\n"); log_fp.flush()

    student_dev = f"cuda:{args.student_gpu}"
    ref_dev = f"cuda:{args.ref_gpu}"

    log(f"[1/5] loading trainable student on {student_dev}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=student_dev,
    )
    student = PeftModel.from_pretrained(base, args.student_lora, is_trainable=True)
    # Gradient checkpointing — trades compute for memory.
    # Needed because each trajectory is a multi-turn rollout (3-4K tokens),
    # and full-graph backprop on a single trajectory blows past 80 GB otherwise.
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()  # required by PEFT + gradient_checkpointing
    student.train()

    log(f"[2/5] loading frozen reference on {ref_dev}")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=ref_dev,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.student_lora, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_lp_fn = make_full_ref_lp_fn(ref_model, ref_dev, student_dev)

    log(f"[3/5] loading tools (web={args.use_web_search}, v2_tools={args.use_v2_tools})")
    if args.use_web_search:
        search_tool = WebSearchTool()
        read_tool = HybridReadTool()
    else:
        search_tool = SearchTool()
        read_tool = ReadTool()
    extract_tool = GenomeExtractTool() if args.use_v2_tools else None
    diff_tool = GenomeDiffTool(extract_tool=extract_tool) if args.use_v2_tools else None
    novelty_tool = NoveltyCheckTool() if args.use_v2_tools else None
    sys_prompt = ROLLOUT_SYS_PROMPT_V2 if args.system_prompt == "v2" else None

    log(f"[4/5] loading prompts")
    prompts = []
    with Path(args.prompts).open() as f:
        for line in f:
            prompts.append(json.loads(line))
    log(f"  {len(prompts)} prompts; K={args.K} max_turns={args.max_turns}")

    log(f"[5/5] judge client (only used if α_arena > 0)")
    judge_client = build_judge_client() if args.alpha_arena > 0 else None

    cfg = AgenticRewardConfig(
        alpha_lineage=args.alpha_lineage,
        alpha_struct=args.alpha_struct,
        alpha_arena=args.alpha_arena,
        alpha_efficiency=args.alpha_efficiency,
        alpha_format=args.alpha_format,
    )
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    log(f"\n=== agentic-OPD loop: {args.steps} steps "
        f"(α_L={cfg.alpha_lineage} α_S={cfg.alpha_struct} α_A={cfg.alpha_arena} "
        f"α_F={cfg.alpha_format} α_E={cfg.alpha_efficiency}) ===")
    log(f"{'step':>4} {'R_mean':>7} {'R_std':>7} {'L̄':>5} {'S̄':>5} {'F̄':>5} {'kl_ref':>8} "
        f"{'loss':>9} {'gen_toks':>8} {'sec':>6}")

    t_start = time.time()
    for step in range(args.steps):
        prompt = prompts[step % len(prompts)]
        prompt_id = prompt["prompt_id"]
        t0 = time.time()

        # 1. Roll out K trajectories sequentially (could parallelize across GPUs later)
        trajectories = []
        rewards = []
        diags = []
        for k in range(args.K):
            traj = run_rollout(
                model=student, tokenizer=tok, device=student_dev,
                prompt=prompt,
                search_tool=search_tool, read_tool=read_tool,
                max_turns=args.max_turns,
                max_new_tokens_per_turn=args.max_new_tokens_per_turn,
                temperature=args.temperature,
                extract_tool=extract_tool,
                diff_tool=diff_tool,
                novelty_tool=novelty_tool,
                system_prompt=sys_prompt,
            )
            r = compute_trajectory_reward(
                traj,
                gold_lineage=prompt.get("gold_lineage", []),
                parent_card=prompt.get("parent_card_compressed") or {},
                config=cfg,
            )
            trajectories.append(traj)
            rewards.append(r)
            diags.append(r.diagnostics)

        # 2. Optional: tournament over final proposals → arena advantage
        if args.alpha_arena > 0:
            # build text list (skip trajectories with no proposal)
            cand_texts = []
            cand_idx = []
            for i, t in enumerate(trajectories):
                if t.final_proposal:
                    cand_texts.append(json.dumps(t.final_proposal, ensure_ascii=False)[:2500])
                    cand_idx.append(i)
            if len(cand_texts) >= 2:
                t_out = run_tournament(
                    prompt["topic"], cand_texts, prompt_id=prompt_id,
                    client=judge_client, workers=args.judge_workers,
                )
                # map z back to original indices
                arena_z_per_traj = [0.0] * args.K
                for j, ci in enumerate(cand_idx):
                    arena_z_per_traj[ci] = t_out.z_advantage[j]
                for i, r in enumerate(rewards):
                    r.R_arena = arena_z_per_traj[i]
                    r.R_total = (
                        cfg.alpha_lineage * r.R_lineage
                        + cfg.alpha_struct * r.R_struct
                        + cfg.alpha_arena * r.R_arena
                        + cfg.alpha_efficiency * r.R_efficiency
                        + cfg.alpha_format * r.R_format
                    )

        # 3. Group z-normalize R_total
        rs = [r.R_total for r in rewards]
        mu = sum(rs) / len(rs)
        var = sum((x - mu) ** 2 for x in rs) / len(rs)
        sigma = max(var ** 0.5, 1e-3)
        advantages = [(r.R_total - mu) / sigma for r in rewards]

        # 4. PG loss on each trajectory, mean the losses
        losses = []
        kl_means = []
        n_grad_tokens_total = 0
        for traj, adv in zip(trajectories, advantages):
            if not any(traj.gen_mask):
                continue
            loss_i, kl_i, n_i = trajectory_pg_loss(
                model=student,
                full_ids=traj.full_ids, gen_mask=traj.gen_mask,
                advantage=adv, beta_kl_ref=args.beta_kl_ref,
                ref_lp_fn=ref_lp_fn, device=student_dev,
            )
            losses.append(loss_i)
            kl_means.append(kl_i)
            n_grad_tokens_total += n_i

        if not losses:
            log(f"step {step}: no usable trajectories, skipping")
            continue

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
        )
        opt.step()

        # diagnostics
        L̄ = sum(r.R_lineage for r in rewards) / len(rewards)
        S̄ = sum(r.R_struct for r in rewards) / len(rewards)
        F̄ = sum(r.R_format for r in rewards) / len(rewards)
        kl_ref_mean = sum(kl_means) / max(len(kl_means), 1)
        sec = time.time() - t0

        log(f"{step:>4} {mu:>+7.3f} {sigma:>7.3f} {L̄:>5.2f} {S̄:>5.2f} {F̄:>+5.2f} "
            f"{kl_ref_mean:>+8.3f} {float(loss):>+9.4f} {n_grad_tokens_total:>8d} {sec:>6.1f}")

        # detailed trace
        jsonl_fp.write(json.dumps({
            "step": step, "prompt_id": prompt_id,
            "R_mean": mu, "R_std": sigma,
            "R_components_mean": {
                "L": L̄, "S": S̄, "F": F̄,
                "E": sum(r.R_efficiency for r in rewards) / len(rewards),
                "A": sum(r.R_arena for r in rewards) / len(rewards),
            },
            "rewards": rs,
            "advantages": advantages,
            "kl_ref_mean": kl_ref_mean,
            "loss": float(loss),
            "n_grad_tokens": n_grad_tokens_total,
            "traj_summaries": [t.to_dict() for t in trajectories],
            "sec": sec,
        }) + "\n")
        jsonl_fp.flush()

        if (step + 1) % args.ckpt_every == 0:
            ck = out_dir / f"checkpoint-{step+1}"
            student.save_pretrained(ck); tok.save_pretrained(ck)
            log(f"  ✓ {ck}")

    final = out_dir / "final"
    student.save_pretrained(final); tok.save_pretrained(final)
    log(f"\n✓ final {final}")
    log(f"total: {(time.time() - t_start) / 60:.1f} min")
    log_fp.close(); jsonl_fp.close()


if __name__ == "__main__":
    main()
