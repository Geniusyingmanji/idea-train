"""evo-OPD v6 — arena-rank reward (ports ArenaRL to evo-OPD).

Adds a tournament-based relative-ranking signal to the v3 verifier-anchored
GRPO loop. The pointwise judge_PES that was implicit in v5's verifier was
unstable on open-ended tasks → policy collapsed to a high-mean safe mode (PES
−1.56 vs v3 SFT). The fix: replace pointwise rating with a seeded
single-elimination tournament rank over the K group rollouts. See
`evo_opd_arena_rank.md` for the full design rationale and ablation matrix.

Per-token reward:
  r_t = α · v_advantage_t        # verifier (schema + evidence + lineage-c)
      + β · arena_rank_adv_t     # NEW: tournament-rank z-advantage
      − δ · KL[π_θ || π_ref]_t   # reference-policy KL anchor

Both v_advantage and arena_rank_adv are broadcast onto content tokens via the
field-weight α(φ(t)) machinery from `evo_opd.rewards.EvoOPDReward`.

Defaults (tuned per the spec):
  --lambda-v 0.5  (verifier; α in the spec — kept the v3 name for back-compat)
  --lambda-c 0.3  (lineage)
  --lambda-arena 1.0  (β in the spec; **new**)
  --beta-kl-ref 0.01

Pass `--lambda-arena 0` for the v6-no-arena ablation (reproduces v5 collapse).
Pass `--pointwise-judge` for v6-pointwise ablation (same β weight but uses
pointwise GPT-5.5 rating instead of tournament rank).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.judges.pairwise_pes import build_client
from evo_opd.judges.pointwise_pes import score_group_pointwise
from evo_opd.rewards import EvoOPDReward, EvoOPDRewardConfig, char_uniform_phi_tags
from evo_opd.schemas import GENE_FIELDS
from evo_opd.structural import compute_struct, group_struct_zscore
from evo_opd.trainer.tournament import run_tournament

# Reuse v2/v3's per-task system prompts and sampling utilities
from evo_opd.trainer.evo_opd_loop_v2 import (
    SCHEMA_HINT_BY_TASK, build_system_prompt, build_prompt_pool, sample_one_rollout,
)
from evo_opd.trainer.evo_opd_loop_v3 import make_ref_lp_fn


def rollout_one_prompt_v6(
    student, ref_lp_fn, tokenizer, reward_fn,
    prompt: dict, judge_client, *,
    n_samples: int, max_new_tokens: int, temperature: float,
    device: str, top_k_for_grad: int, beta_kl_ref: float,
    lambda_arena: float, judge_workers: int = 8,
    judge_mode: str = "tournament",
    lambda_struct: float = 0.0,
):
    """v3's flow plus a seeded single-elim tournament on the K rollouts."""
    p = prompt["prompt"]
    tt = prompt.get("task_type")
    parent_card = prompt.get("parent_card")
    sys_prompt = build_system_prompt(tt)

    messages = [{"role": "system", "content": sys_prompt},
                 {"role": "user",   "content": p}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=4096,
    ).input_ids[0]

    sample_completions = []
    for _ in range(n_samples):
        comp_ids = sample_one_rollout(
            student, tokenizer, prompt_text, prompt_ids,
            max_new_tokens, temperature, device,
        )
        if len(comp_ids) == 0:
            continue
        text = tokenizer.decode(comp_ids, skip_special_tokens=True)
        sample_completions.append({"ids": comp_ids, "text": text})

    if not sample_completions:
        return None
    K = len(sample_completions)

    # ---- score each via verifier + lineage ----
    for s in sample_completions:
        phi = char_uniform_phi_tags(s["text"], tt)
        chars_per_tok = max(1, len(phi) // max(len(s["ids"]), 1))
        phi_tokens = [phi[min(i * chars_per_tok, len(phi) - 1)]
                       for i in range(len(s["ids"]))]
        if not phi:
            phi_tokens = ["unknown"] * len(s["ids"])
        s["phi_tokens"] = phi_tokens
        rew_no_kl = reward_fn(
            text=s["text"], per_token_kl=[0.0] * len(s["ids"]),
            phi_per_token=phi_tokens, fld_per_token=None,
            task_type=tt, gold_answer=None, parent_card=parent_card,
        )
        s["v"] = rew_no_kl.verifier.v
        s["c"] = rew_no_kl.lineage.c if rew_no_kl.lineage else 0.0

    # ---- GRPO advantage from verifier ----
    vs = [s["v"] for s in sample_completions]
    v_mean = sum(vs) / K
    v_std = (sum((x - v_mean) ** 2 for x in vs) / K) ** 0.5
    for s in sample_completions:
        s["v_advantage"] = (s["v"] - v_mean) / max(v_std + 1e-3, 1e-3)

    # ---- arena/judge advantage signal (tournament OR pointwise) ----
    arena_adv = [0.0] * K
    n_judge_calls = 0
    judge_meta = None
    if lambda_arena > 0 and K >= 2:
        candidates = [s["text"] for s in sample_completions]
        try:
            if judge_mode == "tournament":
                t_out = run_tournament(
                    p, candidates, prompt_id=prompt.get("prompt_id", "step"),
                    client=judge_client, workers=judge_workers,
                    anchor_idx=0,
                )
                arena_adv = list(t_out.z_advantage)
                n_judge_calls = t_out.n_judge_calls
                judge_meta = {
                    "mode": "tournament",
                    "rank": t_out.tournament_rank,
                    "quantile": [round(q, 3) for q in t_out.quantile_reward],
                    "z": [round(z, 3) for z in t_out.z_advantage],
                    "n_calls": n_judge_calls,
                }
            elif judge_mode == "pointwise":
                p_out = score_group_pointwise(
                    p, candidates, prompt_id=prompt.get("prompt_id", "step"),
                    client=judge_client, workers=judge_workers,
                )
                arena_adv = list(p_out["z_advantage"])
                n_judge_calls = p_out["n_judge_calls"]
                judge_meta = {
                    "mode": "pointwise",
                    "scores": [round(s, 3) for s in p_out["scores"]],
                    "z": [round(z, 3) for z in p_out["z_advantage"]],
                    "n_calls": n_judge_calls,
                }
            else:
                raise ValueError(f"unknown judge_mode: {judge_mode}")
        except Exception as e:
            print(f"  [warn] {judge_mode} judge failed: {type(e).__name__}: {e}", flush=True)
            arena_adv = [0.0] * K
    for s, a in zip(sample_completions, arena_adv):
        s["arena_advantage"] = float(a)

    # ---- NEW: structural Layer-1 advantage (deterministic, free) ----
    struct_adv = [0.0] * K
    struct_meta = None
    if lambda_struct > 0 and K >= 2:
        struct_scores = []
        struct_details = []
        for s in sample_completions:
            sc = compute_struct(s["text"], parent_card)
            struct_scores.append(sc.s)
            struct_details.append({
                "s": round(sc.s, 3),
                "inherit": round(sc.inheritance_match, 3),
                "limit":   round(sc.limitation_chain, 3),
                "novelty": round(sc.balanced_novelty, 3),
                "raw_sim": round(sc.raw_inherit_sim, 3),
            })
        struct_adv = group_struct_zscore(struct_scores)
        struct_meta = {
            "scores": [round(s, 3) for s in struct_scores],
            "z":      [round(z, 3) for z in struct_adv],
            "details": struct_details,
        }
    for s, a in zip(sample_completions, struct_adv):
        s["struct_advantage"] = float(a)

    # ---- top-K selection for grad: combined score (verifier + arena + struct) ----
    for s in sample_completions:
        s["combined_score"] = (s["v"]
                               + lambda_arena * s["arena_advantage"]
                               + lambda_struct * s["struct_advantage"])
    sample_completions.sort(key=lambda x: -x["combined_score"])
    top = sample_completions[:top_k_for_grad]

    losses = []
    kl_ref_means = []
    for s in top:
        comp_ids = s["ids"]
        full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(device)
        out = student(full_ids)
        logits = out.logits[0]
        shift_logits = logits[len(prompt_ids) - 1: len(prompt_ids) + len(comp_ids) - 1]
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        stu_lp = log_probs.gather(1, comp_ids.to(device).unsqueeze(-1)).squeeze(-1)
        n = len(stu_lp)

        # reference policy log-probs (frozen original student)
        ref_lp = ref_lp_fn(full_ids, comp_ids, len(prompt_ids))[:n]
        kl_ref = (stu_lp - ref_lp).detach()                # detached: anchor only

        # base per-token reward (verifier · α(φ) inside `rew.rewards`)
        rew = reward_fn(
            text=s["text"], per_token_kl=[0.0] * n,
            phi_per_token=s["phi_tokens"][:n], fld_per_token=None,
            task_type=tt, gold_answer=None, parent_card=parent_card,
        )
        r_t = torch.tensor(rew.rewards[:n], device=device, dtype=torch.float32)

        # broadcast group-relative verifier advantage
        r_t = r_t + s["v_advantage"]

        # tournament/pointwise judge advantage (λ_arena · arena_z)
        r_t = r_t + lambda_arena * s["arena_advantage"]

        # NEW: structural Layer-1 advantage (λ_struct · struct_z)
        r_t = r_t + lambda_struct * s["struct_advantage"]

        # reference KL penalty (clip extreme values for stability)
        r_t = r_t - beta_kl_ref * kl_ref.to(device)

        pg = -(r_t * stu_lp).sum() / max(n, 1)
        losses.append(pg)
        kl_ref_means.append(float(kl_ref.mean().cpu()))

    return {
        "loss":      torch.stack(losses).mean(),
        "v_mean":    v_mean, "v_max": max(vs), "v_min": min(vs), "v_std": v_std,
        "arena_z":   [round(a, 3) for a in arena_adv],
        "struct_z":  [round(a, 3) for a in struct_adv],
        "v_adv":     [round(s["v_advantage"], 3) for s in sample_completions],
        "kl_ref_mean": sum(kl_ref_means) / max(len(kl_ref_means), 1),
        "n_samples": K,
        "n_tokens":  n if losses else 0,
        "n_judge_calls": n_judge_calls,
        "tournament": judge_meta,
        "struct":    struct_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True)
    ap.add_argument("--student-gpu", default="0")
    ap.add_argument("--ref-gpu",     default="2")
    ap.add_argument("--prompt-pool", default=None)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="rollout sampling temp (higher than v3=0.5 to keep diversity for tournament)")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="K = group size for tournament; 8 → 14 judge calls/prompt")
    ap.add_argument("--top-k-for-grad", type=int, default=2)
    ap.add_argument("--lambda-v", type=float, default=0.5)
    ap.add_argument("--lambda-c", type=float, default=0.3)
    ap.add_argument("--lambda-arena", type=float, default=1.0,
                    help="β: tournament-rank advantage weight (0 = ablation v6-no-arena)")
    ap.add_argument("--lambda-struct", type=float, default=0.0,
                    help="γ_struct: Layer-1 structural advantage weight "
                         "(0 = off / v6 default; 1.0 = v6-struct or v6-full)")
    ap.add_argument("--beta-kl-ref", type=float, default=0.01)
    ap.add_argument("--judge-mode", choices=["tournament", "pointwise"], default="tournament",
                    help="tournament = ArenaRL seeded single-elim (default); "
                         "pointwise = v6-pointwise ablation (isolates judge identity)")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "train.log").open("w")
    jsonl_fp = (out_dir / "trace.jsonl").open("w")
    def log(s): print(s, flush=True); log_fp.write(s + "\n"); log_fp.flush()

    student_dev = f"cuda:{args.student_gpu}"
    ref_dev = f"cuda:{args.ref_gpu}"
    log(f"[1/4] loading trainable student on {student_dev}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=student_dev,
    )
    student = PeftModel.from_pretrained(base, args.student_lora, is_trainable=True)
    student.train()

    log(f"[2/4] loading frozen reference on {ref_dev}")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=ref_dev,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.student_lora, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_lp_fn = make_ref_lp_fn(ref_model, ref_dev)

    log(f"[3/4] judge client (Azure GPT-5.5 keyless)")
    judge_client = build_client() if args.lambda_arena > 0 else None

    log(f"[4/4] prompts")
    pool = build_prompt_pool(args.prompt_pool, fallback_n=args.n_prompts)
    log(f"  pool: {len(pool)} prompts; K={args.n_samples} "
        f"top_k={args.top_k_for_grad} temp={args.temperature}")
    log(f"  λ_v={args.lambda_v} λ_c={args.lambda_c} λ_arena={args.lambda_arena} "
        f"λ_struct={args.lambda_struct} β_kl_ref={args.beta_kl_ref} judge_mode={args.judge_mode}")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    reward = EvoOPDReward(EvoOPDRewardConfig(
        lambda_v=args.lambda_v, lambda_c=args.lambda_c,
    ))

    log(f"\n=== evo-OPD v6 loop: {args.steps} steps (tournament-rank arena reward) ===")
    log(f"{'step':>4} {'v_mean':>7} {'v_std':>7} {'kl_ref':>8} {'arena_z':>20} "
        f"{'loss':>9} {'jdg':>4} {'sec':>6}")
    t_start = time.time()
    total_judge_calls = 0

    for step in range(args.steps):
        prompt = pool[step % len(pool)]
        prompt = {**prompt, "prompt_id": f"step_{step:04d}::p_{step % len(pool):04d}"}

        def _ref_lp_xdev(full_ids, comp_ids, n_prompt):
            return ref_lp_fn(full_ids.to(ref_dev), comp_ids.to(ref_dev), n_prompt).to(student_dev)

        t0 = time.time()
        try:
            result = rollout_one_prompt_v6(
                student, _ref_lp_xdev, tok, reward, prompt, judge_client,
                n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, device=student_dev,
                top_k_for_grad=args.top_k_for_grad,
                beta_kl_ref=args.beta_kl_ref,
                lambda_arena=args.lambda_arena,
                judge_workers=args.judge_workers,
                judge_mode=args.judge_mode,
                lambda_struct=args.lambda_struct,
            )
        except Exception as e:
            log(f"  step {step}: ERROR {type(e).__name__}: {str(e)[:200]}")
            continue
        if result is None:
            log(f"  step {step}: empty rollouts")
            continue
        loss = result["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
        )
        opt.step()
        sec = time.time() - t0
        total_judge_calls += result["n_judge_calls"]
        arena_z_str = ",".join(f"{z:+.2f}" for z in result["arena_z"][:4])
        log(f"{step:>4} {result['v_mean']:>7.3f} {result['v_std']:>7.3f} "
            f"{result['kl_ref_mean']:>+8.3f} [{arena_z_str:>18}] "
            f"{float(loss):>+9.4f} {result['n_judge_calls']:>4} {sec:>6.1f}")
        jsonl_fp.write(json.dumps({
            "step": step, "v_mean": result["v_mean"], "v_std": result["v_std"],
            "kl_ref_mean": result["kl_ref_mean"],
            "arena_z": result["arena_z"], "v_adv": result["v_adv"],
            "loss": float(loss), "n_judge_calls": result["n_judge_calls"],
            "tournament": result["tournament"], "sec": sec,
        }) + "\n")
        jsonl_fp.flush()

        if (step + 1) % args.ckpt_every == 0:
            ck = out_dir / f"checkpoint-{step+1}"
            student.save_pretrained(ck); tok.save_pretrained(ck)
            log(f"  ✓ {ck} (total judge calls so far: {total_judge_calls})")

    final = out_dir / "final"
    student.save_pretrained(final); tok.save_pretrained(final)
    log(f"\n✓ final {final}")
    log(f"total: {(time.time() - t_start) / 60:.1f} min, judge_calls={total_judge_calls}")
    log_fp.close(); jsonl_fp.close()


if __name__ == "__main__":
    main()
