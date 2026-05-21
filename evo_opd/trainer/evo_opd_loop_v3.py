"""evo-OPD v3 — verifier-only RL (no teacher KL term).

Lesson from v2: Qwen3-14B teacher was never SFT'd on our schema, so the
reverse-KL term dragged the student AWAY from our format. Dropping the
teacher entirely turns this into GRPO/DAPO-style verifier-based RL.

Changes vs v2:
  1. **No teacher model.** Removes ~28GB GPU usage and the format-mismatch
     pressure.
  2. **Per-token reward = α(φ(t)) · λ_v · v_adv + α(φ(t)) · λ_c · c_adv.**
     No KL term.
  3. **Reference policy KL** (against initial v3 LoRA) added as a small
     anchor to prevent runaway drift (β=0.01).
  4. **Keeps everything else from v2**: schema-anchored system prompt,
     N-sample GRPO advantage, top-K gradient.

This is the standard verifier-based RL setup with our per-token role
gating + lineage signal as the evo-OPD-specific contributions. If this
works on top of v3, it validates our claim that "verifier-anchored +
lineage-aware decoupling" is the load-bearing piece (not the teacher KL).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.rewards import EvoOPDReward, EvoOPDRewardConfig, char_uniform_phi_tags
from evo_opd.schemas import GENE_FIELDS

# Reuse v2's per-task system prompts
from evo_opd.trainer.evo_opd_loop_v2 import (
    SCHEMA_HINT_BY_TASK, build_system_prompt, build_prompt_pool, sample_one_rollout,
)


def rollout_one_prompt_no_teacher(
    student, ref_lp_fn, tokenizer, reward_fn,
    prompt: dict, *,
    n_samples: int, max_new_tokens: int, temperature: float,
    device: str, top_k_for_grad: int, beta_kl_ref: float,
):
    """v2's flow but without teacher; KL is to a frozen reference (initial student)."""
    p = prompt["prompt"]
    tt = prompt.get("task_type")
    parent_card = prompt.get("parent_card")
    sys_prompt = build_system_prompt(tt)

    messages = [{"role": "system", "content": sys_prompt},
                 {"role": "user",   "content": p}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True,
                                                       enable_thinking=False)
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                            max_length=4096).input_ids[0]

    sample_completions = []
    for _ in range(n_samples):
        comp_ids = sample_one_rollout(student, tokenizer, prompt_text, prompt_ids,
                                        max_new_tokens, temperature, device)
        if len(comp_ids) == 0:
            continue
        text = tokenizer.decode(comp_ids, skip_special_tokens=True)
        sample_completions.append({"ids": comp_ids, "text": text})

    if not sample_completions:
        return None

    # score each
    for s in sample_completions:
        phi = char_uniform_phi_tags(s["text"], tt)
        chars_per_tok = max(1, len(phi) // max(len(s["ids"]), 1))
        phi_tokens = [phi[min(i * chars_per_tok, len(phi) - 1)]
                       for i in range(len(s["ids"]))]
        if not phi:
            phi_tokens = ["unknown"] * len(s["ids"])
        s["phi_tokens"] = phi_tokens
        rew_no_kl = reward_fn(text=s["text"],
                                per_token_kl=[0.0] * len(s["ids"]),
                                phi_per_token=phi_tokens, fld_per_token=None,
                                task_type=tt, gold_answer=None,
                                parent_card=parent_card)
        s["v"] = rew_no_kl.verifier.v
        s["c"] = rew_no_kl.lineage.c if rew_no_kl.lineage else 0.0

    # GRPO advantage
    vs = [s["v"] for s in sample_completions]
    v_mean = sum(vs) / len(vs)
    v_std = (sum((x - v_mean) ** 2 for x in vs) / len(vs)) ** 0.5
    for s in sample_completions:
        s["advantage"] = (s["v"] - v_mean) / max(v_std + 1e-3, 1e-3)
    sample_completions.sort(key=lambda x: -x["v"])
    top = sample_completions[:top_k_for_grad]

    losses = []
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

        # per-token PG reward = α·λ_v·v_adv − β·kl_to_ref
        # broadcast verifier advantage onto content tokens
        v_adv = s["advantage"]
        rew = reward_fn(text=s["text"], per_token_kl=[0.0] * n,
                          phi_per_token=s["phi_tokens"][:n], fld_per_token=None,
                          task_type=tt, gold_answer=None,
                          parent_card=parent_card)
        # composer already includes α·λ_v·v_adv but NOT the kl-to-teacher.
        r_t = torch.tensor(rew.rewards[:n], device=device, dtype=torch.float32)
        r_t = r_t + v_adv                                   # broadcast group advantage
        r_t = r_t - beta_kl_ref * kl_ref.to(device)         # reference KL penalty

        pg = -(r_t * stu_lp).sum() / max(n, 1)
        losses.append(pg)

    return {
        "loss":     torch.stack(losses).mean(),
        "v_mean":   v_mean, "v_max": max(vs), "v_min": min(vs), "v_std": v_std,
        "kl_ref_mean": float(kl_ref.mean().cpu()) if losses else 0.0,
        "n_samples": len(sample_completions),
        "n_tokens": n if losses else 0,
    }


@torch.no_grad()
def make_ref_lp_fn(ref_model, device):
    """Returns a closure(full_ids, comp_ids, n_prompt) -> per-token reference log-probs."""
    def _fn(full_ids: torch.Tensor, comp_ids: torch.Tensor, n_prompt: int):
        out = ref_model(full_ids)
        logits = out.logits[0]
        shift = logits[n_prompt - 1: n_prompt + len(comp_ids) - 1]
        lp = torch.log_softmax(shift.float(), dim=-1)
        return lp.gather(1, comp_ids.to(device).unsqueeze(-1)).squeeze(-1)
    return _fn


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
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--top-k-for-grad", type=int, default=2)
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--lambda-c", type=float, default=0.3)
    ap.add_argument("--beta-kl-ref", type=float, default=0.01)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "train.log").open("w")
    def log(s): print(s, flush=True); log_fp.write(s + "\n"); log_fp.flush()

    student_dev = f"cuda:{args.student_gpu}"
    ref_dev = f"cuda:{args.ref_gpu}"
    log(f"[1/3] loading trainable student on {student_dev}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=student_dev,
    )
    student = PeftModel.from_pretrained(base, args.student_lora, is_trainable=True)
    student.train()

    log(f"[2/3] loading frozen reference on {ref_dev}")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=ref_dev,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.student_lora, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_lp_fn = make_ref_lp_fn(ref_model, ref_dev)

    log(f"[3/3] prompts")
    pool = build_prompt_pool(args.prompt_pool, fallback_n=args.n_prompts)
    log(f"  pool: {len(pool)} prompts; n_samples={args.n_samples} "
        f"top_k={args.top_k_for_grad} temp={args.temperature} β_kl_ref={args.beta_kl_ref}")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                              lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    reward = EvoOPDReward(EvoOPDRewardConfig(lambda_v=args.lambda_v, lambda_c=args.lambda_c))

    log(f"\n=== evo-OPD v3 loop: {args.steps} steps (no teacher KL, ref-anchored) ===")
    log(f"{'step':>4} {'v_mean':>7} {'v_max':>7} {'v_std':>7} {'kl_ref':>8} {'loss':>9} {'sec':>6}")
    t_start = time.time()

    for step in range(args.steps):
        prompt = pool[step % len(pool)]
        # full_ids is computed inside rollout_one_prompt_no_teacher; we need ref_lp computed on same device as ref_model
        # Cross-device: full_ids built on student_dev, ref_lp_fn must transfer to ref_dev.
        # Wrap ref_lp_fn to handle device transfer:
        def _ref_lp_xdev(full_ids, comp_ids, n_prompt):
            return ref_lp_fn(full_ids.to(ref_dev), comp_ids.to(ref_dev), n_prompt).to(student_dev)

        t0 = time.time()
        result = rollout_one_prompt_no_teacher(
            student, _ref_lp_xdev, tok, reward, prompt,
            n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, device=student_dev,
            top_k_for_grad=args.top_k_for_grad,
            beta_kl_ref=args.beta_kl_ref,
        )
        if result is None:
            log(f"  step {step}: empty rollouts")
            continue
        loss = result["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], max_norm=1.0
        )
        opt.step()
        sec = time.time() - t0
        log(f"{step:>4} {result['v_mean']:>7.3f} {result['v_max']:>7.3f} "
            f"{result['v_std']:>7.3f} {result['kl_ref_mean']:>+8.3f} "
            f"{float(loss):>+9.4f} {sec:>6.1f}")

        if (step + 1) % args.ckpt_every == 0:
            ck = out_dir / f"checkpoint-{step+1}"
            student.save_pretrained(ck); tok.save_pretrained(ck)
            log(f"  ✓ {ck}")

    final = out_dir / "final"
    student.save_pretrained(final); tok.save_pretrained(final)
    log(f"\n✓ final {final}")
    log(f"total: {(time.time() - t_start) / 60:.1f} min")
    log_fp.close()


if __name__ == "__main__":
    main()
