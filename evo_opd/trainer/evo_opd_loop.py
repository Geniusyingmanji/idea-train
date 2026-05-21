"""Minimal evo-OPD trainer loop.

End-to-end correctness-focused implementation:

    prompt → student rollout y
           → teacher per-token log_probs (Qwen3LocalTeacher)
           → verifier v(y), lineage c(y, p)        (evo_opd.rewards)
           → per-token reward r_t                  (evo_opd.rewards)
           → policy-gradient update on student LoRA

NOT optimised for throughput. ~1 sample/sec on a single GPU. Validates that
the algorithm runs end-to-end on real data with a real teacher before we
swap in vLLM + FSDP.

Default checkpoint cadence: every 50 steps to `output_dir/checkpoint-NNN`.
Default eval cadence: post-training only (kept simple; run eval_gene_exam
script separately on saved checkpoints).
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
from evo_opd.rewards import EvoOPDReward, EvoOPDRewardConfig, char_uniform_phi_tags
from evo_opd.teachers.qwen3_local import Qwen3LocalTeacher

SYS_PROMPT = (
    "You are a scientific lineage analyst. Answer concisely in the requested "
    "JSON schema. No commentary outside the fenced code block."
)


def build_prompt_pool(prompt_pool_path: str | None, fallback_n: int = 200) -> list[dict]:
    """Load prompts from a JSONL (one row per prompt with keys: prompt_id, prompt,
    task_type, optional parent_card). If path is None, build a fallback pool by
    reusing v3 SFT prompts (these are in the training distribution so they're
    safe to roll out against)."""
    if prompt_pool_path:
        return [json.loads(l) for l in open(prompt_pool_path)]
    # fallback: use first N v3 SFT prompts
    out = []
    src = "/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train_v7.jsonl"
    with open(src) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append({
                "prompt_id": f"v3p::{i:05d}",
                "prompt":    r["prompt"],
                "task_type": r.get("task_type"),
                "parent_card": None,                       # closed-form, no lineage anchor
            })
            if len(out) >= fallback_n:
                break
    return out


@torch.enable_grad()
def rollout_and_score(
    student: torch.nn.Module,
    tokenizer,
    teacher: Qwen3LocalTeacher,
    reward_fn: EvoOPDReward,
    prompt: dict,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
):
    """Run one rollout, score it, compute per-token reward + PG-trainable loss."""
    p = prompt["prompt"]
    tt = prompt.get("task_type")
    parent_card = prompt.get("parent_card")

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user",   "content": p},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=4096).to(device)

    # Sample student rollout (no grad)
    with torch.no_grad():
        out_ids = student.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    n_prompt = inputs["input_ids"].shape[-1]
    completion_ids = out_ids[0, n_prompt:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    if len(completion_ids) == 0:
        return None                                         # degenerate; skip

    # Score under teacher (gives log π_T(y_t | y_<t, x))
    teacher_score = teacher.score_completion(p, completion_text, system=SYS_PROMPT)

    # Re-compute student log-probs WITH grad (re-feed prompt + completion).
    # This is necessary so the PG loss is differentiable wrt student params.
    full_ids = torch.cat([
        inputs["input_ids"][0],
        completion_ids.to(device),
    ]).unsqueeze(0)
    out = student(full_ids)
    logits = out.logits[0]                                  # (T, V)
    shift_logits = logits[n_prompt - 1: n_prompt + len(completion_ids) - 1]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    student_lp = log_probs.gather(1, completion_ids.to(device).unsqueeze(-1)
                                   ).squeeze(-1)            # (n_comp,)

    # Align teacher log_probs to student's completion tokens. They are computed
    # over the same `completion_text`, but teacher's tokenizer may segment
    # differently in edge cases. Truncate to common length.
    n_align = min(len(teacher_score.log_probs), len(student_lp))
    student_lp = student_lp[:n_align]
    teacher_lp = torch.tensor(teacher_score.log_probs[:n_align],
                               device=device, dtype=torch.float32)
    per_token_kl = (student_lp - teacher_lp).detach().tolist()

    # Per-token phi tags (char-uniform from parser; later: align to student tokens)
    phi_char = char_uniform_phi_tags(completion_text, tt)
    # crude: distribute char tags to tokens by uniform char-per-token approx
    # (good enough for development; production wants offset_mapping)
    if not phi_char:
        phi_tokens = ["unknown"] * n_align
    else:
        chars_per_tok = max(1, len(phi_char) // max(n_align, 1))
        phi_tokens = [phi_char[min(i * chars_per_tok, len(phi_char) - 1)]
                       for i in range(n_align)]

    rew = reward_fn(
        text=completion_text,
        per_token_kl=per_token_kl,
        phi_per_token=phi_tokens,
        fld_per_token=None,
        task_type=tt,
        gold_answer=None,                                   # open-ended in this proto
        parent_card=parent_card,
    )

    # Policy gradient loss: − Σ_t r_t · log π_θ(y_t|·)
    r_t = torch.tensor(rew.rewards[:n_align], device=device, dtype=torch.float32)
    pg_loss = -(r_t * student_lp).sum() / max(n_align, 1)

    return {
        "loss": pg_loss,
        "completion": completion_text,
        "n_tokens": n_align,
        "v": rew.verifier.v,
        "kl_mean": sum(per_token_kl) / max(len(per_token_kl), 1),
        "phi_dist": {tag: phi_tokens.count(tag) for tag in set(phi_tokens)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True, help="LoRA dir to start from")
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--student-gpu", default="0")
    ap.add_argument("--teacher-gpu", default="3")
    ap.add_argument("--prompt-pool", default=None,
                    help="JSONL with prompts (default: v7 SFT prompts).")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100,
                    help="Number of (prompt × 1 rollout) gradient steps to run.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lambda-v", type=float, default=0.5)
    ap.add_argument("--lambda-c", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_fp = log_path.open("w")

    def log(s: str):
        print(s, flush=True); log_fp.write(s + "\n"); log_fp.flush()

    # ---- load student
    student_dev = f"cuda:{args.student_gpu}" if torch.cuda.is_available() else "cpu"
    log(f"[1/3] loading student {args.student_base} + LoRA {args.student_lora} on {student_dev}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=student_dev,
    )
    student = PeftModel.from_pretrained(base, args.student_lora,
                                          is_trainable=True)
    student.train()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log(f"  trainable params: {n_train/1e6:.1f}M")

    # ---- load teacher (on a separate GPU)
    teacher_dev = f"cuda:{args.teacher_gpu}" if torch.cuda.is_available() else "cpu"
    log(f"[2/3] loading teacher {args.teacher_model} on {teacher_dev}")
    teacher = Qwen3LocalTeacher(args.teacher_model, device=teacher_dev)

    # ---- prompts
    log(f"[3/3] loading prompts")
    pool = build_prompt_pool(args.prompt_pool, fallback_n=args.n_prompts)
    log(f"  pool: {len(pool)} prompts")

    # ---- optimiser
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                              lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    reward = EvoOPDReward(EvoOPDRewardConfig(lambda_v=args.lambda_v,
                                              lambda_c=args.lambda_c))

    # ---- training loop
    log(f"\n=== evo-OPD loop: {args.steps} steps ===")
    log(f"{'step':>4} {'v':>6} {'kl':>8} {'loss':>9} {'n_tok':>5} {'sec/step':>8}")
    t_start = time.time()
    for step in range(args.steps):
        prompt = pool[step % len(pool)]
        t0 = time.time()
        result = rollout_and_score(student, tok, teacher, reward,
                                     prompt,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature,
                                     device=student_dev)
        if result is None:
            log(f"  step {step}: degenerate (empty completion), skip")
            continue
        loss = result["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], max_norm=1.0
        )
        opt.step()
        sec = time.time() - t0
        log(f"{step:>4} {result['v']:>6.2f} {result['kl_mean']:>+8.3f} "
            f"{float(loss):>+9.4f} {result['n_tokens']:>5} {sec:>8.2f}")

        if (step + 1) % args.ckpt_every == 0:
            ck = out_dir / f"checkpoint-{step+1}"
            student.save_pretrained(ck)
            tok.save_pretrained(ck)
            log(f"  ✓ saved {ck}")

    # final save
    final = out_dir / "final"
    student.save_pretrained(final); tok.save_pretrained(final)
    log(f"\n✓ final saved {final}")
    log(f"total time: {(time.time() - t_start) / 60:.1f} min")
    log_fp.close()


if __name__ == "__main__":
    main()
