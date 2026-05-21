"""evo-OPD trainer v2 — fixes from v1 lessons.

Changes vs v1 (`evo_opd_loop.py`):
  1. **Lower default temperature** (0.3, was 0.7). v1 rollouts drifted into
     prompt-regurgitation at high temp; v2 stays near the SFT distribution.
  2. **Multi-sample GRPO-style rollouts** (default N=4 per prompt). Compute
     per-sample verifier reward, then advantage as (v_i − mean) / (std + ε).
     Per-token reward uses the advantage signal even when raw v_i are equal.
  3. **Schema-anchoring system prompt**: explicitly lists the required JSON
     keys per task type so the student starts from a closer-to-correct init.
  4. **Reward-aware sample selection**: only the top-K samples (by v) get a
     gradient step. This is the cheap surrogate for GRPO without a full
     critic.
  5. **Per-step diagnostic**: log mean+std of v across N rollouts to detect
     "all v=0" collapse early.

End-to-end correctness still validated against the same parser/verifier/
lineage code as v1.
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
from evo_opd.schemas import GENE_FIELDS

SCHEMA_HINT_BY_TASK = {
    "gene_card_extract": (
        "Return a single JSON object inside ```json ... ``` with exactly these "
        f"keys: {list(GENE_FIELDS)}. Each value is a one-paragraph claim. "
        "Do NOT include any other keys."
    ),
    "T3-01_single_dynamics": (
        'Return JSON: {"driver": "mechanism|niche|observation|limitation", '
        '"dynamics": "Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition"}'
    ),
    "T3-09_relation_classify": (
        'Return JSON: {"label": "...", "dynamics": "Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition"}'
    ),
    "T2-07_lim_delta_match": (
        'Return JSON: {"mapping": {"L1": "D#", "L2": "D#", "L3": "D#"}}'
    ),
    "T4-01_consistency_check": (
        'Return JSON: {"label": "...", "contribution_type": "method|dataset|analysis|system|theory", '
        '"verify": ["T|F", "T|F", "T|F", "T|F"]}'
    ),
    "T1-01_contribution_type": (
        'Return JSON: {"multi_contrib_types": {"G1": "method|dataset|analysis|system|theory", '
        '"G2": "...", "G3": "...", "G4": "..."}}'
    ),
    "T1-03_driver_vs_passenger": (
        'Return JSON: {"driver_gene": "G#", "passenger_gene": "G#"}'
    ),
    "T2-01_ordering_5": (
        'Return JSON: {"correct_order": [g1, g2, g3, g4, g5]}'
    ),
}


def build_system_prompt(task_type: str | None) -> str:
    hint = SCHEMA_HINT_BY_TASK.get(task_type, "Return your answer as a fenced JSON block.")
    return (
        "You are a precise scientific lineage analyst. Read the input below and "
        "respond with exactly one fenced JSON block, no commentary.\n\n" + hint
    )


def build_prompt_pool(prompt_pool_path: str | None, fallback_n: int = 200,
                        fallback_src: str = "/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train_v7.jsonl"
                        ) -> list[dict]:
    if prompt_pool_path:
        return [json.loads(l) for l in open(prompt_pool_path)]
    out = []
    with open(fallback_src) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append({
                "prompt_id": f"v3p::{i:05d}",
                "prompt":    r["prompt"],
                "task_type": r.get("task_type"),
                "parent_card": None,
            })
            if len(out) >= fallback_n:
                break
    return out


@torch.no_grad()
def sample_one_rollout(student, tokenizer, prompt_text: str, prompt_token_ids: torch.Tensor,
                         max_new_tokens: int, temperature: float,
                         device: str):
    inputs = {"input_ids": prompt_token_ids.unsqueeze(0).to(device)}
    out_ids = student.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    n_prompt = inputs["input_ids"].shape[-1]
    return out_ids[0, n_prompt:].cpu()


def rollout_one_prompt_multi(
    student, tokenizer, teacher, reward_fn,
    prompt: dict, *,
    n_samples: int, max_new_tokens: int, temperature: float,
    device: str, top_k_for_grad: int,
):
    """Return list of dicts (one per sample) sorted by descending v, plus the
    chosen loss tensor (top-K mean of PG loss with normalised advantage)."""
    p = prompt["prompt"]
    tt = prompt.get("task_type")
    parent_card = prompt.get("parent_card")
    sys_prompt = build_system_prompt(tt)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": p},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True,
                                                       enable_thinking=False)
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                            max_length=4096).input_ids[0]

    # --- sample N rollouts (no grad)
    sample_completions = []
    for _ in range(n_samples):
        comp_ids = sample_one_rollout(student, tokenizer, prompt_text, prompt_ids,
                                        max_new_tokens, temperature, device)
        if len(comp_ids) == 0:
            continue
        text = tokenizer.decode(comp_ids, skip_special_tokens=True)
        sample_completions.append({"ids": comp_ids, "text": text})

    if not sample_completions:
        return None, None

    # --- score each rollout under teacher + verifier
    for s in sample_completions:
        teach = teacher.score_completion(p, s["text"], system=sys_prompt)
        s["teacher_lp"] = teach.log_probs
        s["v"] = None
    # use rewards_fn just for verifier subscores; per-sample
    for s in sample_completions:
        # compute per-token kl + reward (will recompute student lp below)
        # Use parser + verifier through reward_fn but with dummy kl
        dummy_kl = [0.0] * len(s["ids"])
        phi = char_uniform_phi_tags(s["text"], tt)
        chars_per_tok = max(1, len(phi) // max(len(s["ids"]), 1))
        phi_tokens = [phi[min(i * chars_per_tok, len(phi) - 1)] for i in range(len(s["ids"]))]
        if not phi:
            phi_tokens = ["unknown"] * len(s["ids"])
        rew = reward_fn(text=s["text"], per_token_kl=dummy_kl,
                          phi_per_token=phi_tokens, fld_per_token=None,
                          task_type=tt, gold_answer=None, parent_card=parent_card)
        s["v"] = rew.verifier.v
        s["phi_tokens"] = phi_tokens

    # --- compute group baseline (GRPO style)
    vs = [s["v"] for s in sample_completions]
    v_mean = sum(vs) / len(vs)
    v_std = (sum((x - v_mean) ** 2 for x in vs) / len(vs)) ** 0.5
    for s in sample_completions:
        s["advantage"] = (s["v"] - v_mean) / max(v_std + 1e-3, 1e-3)

    # sort by v desc
    sample_completions.sort(key=lambda x: -x["v"])
    top = sample_completions[:top_k_for_grad]

    # --- compute PG loss only on top-K samples
    losses = []
    for s in top:
        comp_ids = s["ids"]
        full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(device)
        out = student(full_ids)
        logits = out.logits[0]
        shift_logits = logits[len(prompt_ids) - 1 : len(prompt_ids) + len(comp_ids) - 1]
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        stu_lp = log_probs.gather(1, comp_ids.to(device).unsqueeze(-1)).squeeze(-1)
        n = min(len(s["teacher_lp"]), len(stu_lp))
        stu_lp = stu_lp[:n]
        teach_lp = torch.tensor(s["teacher_lp"][:n], device=device, dtype=torch.float32)
        kl_t = (stu_lp - teach_lp).detach().tolist()
        # broadcast advantage as a per-token reward shift
        adv = s["advantage"]
        phi_tokens = s["phi_tokens"][:n]
        rew = reward_fn(text=s["text"], per_token_kl=kl_t,
                          phi_per_token=phi_tokens, fld_per_token=None,
                          task_type=tt, gold_answer=None, parent_card=parent_card)
        # final loss: − Σ (r_t + adv) · log π_θ
        r_t = torch.tensor(rew.rewards[:n], device=device, dtype=torch.float32) + adv
        pg = -(r_t * stu_lp).sum() / max(n, 1)
        losses.append(pg)

    final_loss = torch.stack(losses).mean()
    return {
        "loss":      final_loss,
        "v_mean":    v_mean,
        "v_max":     max(vs),
        "v_min":     min(vs),
        "v_std":     v_std,
        "n_samples": len(sample_completions),
        "kl_mean":   sum(kl_t) / max(len(kl_t), 1) if losses else 0.0,
        "n_tokens":  n if losses else 0,
    }, sample_completions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora", required=True)
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--student-gpu", default="0")
    ap.add_argument("--teacher-gpu", default="3")
    ap.add_argument("--prompt-pool", default=None)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.3)         # v2 default lower
    ap.add_argument("--n-samples", type=int, default=4)               # v2 GRPO-style
    ap.add_argument("--top-k-for-grad", type=int, default=2)
    ap.add_argument("--lambda-v", type=float, default=0.5)
    ap.add_argument("--lambda-c", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "train.log").open("w")
    def log(s): print(s, flush=True); log_fp.write(s + "\n"); log_fp.flush()

    student_dev = f"cuda:{args.student_gpu}"
    log(f"[1/3] loading student on {student_dev}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=student_dev,
    )
    student = PeftModel.from_pretrained(base, args.student_lora, is_trainable=True)
    student.train()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log(f"  trainable params: {n_train/1e6:.1f}M")

    teacher_dev = f"cuda:{args.teacher_gpu}"
    log(f"[2/3] loading teacher on {teacher_dev}")
    teacher = Qwen3LocalTeacher(args.teacher_model, device=teacher_dev)

    log(f"[3/3] prompts")
    pool = build_prompt_pool(args.prompt_pool, fallback_n=args.n_prompts)
    log(f"  pool: {len(pool)} prompts; "
        f"n_samples={args.n_samples} top_k={args.top_k_for_grad} temp={args.temperature}")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                              lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    reward = EvoOPDReward(EvoOPDRewardConfig(lambda_v=args.lambda_v, lambda_c=args.lambda_c))

    log(f"\n=== evo-OPD v2 loop: {args.steps} steps ===")
    log(f"{'step':>4} {'v_mean':>7} {'v_max':>7} {'v_std':>7} {'kl_mean':>8} {'loss':>9} {'n_samp':>6} {'sec':>6}")
    t_start = time.time()
    for step in range(args.steps):
        prompt = pool[step % len(pool)]
        t0 = time.time()
        result, samples = rollout_one_prompt_multi(
            student, tok, teacher, reward, prompt,
            n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, device=student_dev,
            top_k_for_grad=args.top_k_for_grad,
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
            f"{result['v_std']:>7.3f} {result['kl_mean']:>+8.3f} "
            f"{float(loss):>+9.4f} {result['n_samples']:>6} {sec:>6.1f}")

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
