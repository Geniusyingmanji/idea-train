"""SFT on agentic_v2 demos.

Same training loop as v1 but:
  - reads data/agentic_v2/sft_demos.jsonl (287 web-native demos with 6 tools)
  - uses ROLLOUT_SYS_PROMPT_V2 in the chat template
  - base model = qwen3-8b-sft-v3/final (cleaner than v1-agentic-sft)
  - max_len 2048 (median demo ~1510 tokens, max 1890)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT_V2


def build_dataset(sft_demos_path: Path, tokenizer, max_len: int = 2048):
    rows = []
    with sft_demos_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            # use the demo's full_prompt (already structured per source)
            user_msg = d["full_prompt"]
            messages = [
                {"role": "system", "content": ROLLOUT_SYS_PROMPT_V2},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": d["completion"]},
            ]
            try:
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, enable_thinking=False,
                )
            except (TypeError, ValueError):
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            messages_prefix = messages[:2]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages_prefix, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except (TypeError, ValueError):
                prompt_text = tokenizer.apply_chat_template(
                    messages_prefix, tokenize=False, add_generation_prompt=True,
                )
            full_ids = tokenizer(full_text, truncation=True, max_length=max_len,
                                  add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len,
                                    add_special_tokens=False).input_ids
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if len(labels) != len(full_ids):
                continue
            rows.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
                "prompt_id": d.get("prompt_id"),
            })
    return rows


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = max_len - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="Qwen/Qwen3-8B")
    ap.add_argument("--student-lora",
                    default="/home/azureuser/workspace-gzy/zyf/idea_train/train/checkpoints/qwen3-8b-sft-v3/final",
                    help="base LoRA to continue from")
    ap.add_argument("--sft-demos",
                    default="/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v2/sft_demos.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading {args.student_base} + LoRA from {args.student_lora} on {device}")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, args.student_lora, is_trainable=True)
    model.print_trainable_parameters()

    print(f"[2/3] building dataset from {args.sft_demos}")
    rows = build_dataset(Path(args.sft_demos), tok, max_len=args.max_len)
    print(f"  {len(rows)} usable demos")
    if not rows:
        return

    pad_id = tok.pad_token_id

    print(f"[3/3] SFT: epochs={args.num_epochs} lr={args.lr} batch={args.batch_size} grad_accum={args.grad_accum}")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to=[],
            remove_unused_columns=False,
            save_total_limit=2,
        ),
        train_dataset=rows,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()

    final = out_dir / "final"
    model.save_pretrained(final); tok.save_pretrained(final)
    print(f"✓ saved → {final}")


if __name__ == "__main__":
    main()
