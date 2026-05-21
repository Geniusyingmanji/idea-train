"""SFT train Qwen3-8B + LoRA on GPT-5.5-generated gene-card data.

Uses transformers Trainer + peft LoRA. Multi-GPU via accelerate / FSDP
(launch with `accelerate launch`).

Input data shape (one example per line):
  {
    "instance_id": ...,
    "task_type": "gene_card_extract",
    "completion": "```json\n{...}\n```",
    "metadata": {
      "source_text": "PAPER TEXT ...",
      ...
    }
  }

We reconstruct the prompt at training time from `metadata.source_text` plus
the task-specific template, so we can re-use the exact teacher prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.prompts import SYSTEM_PROMPT, build_messages


def example_to_messages(ex: dict) -> list[dict]:
    """Reconstruct the teacher-style messages for training.

    For gene_card_extract: rebuild the user prompt from source_text.
    Otherwise fall back to whatever's in `prompt`.
    """
    task = ex.get("task_type")
    md = ex.get("metadata") or {}
    if task == "gene_card_extract":
        src = md.get("source_text") or ex.get("prompt") or ""
        if src:
            return build_messages("gene_card_extract", paper_text=src)
    if task == "idea_generate":
        return build_messages("idea_generate",
                              lineage_text=md.get("lineage_text", ex.get("prompt", "")),
                              open_question=md.get("open_question", "What's next?"))
    # generic fallback
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex.get("prompt", "")},
    ]


def format_for_training(ex: dict, tokenizer) -> dict:
    """Apply chat template and return tokenized inputs with completion-only loss mask."""
    messages = example_to_messages(ex)
    completion = ex["completion"]

    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = (
            f"<|system|>\n{messages[0]['content']}\n"
            f"<|user|>\n{messages[1]['content']}\n"
            f"<|assistant|>\n"
        )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion + tokenizer.eos_token,
                                add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    # mask the prompt tokens so loss is computed only on completion
    labels = [-100] * len(prompt_ids) + completion_ids

    # truncate from the LEFT of the prompt if too long (keep completion intact)
    max_len = 4096
    if len(input_ids) > max_len:
        keep_completion = len(completion_ids)
        keep_prompt = max_len - keep_completion
        if keep_prompt < 100:
            # if completion alone is too big, hard truncate from right
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        else:
            input_ids = prompt_ids[-keep_prompt:] + completion_ids
            labels = [-100] * keep_prompt + completion_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default="/home/azureuser/workspace-gzy/zyf/idea_train/train/checkpoints/qwen3-8b-sft-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    print(f"[1/4] Loading data from {args.data}")
    examples: list[dict] = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} raw examples")

    print(f"[2/4] Loading tokenizer + model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print(f"[3/4] Adding LoRA r={args.lora_r}")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(f"[4/4] Tokenizing")
    # Skip Dataset.from_list (it chokes on mixed metadata) — tokenize directly.
    tokenized = []
    for ex in examples:
        try:
            tokenized.append(format_for_training(ex, tokenizer))
        except Exception as e:
            print(f"  skip {ex.get('instance_id','?')}: {e}")
            continue
    ds = Dataset.from_list(tokenized)
    print(f"  tokenized: {len(ds)} examples")

    def data_collator(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print(f"Starting training")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print("Done.")


if __name__ == "__main__":
    main()
