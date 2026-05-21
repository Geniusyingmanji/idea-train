"""Qwen3-14B-Thinking local teacher — direct transformers backend.

Provides per-token log-probabilities of a student-completed rollout under the
teacher's conditional distribution. Used by evo-OPD's reverse-KL term.

We use plain transformers (not vLLM) because:
  - We need access to full log-prob tensors per token, not top-K.
  - vLLM had Qwen3 tokenizer-compatibility issues in earlier work
    (`eval/results/OVERNIGHT_REPORT.md`).
  - Throughput at this stage is fine: <1 sample/sec is enough to develop
    + validate the evo-OPD loop end-to-end. Later we can swap in vLLM
    for production training.

Public API:

    teacher = Qwen3LocalTeacher(model_name="Qwen/Qwen3-14B")
    log_probs = teacher.score_completion(prompt, completion)  # list[float], one per completion token
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TeacherScore:
    """Per-token teacher conditional log-probabilities for a completion."""
    completion_token_ids: list[int]
    log_probs: list[float]                           # log π_T(y_t | y_<t, x)
    completion_text: str
    n_prompt_tokens: int
    n_completion_tokens: int


class Qwen3LocalTeacher:
    """Direct-transformers teacher backbone for evo-OPD."""

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-14B",
                 device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 max_seq_len: int = 8192) -> None:
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer must match student family; for cross-tokenizer setups we'd
        # need alignment but Qwen3-14B and Qwen3-8B share tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score_completion(
        self,
        prompt: str,
        completion: str,
        system: str | None = None,
        enable_thinking: bool = False,
    ) -> TeacherScore:
        """Return per-token log π_T(y_t | y_<t, x) for the completion tokens."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except (TypeError, ValueError):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt",
                                      truncation=True, max_length=self.max_seq_len - 512
                                      ).input_ids[0]
        completion_ids = self.tokenizer(completion, return_tensors="pt",
                                          add_special_tokens=False).input_ids[0]
        n_prompt = len(prompt_ids)
        n_comp = len(completion_ids)
        if n_prompt + n_comp > self.max_seq_len:
            completion_ids = completion_ids[: self.max_seq_len - n_prompt]
            n_comp = len(completion_ids)

        full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(self.device)
        out = self.model(full_ids)
        # logits[t] predicts token t+1, so completion tokens are predicted by
        # logits at positions [n_prompt-1, ..., n_prompt+n_comp-2]
        logits = out.logits[0]                              # (T, V)
        shift_logits = logits[n_prompt - 1: n_prompt + n_comp - 1]   # (n_comp, V)
        log_probs_full = torch.log_softmax(shift_logits.float(), dim=-1)
        per_token_lp = log_probs_full.gather(
            1, completion_ids.to(self.device).unsqueeze(-1)
        ).squeeze(-1)                                       # (n_comp,)

        return TeacherScore(
            completion_token_ids=completion_ids.tolist(),
            log_probs=per_token_lp.tolist(),
            completion_text=completion,
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_comp,
        )


# --- smoke -----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"loading teacher {args.model} on {args.device} ...")
    t = Qwen3LocalTeacher(args.model, device=args.device)
    print(f"  ready (vocab={t.tokenizer.vocab_size})")

    prompt = ('Extract a 6-field gene card from this paper.\n'
               'TITLE: Attention Is All You Need\nABSTRACT: We propose a new '
               'simple network architecture, the Transformer, based solely on '
               'attention mechanisms, dispensing with recurrence and convolutions '
               'entirely.')
    completion = '```json\n{"mechanism_genome": "Self-attention transformer architecture."}\n```'

    s = t.score_completion(prompt, completion)
    print(f"completion: {len(s.log_probs)} tokens")
    print(f"  mean log_prob: {sum(s.log_probs) / max(len(s.log_probs), 1):.3f}")
    print(f"  first 5 log_probs: {s.log_probs[:5]}")
    print(f"  any nan? {any(x != x for x in s.log_probs)}")
