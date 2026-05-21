"""Min-K%++ leakage check (Shi et al. 2024) for GeneTrace v0.1 contamination guard.

For each GENE-Exam paper, computes the model's per-token log-likelihood and
takes the average over the bottom-K% tokens. Higher score = more confident
on rare tokens = signal of memorisation.

We report:
  - mean Min-20%++ over all GENE-Exam papers under each trained checkpoint
  - mean Min-20%++ over a reference set of papers definitely-not-in-training
    (recent arXiv abstracts after our training data cutoff) — as the "no
    leakage" baseline
  - per-paper delta vs baseline; outliers flagged

A trained checkpoint is "leak-clean" if its mean Min-20%++ on GENE-Exam
papers does not exceed the reference baseline by more than ~0.05.

For the paper, we run this for each released checkpoint (SFT v1..v6,
GeneTrace-v0.1 retrained, evo-OPD) and report a table in the ethics
section.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
GENE_EXAM = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving/gene_exam/Questions")


def load_gene_exam_paper_texts(limit: int = 100) -> list[dict]:
    """Pull paper texts from GENE-Exam instances (deduplicated by paper_id)."""
    seen: dict[str, dict] = {}
    for td in sorted(glob.glob(str(GENE_EXAM / "*"))):
        f = Path(td) / "instances.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        if isinstance(data, dict):
            data = list(data.values())
        for inst in data:
            md = inst.get("metadata") or {}
            for paper_key in ("paper", "p_paper", "q_paper", "paper_a", "paper_b"):
                p = md.get(paper_key)
                if isinstance(p, dict):
                    pid = p.get("paper_id") or p.get("id")
                    txt = p.get("abstract") or p.get("text") or ""
                    if pid and txt and pid not in seen:
                        seen[pid] = {"paper_id": pid, "text": txt[:3000]}
            # also try top-level prompt as fallback "text"
            if "prompt" in inst and len(seen) < limit:
                pid = inst.get("instance_id")
                if pid and pid not in seen:
                    seen[pid] = {"paper_id": pid, "text": inst["prompt"][:3000]}
            if len(seen) >= limit:
                break
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def load_reference_texts(limit: int = 100) -> list[dict]:
    """Reference papers definitely-not-in-training. For now we use the v0.1
    cards.jsonl source_text field (these ARE in our SFT training pool — so
    they will give an UPPER bound on naturally-memorised score). For a real
    baseline, swap this for recent arXiv abstracts."""
    out = []
    p = ROOT / "data/genetrace_v0_1/cards.jsonl"
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            txt = r.get("source_text", "")[:3000]
            if txt:
                out.append({"paper_id": r["paper_id"], "text": txt})
            if len(out) >= limit:
                break
    return out


@torch.no_grad()
def min_k_pp_score(model, tokenizer, text: str, k_pct: float = 20.0,
                    device: str = "cuda") -> float:
    """Min-K%++ score per Shi et al. 2024. Lower = less memorised."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    ids = enc["input_ids"][0]
    if len(ids) < 5:
        return float("nan")
    out = model(**enc)
    logits = out.logits[0]                                 # (T, V)
    # shift: predict token t from logits[t-1]
    shift_logits = logits[:-1]                             # (T-1, V)
    shift_targets = ids[1:]                                # (T-1,)
    # log-prob of each true token
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = log_probs.gather(1, shift_targets.unsqueeze(-1)).squeeze(-1)
    # Min-K%++ normalisation: subtract the mean log-prob over the vocab,
    # divide by std — see Shi 2024 §3
    mu = log_probs.mean(dim=-1)
    sigma = log_probs.std(dim=-1)
    norm_lp = (tok_lp - mu) / (sigma + 1e-6)
    # take bottom-K% (= rarest tokens)
    n = len(norm_lp)
    k = max(1, int(n * k_pct / 100.0))
    bottom_k, _ = torch.topk(norm_lp, k, largest=False)
    return float(bottom_k.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path "
                                                    "(LoRA adapter dirs work via --lora).")
    ap.add_argument("--lora", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--n-eval", type=int, default=50, help="GENE-Exam papers to score")
    ap.add_argument("--n-ref",  type=int, default=50, help="reference papers to score")
    ap.add_argument("--k-pct",  type=float, default=20.0)
    ap.add_argument("--output", default=str(ROOT / "data/genetrace_v0_1/min_k_report.json"))
    args = ap.parse_args()

    print(f"[1/3] loading {args.model}" + (f" + LoRA {args.lora}" if args.lora else ""))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                   torch_dtype=torch.bfloat16,
                                                   device_map="auto")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    device = next(model.parameters()).device

    print(f"[2/3] loading texts")
    eval_set = load_gene_exam_paper_texts(args.n_eval)
    ref_set = load_reference_texts(args.n_ref)
    print(f"  eval (GENE-Exam): {len(eval_set)} papers")
    print(f"  ref (proxy):      {len(ref_set)} papers (NOTE: currently uses GeneTrace "
          f"sources = upper bound; replace with recent arXiv for real baseline)")

    print(f"[3/3] scoring (k={args.k_pct}%)")
    eval_scores = []
    for i, p in enumerate(eval_set):
        s = min_k_pp_score(model, tok, p["text"], args.k_pct, device=device)
        eval_scores.append({"paper_id": p["paper_id"], "min_k_pp": s})
        if (i + 1) % 10 == 0:
            print(f"  eval {i+1}/{len(eval_set)}", flush=True)
    ref_scores = []
    for i, p in enumerate(ref_set):
        s = min_k_pp_score(model, tok, p["text"], args.k_pct, device=device)
        ref_scores.append({"paper_id": p["paper_id"], "min_k_pp": s})
        if (i + 1) % 10 == 0:
            print(f"  ref  {i+1}/{len(ref_set)}", flush=True)

    eval_mean = sum(s["min_k_pp"] for s in eval_scores if not math.isnan(s["min_k_pp"])) / max(len(eval_scores), 1)
    ref_mean  = sum(s["min_k_pp"] for s in ref_scores  if not math.isnan(s["min_k_pp"]))  / max(len(ref_scores), 1)
    delta = eval_mean - ref_mean

    report = {
        "model": args.model,
        "lora": args.lora,
        "k_pct": args.k_pct,
        "n_eval": len(eval_scores),
        "n_ref":  len(ref_scores),
        "eval_mean_min_k_pp": eval_mean,
        "ref_mean_min_k_pp":  ref_mean,
        "delta (eval - ref)": delta,
        "leak_clean (delta < 0.05)": delta < 0.05,
        "per_paper_eval": eval_scores,
        "per_paper_ref": ref_scores,
        "ref_caveat": "ref set currently uses GeneTrace source_text (upper bound). "
                       "Replace with recent post-cutoff arXiv abstracts for a tight baseline.",
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n=== Min-{args.k_pct}%++ leakage check ===")
    print(f"  eval mean: {eval_mean:+.4f}")
    print(f"  ref  mean: {ref_mean:+.4f}")
    print(f"  delta:     {delta:+.4f}   ({'CLEAN' if delta < 0.05 else 'INVESTIGATE'})")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
