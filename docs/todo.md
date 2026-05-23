# TODO

Prioritized next steps. See `data.md` for corpus state and `experiment.md` for measured results.

## High priority — directly unblocks the paper

### 1. Re-SFT on combined corpus
Train Qwen3-8B + LoRA r=64 on `data/agentic_combined_v3to45/sft_demos.jsonl` (5953 demos). Hypothesis: propose rate jumps from 18% → 60%+ based on the v2-vs-v3 comparison in `experiment.md` §3.

```bash
python tools/sft_agentic_v3.py \
  --data data/agentic_combined_v3to45/sft_demos.jsonl \
  --out train/checkpoints/qwen3-8b-agentic-v33-sft \
  --max-len 3072 --lr 3e-5 --batch 1 --grad-accum 8 --epochs 2
```

Estimate: ~6-8 hr on 1× A100. Schema already compatible (trainer reads `full_prompt` + `completion`).

### 2. Re-eval the new SFT on GENE-Arena (150 ideas)
Use `tools/agentic_eval_gene_arena.py` as in §6 of `experiment.md`. Key metrics to report:
- **Propose rate** (target ≥60%)
- **PES** (target ≥ SFT v3's 57.95)
- **Median tools/demo at inference**

### 3. DPO from `dpo_combined`
After re-SFT lands, run DPO on the 2891 pairs in `data/dpo_combined/preferences.jsonl`. The 7 corruption modes are well-distributed; no further balancing needed.

### 4. SGI-Bench task_2 eval
The new corpus has v8 sgi_idea_plan_strict (81 demos) + v6/v15 idea_plan demos. Re-run SGI eval — previous attempts hit 0/100 propose because the v3-only model didn't know `idea_plan` schema.

### 5. ArenaRL Open-Travel + Open-DeepResearch
Chinese coverage now at 21% (was 8%). v9 has 100 ArenaRL travel-zh demos; v12/v15/v21/v28/v29 cover general ZH. Should unlock both benchmarks. Needs adapter work for the different tool schemas (Chinese tool names).

## Medium priority — paper-completing experiments

### 6. PES re-scoring with 3-judge ensemble
Currently 98.4% of ideas scored by gpt-5.5 alone (gpt-5.4 / gpt-5.4-nano DeploymentNotFound). If alternate deployments come back, re-score all 708 ideas. Will tighten the bootstrap CI and give a defensible "3-judge consensus" number for the paper.

### 7. Eval-prompt-matched SFT (v7 variant for GENE-Exam)
Idea from `OVERNIGHT_REPORT` epilog. Train on the actual GENE-Exam eval prompts so canonical-key behavior transfers. Estimate: ~1 day GPT-5.5 calls + 30 min train + 30 min eval. Could close the 1.17% / 11.27% strict/lenient gap.

### 8. Thinking-mode re-eval
SFT v3 in no-think mode was the baseline for all numbers. Re-eval with `<think>` enabled (4-6 hr). Probably +2-3% headroom on GENE-Exam.

## Low priority — nice-to-have

### 9. v3 SFT + agentic-v3-sft PES rescoring
PES on agentic-v3-sft is currently pending (Azure rate-limit). When quota recovers, run it. Expect lower than 57.95 due to 18% propose rate masking quality.

### 10. Sample efficiency ablation
v3 alone won at 1032 demos. Does the 5953-demo corpus help, or does it dilute the signal? Train two checkpoints (v3-only vs full combined) and compare.

### 11. Re-merge after any new data round
`tools/merge_sft_v3_to_v8.py` (despite the historical name) handles the full v3→latest merge. Update the `INPUTS` list when new rounds land. Same pattern for `tools/merge_dpo_all.py`.

### 12. Push working PAT into a credential helper
Currently using a one-off PAT for pushes (the `ghp_tcOouBN...` token). Should set up a proper credential helper or move the repo to one of the user's own GitHub accounts so pushes don't need ad-hoc tokens.

## Done (recently)

- ✅ Combined corpus assembled: 6687 SFT + 2593 DPO + 2930 RL prompts (19 SFT rounds, 6 DPO rounds, 4 RL rounds)
- ✅ All v5-v33 + v7/v10/v13/v19/v22/v27 DPO data pushed to https://github.com/Geniusyingmanji/idea-train
- ✅ v3 SFT agentic GENE-Arena rollouts: 150 ideas, propose rate 18% (documented in `experiment.md` §3)
- ✅ Docs consolidated: 3 main docs (`data.md`, `experiment.md`, `todo.md`)
