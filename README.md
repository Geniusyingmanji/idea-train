# idea_train

Training pipeline for an 8B scientific-idea reasoning model targeting GENE-bench and related benchmarks.

## Documents

- `survey.md` — methods/benchmarks/data survey (May 2026)
- `plan.md` — full training plan (SFT → GRPO → evo-OPD)
- `lv_opd_plan.md` — evo-OPD algorithm spec (replaces Stage 3 in `plan.md`)

## Repo layout

```
idea_train/
├── survey.md            # research survey
├── plan.md              # training plan
├── lv_opd_plan.md       # evo-OPD spec
│
├── denylist/            # paper IDs to exclude from training (Stage 0)
├── data/                # SFT mixtures, RL prompt sets, OPD prompts
├── tools/               # build_denylist.py, filter_corpus.py, etc.
├── train/               # *.yaml configs for SFT / GRPO / evo-OPD
├── eval/                # full-suite + smoke + contamination probes
└── evo_opd/             # evo-OPD library
    ├── teachers/        # Qwen3-32B-Thinking (primary), GPT-5.5 BB (ablation L7)
    └── trainer/         # verl recipe + custom reward hook
```

## Environment

Dedicated conda env (preferred):
```bash
conda activate idea
```

Contents: torch 2.5.1+cu121, transformers 5.8.0, accelerate 1.13.0, datasets 4.8.5, pyalex 0.21, datasketch 1.10.0, pandas 2.3.3, pyarrow 24.0.0, openai 2.37.0, plus numpy/scipy/sklearn/wandb/pytest. CUDA available.

Stage 1+ adds: vLLM, verl or OpenRLHF, Liger-Kernel, flash-attn, llamafactory/axolotl. To be installed when those stages start.

(Old shared venv at `/home/azureuser/workspace-gzy/zyf/rise-teacher/.venv` still works but is shared with other projects.)

## Data format

See `DATA_FORMAT.md` for concrete JSON schemas at every stage (denylist v0/v1, SFT 8 task types, GRPO prompts, evo-OPD per-rollout inputs/outputs, eval logs, contamination probes).

## Status

- [x] Survey + plans drafted
- [x] Project skeleton
- [ ] **Stage 0 — Contamination firewall** (current)
  - [ ] denylist v0 from local IdeaEvolving assets
  - [ ] OpenAlex 1-hop expansion → denylist v1
  - [ ] filter cascade prototype
- [ ] Stage 1 — SFT data + training
- [ ] Stage 2 — GRPO/DAPO
- [ ] Stage 3 — evo-OPD
- [ ] Stage 4 — Eval + ablations
