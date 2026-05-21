# idea_train

Training pipeline for an 8B agentic scientific-research model. Targets multiple
open-ended idea-generation benchmarks (GENE-Arena PES, SGI-Bench task 2, and the
ArenaRL open-task suite) by teaching a Qwen3-8B + LoRA agent to use 6 research
tools (search, read, extract_genome, genome_diff, novelty_check, propose) along
10 distinct research-workflow archetypes.

## Quick links

- **Design** — [`docs/research_workflows.md`](docs/research_workflows.md): first-principle definition of "research," 10 workflow archetypes (W1-W10), 8 core skills, dataset construction principles.
- **Status** — [`docs/STATUS.md`](docs/STATUS.md): latest snapshot.
- **Doc index** — [`docs/README.md`](docs/README.md): full documentation map (current + historical + specs).
- **Data** — `data/agentic_v3/` is the current authoritative dataset (1032 demos, see [`data/agentic_v3/summary.md`](data/agentic_v3/summary.md) and `exemplars.md` for examples).

## What's in this repo

```
idea_train/
├── README.md
├── docs/                          # all design docs, specs, history — see docs/README.md
│   ├── research_workflows.md      # primary design (v3, 10 archetypes)
│   ├── STATUS.md                  # current state
│   ├── data_format.md             # data schemas
│   ├── genetrace_format.md        # GeneTrace v0.1 schemas
│   ├── paper_positioning.md       # framing vs IdeaEvolving / ArenaRL / SGI-Bench
│   ├── survey.md                  # literature survey
│   ├── specs/                     # algorithm specs (some still load-bearing)
│   └── history/                   # archived plans (pre-agentic, overnight reports)
│
├── evo_opd/                       # core agentic framework
│   ├── agentic/                   # ReAct rollout + trajectory tracking
│   ├── tools/                     # search / read / propose / extract_genome / genome_diff / novelty_check
│   ├── judges/                    # PES judges (pairwise tournament + pointwise)
│   ├── teachers/                  # Azure GPT-5.5 client (keyless)
│   ├── trainer/                   # SFT + RL trainers (with token-mask PG)
│   ├── lineage.py                 # parent-card consistency check
│   ├── rewards.py                 # composite reward (verifier + lineage + arena + struct)
│   ├── structural.py              # deterministic Layer-1 PES
│   ├── verifier.py                # schema/evidence/dynamics validators
│   └── ...
│
├── tools/                         # data construction + eval scripts
│   ├── build_agentic_v{1,2,3}_*.py  # demo generators (v1: simple, v2: web, v3: multi-archetype)
│   ├── build_research_workflow_demos.py  # CURRENT (v3) demo generator
│   ├── analyze_v3_demos.py        # skill / archetype distribution
│   ├── sft_agentic_v{2,3}.py      # SFT scripts
│   ├── agentic_eval_gene_arena.py # GENE-Arena eval adapter
│   ├── agentic_eval_sgi_bench.py  # SGI-Bench task 2 adapter
│   └── ...
│
├── train/                         # train configs (checkpoints excluded via .gitignore)
│
├── eval/                          # eval driver + result snapshots
│   ├── eval_gene_exam_lora.py     # GENE-Exam evaluator
│   ├── lenient_rescore.py         # key-normalized scoring
│   └── results/                   # per-checkpoint metrics
│
├── data/                          # all training/eval data
│   ├── agentic_v1/                # 246 simple SFT demos (early)
│   ├── agentic_v2/                # 287 web-search demos
│   ├── agentic_v3/                # 1032 multi-archetype demos (CURRENT)
│   ├── genetrace_v0_1/            # 855 GenomeCards + 300 DynamicsEdges (lineage corpus)
│   ├── stage1_sft/                # earlier SFT round mixtures
│   └── ...
│
└── paper/                         # NeurIPS-style LaTeX draft + tables
```

## Headline results (full table in `eval/results/OVERNIGHT_REPORT.md`)

| Model | GENE-Arena PES (n=150) |
|---|---|
| 🏆 qwen3-8b-sft-v3 (non-agentic baseline) | **57.95** |
| qwen3-8b-evo-opd-v5 | 56.38 |
| qwen3-8b baseline (no LoRA) | 50.96 |
| qwen3-8b-agentic-v2-sft + web | 46.84 (Azure rate-limit contaminated, needs rerun) |
| qwen3-8b-agentic-v3-sft + web | running |

## Environment

```bash
conda activate idea
```

Core: torch 2.5+cu121, transformers 5.8+, peft, accelerate, datasets, openai (Azure keyless), networkx (for SGI graph_similarity), sentence-transformers (optional), pyalex.

## Data + checkpoints

Code + datasets (~150 MB) are in git. Trained LoRA adapters (~700 MB each) are
NOT in git — see `.gitignore`. To reproduce: load `Qwen/Qwen3-8B` base from HF +
attach the LoRA adapter you train via `tools/sft_agentic_v3.py`.
