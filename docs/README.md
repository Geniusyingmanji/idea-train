# `docs/` — project documentation index

This folder holds all the design docs, specs, and historical plans. The
top-level repo `README.md` is the entry point; this index lets you navigate
deeper.

## Primary references (current)

| File | What it covers |
|---|---|
| [`data_inventory.md`](data_inventory.md) | **What's in each data round (v3 → v33).** SFT/DPO/RL counts, per-round purpose, tool-coverage stats. |
| [`recent_experiments_2026_05.md`](recent_experiments_2026_05.md) | **Recent eval results + methodological lessons** (agentic propose-rate finding, DPO yield comparison, what worked / what broke). |
| [`research_workflows.md`](research_workflows.md) | **Current authoritative design.** First-principle definition of "research," 10 workflow archetypes (W1-W10), 8 core research skills, data-generation principles, and training implications. This is the doc behind the v3 (1032-demo) dataset. |
| [`STATUS.md`](STATUS.md) | Latest status snapshot — what's done, what's blocked, what's next |
| [`paper_positioning.md`](paper_positioning.md) | How this work is framed against IdeaEvolving, ArenaRL, SGI-Bench, etc. |

## Data format references

| File | What it covers |
|---|---|
| [`data_format.md`](data_format.md) | Top-level data format spec (`paper:`, `oa:` ID conventions, gene_genome fields, etc.) |
| [`genetrace_format.md`](genetrace_format.md) | GeneTrace v0.1 corpus: GenomeCard / DynamicsEdge / LineageChain / VerifierBundle, contamination guards, denylist scope |

## Reference / background

| File | What it covers |
|---|---|
| [`survey.md`](survey.md) | Literature survey behind the project — context for the design choices |

## Specifications

| File | What it covers |
|---|---|
| [`specs/evo_opd_open_ended.md`](specs/evo_opd_open_ended.md) | The open-ended-task extension of evo-OPD — partially incorporated into `evo_opd/rewards.py` + `evo_opd/lineage.py`. Some sections (especially component (C) "lineage-consistency self-signal") are still load-bearing. |

## Historical archive (kept for context, not active)

| File | Era | Why archived |
|---|---|---|
| [`history/plan_pre_agentic.md`](history/plan_pre_agentic.md) | Pre-2026-05-19 | Original master plan covering Stages 0-4 (denylist → SFT → GRPO → evo-OPD). Superseded by the agentic pivot. Useful context for the early-stage choices (denylist scope, SFT data mixture). |
| [`history/overnight_2026-05-20.md`](history/overnight_2026-05-20.md) | 2026-05-20 night | Hour-by-hour summary of the night when the 1032-demo v3 dataset was constructed. |

## Things deleted (rationale recorded here)

| Removed | Replaced by |
|---|---|
| ~~`agentic_opd_plan.md`~~ | Subsumed into `research_workflows.md` after v3 redesign |
| ~~`agentic_v2_plan.md`~~ | Subsumed into `research_workflows.md` (v3 is the union of v2 + multi-archetype) |
| ~~`evo_opd_arena_rank.md`~~ | The ArenaRL-style tournament reward was implemented but the approach was abandoned for the agentic direction. Code remains in `evo_opd/judges/` + `evo_opd/trainer/tournament.py` if we want to revisit. |
| ~~`lv_opd_plan.md`~~ | Subsumed into `specs/evo_opd_open_ended.md` |
