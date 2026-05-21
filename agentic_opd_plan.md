# Agentic-OPD: a unified open-task RL recipe

> Drafted 2026-05-19, immediately after killing the v6 stack. We pivot from
> "offline reward signals on a single-turn proposal" to "ReAct-style multi-turn
> agent with tool-augmented lineage discovery + open-ended reward composition."
> Targets: ArenaRL open tasks (Open-DeepResearch, Open-Travel) + SGI-Bench task
> 2 (idea generation, 1k samples) + our GENE-Arena PES. One algorithm, three
> benchmark families.

## 1. Why agentic for open-ended idea generation

Previous attempts (v3 → v5 verifier-anchored RL; v6 tournament rank) all hit a
ceiling around PES ~57. Sub-dim analysis showed the bottleneck is *grounding*
— `parent_grounding`, `gene_inheritance`, `limitation_repair` are saturated on
the parent paper the prompt provides, but the model has no mechanism to seek
related work, integrate counter-evidence, or build a real lineage chain. A
human scientist proposing a follow-up paper would (a) search literature, (b)
read 2-5 papers in the relevant trajectory, (c) identify the *exact* gap that
the most recent paper leaves open, (d) propose a concrete next step.

**Agentic-OPD trains exactly that workflow.** The model becomes an agent with
tools (search, read, score-self) and its trajectories — not just final
outputs — are rewarded.

## 2. Algorithm sketch (per-prompt)

```
prompt = "Topic: <discipline> / <subtopic>. Propose a novel follow-up
          research idea building on the most relevant recent work."

trajectory = []
for step in range(MAX_STEPS):   # MAX_STEPS = 6
    a_t = model(prompt, trajectory)
    if a_t.tool == "search":
        results = search(query=a_t.query, year_min=2018)   # OpenAlex local cache
    elif a_t.tool == "read":
        card = read_card(s2_id=a_t.s2_id)                  # GenomeCard from cache
    elif a_t.tool == "propose":
        final = a_t.proposal                                # gene_genome JSON
        break
    trajectory.append((a_t, observation))

reward = compute_reward(trajectory, final, gold_lineage)
```

Tools (initially):
- `search(query, year_min, year_max, k=5)` — OpenAlex local cache + S2 cache
- `read(s2_id)` — returns GenomeCard from `genome_db` (or "not in cache")
- `propose(idea_genome)` — terminal action; emits final JSON

We start with a closed-vocabulary search (only papers in our 855-card db). This
removes API rate-limit concerns and makes lineage_recall reward well-defined.
Open-vocab search (live OpenAlex) is v2 of the pipeline.

## 3. Reward composition (open-task-agnostic)

Total reward per trajectory:

```
R_total = α_lineage · R_lineage          # how well the agent rebuilt the lineage
        + α_struct · R_struct            # Layer-1 PES (free, deterministic)
        + α_arena · R_arena_rank         # Layer-2 PES (tournament, GPT-5.5 judge)
        + α_efficiency · R_efficiency    # tool-call budget penalty
        − β_kl_ref · KL[π_θ || π_ref]
```

| Component | Defn | Why |
|---|---|---|
| `R_lineage` | recall over gold lineage_chain of s2_ids the agent `read`s | only signal that exercises the *agent* part — without this, model would just emit propose() at step 0 |
| `R_struct` | from `evo_opd/structural.py` (already built) — inheritance + limitation + balanced novelty | free deterministic Layer-1 PES, schema-agnostic |
| `R_arena_rank` | from `evo_opd/judges/pairwise_pes.py` + `tournament.py` — already built | tournament-rank z-advantage over K group rollouts; ports ArenaRL |
| `R_efficiency` | `−γ · max(0, n_tool_calls − T)` where T = budget (≈4) | prevents the agent from running forever; encourages parsimony |
| `KL_ref` | per-token KL vs initial π_θ | stability anchor (β = 0.01, unchanged from v6) |

Default weights: `α_lineage=0.5, α_struct=0.3, α_arena=0.7, α_efficiency=0.1`.

**Note on `R_lineage`:** GeneTrace already has lineage edges (parent → child
s2_id pairs). For each prompt, we know the gold ancestry of the "target"
paper. If the agent reads any subset of those ancestors, that's recall > 0.
If it reads only spurious papers, recall = 0.

## 4. Data construction

We have these primitives in `data/genetrace_v0_1/`:
- 855 GenomeCards (with `s2_id`, fields: driver/mechanism/limitation/...)
- ~300 DynamicsEdges (parent_s2_id → child_s2_id + dynamics label)
- ~50 LineageChains (≥3-hop ancestry)

Convert to agentic training data — one row per prompt:

```json
{
  "prompt_id":        "agentic::v1::p_0042",
  "topic":            "Physics-aware molecule generation",  // from GenomeCard fields
  "discipline":       "chemistry",
  "target_card":      { ...GenomeCard of the recent paper... },
  "gold_lineage":     ["s2_id_grand", "s2_id_parent"],     // ancestors
  "gold_proposal":    { ...the follow-up paper's gene_genome from GeneTrace... },
  "search_seed":      "physics-aware molecule diffusion 2023..2025",  // optional hint
}
```

**Tool corpus** = the 855 GenomeCards indexed by:
- s2_id (for `read`)
- title + abstract + driver_genome + mechanism_genome (for BM25 `search`)

This gives us ~1000 training prompts (one per "potential follow-up"). For each
prompt, the gold trajectory is known (ancestors → final propose) but we don't
demonstrate it — the agent has to find it via search + read.

**Bootstrap with GPT-5.5 demonstrations** (optional, ~$0 Azure keyless):
For each prompt, ask GPT-5.5 to produce a ReAct trajectory using the same
toolset. ~500 demonstrations × ~8 tool calls × ~$0 = a few hours wall-clock.
Use as SFT warm-start before RL.

## 5. Training algorithm

GRPO over trajectories, group size K=4 (we are doing 4-6× more compute per
step due to multi-turn tool calls — keep K small).

For each batch:
1. Sample 4 prompts.
2. For each prompt, sample K=4 trajectories (multi-turn ReAct).
3. Compute per-trajectory `R_total`.
4. Group-relative advantage `A_i = (R_total_i − μ_R) / σ_R`.
5. PG update with per-token mask (only tokens generated by π_θ, not tool
   observations) — same masking infra we have for v6.
6. KL_ref to initial π_θ as anchor.

**Per-token reward decomposition (key implementation point):**

Trajectory tokens fall into 4 roles:
- `thought` — model's reasoning text inside `<think>...</think>`
- `action` — JSON for tool call (`search`/`read`/`propose`)
- `observation` — tool result, NOT generated by model; **mask out** (no gradient)
- `proposal` — final gene_genome JSON

Reward `R_total` is broadcast onto generated tokens (thought + action +
proposal), with proportional weight per role (similar to FIELD_WEIGHT in v6).

## 6. Three eval targets — one algorithm

| Benchmark | Task | Output schema | Eval metric | Our model variant |
|---|---|---|---|---|
| **ArenaRL Open-DeepResearch** | answer a research question via search+synthesis | XML with `<think>`, `<answer>` | 7-dim rubric (their judge) | same agentic-OPD model |
| **ArenaRL Open-Travel** | plan a trip given constraints | structured plan | 2-dim rubric (their judge) | same agentic-OPD model |
| **SGI-Bench task 2** | propose an idea given a question | JSON with `Idea / ImplementationSteps / ...` | graph_similarity + 4-dim judge | same agentic-OPD model |
| **GENE-Arena PES** | propose follow-up paper given parent | gene_genome JSON | PES (Layer 1 + Layer 2) | same agentic-OPD model |

The model output adapts via prompt template — same weights for all four. SGI
and GENE both require structured JSON output; ArenaRL Open-DeepResearch wants
XML `<answer>` tag. We handle this via a prompt-template selector at eval time.

## 7. Implementation plan (5-7 days)

| Day | Artefact | Notes |
|---|---|---|
| 1 | data/agentic_v1/: 855 prompts + lineage labels + tool corpus | from existing GeneTrace files; one-shot script |
| 1-2 | `evo_opd/tools/`: search.py (BM25 over genome_db), read.py (cache lookup), propose.py (terminal) | pure local, no API |
| 2 | `evo_opd/trainer/agentic_rollout.py`: ReAct loop with tool dispatch | reuses sample_one_rollout from v6 |
| 3 | `evo_opd/rewards_agentic.py`: lineage recall, efficiency, hooks into struct/arena modules | composes existing pieces |
| 3-4 | `tools/build_agentic_sft_data.py`: GPT-5.5 demonstrations (optional warm-start) | ~$0 Azure |
| 4 | SFT warm-start on demonstrations (4× A100, ~4 hr) | optional but safer than cold RL |
| 5 | RL training: ~200 steps × K=4 × 4 prompts, 1× A100 student + 1× ref | ~24 hr |
| 5-6 | Eval adapters: SGI-Bench step_1 wrapper, ArenaRL adapter, GENE-Arena gen wrapper | each ~2 hr to write |
| 6-7 | Full eval on 4 benchmarks + ablations | each ~3-6 hr |

## 8. Open design questions (decide as we go)

1. **Tool call format** — function-call JSON vs ReAct prose vs Qwen3's built-in
   `<tool_call>` block? Default: Qwen3 native (best inference-time integration).
2. **Search backend** — closed (genome_db, 855 cards) vs open (live OpenAlex)?
   Default: closed for v1, open for v2. Closed gives clean recall metric.
3. **Group size K** — 4 (cheap, less judge cost) vs 8 (more discriminative)?
   Default: K=4 because trajectories are 4-6× longer than v6's single-turn.
4. **Bootstrap with SFT demos** — yes/no? Default: yes for safety; ablate later.
5. **Lineage gold definition** — strict ancestor s2_id match, or fuzzy
   embedding distance? Default: strict for v1.
6. **Max trajectory length** — 4 / 6 / 8 tool calls? Default: 6.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Closed-vocab search is too restrictive (only 855 papers) | Expand to ~5K via OpenAlex one-time crawl; or use GenomeCard embeddings for BM25 fallback |
| Trajectory rollouts are slow (4 min/step × multi-turn) | Cache tool observations; rollouts stay under 8 min/step |
| Tool-call format errors (model emits malformed JSON) | Hard penalty in `R_efficiency`; SFT warm-start teaches format |
| RL collapse to "propose at step 0" (no tool use) | `R_lineage` requires at least one `read` to score > 0 |
| Judge cost explosion on long trajectories | Judge sees only the final `propose` content, not intermediate steps (cheap) |

## 10. What ships in the paper

**Main result:** agentic-OPD beats v3 SFT (current best) on PES AND beats v6
ablations on PES, while also performing on SGI-Bench + ArenaRL open tasks.

**Two contributions framed cleanly:**
1. **C1 (algorithm)**: agentic-OPD — a unified open-task RL recipe with
   structural + tournament + lineage rewards, schema-agnostic, transfers to
   3 benchmark families with prompt-template-only changes.
2. **C2 (dataset)**: GeneTrace as a lineage-supervised training corpus for
   agentic scientific reasoning (existing C2; positioning carried over).

ArenaRL's contribution gets cited as the closest prior; we differentiate via
(a) multi-source reward composition not just tournament, (b) cross-domain
transfer to scientific idea gen, (c) explicit lineage supervision.
