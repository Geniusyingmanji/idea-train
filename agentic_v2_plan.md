# Agentic-OPD v2: Web-Native + Rich Tool Belt

> Drafted 2026-05-20 after agentic v1 plateaued (RL on local-only data: F̄=+1.50,
> S̄=0.18, propose=100%, but no struct-quality gains beyond SFT). Root cause:
> training distribution = 855-card local corpus, eval distribution = full web /
> diverse arena/SGI prompts. We pivot to web-native data + richer tools ported
> from `IdeaEvolving/agent/`.

## 1. Strategic Decisions Locked In

1. **Stop training on local-only data.** Agentic v1 SFT + RL has hit ceiling.
   v2 starts from the existing `qwen3-8b-agentic-sft` (or possibly v3 SFT)
   as base, re-SFT on the new web-native demos.

2. **Web search is the default backend.** Local 855-card BM25 stays only as
   a fast in-distribution sanity tool; OpenAlex (and maybe arXiv/S2) is the
   primary source.

3. **Tool belt expands from 3 → 6** to give agent richer scientific reasoning
   primitives — beyond just "search → read → propose".

4. **Training prompts mix three benchmark distributions** (GENE-Arena, SGI,
   synthetic) so the SFT model is generalist by construction, not just
   transferred at eval time.

## 2. New Tool Belt (6 tools)

Current 3 tools (kept, but `search` upgraded):

| Tool | Status | What it does |
|---|---|---|
| `search(query, year_min, year_max, k)` | upgraded | OpenAlex (default) or local BM25; returns `[(paper_id, title, year, snippet)]` |
| `read(paper_id)` | upgraded | HybridReadTool: handles `oa:W...`, `paper:...`, ; returns title + year + authors + venue + abstract + cited_by |
| `propose(gene_genome)` | unchanged | terminal action; emits 6-field gene_genome |

New 3 tools (ported from `IdeaEvolving/agent/`):

| Tool | Source | Why it helps the agent |
|---|---|---|
| `extract_genome(paper_id_or_text)` | `genome_extract_agent.py` | Given a paper, return its structured 6-field gene_genome (parent context for inheritance reasoning). Different from `read` which returns raw abstract — `extract_genome` returns the already-structured driver/mechanism/limit/etc. |
| `genome_diff(parent_id, proposed_genome)` | `genome_differ.align_gene_cards` + classify_dynamics | Given parent paper + agent's proposed genome, return per-gene fates (INHERITED/MUTATED/LOST/NOVEL/HYBRIDIZED) + inferred dynamics. Agent can iterate: propose → diff → revise. |
| `novelty_check(proposed_mechanism)` | new (uses local + OpenAlex similarity) | Returns nearest 3 papers + similarity score. If sim > 0.85 → "redundant" warning; if < 0.15 → "disconnected from any lineage" warning. |

These give the agent **structured reasoning** beyond just retrieval — it can check its own proposal for redundancy / weak inheritance / dynamics mismatch before committing.

### Tool call format (JSON in fenced ```action block):

```
```action
{"tool": "extract_genome", "paper_id": "oa:W4281790889"}
```

```action
{"tool": "genome_diff", "parent_id": "oa:W4281790889",
 "proposed_genome": {"mechanism_genome": "...", ...}}
```

```action
{"tool": "novelty_check", "mechanism": "Foundation model for ..."}
```

Action: result is a small structured JSON observation.

## 3. Training Data: `agentic_v2`

### 3.1 Prompt pool (~500 prompts)

| Source | Count | Recipe |
|---|---|---|
| GENE-Arena tasks | 50 × 3 settings = 150 | Direct from `IdeaEvolving/gene_arena/tasks/*.json` via PromptBuilder; Library/Lineage/Question split |
| SGI-Bench questions | 200 (sample from 315) | Direct from HF `InternScience/SGI-IdeaGeneration`; stratified across 10 disciplines |
| Synthetic | 150 | GPT-5.5 generates 15 research topics per discipline × 10 disciplines (covers gaps in above two) |

Each prompt has:
- `topic` (or full structured arena/SGI prompt)
- `discipline`
- `year_min_hint`, `year_max_hint`
- NO `gold_lineage` — agent uses real web search; we judge by struct + arena rank only.

### 3.2 Demo trajectories (~500 demos)

For each prompt, GPT-5.5 plays the agent role with our 6 tools. Demo loop:

```python
for prompt in pool:
    history = []
    for turn in range(MAX_TURNS):
        action = gpt55.next_action(prompt, history, ROLLOUT_SYS_PROMPT_V2)
        if action.tool == "search":
            obs = openalex.search(action.query, ...)
        elif action.tool == "read":
            obs = openalex_read(action.paper_id)
        elif action.tool == "extract_genome":
            obs = genome_extract_agent.extract(action.paper_id_or_text)
        elif action.tool == "genome_diff":
            obs = genome_differ.align_gene_cards(parent_card, proposed_card)
        elif action.tool == "novelty_check":
            obs = compute_novelty(action.mechanism, openalex_index)
        elif action.tool == "propose":
            break
        history.append((action, obs))
    save_demo(prompt, history)
```

GPT-5.5 system prompt should encourage **diverse tool sequencing patterns**:
- some demos: search → read → propose (simple)
- some demos: search → read → extract_genome → genome_diff → propose (deeper)
- some demos: search → read → propose → genome_diff(self-check) → propose-revised
- some demos: novelty_check first → search → read → propose

Cost: ~500 demos × ~6 GPT-5.5 calls each = ~3000 GPT-5.5 calls (Azure free)
     + ~5000 OpenAlex calls (free)
Wall-clock: ~3 hours at 16 workers parallel.

### 3.3 Quality gates

- Drop demos where `propose` not reached
- Drop demos with > 3 malformed actions
- Drop demos with < 1 `read` or < 1 `extract_genome` (must use tools)
- Manual spot-check 10 random demos for sanity

Target: ~400 usable demos after filtering.

## 4. SFT Training (~3 hr on 1 GPU)

Start from `qwen3-8b-sft-v3/final` (the strongest pure-SFT base, PES 57.95 on
GENE-Arena before we got fancy). NOT from `qwen3-8b-agentic-sft` because v1
has imprinted on local paper IDs — re-SFT from v3 is cleaner.

Config (same as v1, just new data + new system prompt with 6 tools):
- LoRA r=64, α=128
- 2 epochs
- LR 5e-5, batch 2 × grad_accum 4
- max_len 4096 (demos with 6-tool trajectories are longer than v1)
- `CUDA_VISIBLE_DEVICES=2` to avoid the cross-GPU leak we hit before

Output: `train/checkpoints/qwen3-8b-agentic-v2-sft/final`.

## 5. RL Training (~30 hr on 4 GPUs)

Same trainer (`evo_opd/trainer/agentic_loop.py`) — only changes:
- Pass `--use-web-search` (new flag we'll add)
- Use new prompt pool (mix of arena + SGI + synthetic + agentic_v1)
- α_lineage drops to 0.0 (no gold ancestors in web prompts)
- α_struct = 0.6 (primary signal)
- α_format = 0.3 (keep)
- α_arena = 0.0 → 0.5 after step 50 (introduce tournament once format is solid)

Steps: 200, K=4, max_turns=8 (more for richer tools), max_new_tokens=512.

Wall-clock per step: ~15 min (slower than v1 due to web latency + more tools).
Total: ~50 hours. Will do partial run (100 steps, ~25 hr) for first signal.

## 6. Eval Pipeline (no change in script structure, just point to new model)

| Benchmark | Adapter | Metric | ETA |
|---|---|---|---|
| GENE-Arena | `agentic_eval_gene_arena.py --search-backend web` | PES via Azure GPT-5.5 | 9 hr (gen) + 30 min (PES) |
| SGI-Bench task 2 | `agentic_eval_sgi_bench.py --search-backend web` | graph_similarity (their script) | 15 hr (gen) + 1 hr (score) |
| ArenaRL Open-DeepResearch (defer) | needs adapter for their schema | their 7-dim rubric | TBD |

## 7. Implementation Plan (3 phases, ~4 days total)

### Phase A — Tools + Data (1 day)
| Step | Hours | Artefact |
|---|---|---|
| A1. Implement `extract_genome` tool | 2 | `evo_opd/tools/genome_tool.py` (port from IdeaEvolving) |
| A2. Implement `genome_diff` tool | 2 | same file or `evo_opd/tools/diff_tool.py` |
| A3. Implement `novelty_check` tool | 1 | uses OpenAlex search + embedding distance |
| A4. Update rollout to dispatch 6 tools | 2 | `evo_opd/agentic/rollout.py` + system prompt v2 |
| A5. Build prompt pool (arena + SGI + synthetic) | 1 | `tools/build_agentic_v2_prompts.py` |
| A6. Generate demos via GPT-5.5 (3-4 hr wall) | 4 | `tools/build_agentic_v2_demos.py` |

### Phase B — SFT (0.5 day)
| Step | Hours | Artefact |
|---|---|---|
| B1. Re-SFT from v3 base | 3 | `train/checkpoints/qwen3-8b-agentic-v2-sft/final` |
| B2. Smoke test on 5 each (arena/SGI/synthetic) | 0.5 | sanity: propose rate, tool diversity |

### Phase C — RL + Eval (2-3 days)
| Step | Hours | Artefact |
|---|---|---|
| C1. RL: 100 steps initial | 25 | first signal on web-native RL |
| C2. PES eval ckpt-50, ckpt-100 | 10 | comparison curve |
| C3. SGI-Bench eval ckpt-100 | 15 | graph_similarity number |
| C4. If C2-C3 promising: RL 200 steps | 30 | full RL run |
| C5. Write up results | 4 | STATUS + paper section update |

## 8. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| OpenAlex rate-limits during RL (200 steps × 4 K × ~6 tools = ~5K calls) | medium | cache is per-query; re-runs hit cache; if rate-limit, drop to k=3 results |
| Demos too long (>4K tokens) | high | truncate observation text; cap at 5 turns for SFT, 8 for RL |
| Tool dispatch parser breaks on 6 tools | medium | add unit tests for each tool's action format |
| Re-SFT from v3 vs from agentic-v1: which is better base? | medium | smoke-test both; pick the one with higher propose rate on web prompts |
| GENE-Arena PES still bounded by judge variance | low | accept; bootstrap CI is our defense |

## 9. Open Design Questions (decide during execution)

1. Should `extract_genome` cache results? Each extract is ~5s on GPT-5.5; caching saves ~30% of wall-clock.
2. Should `novelty_check` use embedding distance or lexical-only? Embedding is more accurate but adds dependency (sentence-transformers). Default lexical for v1.
3. Should agent see "reward intermediate signals" during rollout (e.g., genome_diff results revealing weak inheritance)? This would let agent self-correct — but pollutes the training distribution if SFT demos don't show it.
4. What % of demos should use ≥4 tools (deep) vs ≤3 tools (shallow)? Default: 50/50.

## 10. What v2 Doesn't Touch

- Reward weights (α_lineage/α_struct/etc.) — unchanged from v1 except α_lineage = 0
- Trainer code structure (`agentic_loop.py`) — only flag additions
- Eval scripts — only `--search-backend web` already supported
- LoRA hyperparams (r=64, α=128) — unchanged

## 11. Success Criteria

| Metric | Current (v1 SFT, eval with web smoke n=9) | v2 Target |
|---|---|---|
| GENE-Arena PES | 51.25 | **≥ 60** (cleanly beat old v3 SFT 57.95) |
| GENE-Arena propose rate | 6/9 = 67% | **≥ 90%** |
| SGI-Bench task_2 graph_similarity | (no number yet) | **≥ GPT-4.1 baseline** |
| Tool diversity (avg tools per traj) | 3.0 | **3.5–4** (use all 6 sometimes) |

---

If you approve, I'll start with Phase A1 (port `extract_genome` and `genome_diff` as tools) and proceed sequentially.
