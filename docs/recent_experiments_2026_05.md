# Recent experimental results (2026-05-17 → 2026-05-22)

Compiled 2026-05-22. Covers the agentic-OPD phase (post the original GENE-Exam
SFT runs documented in `eval/results/OVERNIGHT_REPORT.md`).

## TL;DR

- **Non-agentic SFT v3 remains the GENE-Arena PES champion at 57.95** (vs 50.96 baseline / 56.38 evo-OPD-v5). All later evo-OPD RL variants traded PES for tool-task gains.
- **Agentic v3 SFT model has an 18% propose rate** on GENE-Arena (27/150) — confirmed over-tooling. Median 1180 output tokens, median 466 s/idea wall time. This is the empirical motivation for the v5-v33 data expansion.
- **Agentic v2 SFT had 60% propose rate** (70/116) — interesting comparison point: v2 data was shorter/simpler, v3 data was longer with more multi-tool archetypes. The model over-fitted to long trajectories.
- Data expansion: **5953 SFT demos + 2593 DPO pairs + 2930 RL prompts**, 73 disciplines, 21% Chinese. Now sitting in `data/agentic_combined_v3to33/` + `dpo_combined/` + `rl_prompts_combined/`. See `docs/data_inventory.md` for per-round detail.

## 1. GENE-Arena PES — head-to-head

| Model | n | PES | Δ vs baseline | Notes |
|---|---:|---:|---:|---|
| 🏆 Qwen3-8B SFT v3 (non-agentic) | 150 | **57.95** | +7.0 | Original 1607-example SFT, no agentic tools |
| Qwen3-8B evo-OPD v5 | 150 | 56.38 | +5.4 | Best RL variant, slightly under SFT v3 |
| Qwen3-8B baseline (no LoRA) | 150 | 50.96 | — | No-think mode |
| Qwen3-8B agentic-v2-sft + web | 116 | ~46.84 | −4.1 | Azure rate-limit contaminated 133 scores |
| Qwen3-8B agentic-v3-sft + web | 150 | **pending PES** | — | Rollouts done 2026-05-21; propose rate 18% blocks fair PES |

The PES gap between SFT v3 (57.95) and evo-OPD v5 (56.38) is the "trade-off" finding: evo-OPD lost 1.56 PES on T1/T2 but gained on T3 (tool-task) by 2× and unlocked T4 reasoning. SFT v3 is the strict GENE-Arena PES winner.

## 2. Agentic propose-rate finding (key motivator)

Rollouts on 150 GENE-Arena prompts using `agentic_eval_gene_arena.py`:

| Adapter | n | propose_emitted | rate | median tokens | median latency |
|---|---:|---:|---:|---:|---:|
| `qwen3-8b-agentic-v2-sft + web` | 116 | 70 | **60%** | — | — |
| `qwen3-8b-agentic-v3-sft + web` | 150 | 27 | **18%** | 1180 | 466 s |

**Diagnosis**: v3 SFT was trained on 1032 demos with **median 10 tools/demo and 77% long trajectories**. The model learned "research = many tool calls", and ran out of `max_turns` (9) before reaching propose 82% of the time. Despite richer demonstrations, the trained behavior over-deliberated to the point of failure.

**Fix direction**: v5-v33 data expansion inverts the bias. The new combined corpus is **64% short (1-3 tools) / 14% med / 22% long**, median 3 tools/demo. v11 (471 direct-propose demos), v14 (275 clean 2-turn refinement), and v18 (173 self-correction demos with mid-trajectory pivot) directly target the over-tooling failure mode.

## 3. Data corpus snapshot (post-expansion)

Total **5953 SFT + 2593 DPO + 2930 RL prompts** across 18 SFT rounds + 5 DPO rounds + 2 RL rounds.

| Pool | Count | Median tools/demo | Short % | ZH share |
|---|---:|---:|---:|---:|
| `data/agentic_combined_v3to33/sft_demos.jsonl` | 5953 | 3 | 64% | 21% |
| `data/dpo_combined/preferences.jsonl` | 2593 | — | — | — |
| `data/rl_prompts_combined/rl_prompts.jsonl` | 2930 | — | — | — |

Full per-round purpose in `docs/data_inventory.md`. The expansion targeted four under-served axes:

1. **Length distribution**: from 11/12/77 (short/med/long) in v3 → 64/14/22 now.
2. **Schema rigor**: v6 + v15 + v8-sgi-strict added explicit `idea_plan` + `free_text_answer` schemas (was gene_genome-only).
3. **Chinese coverage**: from 8% in v3 → 21% now (v6, v9-travel, v12, v15, v21, v28, v29 — plus cross-lingual v30/v31).
4. **DPO yield**: v7 / v10 had 34-41% pair yield (independent calls); v13 corruption-style hit **95% yield** by taking existing SFT demos as `chosen` and corrupting them as `rejected`.

## 4. Methodological lessons

### What worked

| Approach | Outcome |
|---|---|
| **Corruption-based DPO** (v13/v19/v22/v27) | 95% pair yield vs 34-41% for tournament-style |
| **Schema-strict system prompts** | v5 + v6 + v15 all hit 95%+ valid; loose prompts (v32 retry-1) returned 0% |
| **Explicit length-tier hints** | v5/v11 produced exactly the short distribution requested |
| **2-turn dialogue with simpler structure** | v14 yielded 100% (vs v9's 22%); the cleaner format made parsing trivial |

### What broke

| Issue | Round | Root cause | Fix |
|---|---|---|---|
| 0 valid demos | v11 first run | `max_tokens=1800` truncated `n=40` prompts mid-JSON | Bumped to 4000, fine |
| 0 valid demos | v32 first run | `max_tokens=2200` truncated `n=15` long-context prompts | n=10, max_tokens=4000 |
| 0 valid demos | v32 retry-1 | Loose SYS prompt let GPT-5.5 use markdown headers (`Action 1 —`) instead of ` ```action ``` ` code blocks | Strict format SYS + sample format |
| 0 valid demos | v7 first run | `.format()` interpreted literal `{6 fields}` braces in template | Replaced `{...}` with `[...]` in non-template parts |
| GPT-5.5 reasoning truncation | early v3 | `max_tokens=1800` left ~0 tokens for content | Standard now: 2500-4000 max_tokens for demo gen |

### What didn't pay off

- **v3 to v7 SFT GENE-Exam ablations** (more data, bigger LoRA, strict-key prompts): all regressed below v3. Documented in `OVERNIGHT_REPORT.md`. Data quality + prompt-schema mismatch is the bottleneck, not adapter capacity.
- **Tournament-style DPO** (v7, v10): low pair yield because both `chosen` and `rejected` had to independently succeed. v13 corruption approach (one-shot rewrite of an existing demo) is now the template.

## 5. Open questions / next steps

1. **Re-SFT on the combined corpus** (`agentic_combined_v3to33/sft_demos.jsonl`, 5953 demos). Hypothesis: propose rate jumps from 18% → 60-80% based on the v2 vs v3 comparison.
2. **DPO from `dpo_combined/preferences.jsonl`** (2593 pairs) after the re-SFT.
3. **PES on the re-SFT model**: does it match SFT v3's 57.95, or does the agentic format still cost points?
4. **ArenaRL Open-Travel + Open-DeepResearch eval**: Chinese now at 21% of corpus — should unlock these benchmarks that we've been blocked on.
5. **Sample efficiency study**: v3 alone won at 1032 demos. Does the 5953-demo corpus help, or does it dilute the signal?

## 6. Reproducing the rollouts

The agentic_v3_sft rollouts (150 ideas, ~19 GPU-hours on 1× A100) are in:
- `eval/results/arena_agentic_v3_sft/manifest.jsonl` — per-idea metadata
- `eval/results/arena_agentic_v3_sft/ideas/<topic>/qwen3-8b-agentic-v3-sft-web_<setting>.json` — per-rollout trajectories

To re-run on a new checkpoint:

```bash
python tools/agentic_eval_gene_arena.py \
  --student-lora <new_lora_path> \
  --participant <participant_name> \
  --gpu <id> --workers 1 --max-turns 9 \
  --max-new-tokens-per-turn 512 --temperature 0.5 \
  --search-backend web \
  --output-dir eval/results/arena_<name>
```

PES judging (separate Azure GPT-5.5 step) is currently blocked on Azure rate-limit. Will run when quota recovers.
