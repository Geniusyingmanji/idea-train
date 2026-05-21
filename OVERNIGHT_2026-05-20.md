# Overnight 2026-05-20 — Research Workflow Demo Dataset (v3)

> 用户 ~17:50 睡眠前嘱：用 web search + 免 key GPT-5.5 持续构造高质量科研 agentic
> 数据，模拟真实科研轨迹，多样化思路。同时其他人占用 GPU，RL 不能跑。

## 完成的事

### 1. 第一性原理 + 10 工作流原型（已写入 `research_workflows.md`）

科研第一性原理：
> 科研 = 在不确定性中迭代构建可验证因果模型，并在已有知识网络上找到合适连接点

8 核心 skill：问题澄清 / 谱系建构 / 缺口识别 / 方法迁移 / 批判评估 / 多源综合 / 假设具体化 / 反思修正

10 工作流原型（archetypes）：
- W1 Pure Discovery（纯探索式：模糊问题 + 宽搜）
- W2 Single-Paper Extension（基于 1 篇论文延伸）
- W3 Multi-Paper Synthesis（多篇桥接）
- W4 Literature Review（综述式）
- W5 Critical Analysis（批判式）
- W6 Cross-Domain Bridge（跨域迁移）
- W7 Hypothesis Refinement（迭代精炼）
- W8 Reproduction Doubt（复现质疑）
- W9 Tool-Heavy Comparison（密集对比）
- W10 Vague-to-Concrete（模糊→具体）

### 2. v3 Demo 数据集（已生成 1032 demos）

| 维度 | v3 | vs v1 | vs v2 |
|---|---|---|---|
| 总 demos | **1032** | 246 (4.2×) | 287 (3.6×) |
| 学科覆盖 | 10 主 + 12 sub | 5-6 | 10 |
| Archetypes | **10 种** | 1 (隐式) | 1 (隐式) |
| Avg tools/demo | **7.78** | 3.74 | 3.74 |
| 6 工具全用 | ✓ | ✗ (3 工具) | ~✓ |
| Median 长度 | 10256 chars (~2.5K tok) | ~885 tok | ~1510 tok |

### 3. 工具使用分布（v3 vs 之前）

| Tool | v3 total | v3 avg/demo | 之前 v2 avg |
|---|---|---|---|
| read | 2523 | 2.44 | 1.10 |
| search | 2312 | 2.24 | 0.76 |
| propose | 1128 | 1.09 | 0.75 |
| extract_genome | 727 | 0.70 | 0.32 |
| novelty_check | 615 | 0.60 | 0.76 |
| genome_diff | 575 | 0.56 | 0.04 |

genome_diff 用量从 0.04 → 0.56 暴涨 14×（archetype W7/W2 主推 self-check）。

### 4. Skill 覆盖

| Skill | demos demo'd (%) |
|---|---|
| gap_identification | 1032 (100%) |
| multi_source_synthesis | 936 (91%) |
| critical_evaluation | 720 (70%) |
| problem_clarification | 736 (71%) |
| hypothesis_specification | 593 (57%) |
| lineage_construction | 567 (55%) |
| method_transfer | 480 (47%) |
| reflection_revision | 343 (33%) |

reflection_revision 偏低（仅 W7 archetype 专攻），但仍有 33% 覆盖。

### 5. 文件清单

```
data/agentic_v3/
├── prompts.jsonl         (1478 prompts: 1000 v3-synthetic + 478 v2-tagged)
├── sft_demos.jsonl       (1032 demos, ~10 MB)
├── skill_breakdown.json  (machine-readable stats)
├── summary.md            (人读总结)
└── exemplars.md          (每 archetype 1 个最佳样例)

idea_train/
├── research_workflows.md         (第一性原理 + archetype 设计)
├── OVERNIGHT_2026-05-20.md       (本文档)
└── tools/build_research_workflow_demos.py  (生成脚本)
```

## 同时跑的 baseline 评测（仍在跑）

| Job | Progress | Notes |
|---|---|---|
| GENE-Arena (v2-sft + web, n=150) | 75/150 (50%) | propose rate looks healthy, ETA ~3hr |
| SGI-Bench task 2 (v2-sft + web, n=100) | 30/100 (30%) | ~7hr |

完成后会给出 v2-sft 在两个 benchmark 上的具体数字。

## 我目前学到的关键认识（写入 research_workflows.md）

1. **工作流多样性 ≠ 工具种类多** — 真正的多样性是"工具用法序列"的多样性。同样 6 个工具，W4 综述会 read × 5-8，W2 单篇延伸只 read 1-2。
2. **archetype 是 prompt 的属性，不是模型的属性** — 模型应该根据 prompt 自动选择工作流。SFT 把 prompt-to-archetype 关联训进去。
3. **GPT-5.5 是推理模型** — 需要充足 max_tokens（4000+）才能完成"思考 + 输出"全流程。一开始用 1800 导致 99% 空响应。
4. **synthetic prompt 仍可控** — 通过 archetype-specific prompt style hint，让 GPT-5.5 生成结构清晰、可被 archetype 解析的训练 prompt。
5. **真实 OpenAlex paper_id 锚定关键** — 把搜索结果（5 篇真论文）放在 demo gen 的 system prompt 里，让 GPT-5.5 用真 ID，避免模型学到 fake ID。

## 接下来要做的（用户醒后建议）

| 优先级 | 任务 | 预计 |
|---|---|---|
| **P0** | 等 GPU 空（Ray Serve 退） → SFT v3 on 1032 demos (max_len=3072, LR=3e-5) | ~3 hr |
| **P0** | v2-sft eval 结果出来 → 看 v2 vs old baselines | 已在跑 |
| **P1** | SFT v3 完成 → eval on GENE-Arena + SGI + ArenaRL → 终极对比 | ~10 hr |
| **P2** | RL on v3-sft（若 GPU 释放）→ 验证 v3 是否能上 60+ PES | ~20 hr |
| **P3** | 再补 W3/W9（仅 ~80 each）至 150 each | 0.5 hr |

## 第一性原理回顾

我做这批数据时，心中 anchor 是用户的话：**"第一性原理就是训练出通用的科研模型"**。

为此 v3 数据的设计原则：
1. **覆盖广度**：10 archetypes × 10 disciplines × 多种 prompt 风格
2. **覆盖深度**：每 demo 平均 7.78 工具调用，比 v1/v2 高 2× — 模型见过"深度推理"的样子
3. **结构化技能传授**：8 skill 通过 archetype-skill matrix 系统性覆盖
4. **真实工具锚定**：所有 paper_id 来自真实 OpenAlex 搜索结果，模型不能学到伪造
5. **梯度难度**：W2/W8（简单单源）+ W3/W4/W9（复杂多源）+ W1/W10（模糊探索）

数据生成总成本：**$0**（Azure GPT-5.5 keyless + OpenAlex 免费）。Wall-clock 约 2 小时（synthesizing prompts + prefetching + GPT-5.5 calls）。

睡好。下次问"评测怎么样"会有具体 PES + graph_similarity 数字。
