# Research Workflows — 第一性原理思考

> 这份 doc 是 v3 agentic 数据集的核心设计思想。要训出一个**通用科研模型**，
> 必须先理解人做科研时的真实轨迹，再让数据反映这些轨迹的全部多样性。
> 不是简单的 "search→read→propose"，而是多种工作流原型 (archetypes) 的混合。

## 1. 第一性原理：什么是科研？

科研 ≠ 信息检索 + 写一篇综述。
科研 = **在不确定性中迭代构建一个能被验证或证伪的因果模型，并在已有知识网络上找到合适的连接点。**

这意味着模型需要的核心能力：

1. **问题澄清 (problem clarification)**：把模糊问题转化为可执行的子问题
2. **谱系建构 (lineage construction)**：在已有文献中找到自己想法的祖先和兄弟
3. **缺口识别 (gap identification)**：发现哪里没人做、哪里做错了、哪里不完整
4. **方法迁移 (method transfer)**：把领域 A 的方法移植到领域 B
5. **批判性评估 (critical evaluation)**：不接受表面论断，追问机制和证据
6. **多源综合 (multi-source synthesis)**：跨论文/跨子领域提取共同模式
7. **假设具体化 (hypothesis specification)**：从模糊 idea 走到可验证机制
8. **反思与修正 (reflection and revision)**：用 diff/novelty 对自己的提议自查

这 8 项是 v3 数据集要传授给模型的能力。每条 demo 应该体现 1-3 项。

## 2. 工作流原型（10 类）

每种 archetype 对应一种现实中常见的科研活动，明确：
- **起点状态**（手上有什么）
- **目标产出**（要做出什么）
- **工具序列模板**（典型的 ReAct 模式）
- **关键 skill**（demo 主要教给模型的能力）

### W1. Pure Discovery（纯探索式）

起点：一个模糊的研究问题 ("如何让 LLM 在数学竞赛上接近 IMO 选手？")
产出：基于多篇文献综合的研究方向

序列模板：
```
search(broad query) → read 2-3 top results → search(narrower 1) → read → ...
→ propose(grounded in 3-5 papers)
```

Skill：问题澄清、谱系建构、多源综合

### W2. Single-Paper Extension（单篇延伸）

起点：1 篇 anchor 论文
产出：基于其 limitation 的 follow-up

序列：
```
extract_genome(anchor) → search(related)  → read 1-2 closest → 
genome_diff(anchor, candidate proposal) → propose 修正版
```

Skill：谱系建构、缺口识别、反思修正

### W3. Multi-Paper Synthesis（多篇合成）

起点：2-4 篇 anchor 论文
产出：发现它们共同的潜在模式，提出统一框架

序列：
```
extract × N → genome_diff(pairwise) → 找重叠的 mechanism →
search 一个 "桥梁" 概念 → propose 统一框架
```

Skill：多源综合、方法迁移

### W4. Literature Review（综述式）

起点：领域 + 时间窗
产出：结构化 mini-review

序列：
```
search × 多 (不同关键词) → read × 5-8 → 按 mechanism / niche / dataset 分组 →
propose 综述 (但用 gene_genome 格式作为结构骨架)
```

Skill：多源综合、问题澄清

### W5. Critical Analysis（批判式）

起点：1 篇 anchor 论文，可能有可疑的 claim
产出：指出方法学缺陷 + 提议修正实验

序列：
```
read(anchor) → extract_genome → novelty_check(其 mechanism) →
search "limitations of X" → read 2-3 反驳/补充 → propose 反例实验
```

Skill：批判性评估、假设具体化

### W6. Cross-Domain Bridge（跨域迁移）

起点：领域 A 的某种成熟方法 + 领域 B 的某个问题
产出：将 A 的方法移植到 B 的提议

序列：
```
search(A 方法) → read → extract_genome → 
search(B 问题域) → read → extract_genome → 
genome_diff(A_mech, B_niche) → propose 迁移
```

Skill：方法迁移、跨域思维

### W7. Hypothesis Refinement（假设迭代）

起点：一个粗糙的初始 idea
产出：经过多轮自查后的精炼版

序列：
```
propose(粗糙版) → genome_diff(自查 vs 文献) → novelty_check →
search 缺失的成分 → read → propose(精炼版)
```

Skill：反思修正、假设具体化

### W8. Reproduction Doubt（复现质疑）

起点：1 篇有可疑 claim 的论文
产出：验证或证伪的实验设计

序列：
```
read(anchor) → search(replication of X) / search(criticism of X) → 
read 2-3 → extract → 评估 → propose 验证实验
```

Skill：批判性评估、假设具体化

### W9. Tool-Heavy Comparison（密集对比）

起点：3+ 个待对比的方法
产出：用 diff/novelty 量化比较，给出选择建议

序列：
```
search → read × 3+ → extract × 3+ → genome_diff × N pairs →
novelty_check on each → propose 比较结论或下一步实验
```

Skill：多源综合、批判性评估

### W10. Vague-to-Concrete（模糊到具体）

起点：极度模糊的科学问题 ("如何让蛋白质设计更可控？")
产出：具体可执行的研究问题 + 初步方案

序列：
```
search(broad) → 看结果，重写 query → search(narrower) → read 1-2 →
extract → 收窄问题域 → search 具体方法 → read → propose 具体方案
```

Skill：问题澄清、谱系建构

## 3. 数据生成原则

1. **每个 prompt 对应 1 个 archetype（主要）+ 偶尔混合 1-2 个**
2. **GPT-5.5 system prompt 明确描述当前 archetype 的目标 + 序列模板**
3. **使用真实 OpenAlex 候选作为 search 锚点**（不让模型造假 ID）
4. **demos 长度允许 1500-3000 tokens**（W3/W4/W9 较长）
5. **覆盖学科**：CS / Physics / Bio / Chem / Materials / Earth / Energy / Math / Neuro / Astro，每域 ≥ 50 demos
6. **覆盖语义难度**：简单（W2）/ 中等（W1, W6, W7）/ 复杂（W3, W4, W9）
7. **保留失败案例的少量样本**（5-10%）：让模型学会识别 dead-end 并 pivot

## 4. 数据生成流程

```
[1] build prompt pool (~1500)
    ├── re-use agentic_v2/prompts.jsonl (478)
    ├── + synthesize new prompts via GPT-5.5 for each archetype (10 × ~100 = 1000)
    └── tag each prompt with primary archetype

[2] for each (prompt, archetype):
    ├── pre-fetch 5 OpenAlex candidates (for search anchoring)
    ├── construct GPT-5.5 system prompt with archetype description + tool list
    ├── GPT-5.5 generates trajectory (one shot, max 2500 tokens)
    └── parse + quality-gate

[3] post-process: extract skills demonstrated per demo
    ├── parser: scan tool sequence → infer archetype from pattern
    ├── classify skills (8-class soft labels)
    └── write skill_breakdown.json

[4] write summary doc
    └── statistics, exemplar trajectories, distribution analysis
```

## 5. 评估指标（构造完毕后）

- Total demos
- Per-archetype distribution (should be roughly uniform)
- Per-discipline distribution
- Avg tool calls per demo
- Avg tools-per-demo distribution (should NOT be uniform — W2 is short, W9 is long)
- Skill coverage matrix (each skill should appear in ≥ 100 demos)

## 6. 训练 implications

这批 v3 demos 用于 SFT v3：
- LR 略低 (3e-5 vs 5e-5)，因为更长上下文
- max_len 4096（W9 demos 可能 ~2.5K tokens）
- 2 个 epoch
- 在 v2-sft checkpoint 基础上继续训（保留已学的格式 + 新加多样性）

期待 v3-sft 在 GENE-Arena + SGI + ArenaRL 都更稳健，因为见过类似 prompt 的分布。
