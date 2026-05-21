# GeneTrace v0.1

> The first training-grade release of the genome-centric paper-lineage corpus.
> Companion artefact to the genetrace-v0.1 paper (anon).

## Statistics

| Level | Records | File |
|---|---|---|
| 1 GenomeCard     | 855   | `cards.jsonl` |
| 2 DynamicsEdge   | 300   | `edges.jsonl` |
| 3 LineageChain   | 0  | `chains.jsonl` (omitted from v0.1; see Known Limitations.) |
| 4 VerifierBundle | 1155 | `verifier_bundle.jsonl` |

Build timestamp (UTC): 2026-05-18T15:21:33Z
Verifier version: `evo_opd.verifier@v0.1`
Schema version: `genetrace-v0.1`

## Known limitations (v0.1)

- **DynamicsEdges are SFT-synthesised, not citation-grounded.** The 300 edges
  in v0.1 were synthesised by sampling random paper pairs from the safe pool
  and asking GPT-5.5 to predict the dynamics label. They are useful for
  training a model to recognise dynamics patterns (which is what SFT v3 did),
  but they do not correspond to real citation links. v0.2 will rebuild edges
  from the S2/OpenAlex citation graph restricted to the safe pool.
- **Evidence quotes are not yet extracted.** All `evidence` fields in v0.1
  are empty arrays. Stage E of the build script (GPT-5.5 evidence extraction)
  must be run separately and is gated by cost approval; planned for v0.1
  freeze.
- **No Min-K%++ leakage report yet.** The report script lives in
  `tools/min_k_check.py` (planned); v0.1 release will include the report.

## Contamination guard

- **Denylist:** `denylist_v0.jsonl` (15698 ids across
  paper_id, internal_paper_id, and s2_id namespaces from the IdeaEvolving
  paper_db) — ships in `denylist/` next to the corpus.
- **Temporal cut:** all source papers are pre-2017.
- **Min-K%++ leakage report:** see `min_k_report.json` once generated
  (target Min-20%++ < 0.05).
- **Verified at build time:** 0 records in this release had a paper_id in
  the denylist (dropped at filter time, count logged to build stats).

## Licence

- Code: MIT.
- Data (annotations): CC-BY-4.0.
- Source paper texts: NOT redistributed — paper IDs only.

## Citation

```bibtex
@inproceedings{genetrace2026,
  title  = {GeneTrace and evo-OPD: A Training-Grade Genome-Centric Corpus
            and Lineage-Aware Distillation for Scientific Idea Models},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
