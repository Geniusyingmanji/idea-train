# GENE-Exam comparison: baseline vs trained

- Baseline: `/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/qwen3-8b_baseline_nothink`
- Trained : `/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/qwen3-8b-sft-v1_nothink`

## Headline

| metric          | baseline | trained | delta |
|-----------------|---------:|--------:|------:|
| n_instances     |    1029 |    1029 | +0 |
| macro_accuracy  |   0.68% |   1.07% | +0.39 pts |

## Per tier

| tier | baseline | trained | delta |
|------|---------:|--------:|------:|
| T1   |   4.00% |   8.80% | +4.80 pts |
| T2   |   0.00% |   0.00% | +0.00 pts |
| T3   |   0.47% |   0.00% | -0.47 pts |
| T4   |   0.00% |   0.00% | +0.00 pts |

## Per task

| task | baseline | trained | delta |
|------|---------:|--------:|------:|
| T1-01_contribution_type          |   0.00% |   0.00% |  +0.00   |
| T1-02_genome_field_type          |  20.00% |  44.00% | +24.00 ↑ |
| T1-03_driver_vs_passenger        |   0.00% |   0.00% |  +0.00   |
| T1-04_lineage_position           |   0.00% |   0.00% |  +0.00   |
| T1-05_cross_lineage_bridge       |   0.00% |   0.00% |  +0.00   |
| T2-01_ordering_5                 |   0.00% |   0.00% |  +0.00   |
| T2-02_ordering_6                 |   0.00% |   0.00% |  +0.00   |
| T2-03_ordering_7                 |   0.00% |   0.00% |  +0.00   |
| T2-04_grouping_8                 |   0.00% |   0.00% |  +0.00   |
| T2-05_grouping_8_medium          |   0.00% |   0.00% |  +0.00   |
| T2-06_grouping_9_triple          |   0.00% |   0.00% |  +0.00   |
| T2-07_lim_delta_match            |   0.00% |   0.00% |  +0.00   |
| T2-08_lim_delta_mixed            |   0.00% |   0.00% |  +0.00   |
| T2-09_lim_delta_chain            |   0.00% |   0.00% |  +0.00   |
| T2-10_genome_field_assign_2p     |   0.00% |   0.00% |  +0.00   |
| T2-11_genome_field_assign_3p_9a  |   0.00% |   0.00% |  +0.00   |
| T2-12_gene_alignment             |   0.00% |   0.00% |  +0.00   |
| T3-01_single_dynamics            |   0.00% |   0.00% |  +0.00   |
| T3-02_genome_field_change        |   0.00% |   0.00% |  +0.00   |
| T3-03_driver_dynamics            |   0.00% |   0.00% |  +0.00   |
| T3-04_genome_field_change_shown  |   0.00% |   0.00% |  +0.00   |
| T3-05_driver_summary             |   0.00% |   0.00% |  +0.00   |
| T3-06_dynamics_mech              |   8.00% |   0.00% |  -8.00 ↓ |
| T3-07_blind_change               |   0.00% |   0.00% |  +0.00   |
| T3-08_driver_unlabeled           |   0.00% |   0.00% |  +0.00   |
| T3-09_relation_classify          |   0.00% |   0.00% |  +0.00   |
| T3-10_genome_direction           |   0.00% |   0.00% |  +0.00   |
| T3-11_evo_tempo                  |   0.00% |   0.00% |  +0.00   |
| T3-12_evo_pattern                |   0.00% |   0.00% |  +0.00   |
| T3-13_hidden_gene_fate           |   0.00% |   0.00% |  +0.00   |
| T3-14_hybrid_provenance          |   0.00% |   0.00% |  +0.00   |
| T3-15_gene_tracking              |   0.00% |   0.00% |  +0.00   |
| T3-16_dynamics_boundary          |   0.00% |   0.00% |  +0.00   |
| T3-17_multi_citation             |   0.00% |   0.00% |  +0.00   |
| T4-01_consistency_check          |   0.00% |   0.00% |  +0.00   |
| T4-02_intruder_domain            |   0.00% |   0.00% |  +0.00   |
| T4-03_wrong_step                 |   0.00% |   0.00% |  +0.00   |
| T4-04_next_hop                   |   0.00% |   0.00% |  +0.00   |
| T4-05_parent_genome              |   0.00% |   0.00% |  +0.00   |
| T4-06_missing_link               |   0.00% |   0.00% |  +0.00   |
| T4-07_gene_bridge                |   0.00% |   0.00% |  +0.00   |
| T4-08_citation_consistency       |   0.00% |   0.00% |  +0.00   |