"""Re-score per_instance.jsonl with key-name normalization.

Maps the model's creative JSON keys to canonical schema keys, then re-runs
exact_match. Doesn't require re-running the model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.schemas import TASK_SCHEMAS


def normalize_key(k: str) -> str:
    """Normalize for fuzzy matching: lowercase, drop punctuation/spaces/underscores."""
    return re.sub(r"[\s_\-\.]+", "", k).lower()


# Per-task remapping of model's likely keys → canonical schema keys.
# Built from observation of v2 outputs.
KEY_ALIASES: dict[str, dict[str, str]] = {
    # T1
    "T1-01_contribution_type": {"multi_contrib_types": "multi_contrib_types"},
    "T1-02_genome_field_type": {"GenomeField1Type": "GenomeField1Type"},
    "T1-03_driver_vs_passenger": {
        "driver_gene": "driver_gene", "DriverGene": "driver_gene", "driverGene": "driver_gene",
        "passenger_gene": "passenger_gene", "PassengerGene": "passenger_gene", "passengerGene": "passenger_gene",
    },
    "T1-04_lineage_position": {
        "label": "label", "Label": "label",
        "contribution_type": "contribution_type", "Type": "contribution_type", "type": "contribution_type",
    },
    "T1-05_cross_lineage_bridge": {"label": "label", "Label": "label"},
    # T2
    "T2-01_ordering_5": {"correct_order": "correct_order", "Order": "correct_order", "order": "correct_order", "Ordering": "correct_order"},
    "T2-02_ordering_6": {"correct_order": "correct_order", "Order": "correct_order", "order": "correct_order"},
    "T2-03_ordering_7": {"correct_order": "correct_order", "Order": "correct_order", "order": "correct_order"},
    "T2-04_grouping_8": {
        "ordered_group_a": "ordered_group_a", "GroupA": "ordered_group_a", "group_a": "ordered_group_a", "groupA": "ordered_group_a",
        "ordered_group_b": "ordered_group_b", "GroupB": "ordered_group_b", "group_b": "ordered_group_b", "groupB": "ordered_group_b",
    },
    "T2-05_grouping_8_medium": {
        "ordered_group_a": "ordered_group_a", "GroupA": "ordered_group_a", "group_a": "ordered_group_a",
        "ordered_group_b": "ordered_group_b", "GroupB": "ordered_group_b", "group_b": "ordered_group_b",
    },
    "T2-06_grouping_9_triple": {"ordered_groups": "ordered_groups", "Groups": "ordered_groups", "groups": "ordered_groups",
                                  "Group1": "ordered_groups", "Group2": "ordered_groups", "Group3": "ordered_groups"},
    "T2-07_lim_delta_match": {"mapping": "mapping", "Mapping": "mapping", "match": "mapping"},
    "T2-08_lim_delta_mixed": {"mapping": "mapping", "Mapping": "mapping"},
    "T2-09_lim_delta_chain": {"mapping": "mapping", "Mapping": "mapping"},
    "T2-10_genome_field_assign_2p": {
        "assignments_with_types": "assignments_with_types", "Assignments": "assignments_with_types",
        "dynamics": "dynamics", "Dynamics": "dynamics",
    },
    "T2-11_genome_field_assign_3p_9a": {"dynamics": "dynamics", "Dynamics": "dynamics"},
    "T2-12_gene_alignment": {"assignments": "assignments", "Assignments": "assignments", "Alignment": "assignments"},
    # T3
    "T3-01_single_dynamics": {"driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics"},
    "T3-02_genome_field_change": {
        "driver": "driver", "Driver": "driver",
        "dynamics": "dynamics", "Dynamics": "dynamics",
        "source_genome_status": "source_genome_status", "SourceGenomeStatus": "source_genome_status",
    },
    "T3-03_driver_dynamics": {"driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics"},
    "T3-04_genome_field_change_shown": {
        "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
        "source_genome_status": "source_genome_status", "SourceGenomeStatus": "source_genome_status",
    },
    "T3-05_driver_summary": {
        "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
        "source_genome_status": "source_genome_status",
    },
    "T3-06_dynamics_mech": {"dynamics": "dynamics", "Dynamics": "dynamics"},
    "T3-07_blind_change": {
        "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
        "source_genome_status": "source_genome_status",
    },
    "T3-08_driver_unlabeled": {
        "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
        "source_genome_status": "source_genome_status",
    },
    "T3-09_relation_classify": {"dynamics": "dynamics", "Dynamics": "dynamics", "label": "label", "Label": "label"},
    "T3-10_genome_direction": {
        "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
        "label": "label", "Label": "label",
    },
    "T3-11_evo_tempo": {"label": "label", "Label": "label"},
    "T3-12_evo_pattern": {"label": "label", "Label": "label",
                           "correct_dynamics": "correct_dynamics", "Dynamics": "correct_dynamics",
                           "dynamics": "correct_dynamics"},
    "T3-13_hidden_gene_fate": {
        "G1_status": "G1_status", "G1Status": "G1_status",
        "G2_status": "G2_status", "G2Status": "G2_status",
        "dynamics": "dynamics", "Dynamics": "dynamics",
    },
    "T3-14_hybrid_provenance": {"gene_sources": "gene_sources", "GeneSources": "gene_sources"},
    "T3-15_gene_tracking": {"correct_dynamics": "correct_dynamics", "Dynamics": "correct_dynamics", "dynamics": "correct_dynamics"},
    "T3-16_dynamics_boundary": {"correct_dynamics": "correct_dynamics", "Dynamics": "correct_dynamics", "dynamics": "correct_dynamics"},
    "T3-17_multi_citation": {"relation": "relation", "Relation": "relation"},
    # T4
    "T4-01_consistency_check": {
        "label": "label", "Label": "label",
        "contribution_type": "contribution_type", "ContributionType": "contribution_type",
        "verify": "verify", "Verify": "verify", "verifications": "verify",
    },
    "T4-02_intruder_domain": {"intruder": "intruder", "Intruder": "intruder",
                                "lineage_members": "lineage_members", "LineageMembers": "lineage_members"},
    "T4-03_wrong_step": {"correct_dynamics": "correct_dynamics", "Dynamics": "correct_dynamics",
                          "correct_order": "correct_order", "Order": "correct_order",
                          "label": "label", "Label": "label"},
    "T4-04_next_hop": {"driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics",
                        "label": "label", "Label": "label"},
    "T4-05_parent_genome": {"driver": "driver", "Driver": "driver", "label": "label", "Label": "label"},
    "T4-06_missing_link": {"bridge_paper": "bridge_paper", "BridgePaper": "bridge_paper",
                            "driver": "driver", "Driver": "driver", "dynamics": "dynamics", "Dynamics": "dynamics"},
    "T4-07_gene_bridge": {"correct_dynamics": "correct_dynamics", "Dynamics": "correct_dynamics", "label": "label", "Label": "label"},
    "T4-08_citation_consistency": {"claim_label": "claim_label", "ClaimLabel": "claim_label",
                                     "source_genome_status": "source_genome_status"},
}


def remap_keys(obj, task_type: str):
    """Recursively remap keys according to KEY_ALIASES."""
    aliases = KEY_ALIASES.get(task_type, {})
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # Try exact alias, then case-insensitive normalized
            canon = aliases.get(k)
            if canon is None:
                # try normalized lookup
                norm_k = normalize_key(k)
                for alias_k, alias_v in aliases.items():
                    if normalize_key(alias_k) == norm_k:
                        canon = alias_v; break
            new[canon or k] = v
        return new
    return obj


def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    # last bare object
    m2 = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m2:
        try: return json.loads(m2[-1])
        except: pass
    return None


# ---------------------------------------------------------------------------
# v9-style plain-text parsers — convert "Key = value" / "Key: value" lines
# into a dict so the same is_correct() path can score them.
# ---------------------------------------------------------------------------

def _parse_kv_lines(text: str) -> dict:
    """Best-effort 'Key = value' / 'Key: value' line extractor."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip().rstrip(".,")
        if not line:
            continue
        m = re.match(r"\*?\*?([A-Za-z][\w ]*?)\*?\*?\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip().strip("[]")
        # try to coerce list-shaped values: e.g. "[2, 1, 4, 5, 3]" or "T, F, F, T"
        if "," in v:
            parts = [p.strip() for p in v.split(",")]
            # all ints? coerce
            if all(re.fullmatch(r"-?\d+", p) for p in parts):
                v = [int(p) for p in parts]
            else:
                v = parts
        out[k] = v
    return out


# normalized-key alias table for v9 plain-text outputs
PLAIN_TEXT_ALIASES = {
    # T3-01
    "Driver":          "driver",
    "Dynamics":        "dynamics",
    # T3-09
    "Label":           "label",
    # T1-03
    "DriverGene":      "driver_gene",
    "PassengerGene":   "passenger_gene",
    "Driver Gene":     "driver_gene",
    "Passenger Gene":  "passenger_gene",
    # T2-01..03
    "Order":           "correct_order",
    # T4-01
    "Type":            "contribution_type",
    "Verify":          "verify",
    # T1-01 (multi-G labels)
    # T2-07..09 (multi-L labels)
}


def extract_plain_text(text: str, task_type: str):
    """Parse v9-style outputs into a dict in the same shape as gold_answer."""
    kv = _parse_kv_lines(text)
    if not kv:
        return None
    # canonical key remap
    out = {}
    for k, v in kv.items():
        canon = PLAIN_TEXT_ALIASES.get(k, k.lower())
        out[canon] = v

    # task-specific nesting
    if task_type == "T1-01_contribution_type":
        # G1, G2, G3, G4 -> {"multi_contrib_types": {"G1": ..., ...}}
        gs = {k: v for k, v in out.items() if re.fullmatch(r"g\d+", k, re.I)}
        if gs:
            return {"multi_contrib_types": {k.upper(): v for k, v in gs.items()}}
    if task_type.startswith("T2-0") and ("lim_delta" in task_type):
        # L1, L2, L3 -> {"mapping": {"L1": "D#", ...}}
        ls = {k: v for k, v in out.items() if re.fullmatch(r"l\d+", k, re.I)}
        if ls:
            return {"mapping": {k.upper(): v for k, v in ls.items()}}
    if task_type == "T4-01_consistency_check":
        # ensure verify is a list of T/F
        if "verify" in out and isinstance(out["verify"], str):
            out["verify"] = [x.strip() for x in re.split(r"[,\s]+", out["verify"]) if x.strip()]
    return out or None


def extract_any(text: str, task_type: str):
    """Try JSON first, then v9-style plain text."""
    js = extract_json(text)
    if js is not None:
        return js
    return extract_plain_text(text, task_type)


def eq_ci(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def is_correct(pred, gold, task_type: str) -> bool:
    """Lenient exact-match with key remapping + value normalization."""
    if not isinstance(gold, dict):
        return False
    if pred is None:
        return False

    # Special case: T2-01/02/03 — model often emits a raw list instead of {correct_order: [...]}
    if task_type.startswith("T2-0") and "ordering" in task_type and isinstance(pred, list):
        pred = {"correct_order": pred}

    # T2-06 ordered_groups: model may emit as Group1, Group2, Group3 keys
    if task_type == "T2-06_grouping_9_triple" and isinstance(pred, dict):
        groups = [v for k, v in sorted(pred.items()) if "group" in k.lower()]
        if groups and "ordered_groups" not in pred:
            pred = {"ordered_groups": groups}

    if not isinstance(pred, dict):
        return False
    pred = remap_keys(pred, task_type)

    for k, gv in gold.items():
        pv = pred.get(k)
        if isinstance(gv, list) and isinstance(pv, list):
            if len(gv) != len(pv) or not all(eq_ci(x, y) for x, y in zip(gv, pv)):
                return False
        elif isinstance(gv, dict) and isinstance(pv, dict):
            # nested compare (case-insensitive on values)
            if set(gv.keys()) != set(pv.keys()):
                return False
            if not all(eq_ci(gv[k2], pv.get(k2)) for k2 in gv):
                return False
        else:
            if not eq_ci(gv, pv):
                return False
    return True


def rescore(eval_dir: Path):
    """Re-score all per_instance_shard*.jsonl in a result directory + sub-shards."""
    per_task_correct = Counter()
    per_task_total = Counter()
    files = list(eval_dir.glob("shard*/per_instance_shard*.jsonl")) or \
             list(eval_dir.glob("per_instance*.jsonl"))
    print(f"Re-scoring {len(files)} per_instance files in {eval_dir}")

    # also need to load gold answers from gene_exam Questions/
    gold_lookup = {}
    import glob
    for td in glob.glob("/home/azureuser/workspace-gzy/zyf/IdeaEvolving/gene_exam/Questions/*"):
        with open(f"{td}/instances.json") as f:
            data = json.load(f)
        if isinstance(data, dict): data = list(data.values())
        for inst in data:
            iid = inst.get("instance_id")
            if iid:
                gold_lookup[iid] = inst.get("gold_answer")

    for fp in files:
        with fp.open() as f:
            for line in f:
                r = json.loads(line)
                tt = r["task_type"]
                gold = gold_lookup.get(r["instance_id"])
                pred = extract_any(r["completion"], tt)
                ok = is_correct(pred, gold, tt)
                per_task_total[tt] += 1
                if ok:
                    per_task_correct[tt] += 1

    # report
    n_total = sum(per_task_total.values())
    n_correct = sum(per_task_correct.values())
    print(f"  n: {n_total}  correct: {n_correct}  macro: {100*n_correct/max(n_total,1):.2f}%")

    tiers = defaultdict(list)
    for t in sorted(per_task_total):
        acc = per_task_correct[t] / per_task_total[t]
        if t.startswith("T1-"): tiers["T1"].append(acc)
        elif t.startswith("T2-"): tiers["T2"].append(acc)
        elif t.startswith("T3-"): tiers["T3"].append(acc)
        elif t.startswith("T4-"): tiers["T4"].append(acc)
    for k in ("T1", "T2", "T3", "T4"):
        if tiers[k]:
            print(f"  {k}: {100*sum(tiers[k])/len(tiers[k]):.2f}%")
    # write summary
    out = eval_dir / "lenient_summary.json"
    out.write_text(json.dumps({
        "n_instances": n_total, "n_correct": n_correct,
        "macro_accuracy": n_correct / max(n_total, 1),
        "per_tier_macro": {k: sum(v)/len(v) for k, v in tiers.items() if v},
        "per_task_accuracy": {t: per_task_correct[t]/per_task_total[t] for t in sorted(per_task_total)},
        "per_task_n": dict(per_task_total),
    }, indent=2))
    print(f"  Wrote {out}")
    return per_task_correct, per_task_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for d in args.dirs:
        rescore(Path(d))
        print()
