"""Real-data sanity test of evo_opd parser + verifier on all GENE-Exam task types.

For each task in gene_exam/Questions/{T1,T2,T3,T4}-*:
  - Load up to N instances
  - For each: synthesize a "perfect" model response by wrapping the gold_answer in
    JSON code fences
  - Run parser → expect schema_valid=True
  - Run verifier with gold_answer → expect v=1.0 (perfect score)

Output: per-task pass rate + missing-schema list + sample failures.

Reveals exactly which task_types we still need to register in evo_opd/schemas.py
before evo-OPD training can adjudicate them.

Usage:
  python tools/sanity_check_parser_verifier.py [--n 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from idea_train.evo_opd.parser import parse_rollout
from idea_train.evo_opd.schemas import TASK_SCHEMAS
from idea_train.evo_opd.verifier import compute_verifier

REPO = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")


def synth_perfect_response(gold_answer: dict) -> str:
    """Wrap gold_answer in fenced JSON — simulates a model that nails the task."""
    return "```json\n" + json.dumps(gold_answer, indent=2) + "\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="instances per task")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    task_dirs = sorted(glob.glob(str(REPO / "gene_exam/Questions/T*")))
    print(f"Found {len(task_dirs)} task directories\n")

    # results[(tier,task)] = {pass, schema_valid, v_eq_1, parse_fail, schema_unknown, v_actual}
    results: dict[str, dict] = defaultdict(lambda: {
        "n_tested": 0,
        "n_schema_registered": 0,
        "n_parse_ok": 0,
        "n_schema_valid": 0,
        "n_v_eq_1": 0,
        "n_v_ge_0p5": 0,
        "v_samples": [],
        "sample_failure": None,
    })

    for td in task_dirs:
        name = Path(td).name
        inst_file = Path(td) / "instances.json"
        if not inst_file.exists():
            continue
        try:
            with inst_file.open() as f:
                insts = json.load(f)
        except Exception as e:
            print(f"  [WARN] {name}: cannot load instances.json: {e}")
            continue
        if isinstance(insts, dict):
            insts = list(insts.values())
        sample = random.sample(insts, min(args.n, len(insts)))

        for inst in sample:
            task_type = inst.get("task_type", name)
            gold = inst.get("gold_answer")
            r = results[task_type]
            r["n_tested"] += 1
            if task_type in TASK_SCHEMAS:
                r["n_schema_registered"] += 1

            if not gold:
                continue

            response = synth_perfect_response(gold)
            pr = parse_rollout(response, task_type)
            if pr.parsed_json is not None:
                r["n_parse_ok"] += 1
            if pr.schema_valid:
                r["n_schema_valid"] += 1

            try:
                vs, _ = compute_verifier(response, task_type, gold_answer=gold)
                r["v_samples"].append(round(vs.v, 3))
                if vs.v >= 0.99:
                    r["n_v_eq_1"] += 1
                if vs.v >= 0.5:
                    r["n_v_ge_0p5"] += 1
                if vs.v < 0.99 and r["sample_failure"] is None:
                    r["sample_failure"] = {
                        "instance_id": inst.get("instance_id"),
                        "v": vs.v,
                        "schema_valid": vs.schema_valid,
                        "exact_match": vs.exact_match,
                        "notes": vs.notes[:2],
                        "gold_keys": list(gold.keys()),
                    }
            except Exception as e:
                if r["sample_failure"] is None:
                    r["sample_failure"] = {"instance_id": inst.get("instance_id"), "error": str(e)[:120]}

    # --- report ----------------------------------------------------------
    print(f"{'task_type':<32} {'n':>3} {'sch_reg':>7} {'parse_ok':>8} {'sch_val':>7} {'v=1':>4} {'v≥0.5':>5}  failure")
    print("-" * 110)
    tested = parse_ok = schema_valid = v_eq_1 = v_ge_0p5 = 0
    schema_unregistered_tasks: list[str] = []
    schema_unparseable: list[tuple[str, dict]] = []
    schema_v_failures: list[tuple[str, dict]] = []

    for t in sorted(results):
        r = results[t]
        n = r["n_tested"]
        if n == 0:
            continue
        tested += n
        parse_ok += r["n_parse_ok"]
        schema_valid += r["n_schema_valid"]
        v_eq_1 += r["n_v_eq_1"]
        v_ge_0p5 += r["n_v_ge_0p5"]
        if r["n_schema_registered"] == 0:
            schema_unregistered_tasks.append(t)
        if r["n_v_eq_1"] < n:
            schema_v_failures.append((t, r["sample_failure"]))
        failure = ""
        if r["sample_failure"]:
            sf = r["sample_failure"]
            if "error" in sf:
                failure = f"ERR {sf['error']}"
            else:
                failure = f"v={sf['v']} sch={sf['schema_valid']} ex={sf['exact_match']}"
        flag = " *" if t not in TASK_SCHEMAS else "  "
        print(f"{t:<32}{flag}{n:>3} {r['n_schema_registered']:>7} "
              f"{r['n_parse_ok']:>8} {r['n_schema_valid']:>7} "
              f"{r['n_v_eq_1']:>4} {r['n_v_ge_0p5']:>5}  {failure[:50]}")

    print()
    print("=" * 110)
    print(f"TOTAL  n={tested}  parse_ok={parse_ok}/{tested} ({100*parse_ok/max(tested,1):.0f}%)  "
          f"schema_valid={schema_valid}/{tested} ({100*schema_valid/max(tested,1):.0f}%)  "
          f"v=1.0={v_eq_1}/{tested} ({100*v_eq_1/max(tested,1):.0f}%)  "
          f"v≥0.5={v_ge_0p5}/{tested} ({100*v_ge_0p5/max(tested,1):.0f}%)")
    print()
    if schema_unregistered_tasks:
        print(f"*** UNREGISTERED schemas ({len(schema_unregistered_tasks)}): need TASK_SCHEMAS entries in evo_opd/schemas.py")
        for t in schema_unregistered_tasks:
            r = results[t]
            sf = r["sample_failure"] or {}
            gold_keys = sf.get("gold_keys", [])
            print(f"  - {t:<32}  gold_keys={gold_keys}")
    print()
    # save full report
    out = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/parser_sanity_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "n_tested": tested,
            "parse_ok": parse_ok,
            "schema_valid": schema_valid,
            "v_eq_1": v_eq_1,
            "v_ge_0p5": v_ge_0p5,
            "per_task": {t: dict(r) for t, r in results.items() if r["n_tested"]},
            "unregistered_schemas": schema_unregistered_tasks,
        }, f, indent=2)
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
