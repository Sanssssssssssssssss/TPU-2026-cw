"""Aggregate per-run artifacts for a checkpoint continuation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNS = [
    ("C1_R1_no_approx_from256", "no_approx", "R1_no_approx", 256),
    ("C2_R3_numeric_primary_no_len_from768", "numeric_primary_no_len", "R3_numeric_primary_no_len", 768),
    (
        "C3_R5_numeric_primary_answer_only_len1200_from512",
        "numeric_primary_answer_only_len1200",
        "R5_numeric_primary_answer_only_len1200",
        512,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reward continuation artifacts.")
    parser.add_argument("--input-dir", required=True, help="Continuation run root.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    print(f"Wrote {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def run_specs(input_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(input_dir / "artifacts" / "reward_continuation_manifest.json")
    specs = manifest.get("runs") if isinstance(manifest, dict) else None
    if specs:
        return list(specs)
    return [
        {
            "run_id": run_id,
            "reward_mode": mode,
            "source_run": source_run,
            "source_step": source_step,
        }
        for run_id, mode, source_run, source_step in DEFAULT_RUNS
    ]


def add_common(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["run_id"] = spec["run_id"]
    out["reward_mode"] = spec["reward_mode"]
    out["source_run"] = spec.get("source_run")
    out["source_step"] = spec.get("source_step")
    return out


def aggregate_scalars(input_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = input_dir / "runs" / spec["run_id"] / "artifacts" / "analysis" / "scalar_metrics.csv"
        for row in read_csv(path):
            rows.append(add_common(row, spec))
    rows.sort(key=lambda row: (row.get("run_id", ""), row.get("metric", ""), float(row.get("step") or 0)))
    return rows


def scalar_pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    meta: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            step = int(float(row.get("step") or 0))
            value = float(row.get("value") or 0)
        except ValueError:
            continue
        key = (row["run_id"], row["reward_mode"], row.get("metric", ""), step)
        grouped.setdefault(key, []).append(value)
        meta.setdefault(
            (row["run_id"], row["reward_mode"], step),
            {
                "run_id": row["run_id"],
                "reward_mode": row["reward_mode"],
                "source_run": row.get("source_run"),
                "source_step": row.get("source_step"),
                "step": step,
            },
        )

    out_by_step: dict[tuple[str, str, int], dict[str, Any]] = dict(meta)
    for (run_id, mode, metric, step), values in grouped.items():
        if not metric:
            continue
        out_by_step[(run_id, mode, step)][metric] = sum(values) / len(values)
    return sorted(out_by_step.values(), key=lambda row: (row["run_id"], row["step"]))


def aggregate_trace_summary(input_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = input_dir / "runs" / spec["run_id"] / "artifacts" / "analysis" / "trace_summary.csv"
        for row in read_csv(path):
            rows.append(add_common(row, spec))
    rows.sort(key=lambda row: (row.get("run_id", ""), float(row.get("call_index") or 0)))
    return rows


def aggregate_checkpoint_eval(input_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = read_json(
            input_dir / "runs" / spec["run_id"] / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json"
        )
        if not summary:
            continue
        for row in summary.get("rows") or []:
            rows.append(add_common(row, spec))
    rows.sort(key=lambda row: (row.get("run_id", ""), float(row.get("step") or 0)))
    return rows


def selection_summary(ckpt_rows: list[dict[str, Any]], scalar_rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    wanted = {
        "train_kl",
        "eval_kl",
        "rollout_empty_response_rate",
        "rollout_extracted_none_rate",
        "rollout_overlong_rate_1600",
        "train_grpo_frac_reward_zero_std",
        "audit_reward_numeric_margin",
        "audit_reward_hacking_rate",
    }
    for row in scalar_rows:
        metric = row.get("metric")
        if metric not in wanted:
            continue
        try:
            step = float(row.get("step") or -1)
            value = float(row.get("value") or 0)
        except ValueError:
            continue
        key = (row["run_id"], metric)
        prev = latest_metrics.get(key)
        if prev is None or step >= prev["step"]:
            latest_metrics[key] = {"step": step, "value": value}

    rows = []
    for spec in specs:
        run_id = spec["run_id"]
        ckpts = [row for row in ckpt_rows if row.get("run_id") == run_id]
        best = None
        if ckpts:
            best = max(
                ckpts,
                key=lambda row: (
                    float(row.get("accuracy") or 0),
                    float(row.get("partial_accuracy") or 0),
                    float(row.get("format_accuracy") or 0),
                    int(float(row.get("step") or 0)),
                ),
            )
        reasons = []
        latest = {metric: latest_metrics.get((run_id, metric), {}).get("value") for metric in wanted}
        if (latest.get("rollout_empty_response_rate") or 0) > 0.15:
            reasons.append("empty_response_rate>0.15")
        if (latest.get("rollout_extracted_none_rate") or 0) > 0.35:
            reasons.append("extracted_none_rate>0.35")
        if (latest.get("train_grpo_frac_reward_zero_std") or 0) > 0.65:
            reasons.append("frac_reward_zero_std>0.65")
        if (latest.get("train_kl") or 0) > 0.8:
            reasons.append("KL>0.8")
        rows.append(
            {
                "run_id": run_id,
                "reward_mode": spec["reward_mode"],
                "source_run": spec.get("source_run"),
                "source_step": spec.get("source_step"),
                "best_checkpoint_step": best.get("step") if best else "",
                "best_checkpoint_accuracy": best.get("accuracy") if best else "",
                "best_checkpoint_partial_accuracy": best.get("partial_accuracy") if best else "",
                "best_checkpoint_format_accuracy": best.get("format_accuracy") if best else "",
                "screening_status": "candidate" if not reasons else "guardrail_warn",
                "guardrail_reasons": ";".join(reasons),
                "latest_train_kl": latest.get("train_kl"),
                "latest_eval_kl": latest.get("eval_kl"),
                "latest_empty_response_rate": latest.get("rollout_empty_response_rate"),
                "latest_extracted_none_rate": latest.get("rollout_extracted_none_rate"),
                "latest_overlong_rate_1600": latest.get("rollout_overlong_rate_1600"),
                "latest_frac_reward_zero_std": latest.get("train_grpo_frac_reward_zero_std"),
                "latest_reward_numeric_margin": latest.get("audit_reward_numeric_margin"),
                "latest_reward_hacking_rate": latest.get("audit_reward_hacking_rate"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    specs = run_specs(input_dir)

    scalar_rows = aggregate_scalars(input_dir, specs)
    pivot_rows = scalar_pivot_rows(scalar_rows)
    trace_rows = aggregate_trace_summary(input_dir, specs)
    ckpt_rows = aggregate_checkpoint_eval(input_dir, specs)
    selection_rows = selection_summary(ckpt_rows, scalar_rows, specs)

    write_csv(table_dir / "scalar_long.csv", scalar_rows)
    write_csv(table_dir / "scalar_pivot.csv", pivot_rows)
    write_csv(table_dir / "trace_summary_long.csv", trace_rows)
    write_csv(table_dir / "checkpoint_eval_long.csv", ckpt_rows)
    write_csv(table_dir / "selection_summary.csv", selection_rows)
    write_json(table_dir / "selection_summary.json", selection_rows)
    write_json(
        output_dir / "manifest_analysis.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "runs": specs,
            "tables": [
                "tables/scalar_long.csv",
                "tables/scalar_pivot.csv",
                "tables/trace_summary_long.csv",
                "tables/checkpoint_eval_long.csv",
                "tables/selection_summary.csv",
            ],
        },
    )


if __name__ == "__main__":
    main()
