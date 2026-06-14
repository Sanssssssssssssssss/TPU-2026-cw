"""Package rollout320 official lines plus R4 alternatives.

This keeps the original three-line package immutable while adding the official
R2 line and side-by-side R4 alternatives for final selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_rollout320_three_line_package as base


RUNS = [
    base.OfficialRun(
        key="R0",
        run_id="baseline-rollout320-full-001",
        branch="R0_baseline_rollout320",
        legend="R0 baseline dense32-rollout",
        reward_mode="baseline",
        num_generations=2,
        max_steps=3364,
        checkpoint_steps=base.EXPECTED_K2_STEPS,
        learning_rate="3e-6",
        beta="0.08",
        rank="64",
        alpha="64",
    ),
    base.OfficialRun(
        key="R1",
        run_id="r1-format-rollout320-full-001",
        branch="R1_format_reward_rollout320",
        legend="R1 format-aware reward-only dense32-rollout",
        reward_mode="gsm8k_verifiable_format",
        num_generations=2,
        max_steps=3364,
        checkpoint_steps=base.EXPECTED_K2_STEPS,
        learning_rate="3e-6",
        beta="0.08",
        rank="64",
        alpha="64",
    ),
    base.OfficialRun(
        key="R2",
        run_id="r2-k8-beta004-rollout320-full-001",
        branch="R2_k8_beta004_rollout320",
        legend="R2 baseline K=8 beta=0.04 dense32-rollout",
        reward_mode="baseline",
        num_generations=8,
        max_steps=841,
        checkpoint_steps=base.EXPECTED_K8_STEPS,
        learning_rate="3e-6",
        beta="0.04",
        rank="64",
        alpha="64",
    ),
    base.OfficialRun(
        key="R3",
        run_id="r3-loo-advantage-rollout320-full-001",
        branch="R3_loo_advantage_rollout320",
        legend="R3 baseline reward + RLOO advantage dense32-rollout",
        reward_mode="baseline",
        num_generations=2,
        max_steps=3364,
        checkpoint_steps=base.EXPECTED_K2_STEPS,
        learning_rate="3e-6",
        beta="0.08",
        rank="64",
        alpha="64",
        advantage_estimator="rloo",
    ),
    base.OfficialRun(
        key="R4_lr1e6",
        run_id="r4-r12-full-rollout320-lr1e6-001",
        branch="R4_r12_full_lr1e-6_rollout320",
        legend="R4 full-from-zero lr1e-6 dense32-rollout",
        reward_mode="gsm8k_verifiable_simple",
        num_generations=8,
        max_steps=841,
        checkpoint_steps=base.EXPECTED_K8_STEPS,
        learning_rate="1e-6",
        beta="0.04",
        rank="64",
        alpha="64",
    ),
    base.OfficialRun(
        key="R4_format_lr3e6",
        run_id="r4-r12-format-rollout320-lr3e6-001",
        branch="R4_r12_format_lr3e-6_rollout320",
        legend="R4 format-aware full-from-zero lr3e-6 dense32-rollout",
        reward_mode="gsm8k_verifiable_format",
        num_generations=8,
        max_steps=841,
        checkpoint_steps=base.EXPECTED_K8_STEPS,
        learning_rate="3e-6",
        beta="0.04",
        rank="64",
        alpha="64",
    ),
]

SERIES_COLORS = {
    "R0": "#6F768A",
    "R1": "#CC6F47",
    "R2": "#4B9A62",
    "R3": "#BD569B",
    "R4_lr1e6": "#5477C4",
    "R4_format_lr3e6": "#8B6BBE",
}

OFFICIAL_SCALAR_METRICS = {
    "train_reward_score",
    "eval_reward_score",
    "train_kl",
    "eval_kl",
    "train_loss",
    "eval_loss",
    "train_official_numeric_exact_rate",
    "train_format_accuracy",
    "train_rollout_extracted_none_rate",
    "train_rollout_empty_response_rate",
    "train_grpo_frac_reward_zero_std",
    "train_grpo_group_reward_std_mean",
}

REWARD_NATIVE_MAX = {
    "baseline": 10.0,
    "gsm8k_verifiable_simple": 1.2,
    "gsm8k_verifiable_format": 1.8,
}

REWARD_MECHANISMS = {
    "baseline": {
        "native_max": 10.0,
        "components": {
            "match_format_exactly": 3.0,
            "match_format_approximately": 2.5,
            "check_answer": 3.0,
            "check_numbers": 1.5,
        },
    },
    "gsm8k_verifiable_simple": {
        "native_max": 1.2,
        "components": {
            "gsm8k_simple_numeric": 1.0,
            "gsm8k_simple_format": 0.2,
        },
    },
    "gsm8k_verifiable_format": {
        "native_max": 1.8,
        "components": {
            "gsm8k_simple_numeric": 1.0,
            "gsm8k_simple_format": 0.2,
            "reasoning_structure_format": 0.6,
        },
    },
}


def normalize_scalar_reward_rows(rows: list[dict[str, Any]], run: base.OfficialRun) -> list[dict[str, Any]]:
    """Rewrite report-facing reward scalar values to the shared baseline 0-10 scale."""
    native_max = REWARD_NATIVE_MAX[run.reward_mode]
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if updated.get("metric") in {"train_reward_score", "eval_reward_score"}:
            native_value = base.metric_float(updated.get("value"))
            updated["reward_mode"] = run.reward_mode
            updated["reward_native_max"] = native_max
            updated["reward_score_scale"] = "baseline_0_10"
            if native_value is not None:
                updated["value_native"] = native_value
                updated["value"] = native_value / native_max * 10.0
                updated["value_native_pct"] = native_value / native_max * 100.0
        out.append(updated)
    return out


def compact_official_scalar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("metric") in OFFICIAL_SCALAR_METRICS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-root", type=Path, default=Path("artifacts/cloud"))
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/grpo-rollout320-official-comparison-001"),
    )
    parser.add_argument("--allow-missing", action="store_true", help="Build from available fetched runs.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create the final .zip package.")
    return parser.parse_args()


def copy_raw_refs(output_dir: Path, cloud_root: Path, reports_root: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for run in RUNS:
        run_dir = cloud_root / run.run_id
        branch_dir = run_dir / "runs" / run.branch
        report_dir = reports_root / f"{run.run_id}-clean"
        for src, dst in [
            (run_dir / "pipeline.log", output_dir / "raw_refs" / run.key / "pipeline.log"),
            (run_dir / "checkpoint_archives.txt", output_dir / "raw_refs" / run.key / "checkpoint_archives.txt"),
            (run_dir / "artifacts" / "reward_k8_pilot_manifest.json", output_dir / "raw_refs" / run.key / "reward_k8_pilot_manifest.json"),
            (run_dir / "artifacts" / "reward_k8_pilot_manifest.json.pre_repair", output_dir / "raw_refs" / run.key / "reward_k8_pilot_manifest.json.pre_repair"),
            (branch_dir / "run_env.txt", output_dir / "raw_refs" / run.key / "run_env.txt"),
            (branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.csv", output_dir / "raw_refs" / run.key / "checkpoint_eval_summary.csv"),
            (branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json", output_dir / "raw_refs" / run.key / "checkpoint_eval_summary.json"),
            (report_dir / "manifest_clean_plots.json", output_dir / "raw_refs" / run.key / "manifest_clean_plots.json"),
        ]:
            item = base.copy_ref(src, dst)
            if item:
                copied.append(item)
    return copied


def build_official_comparison_package(
    output_dir: Path,
    statuses: list[dict[str, Any]],
    ckpt_rows: list[dict[str, Any]],
    scalar_rows: list[dict[str, Any]],
    cloud_root: Path,
    reports_root: Path,
    make_zip: bool,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    base.write_csv(output_dir / "tables" / "checkpoint_eval_rollout_aligned.csv", ckpt_rows)
    base.write_csv(output_dir / "tables" / "scalar_long_rollout_aligned.csv", scalar_rows)
    base.write_csv(output_dir / "tables" / "verification_summary.csv", statuses)

    base.plot_checkpoint_metric(
        ckpt_rows,
        "accuracy",
        "Checkpoint exact accuracy by aligned rollouts",
        "R0/R1/R2/R3 official lines plus R4 alternatives; x-axis is generated rollouts.",
        output_dir / "figures" / "01_checkpoint_exact_accuracy_by_rollouts.png",
    )
    base.plot_checkpoint_metric(
        ckpt_rows,
        "partial_accuracy",
        "Checkpoint partial accuracy by aligned rollouts",
        "All available lines use the same 22 rollout-aligned eval points.",
        output_dir / "figures" / "02_checkpoint_partial_accuracy_by_rollouts.png",
    )
    base.plot_checkpoint_metric(
        ckpt_rows,
        "format_accuracy",
        "Checkpoint format accuracy by aligned rollouts",
        "Format accuracy from checkpoint eval CSV at matched rollout budgets.",
        output_dir / "figures" / "03_checkpoint_format_accuracy_by_rollouts.png",
    )
    base.plot_scalar_metric(
        scalar_rows,
        "train_reward_score",
        "Train reward by aligned rollouts",
        "Report-facing reward values are rewritten to the shared baseline 0-10 scale; native values are retained as value_native.",
        output_dir / "figures" / "04_train_reward_by_rollouts.png",
    )
    base.plot_scalar_metric(
        scalar_rows,
        "train_kl",
        "Train KL by aligned rollouts",
        "Policy KL scalar transformed with rollouts_seen = step x NUM_GENERATIONS.",
        output_dir / "figures" / "05_train_kl_by_rollouts.png",
    )
    base.plot_scalar_metric(
        scalar_rows,
        "train_loss",
        "Train loss by aligned rollouts",
        "Actor loss scalar transformed with rollouts_seen = step x NUM_GENERATIONS.",
        output_dir / "figures" / "06_train_loss_by_rollouts.png",
    )
    base.plot_scalar_metric(
        scalar_rows,
        "train_rollout_extracted_none_rate",
        "Extracted-none rate by aligned rollouts",
        "Parser health scalar; lower is better.",
        output_dir / "figures" / "07_extracted_none_by_rollouts.png",
        ylim=(0, 1),
    )
    base.plot_scalar_metric(
        scalar_rows,
        "train_rollout_empty_response_rate",
        "Empty response rate by aligned rollouts",
        "Response health scalar; lower is better.",
        output_dir / "figures" / "08_empty_response_by_rollouts.png",
        ylim=(0, 1),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Rollout-aligned official comparison package for R0/R1/R2/R3 and R4 alternatives.",
        "rollout_axis": "rollouts_seen = step * num_generations",
        "expected_rollouts": base.EXPECTED_ROLLOUTS,
        "r1_status": "new R1 uses the current format-aware reward; old simple R1 is superseded and kept only as historical evidence.",
        "r3_status": "R3 changes only the Tunix GRPO advantage estimator from grpo to rloo.",
        "r4_selection_status": "R4_lr1e6 is the accuracy reference; format-aware R4 alternatives repair strict format reward before final R4 selection.",
        "reward_score_policy": {
            "report_primary_reward_values_are_rewritten": True,
            "target_scale": "baseline_0_10",
            "formula": "value = value_native / reward_native_max * 10.0 for train_reward_score and eval_reward_score rows",
            "native_values_retained_as": "value_native",
            "native_mechanisms": REWARD_MECHANISMS,
        },
        "scalar_table_policy": {
            "tables/scalar_long_rollout_aligned.csv": "Compact long table with core metrics used by the official comparison figures.",
            "full_width_scalar_tables": "See artifacts/reports/grpo-rollout320-report-figures-001/data/<line>/tensorboard_derived/scalar_pivot.csv.",
            "included_metrics": sorted(OFFICIAL_SCALAR_METRICS),
        },
        "lines": [
            {
                "key": run.key,
                "slot": "R4" if run.key.startswith("R4_") else run.key,
                "selection_status": (
                    "current_r4_reference"
                    if run.key == "R4_lr1e6"
                    else "r4_candidate"
                    if run.key.startswith("R4_format_")
                    else "official_fixed"
                ),
                "run_id": run.run_id,
                "branch": run.branch,
                "legend": run.legend,
                "reward_mode": run.reward_mode,
                "num_generations": run.num_generations,
                "max_steps": run.max_steps,
                "checkpoint_steps": run.checkpoint_steps,
                "checkpoint_rollouts": base.EXPECTED_ROLLOUTS,
                "learning_rate": run.learning_rate,
                "beta": run.beta,
                "rank": run.rank,
                "alpha": run.alpha,
                "advantage_estimator": run.advantage_estimator,
                "source_checkpoint": None,
            }
            for run in RUNS
        ],
        "verification": statuses,
        "copied_files": copy_raw_refs(output_dir, cloud_root, reports_root),
    }
    (output_dir / "manifest_rollout320_official_comparison.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# GRPO Rollout320 Official Comparison\n\n"
        "This package compares fixed official R0/R1/R2/R3 lines and the R4 slot. "
        "R1 now uses the current format-aware reward, while the earlier simple-reward R1 is superseded. "
        "R3 changes only the advantage estimator from Tunix GRPO to RLOO. "
        "R4 currently has the lr1e-6 reference plus format-aware full-from-zero alternatives until final selection. "
        "The primary reward scalar values in `tables/scalar_long_rollout_aligned.csv` are already on the shared baseline 0-10 scale; "
        "native TensorBoard reward values are retained as `value_native`. "
        "The official scalar long table is intentionally compact; full per-run scalar pivots live in the report-figures package. "
        "The original three-line evidence package is not overwritten. "
        "All comparison figures use `rollouts_seen`, not raw step.\n",
        encoding="utf-8",
    )
    if make_zip:
        shutil.make_archive(str(output_dir.with_suffix("")), "zip", output_dir)


def main() -> int:
    args = parse_args()
    base.RUNS = RUNS
    base.SERIES_COLORS = SERIES_COLORS

    statuses: list[dict[str, Any]] = []
    all_ckpt_rows: list[dict[str, Any]] = []
    all_scalar_rows: list[dict[str, Any]] = []

    for run in RUNS:
        run_dir = args.cloud_root / run.run_id
        if not run_dir.exists():
            statuses.append(
                {
                    "line": run.key,
                    "run_id": run.run_id,
                    "branch": run.branch,
                    "raw_dir": str(run_dir),
                    "passed": False,
                    "errors": [f"missing fetched run directory: {run_dir}"],
                    "warnings": [],
                }
            )
            continue
        status, ckpt_rows, scalar_rows = base.verify_one(run_dir, run)
        scalar_rows = normalize_scalar_reward_rows(scalar_rows, run)
        statuses.append(status)
        all_ckpt_rows.extend(ckpt_rows)
        all_scalar_rows.extend(compact_official_scalar_rows(scalar_rows))
        if status["passed"]:
            base.build_clean_report(run, args.reports_root / f"{run.run_id}-clean", status, ckpt_rows, scalar_rows)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    verification_path = args.output_dir.parent / "grpo-rollout320-official-comparison-verification.json"
    verification_path.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "runs": statuses}, indent=2), encoding="utf-8")

    failed = [status for status in statuses if not status.get("passed")]
    if failed and not args.allow_missing:
        print(f"Wrote verification report to {verification_path}")
        for status in failed:
            print(f"{status['run_id']}: FAILED")
            for error in status.get("errors", []):
                print(f"  - {error}")
        return 1

    if failed:
        hard_failed = [
            status
            for status in failed
            if status.get("errors") != [f"missing fetched run directory: {args.cloud_root / status['run_id']}"]
        ]
        if hard_failed:
            print(f"Wrote verification report to {verification_path}")
            for status in hard_failed:
                print(f"{status['run_id']}: FAILED")
                for error in status.get("errors", []):
                    print(f"  - {error}")
            return 1

    build_official_comparison_package(
        args.output_dir,
        statuses,
        all_ckpt_rows,
        all_scalar_rows,
        args.cloud_root,
        args.reports_root,
        make_zip=not args.no_zip,
    )
    print(f"Wrote rollout320 official comparison package to {args.output_dir}")
    if not args.no_zip:
        print(f"Wrote zip to {args.output_dir}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
