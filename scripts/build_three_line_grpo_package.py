"""Create a local evidence index for baseline, R12 full, and R12 reward-only."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_files(src: Path, dst: Path, patterns: list[str]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    if not src.exists():
        return copied
    ensure(dst)
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            if path.is_file():
                target = dst / path.name
                shutil.copy2(path, target)
                copied.append({"source": str(path), "output": str(target)})
    return copied


def copy_one(src: Path, dst: Path) -> dict[str, str] | None:
    if not src.exists() or not src.is_file():
        return None
    ensure(dst.parent)
    shutil.copy2(src, dst)
    return {"source": str(src), "output": str(dst)}


def file_count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/grpo-three-line-evidence-001"))
    parser.add_argument("--baseline-raw", type=Path, default=Path("artifacts/cloud/course-baseline-001"))
    parser.add_argument("--baseline-report", type=Path, default=Path("artifacts/reports/course-baseline-001"))
    parser.add_argument("--r12-full-raw", type=Path, default=Path("artifacts/cloud/reward-k8-beta004-r12-full-001"))
    parser.add_argument("--r12-full-report", type=Path, default=Path("artifacts/reports/reward-k8-beta004-r12-full-001-clean"))
    parser.add_argument("--reward-only-raw", type=Path, default=Path("artifacts/cloud/reward-only-r12-full-complete-001"))
    parser.add_argument("--reward-only-report", type=Path, default=Path("artifacts/reports/reward-only-r12-full-complete-001-clean"))
    parser.add_argument("--reward-only-source-raw", type=Path, default=Path("artifacts/cloud/reward-only-r12-full-001"))
    parser.add_argument(
        "--reward-only-status",
        default="complete 3364-step reward-only ablation; seeded from reward-only-r12-full-001 checkpoint 500",
    )
    parser.add_argument("--r12-final-report", type=Path, default=Path("artifacts/reports/r12-final-evidence-001"))
    args = parser.parse_args()

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    ensure(out)

    copied: list[dict[str, str]] = []
    copied += copy_tree_files(args.baseline_report / "figures", out / "figures" / "baseline", ["*.png"])
    copied += copy_tree_files(args.r12_full_report / "figures" / "combined", out / "figures" / "r12_full", ["*.png"])
    copied += copy_tree_files(args.reward_only_report / "figures" / "combined", out / "figures" / "r12_reward_only", ["*.png"])

    copied += copy_tree_files(
        args.baseline_report / "tables",
        out / "tables" / "baseline",
        [
            "checkpoint_eval_summary.csv",
            "eval_summary.csv",
            "metric_snapshot.csv",
            "run_summary.csv",
            "trace_phase_summary.csv",
            "baseline_config.csv",
        ],
    )
    copied += copy_tree_files(
        args.r12_full_report / "tables",
        out / "tables" / "r12_full",
        [
            "checkpoint_eval_long.csv",
            "clean_selection_summary.csv",
            "selection_summary.csv",
            "trace_audit_by_call.csv",
            "reward_composition_latest.csv",
            "reward_composition_by_call.csv",
        ],
    )
    copied += copy_tree_files(
        args.reward_only_report / "tables",
        out / "tables" / "r12_reward_only",
        [
            "alert_counts.csv",
            "checkpoint_archives_summary.csv",
            "checkpoint_eval_long.csv",
            "clean_selection_summary.csv",
            "reward_composition_latest.csv",
            "reward_composition_by_call.csv",
            "selection_summary.csv",
            "trace_audit_by_call.csv",
        ],
    )
    copied += copy_tree_files(
        args.r12_final_report / "tables",
        out / "tables" / "r12_large_eval",
        ["r12_large_eval_summary.csv", "r12_full_checkpoint_eval.csv", "reward_only_alert_counts.csv"],
    )

    raw_refs = out / "raw_refs"
    refs = [
        (args.baseline_raw / "pipeline.log", raw_refs / "baseline" / "pipeline.log"),
        (args.baseline_raw / "artifacts" / "run_manifest.json", raw_refs / "baseline" / "run_manifest.json"),
        (args.baseline_raw / "artifacts" / "base_eval.json", raw_refs / "baseline" / "base_eval.json"),
        (args.baseline_raw / "artifacts" / "baseline_lora_eval.json", raw_refs / "baseline" / "baseline_lora_eval.json"),
        (args.r12_full_raw / "pipeline.log", raw_refs / "r12_full" / "pipeline.log"),
        (args.r12_full_raw / "artifacts" / "reward_k8_pilot_manifest.json", raw_refs / "r12_full" / "reward_k8_pilot_manifest.json"),
        (args.r12_full_raw / "checkpoint_archives.txt", raw_refs / "r12_full" / "checkpoint_archives.txt"),
        (args.r12_full_raw / "runs" / "R12_gsm8k_verifiable_simple" / "run_env.txt", raw_refs / "r12_full" / "run_env.txt"),
        (args.reward_only_raw / "pipeline.log", raw_refs / "r12_reward_only" / "pipeline.log"),
        (args.reward_only_raw / "artifacts" / "reward_k8_pilot_manifest.json", raw_refs / "r12_reward_only" / "reward_k8_pilot_manifest.json"),
        (args.reward_only_raw / "checkpoint_archives.txt", raw_refs / "r12_reward_only" / "checkpoint_archives.txt"),
        (args.reward_only_raw / "runs" / "R12_reward_only_baseline_kkl" / "run_env.txt", raw_refs / "r12_reward_only" / "run_env.txt"),
        (args.reward_only_raw / "runs" / "R12_reward_only_baseline_kkl" / "artifacts" / "run_manifest.json", raw_refs / "r12_reward_only" / "run_manifest.json"),
        (
            args.reward_only_raw
            / "runs"
            / "R12_reward_only_baseline_kkl"
            / "artifacts"
            / "checkpoint_eval"
            / "checkpoint_eval_summary.csv",
            raw_refs / "r12_reward_only" / "checkpoint_eval_summary.csv",
        ),
        (
            args.reward_only_raw
            / "runs"
            / "R12_reward_only_baseline_kkl"
            / "artifacts"
            / "checkpoint_eval"
            / "checkpoint_eval_summary.json",
            raw_refs / "r12_reward_only" / "checkpoint_eval_summary.json",
        ),
        (args.reward_only_source_raw / "pipeline.log", raw_refs / "r12_reward_only_source" / "pipeline.log"),
        (args.reward_only_source_raw / "checkpoint_archives.txt", raw_refs / "r12_reward_only_source" / "checkpoint_archives.txt"),
        (args.reward_only_source_raw / "runs" / "R12_reward_only_baseline_kkl" / "run_env.txt", raw_refs / "r12_reward_only_source" / "run_env.txt"),
        (
            args.reward_only_source_raw
            / "runs"
            / "R12_reward_only_baseline_kkl"
            / "branch_metadata.json",
            raw_refs / "r12_reward_only_source" / "branch_metadata.json",
        ),
    ]
    for src, dst in refs:
        item = copy_one(src, dst)
        if item:
            copied.append(item)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local evidence index for the three main GRPO lines: baseline, R12 full, R12 reward-only.",
        "lines": {
            "baseline": {
                "raw_dir": str(args.baseline_raw),
                "report_dir": str(args.baseline_report),
                "figures": file_count(out / "figures" / "baseline", "*.png"),
            },
            "r12_full": {
                "raw_dir": str(args.r12_full_raw),
                "report_dir": str(args.r12_full_report),
                "checkpoint_archive_dir": str(args.r12_full_raw / "checkpoint_archives"),
                "checkpoint_archive_count": file_count(args.r12_full_raw / "checkpoint_archives", "*.tar.gz"),
                "figures": file_count(out / "figures" / "r12_full", "*.png"),
            },
            "r12_reward_only": {
                "raw_dir": str(args.reward_only_raw),
                "source_raw_dir": str(args.reward_only_source_raw),
                "source_checkpoint_step": 500,
                "report_dir": str(args.reward_only_report),
                "checkpoint_archive_dir": str(args.reward_only_raw / "checkpoint_archives"),
                "checkpoint_archive_count": file_count(args.reward_only_raw / "checkpoint_archives", "*.tar.gz"),
                "figures": file_count(out / "figures" / "r12_reward_only", "*.png"),
                "status": args.reward_only_status,
            },
            "r12_large_eval": {
                "report_dir": str(args.r12_final_report),
                "selection_basis": "256-prompt large eval; R12 step 512 is recommended",
            },
        },
        "copied_files": copied,
    }
    (out / "manifest_three_line.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# GRPO Three-Line Evidence Package

This folder indexes the three main report lines:

1. `baseline`: course baseline full run and report-ready figures.
2. `r12_full`: successful R12 full run, using `gsm8k_verifiable_simple`, `K=8`, `BETA=0.04`, `RANK=64`, `ALPHA=64`.
3. `r12_reward_only`: reward-only ablation, using the R12 reward but baseline-style `K=2`, `BETA=0.08`.

Reward-only status: {args.reward_only_status}

The package copies report-ready figures and small tables, and keeps raw logs,
manifests, run env files, and checkpoint archive lists under `raw_refs/`.
Large checkpoint archives remain in their original raw artifact directories and
are linked in `manifest_three_line.json` so they are not duplicated.

Use `r12_large_eval/r12_large_eval_summary.csv` for the final checkpoint choice:
R12 step 512 is the recommended checkpoint based on the 256-prompt large eval.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote three-line evidence package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
