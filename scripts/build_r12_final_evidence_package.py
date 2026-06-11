"""Build the final local evidence package for the R12 GRPO line.

This script is intentionally conservative: it does not connect to the TPU and
does not invent metrics. It collects the fetched R12 full run, the stopped
reward-only ablation, and optional baseline references into one auditable folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_tree_files(src: Path, dst: Path, patterns: tuple[str, ...]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    if not src.exists():
        return copied
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(src)
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
                copied.append({"source": str(path), "output": str(out)})
    return copied


def best_checkpoint(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    def key(row: dict[str, str]) -> tuple[float, float, int]:
        return (
            float(row.get("accuracy") or 0.0),
            float(row.get("partial_accuracy") or 0.0),
            -int(float(row.get("step") or 0)),
        )

    best = max(rows, key=key)
    return {
        "step": int(float(best.get("step") or 0)),
        "accuracy": float(best.get("accuracy") or 0.0),
        "partial_accuracy": float(best.get("partial_accuracy") or 0.0),
        "format_accuracy": float(best.get("format_accuracy") or 0.0),
        "robust_numeric_exact_rate": float(best.get("robust_numeric_exact_rate") or 0.0),
    }


def best_large_eval(rows: list[dict[str, str]]) -> dict[str, Any]:
    r12_rows = [row for row in rows if row.get("label", "").startswith("R12_") and row.get("step")]
    if not r12_rows:
        return {}

    def key(row: dict[str, str]) -> tuple[float, float, int]:
        return (
            float(row.get("accuracy") or 0.0),
            float(row.get("partial_accuracy") or 0.0),
            -int(float(row.get("step") or 0)),
        )

    best = max(r12_rows, key=key)
    return {
        "label": best.get("label", ""),
        "step": int(float(best.get("step") or 0)),
        "correct": int(float(best.get("correct") or 0)),
        "total": int(float(best.get("total") or 0)),
        "accuracy": float(best.get("accuracy") or 0.0),
        "partial_accuracy": float(best.get("partial_accuracy") or 0.0),
        "format_accuracy": float(best.get("format_accuracy") or 0.0),
        "robust_numeric_exact_rate": float(best.get("robust_numeric_exact_rate") or 0.0),
        "no_close_answer_rate": float(best.get("no_close_answer_rate") or 0.0),
        "text_after_close_rate": float(best.get("text_after_close_rate") or 0.0),
    }


def plot_checkpoint_accuracy(rows: list[dict[str, str]], out: Path) -> None:
    if not rows:
        return
    steps = [int(float(row["step"])) for row in rows]
    acc = [float(row["accuracy"]) for row in rows]
    partial = [float(row.get("partial_accuracy") or 0.0) for row in rows]
    low = [float(row.get("accuracy_ci95_low") or a) for row, a in zip(rows, acc)]
    high = [float(row.get("accuracy_ci95_high") or a) for row, a in zip(rows, acc)]
    yerr = [[a - l for a, l in zip(acc, low)], [h - a for a, h in zip(acc, high)]]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(steps, acc, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label="accuracy")
    ax.plot(steps, partial, marker="s", linewidth=1.2, label="partial_accuracy")
    best_idx = max(range(len(acc)), key=lambda i: (acc[i], partial[i], -steps[i]))
    ax.axvline(steps[best_idx], color="0.3", linewidth=1, linestyle="--")
    ax.text(steps[best_idx], max(acc + partial) + 1, f"best {steps[best_idx]}", ha="center", va="bottom", fontsize=9)
    ax.set_title("R12 full checkpoint evaluation")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("rate (%)")
    ax.set_ylim(max(0, min(acc + partial) - 6), min(100, max(acc + partial) + 8))
    ax.grid(True, axis="y", color="0.88")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_large_eval_summary(rows: list[dict[str, str]], out: Path) -> None:
    r12_rows = [row for row in rows if row.get("label", "").startswith("R12_") and row.get("step")]
    if not r12_rows:
        return
    r12_rows = sorted(r12_rows, key=lambda row: int(float(row.get("step") or 0)))
    steps = [int(float(row["step"])) for row in r12_rows]
    acc = [float(row.get("accuracy") or 0.0) for row in r12_rows]
    partial = [float(row.get("partial_accuracy") or 0.0) for row in r12_rows]
    robust = [100.0 * float(row.get("robust_numeric_exact_rate") or 0.0) for row in r12_rows]
    base_rows = [row for row in rows if row.get("label") == "base"]
    base_acc = float(base_rows[0].get("accuracy") or 0.0) if base_rows else None
    base_partial = float(base_rows[0].get("partial_accuracy") or 0.0) if base_rows else None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, acc, marker="o", linewidth=1.5, label="large_eval accuracy")
    ax.plot(steps, partial, marker="s", linewidth=1.3, label="large_eval partial_accuracy")
    ax.plot(steps, robust, marker="^", linewidth=1.1, label="robust_numeric_exact_rate")
    if base_acc is not None:
        ax.axhline(base_acc, color="0.45", linewidth=1, linestyle="--", label="base accuracy")
    if base_partial is not None:
        ax.axhline(base_partial, color="0.7", linewidth=1, linestyle=":", label="base partial_accuracy")
    best_idx = max(range(len(acc)), key=lambda i: (acc[i], partial[i], -steps[i]))
    ax.axvline(steps[best_idx], color="0.25", linewidth=1, linestyle="--")
    ax.text(
        steps[best_idx],
        max(acc + partial + robust) + 1,
        f"large best {steps[best_idx]}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_title("R12 256-prompt large eval summary")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("rate (%)")
    y_values = acc + partial + robust
    if base_acc is not None:
        y_values.append(base_acc)
    if base_partial is not None:
        y_values.append(base_partial)
    ax.set_ylim(max(0, min(y_values) - 6), min(100, max(y_values) + 8))
    ax.grid(True, axis="y", color="0.88")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def alert_counts(log_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in re.finditer(r"OBS ALERT ([^:]+):", log_text):
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


def plot_ablation_alerts(counts: dict[str, int], out: Path) -> None:
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(labels, values, color="#777777")
    ax.set_title("Reward-only R12 stopped ablation: alert counts")
    ax.set_ylabel("count in logs")
    ax.grid(True, axis="y", color="0.88")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r12-full-dir", type=Path, default=Path("artifacts/cloud/reward-k8-beta004-r12-full-001"))
    parser.add_argument("--reward-only-dir", type=Path, default=Path("artifacts/cloud/reward-only-r12-full-001"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/cloud/course-baseline-001"))
    parser.add_argument("--r12-clean-dir", type=Path, default=Path("artifacts/reports/reward-k8-beta004-r12-full-001-clean"))
    parser.add_argument("--large-eval-dir", type=Path, default=Path("artifacts/cloud/r12-best-large-eval-001/artifacts/eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/r12-final-evidence-001"))
    args = parser.parse_args()

    out = args.output_dir
    figures = out / "figures"
    tables = out / "tables"
    raw_refs = out / "raw_refs"
    for d in (figures, tables, raw_refs):
        d.mkdir(parents=True, exist_ok=True)

    eval_csv = args.r12_full_dir / "runs/R12_gsm8k_verifiable_simple/artifacts/checkpoint_eval/checkpoint_eval_summary.csv"
    rows = read_csv(eval_csv)
    best = best_checkpoint(rows)
    write_csv(tables / "r12_full_checkpoint_eval.csv", rows)
    plot_checkpoint_accuracy(rows, figures / "01_r12_checkpoint_accuracy.png")

    large_rows = read_csv(args.large_eval_dir / "large_eval_summary.csv")
    large_best = best_large_eval(large_rows)
    write_csv(tables / "r12_large_eval_summary.csv", large_rows)
    plot_large_eval_summary(large_rows, figures / "03_r12_large_eval_summary.png")

    reward_only_log = read_text(args.reward_only_dir / "pipeline.log") + "\n" + read_text(
        args.reward_only_dir / "runs/R12_reward_only_baseline_kkl/train.log"
    )
    alerts = alert_counts(reward_only_log)
    alert_rows = [{"alert": k, "count": v} for k, v in sorted(alerts.items())]
    write_csv(tables / "reward_only_alert_counts.csv", alert_rows)
    plot_ablation_alerts(alerts, figures / "02_reward_only_alert_counts.png")

    copied: list[dict[str, str]] = []
    copied += copy_tree_files(
        args.r12_clean_dir,
        out,
        (
            "figures/combined/*.png",
            "figures/by_run/*.png",
            "tables/*.csv",
            "manifest_clean_plots.json",
            "README.md",
        ),
    )
    copied += copy_tree_files(
        args.r12_full_dir,
        raw_refs / "reward-k8-beta004-r12-full-001",
        (
            "pipeline.log",
            "checkpoint_archives.txt",
            "artifacts/*.json",
            "runs/R12_gsm8k_verifiable_simple/run_env.txt",
            "runs/R12_gsm8k_verifiable_simple/reward_mode.txt",
            "runs/R12_gsm8k_verifiable_simple/artifacts/checkpoint_eval/*.csv",
            "runs/R12_gsm8k_verifiable_simple/artifacts/checkpoint_eval/*.json",
        ),
    )
    copied += copy_tree_files(
        args.large_eval_dir,
        raw_refs / "r12-best-large-eval-001" / "artifacts" / "eval",
        (
            "*.csv",
            "*.json",
            "*.jsonl",
        ),
    )
    copied += copy_tree_files(
        args.reward_only_dir,
        raw_refs / "reward-only-r12-full-001",
        (
            "pipeline.log",
            "checkpoint_archives.txt",
            "artifacts/*.json",
            "runs/R12_reward_only_baseline_kkl/run_env.txt",
            "runs/R12_reward_only_baseline_kkl/reward_mode.txt",
            "runs/R12_reward_only_baseline_kkl/train.log",
        ),
    )
    if args.baseline_dir.exists():
        copied += copy_tree_files(
            args.baseline_dir,
            raw_refs / "course-baseline-001",
            ("pipeline.log", "artifacts/*.json", "meta/*.txt"),
        )

    checkpoint_archives = [
        str(path.relative_to(args.r12_full_dir))
        for path in sorted((args.r12_full_dir / "checkpoint_archives").glob("*.tar.gz"))
        if path.is_file()
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "r12_full_dir": str(args.r12_full_dir),
        "reward_only_dir": str(args.reward_only_dir),
        "baseline_dir": str(args.baseline_dir),
        "large_eval_dir": str(args.large_eval_dir),
        "recommended_checkpoint": large_best or best,
        "selection_basis": "large_eval_256_prompts" if large_best else "small_eval_64_prompts",
        "best_r12_checkpoint_small_eval": best,
        "best_r12_checkpoint_large_eval": large_best,
        "checkpoint_archives_local_dir": str(args.r12_full_dir / "checkpoint_archives"),
        "checkpoint_archives": checkpoint_archives,
        "reward_only_ablation": {
            "status": "stopped_negative_evidence",
            "reason": "sustained zero_reward_std_spike and extracted_none_spike through checkpoint 500",
            "checkpoints_preserved": ["1", "500"],
            "alert_counts": alerts,
        },
        "copied_refs": copied,
        "figures": [
            "figures/01_r12_checkpoint_accuracy.png",
            "figures/02_reward_only_alert_counts.png",
            "figures/03_r12_large_eval_summary.png",
            "figures/combined/*.png",
            "figures/by_run/*.png",
        ],
        "tables": [
            "tables/r12_full_checkpoint_eval.csv",
            "tables/r12_large_eval_summary.csv",
            "tables/reward_only_alert_counts.csv",
            "tables/*.csv",
        ],
    }
    write_json(out / "manifest_report.json", manifest)

    large_line = (
        f"- Recommended checkpoint from 256-prompt large eval: step `{large_best.get('step')}`, "
        f"accuracy `{large_best.get('accuracy')}%`, partial `{large_best.get('partial_accuracy')}%`, "
        f"correct `{large_best.get('correct')}/{large_best.get('total')}`."
        if large_best
        else "- Recommended checkpoint falls back to the 64-prompt checkpoint eval because large eval was not found."
    )

    readme = f"""# R12 Final Evidence Package

This folder consolidates the R12 GRPO evidence used for the report.

## Key result

- Best 64-prompt checkpoint eval: step `{best.get('step')}`, accuracy `{best.get('accuracy')}%`, partial `{best.get('partial_accuracy')}%`.
{large_line}
- Full R12 config: `REWARD_MODE=gsm8k_verifiable_simple`, `NUM_GENERATIONS=8`, `BETA=0.04`, `RANK=64`, `ALPHA=64`, `MAX_STEPS=841`.
- Local checkpoint archives: `{args.r12_full_dir / "checkpoint_archives"}` ({len(checkpoint_archives)} files).
- Reward-only ablation was stopped at checkpoint `500`: it kept the R12 reward but used baseline `NUM_GENERATIONS=2`, `BETA=0.08`; logs showed sustained zero reward std and extraction failure alerts.

## How to read

- `figures/01_r12_checkpoint_accuracy.png`: checkpoint accuracy and partial accuracy.
- `figures/03_r12_large_eval_summary.png`: large eval comparison for base and R12 checkpoints.
- `figures/combined/`: copied clean training diagnostics from the R12 full run.
- `tables/r12_full_checkpoint_eval.csv`: source table for checkpoint results.
- `tables/r12_large_eval_summary.csv`: 256-prompt large eval summary for base, R12 step 384, step 512, and step 841.
- `tables/reward_only_alert_counts.csv`: stopped ablation alert counts.
- `raw_refs/`: copied logs, env files, checkpoint eval JSON/CSV, and manifests.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote R12 evidence package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
