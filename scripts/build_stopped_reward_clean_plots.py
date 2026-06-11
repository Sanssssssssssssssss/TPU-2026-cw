"""Build clean plots for a stopped reward run when scalar tables are absent.

This is a local recovery helper for partial runs such as reward-only R12. It
uses rollout trace rows, pipeline alerts, run env, and checkpoint archive lists
to produce the same 01-08 clean plot set used by full reward runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIGSIZE = (10, 5.6)
BLUE = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GRAY = "#6b7280"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_alert_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    pattern = re.compile(r"OBS ALERT ([A-Za-z0-9_]+):")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finish_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_lines(
    df: pd.DataFrame,
    y_cols: list[tuple[str, str, str]],
    title: str,
    path: Path,
    ylim: tuple[float, float] | None = None,
    rolling_window: int = 64,
) -> None:
    plt.figure(figsize=FIGSIZE)
    if df.empty:
        plt.text(0.5, 0.5, "No trace rows available", ha="center", va="center")
        plt.axis("off")
    else:
        for col, label, color in y_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                x = pd.to_numeric(df["call_index"], errors="coerce")
                if len(series) > rolling_window and rolling_window > 1:
                    plt.plot(x, series, linewidth=0.55, color=color, alpha=0.18)
                    plotted = series.rolling(rolling_window, min_periods=1).mean()
                    plt.plot(x, plotted, label=f"{label} (rolling {rolling_window})", linewidth=2.0, color=color)
                else:
                    plt.plot(x, series, label=label, linewidth=1.8, color=color)
        plt.xlabel("metric call index")
        plt.ylabel("rate / value")
        if ylim:
            plt.ylim(*ylim)
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
    plt.title(title)
    finish_plot(path)


def plot_text_panel(lines: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=FIGSIZE)
    plt.axis("off")
    plt.title(title, loc="left", pad=12)
    plt.text(0.02, 0.92, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    finish_plot(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_group_health(trace_rows: pd.DataFrame) -> pd.DataFrame:
    if trace_rows.empty:
        return pd.DataFrame()
    needed = ["call_index", "group_index", "reward_total"]
    if any(col not in trace_rows.columns for col in needed):
        return pd.DataFrame()
    work = trace_rows[needed].copy()
    work["call_index"] = pd.to_numeric(work["call_index"], errors="coerce")
    work["group_index"] = pd.to_numeric(work["group_index"], errors="coerce")
    work["reward_total"] = pd.to_numeric(work["reward_total"], errors="coerce")
    work = work.dropna(subset=needed)
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(["call_index", "group_index"])["reward_total"].agg(["std", "mean", "count"]).reset_index()
    grouped["zero_std"] = grouped["std"].fillna(0.0).abs() <= 1e-8
    per_call = grouped.groupby("call_index").agg(
        reward_std_mean=("std", "mean"),
        frac_reward_zero_std=("zero_std", "mean"),
        group_reward_mean=("mean", "mean"),
        groups=("group_index", "count"),
    )
    return per_call.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--rolling-window", type=int, default=64)
    args = parser.parse_args()

    root = args.input_dir
    out = args.output_dir
    figures = out / "figures" / "combined"
    tables = out / "tables"
    ensure_dir(figures)
    ensure_dir(tables)

    analysis_tables = root / "artifacts" / "sweep_analysis" / "tables"
    trace_path = analysis_tables / "trace_rows_flat.csv"
    audit_path = analysis_tables / "trace_audit_by_call.csv"
    selection_path = analysis_tables / "selection_summary.csv"
    if not trace_path.exists() or not audit_path.exists():
        raise SystemExit(f"Missing trace tables under {analysis_tables}")

    trace = pd.read_csv(trace_path)
    audit = pd.read_csv(audit_path)
    run_ids = [rid for rid in trace.get("run_id", pd.Series(dtype=str)).dropna().unique().tolist() if rid != "R0_baseline"]
    run_id = args.run_id or (run_ids[0] if run_ids else "")
    if run_id:
        trace_run = trace[trace["run_id"] == run_id].copy()
        audit_run = audit[audit["run_id"] == run_id].copy()
    else:
        trace_run = trace.copy()
        audit_run = audit.copy()

    numeric_cols = [
        "call_index",
        "reward_mean",
        "empty_response_rate",
        "extracted_none_rate",
        "format_accuracy",
        "numeric_exact_rate",
        "robust_numeric_exact_rate",
        "no_close_answer_rate",
        "overlong_rate_1200",
        "overlong_rate_1600",
        "parser_false_negative_rate",
        "answer_single_number_rate",
    ]
    audit_run = safe_numeric(audit_run, numeric_cols).sort_values("call_index")
    trace_run = safe_numeric(
        trace_run,
        [
            "call_index",
            "group_index",
            "reward_total",
            "reward_total_recomputed",
            "component_gsm8k_simple_numeric",
            "component_gsm8k_simple_format",
            "completion_chars",
        ],
    )
    group_health = build_group_health(trace_run)

    shutil.copy2(trace_path, tables / "trace_rows_flat.csv")
    shutil.copy2(audit_path, tables / "trace_audit_by_call.csv")
    if selection_path.exists():
        shutil.copy2(selection_path, tables / "selection_summary.csv")

    alert_counts = read_alert_counts(root / "pipeline.log")
    write_csv(tables / "alert_counts.csv", [{"alert": k, "count": v} for k, v in sorted(alert_counts.items())])

    archives = []
    archive_list = root / "checkpoint_archives.txt"
    if archive_list.exists():
        for line in archive_list.read_text(encoding="utf-8", errors="replace").splitlines():
            value = line.strip()
            if value:
                archive_path = root / value
                archives.append({"archive": value, "exists": archive_path.exists(), "bytes": archive_path.stat().st_size if archive_path.exists() else 0})
    write_csv(tables / "checkpoint_archives_summary.csv", archives)

    run_env_path = next(root.glob("runs/*/run_env.txt"), None)
    env = read_env(run_env_path) if run_env_path else {}

    latest = audit_run.dropna(subset=["call_index"]).tail(1).to_dict("records")
    latest_row = latest[0] if latest else {}
    checkpoint_lines = [
        f"run: {root.name}",
        f"branch: {run_id or 'unknown'}",
        f"status: stopped partial run",
        f"reward_mode: {env.get('REWARD_MODE', latest_row.get('reward_mode', 'unknown'))}",
        f"max_steps: {env.get('MAX_STEPS', 'unknown')}",
        f"num_generations: {env.get('NUM_GENERATIONS', 'unknown')}",
        f"beta: {env.get('BETA', 'unknown')}",
        f"rank/alpha: {env.get('RANK', 'unknown')}/{env.get('ALPHA', 'unknown')}",
        f"checkpoint archives: {sum(1 for row in archives if row['exists'])}/{len(archives)}",
        "checkpoint eval: not available; run was stopped before scheduled eval completed",
    ]
    if alert_counts:
        checkpoint_lines.append("alerts: " + ", ".join(f"{k}={v}" for k, v in alert_counts.most_common()))
    plot_text_panel(checkpoint_lines, "01 checkpoint eval / run status", figures / "01_checkpoint_eval.png")

    plot_lines(
        audit_run,
        [("reward_mean", "reward mean", BLUE), ("numeric_exact_rate", "numeric exact", GREEN), ("extracted_none_rate", "extracted none", RED)],
        "02 reward score and extraction health",
        figures / "02_reward_score.png",
        rolling_window=args.rolling_window,
    )

    plot_text_panel(
        [
            "KL/loss/clipfrac scalar timeline is not in the recovered scalar table.",
            "Raw TensorBoard event files are preserved under:",
            str(next(root.glob("runs/*/tensorboard"), root / "runs")),
            "",
            "Use this stopped-run package for trace-based reward and response analysis.",
        ],
        "03 KL / loss / clipfrac",
        figures / "03_kl_loss_clipfrac.png",
    )

    plot_lines(
        group_health,
        [("frac_reward_zero_std", "frac reward zero std", RED), ("reward_std_mean", "reward std mean", BLUE)],
        "04 GRPO group health reconstructed from trace",
        figures / "04_grpo_health.png",
        rolling_window=args.rolling_window,
    )

    plot_lines(
        audit_run,
        [
            ("empty_response_rate", "empty", GRAY),
            ("extracted_none_rate", "extracted none", RED),
            ("no_close_answer_rate", "no close answer", ORANGE),
            ("overlong_rate_1600", "overlong 1600", PURPLE),
        ],
        "05 response health",
        figures / "05_response_health.png",
        ylim=(0, 1),
        rolling_window=args.rolling_window,
    )

    plot_lines(
        audit_run,
        [
            ("numeric_exact_rate", "official numeric exact", GREEN),
            ("robust_numeric_exact_rate", "robust numeric exact", BLUE),
            ("format_accuracy", "format accuracy", PURPLE),
            ("parser_false_negative_rate", "parser false negative", ORANGE),
        ],
        "06 reward audit",
        figures / "06_reward_audit.png",
        ylim=(0, 1),
        rolling_window=args.rolling_window,
    )

    components = trace_run.groupby("call_index", dropna=True).agg(
        gsm8k_simple_numeric=("component_gsm8k_simple_numeric", "mean"),
        gsm8k_simple_format=("component_gsm8k_simple_format", "mean"),
        reward_total=("reward_total", "mean"),
    ).reset_index() if not trace_run.empty else pd.DataFrame()
    plot_lines(
        components,
        [
            ("gsm8k_simple_numeric", "gsm8k simple numeric", BLUE),
            ("gsm8k_simple_format", "gsm8k simple format", GREEN),
            ("reward_total", "reward total", GRAY),
        ],
        "07 reward components",
        figures / "07_reward_components.png",
        rolling_window=args.rolling_window,
    )

    if not components.empty:
        comp = components.copy()
        denom = comp["gsm8k_simple_numeric"].abs().fillna(0) + comp["gsm8k_simple_format"].abs().fillna(0)
        comp["format_share_abs"] = comp["gsm8k_simple_format"].abs().where(denom > 0, math.nan) / denom.where(denom > 0, math.nan)
    else:
        comp = pd.DataFrame()
    plot_lines(
        comp,
        [("format_share_abs", "format component share", PURPLE), ("gsm8k_simple_numeric", "numeric component", BLUE), ("gsm8k_simple_format", "format component", GREEN)],
        "08 reward composition format share",
        figures / "08_reward_composition_format_share.png",
        rolling_window=args.rolling_window,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(root),
        "output_dir": str(out),
        "run_id": run_id,
        "source": "trace_rows_flat.csv + trace_audit_by_call.csv + pipeline alerts",
        "figures": sorted(str(path.relative_to(out)) for path in figures.glob("*.png")),
        "tables": sorted(str(path.relative_to(out)) for path in tables.glob("*.csv")),
        "note": "Stopped-run clean package; checkpoint eval and scalar KL/loss plots are marked unavailable when absent.",
        "rolling_window": args.rolling_window,
    }
    (out / "manifest_clean_plots.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                f"# {root.name} Clean Stopped-Run Package",
                "",
                "This package preserves the same 01-08 clean plot naming used by full reward runs.",
                "It is generated from rollout trace rows, pipeline alerts, run env, and checkpoint archive metadata.",
                "",
                "Important limitation: this stopped run does not have checkpoint eval results in the fetched package.",
                "KL/loss/clipfrac raw TensorBoard event files are preserved, but no scalar table was available locally.",
                "",
                "Key files:",
                "- `figures/combined/*.png`",
                "- `tables/trace_rows_flat.csv`",
                "- `tables/trace_audit_by_call.csv`",
                "- `tables/alert_counts.csv`",
                "- `tables/checkpoint_archives_summary.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote stopped-run clean package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
