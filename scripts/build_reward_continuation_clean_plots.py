"""Build plain matplotlib plots for reward-continuation experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    "C1_R1_no_approx_from256",
    "C2_R3_numeric_primary_no_len_from768",
    "C3_R5_numeric_primary_answer_only_len1200_from512",
]

RUN_LABEL = {
    "C1_R1_no_approx_from256": "C1 R1 no_approx from 256",
    "C2_R3_numeric_primary_no_len_from768": "C2 R3 numeric_primary from 768",
    "C3_R5_numeric_primary_answer_only_len1200_from512": "C3 R5 answer_only from 512",
}

RUN_COLOR = {
    "C1_R1_no_approx_from256": "#1f77b4",
    "C2_R3_numeric_primary_no_len_from768": "#2ca02c",
    "C3_R5_numeric_primary_answer_only_len1200_from512": "#d62728",
}

COMPONENTS_BY_MODE = {
    "no_approx": ["match_format_exactly", "check_answer", "check_numbers"],
    "numeric_primary_no_len": ["numeric_primary", "format_strict_light", "answer_tag_light"],
    "numeric_primary_answer_only_len1200": ["numeric_primary", "answer_tag_light", "length_penalty_1200"],
}

NUMERIC_COMPONENTS = {"check_answer", "check_numbers", "numeric_primary"}
FORMAT_COMPONENTS = {"match_format_exactly", "match_format_approximately", "format_strict_light", "answer_tag_light"}
LENGTH_COMPONENTS = {"length_penalty_1200"}

METRIC_COLOR = {
    "train_score_mean": "#1f77b4",
    "eval_score_mean": "#ff7f0e",
    "train_kl": "#1f77b4",
    "eval_kl": "#ff7f0e",
    "train_loss": "#1f77b4",
    "train_pg_clipfrac": "#2ca02c",
    "reward_std": "#1f77b4",
    "frac_reward_zero_std": "#d62728",
    "advantage_std": "#9467bd",
    "grpo_group_reward_std_mean": "#2ca02c",
    "empty_response_rate": "#d62728",
    "extracted_none_rate": "#ff7f0e",
    "format_accuracy": "#9467bd",
    "numeric_exact_rate": "#2ca02c",
    "overlong_rate_1600": "#8c564b",
    "completion_chars": "#1f77b4",
    "reward_numeric_margin": "#1f77b4",
    "reward_hacking_rate": "#d62728",
    "reward_numeric_primary_mean": "#1f77b4",
    "reward_format_light_mean": "#ff7f0e",
    "reward_length_penalty_1200_mean": "#d62728",
    "reward_match_format_exactly_mean": "#2ca02c",
    "reward_match_format_approximately_mean": "#9467bd",
    "composition_total_reward_mean": "#1f77b4",
    "composition_numeric_component_mean": "#2ca02c",
    "composition_format_component_mean": "#ff7f0e",
    "composition_length_component_mean": "#d62728",
    "composition_other_component_mean": "#7f7f7f",
    "composition_format_abs_share": "#9467bd",
}

COMBINED_GROUPS = [
    ("02_reward_score.png", "Reward score", [("train_score_mean", None), ("eval_score_mean", None)]),
    ("03_kl_loss_clipfrac.png", "KL / loss / clip fraction", [("train_kl", None), ("eval_kl", None), ("train_loss", None), ("train_pg_clipfrac", (0, 1))]),
    ("04_grpo_health.png", "GRPO group health", [("reward_std", None), ("frac_reward_zero_std", (0, 1)), ("advantage_std", None), ("grpo_group_reward_std_mean", None)]),
    ("05_response_health.png", "Response / parser health", [("empty_response_rate", (0, 1)), ("extracted_none_rate", (0, 1)), ("numeric_exact_rate", (0, 1)), ("format_accuracy", (0, 1)), ("overlong_rate_1600", (0, 1)), ("completion_chars", None)]),
    ("06_reward_audit.png", "Reward audit", [("reward_numeric_margin", None), ("reward_hacking_rate", (0, 1))]),
    ("07_reward_components.png", "Reward components", [("reward_numeric_primary_mean", None), ("reward_format_light_mean", None), ("reward_length_penalty_1200_mean", None), ("reward_match_format_exactly_mean", None), ("reward_match_format_approximately_mean", None)]),
]

COMPOSITION_PANELS = [
    ("total_reward_mean", "total reward", None),
    ("numeric_component_mean", "numeric component", None),
    ("format_component_mean", "format/tag component", None),
    ("format_abs_share", "format share of active reward", (0, 1)),
    ("length_component_mean", "length penalty", None),
    ("other_component_mean", "other active component", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean continuation plots.")
    parser.add_argument("--input-dir", default="artifacts/cloud/reward-continuation-001")
    parser.add_argument("--output-dir", default="artifacts/reports/reward-continuation-001-clean")
    parser.add_argument("--rolling-window", type=int, default=64)
    return parser.parse_args()


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_dir = input_dir / "artifacts" / "continuation_analysis" / "tables"
    if not table_dir.exists():
        raise SystemExit(f"Missing continuation analysis tables: {table_dir}")
    scalars = pd.read_csv(table_dir / "scalar_pivot.csv", low_memory=False)
    ckpt = pd.read_csv(table_dir / "checkpoint_eval_long.csv")
    selection = pd.read_csv(table_dir / "selection_summary.csv")
    for df in (scalars, ckpt, selection):
        for col in df.columns:
            if col not in {"run_id", "reward_mode", "source_run", "screening_status", "guardrail_reasons", "label", "policy", "file"}:
                converted = pd.to_numeric(df[col], errors="coerce")
                if not converted.isna().all():
                    df[col] = converted
    return scalars, ckpt, selection


def save_fig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def rolling(series: pd.Series, window: int) -> pd.Series:
    if len(series) < window * 2:
        return series
    return series.rolling(window=window, min_periods=max(4, window // 4), center=True).mean()


def tidy(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_ids(scalars: pd.DataFrame, ckpt: pd.DataFrame) -> list[str]:
    seen = list(dict.fromkeys(list(ckpt["run_id"].dropna()) + list(scalars["run_id"].dropna())))
    return [run for run in DEFAULT_RUNS if run in seen] + [run for run in seen if run not in DEFAULT_RUNS]


def scalar_series(scalars: pd.DataFrame, run_id: str, metric: str) -> pd.DataFrame:
    if metric not in scalars.columns:
        return pd.DataFrame(columns=["step", metric])
    sub = scalars.loc[scalars["run_id"].eq(run_id), ["step", metric]].copy()
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce")
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    return sub.dropna().groupby("step", as_index=False)[metric].mean().sort_values("step")


def component_column_options(component: str) -> list[str]:
    return [
        f"reward_{component}_mean",
        f"rewards_train_{component}",
        f"rewards_eval_{component}",
    ]


def first_existing_component_value(row: pd.Series, component: str) -> float:
    for column in component_column_options(component):
        if column in row.index and pd.notna(row[column]):
            try:
                return float(row[column])
            except Exception:
                return 0.0
    return 0.0


def build_reward_composition(scalars: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in scalars.iterrows():
        run_id = str(row.get("run_id", ""))
        reward_mode = str(row.get("reward_mode", "")).strip().lower()
        step = pd.to_numeric(row.get("step"), errors="coerce")
        if not run_id or pd.isna(step) or reward_mode not in COMPONENTS_BY_MODE:
            continue
        numeric = 0.0
        format_part = 0.0
        length = 0.0
        other = 0.0
        for component in COMPONENTS_BY_MODE[reward_mode]:
            value = first_existing_component_value(row, component)
            if component in NUMERIC_COMPONENTS:
                numeric += value
            elif component in FORMAT_COMPONENTS:
                format_part += value
            elif component in LENGTH_COMPONENTS:
                length += value
            else:
                other += value
        total = numeric + format_part + length + other
        denom = abs(numeric) + abs(format_part) + abs(length) + abs(other)
        rows.append(
            {
                "run_id": run_id,
                "reward_mode": reward_mode,
                "source_run": row.get("source_run", ""),
                "source_step": row.get("source_step", ""),
                "step": float(step),
                "total_reward_mean": total,
                "numeric_component_mean": numeric,
                "format_component_mean": format_part,
                "length_component_mean": length,
                "other_component_mean": other,
                "numeric_abs_share": abs(numeric) / denom if denom else 0.0,
                "format_abs_share": abs(format_part) / denom if denom else 0.0,
                "length_abs_share": abs(length) / denom if denom else 0.0,
                "other_abs_share": abs(other) / denom if denom else 0.0,
                "active_abs_denominator_mean": denom,
            }
        )
    composition = pd.DataFrame(rows)
    if composition.empty:
        return composition
    composition = (
        composition.groupby(["run_id", "reward_mode", "source_run", "source_step", "step"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["run_id", "step"])
    )
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    composition.to_csv(tables / "reward_composition_by_step.csv", index=False)
    latest_rows = []
    for run_id, sub in composition.groupby("run_id"):
        sub = sub.dropna(subset=["step"]).sort_values("step")
        if not sub.empty:
            latest_rows.append(sub.iloc[-1].to_dict())
    if latest_rows:
        pd.DataFrame(latest_rows).to_csv(tables / "reward_composition_latest.csv", index=False)
    return composition


def composition_series(composition: pd.DataFrame, run_id: str, metric: str) -> pd.DataFrame:
    if composition.empty or metric not in composition.columns:
        return pd.DataFrame(columns=["step", metric])
    sub = composition.loc[composition["run_id"].eq(run_id), ["step", metric]].copy()
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce")
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    return sub.dropna().groupby("step", as_index=False)[metric].mean().sort_values("step")


def plot_checkpoint(ckpt: pd.DataFrame, runs: list[str], out: Path) -> None:
    metrics = [("accuracy", "accuracy (%)"), ("partial_accuracy", "partial_accuracy (%)"), ("format_accuracy", "format_accuracy (%)")]
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 7.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for run_id in runs:
            sub = ckpt.loc[ckpt["run_id"].eq(run_id)].sort_values("step")
            if sub.empty:
                continue
            if metric == "accuracy" and {"accuracy_ci95_low", "accuracy_ci95_high"}.issubset(sub.columns):
                ax.errorbar(
                    sub["step"],
                    sub[metric],
                    yerr=[sub[metric] - sub["accuracy_ci95_low"], sub["accuracy_ci95_high"] - sub[metric]],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    capsize=2,
                    color=RUN_COLOR.get(run_id),
                    label=RUN_LABEL.get(run_id, run_id),
                )
            else:
                ax.plot(sub["step"], sub[metric], marker="o", markersize=3.5, linewidth=1.1, color=RUN_COLOR.get(run_id), label=RUN_LABEL.get(run_id, run_id))
        tidy(ax, ylabel, (0, 100))
    axes[0].set_title("Continuation checkpoint eval")
    axes[-1].set_xlabel("checkpoint step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_fig(fig, out)


def plot_combined(scalars: pd.DataFrame, runs: list[str], out: Path, title: str, metrics: list[tuple[str, tuple[float, float] | None]], window: int) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, max(2.2 * len(metrics), 4.8)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for ax, (metric, ylim) in zip(axes, metrics):
        for run_id in runs:
            sub = scalar_series(scalars, run_id, metric)
            if sub.empty:
                continue
            ax.plot(sub["step"], rolling(sub[metric], window), color=RUN_COLOR.get(run_id), linewidth=1.2, label=RUN_LABEL.get(run_id, run_id))
        tidy(ax, metric, ylim)
    axes[0].set_title(f"{title} (rolling mean, window={window})")
    axes[-1].set_xlabel("scalar step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_fig(fig, out)


def plot_per_run(scalars: pd.DataFrame, run_id: str, out: Path, title: str, metrics: list[tuple[str, tuple[float, float] | None]], window: int) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, max(2.05 * len(metrics), 4.8)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for ax, (metric, ylim) in zip(axes, metrics):
        sub = scalar_series(scalars, run_id, metric)
        if not sub.empty:
            color = METRIC_COLOR.get(metric, "#1f77b4")
            ax.plot(sub["step"], sub[metric], color=color, linewidth=0.45, alpha=0.32, label="raw")
            ax.plot(sub["step"], rolling(sub[metric], window), color=color, linewidth=1.2, label=f"rolling {window}")
        tidy(ax, metric, ylim)
    axes[0].set_title(f"{RUN_LABEL.get(run_id, run_id)}: {title}")
    axes[-1].set_xlabel("scalar step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out)


def plot_combined_reward_composition(composition: pd.DataFrame, runs: list[str], out: Path, window: int) -> None:
    if composition.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.2), sharex=True)
    axes_flat = axes.flat
    for ax, (metric, ylabel, ylim) in zip(axes_flat, COMPOSITION_PANELS):
        for run_id in runs:
            sub = composition_series(composition, run_id, metric)
            if sub.empty:
                continue
            ax.plot(
                sub["step"],
                rolling(sub[metric], window),
                color=RUN_COLOR.get(run_id, "#1f77b4"),
                linewidth=1.2,
                label=RUN_LABEL.get(run_id, run_id),
            )
        if metric.endswith("_component_mean") or metric == "total_reward_mean":
            ax.axhline(0, color="#444444", linewidth=0.65, alpha=0.65)
        tidy(ax, ylabel, ylim)
    axes_flat[0].set_title("Continuation active reward composition")
    axes_flat[-2].set_xlabel("scalar step")
    axes_flat[-1].set_xlabel("scalar step")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.text(
        0.01,
        0.015,
        "Format share = mean(abs(active format/tag component) / (abs(numeric) + abs(format/tag) + abs(length) + abs(other))).",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    save_fig(fig, out)


def plot_per_run_reward_composition(composition: pd.DataFrame, run_id: str, out: Path, window: int) -> None:
    if composition.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.2), sharex=True)
    axes_flat = axes.flat
    for ax, (metric, ylabel, ylim) in zip(axes_flat, COMPOSITION_PANELS):
        sub = composition_series(composition, run_id, metric)
        if not sub.empty:
            color = METRIC_COLOR.get(f"composition_{metric}", "#1f77b4")
            ax.plot(sub["step"], sub[metric], color=color, linewidth=0.25, alpha=0.12, label="raw")
            ax.plot(sub["step"], rolling(sub[metric], window), color=color, linewidth=1.2, label=f"rolling {window}")
        if metric.endswith("_component_mean") or metric == "total_reward_mean":
            ax.axhline(0, color="#444444", linewidth=0.65, alpha=0.65)
        tidy(ax, ylabel, ylim)
    axes_flat[0].set_title(f"{RUN_LABEL.get(run_id, run_id)}: active reward composition")
    axes_flat[-2].set_xlabel("scalar step")
    axes_flat[-1].set_xlabel("scalar step")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.text(
        0.01,
        0.015,
        "Only components enabled by this run's REWARD_MODE are counted; audit-only component columns are ignored.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_fig(fig, out)


def copy_tables(input_dir: Path, output_dir: Path) -> list[str]:
    src = input_dir / "artifacts" / "continuation_analysis" / "tables"
    dst = output_dir / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in src.glob("*"):
        if path.is_file():
            target = dst / path.name
            shutil.copy2(path, target)
            copied.append(str(target.relative_to(output_dir)))
    return sorted(copied)


def write_readme(output_dir: Path, input_dir: Path, figures: list[str], selection: pd.DataFrame, copied: list[str], window: int) -> None:
    lines = [
        "# reward-continuation-001 clean plots",
        "",
        "Plain matplotlib plots for R1/R3/R5 checkpoint-dense continuation.",
        "",
        f"- Raw fetched run: `{input_dir.resolve()}`",
        f"- Clean plot package: `{output_dir.resolve()}`",
        f"- Rolling window: `{window}`",
        "- Reward composition plots count only the components enabled by each run's `REWARD_MODE`; audit-only component columns are ignored.",
        "- Format share is `abs(format_or_tag_hygiene) / (abs(numeric) + abs(format_or_tag_hygiene) + abs(length) + abs(other))`.",
        "- Derived reward composition tables: `tables/reward_composition_by_step.csv`, `tables/reward_composition_latest.csv`.",
        "",
        "## Selection snapshot",
        "",
        "| run_id | source | best_step | acc | partial | format | guardrails |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in selection.to_dict("records"):
        guardrails = "" if pd.isna(row.get("guardrail_reasons")) else row.get("guardrail_reasons")
        lines.append(
            f"| {row.get('run_id')} | {row.get('source_run')}@{row.get('source_step')} | {row.get('best_checkpoint_step')} | {float(row.get('best_checkpoint_accuracy') or 0):.2f} | {float(row.get('best_checkpoint_partial_accuracy') or 0):.2f} | {float(row.get('best_checkpoint_format_accuracy') or 0):.2f} | {guardrails} |"
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend(["", "## Tables", ""])
    lines.extend(f"- `{table}`" for table in copied)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scalars, ckpt, selection = load_inputs(input_dir)
    runs = run_ids(scalars, ckpt)
    composition = build_reward_composition(scalars, output_dir)
    figures: list[str] = []

    out = output_dir / "figures" / "combined" / "01_checkpoint_eval.png"
    plot_checkpoint(ckpt, runs, out)
    figures.append(str(out.relative_to(output_dir)))
    for filename, title, metrics in COMBINED_GROUPS:
        out = output_dir / "figures" / "combined" / filename
        plot_combined(scalars, runs, out, title, metrics, args.rolling_window)
        figures.append(str(out.relative_to(output_dir)))
    out = output_dir / "figures" / "combined" / "08_reward_composition_format_share.png"
    plot_combined_reward_composition(composition, runs, out, args.rolling_window)
    if out.exists():
        figures.append(str(out.relative_to(output_dir)))
    for run_id in runs:
        out = output_dir / "figures" / "by_run" / run_id / "01_checkpoint_eval.png"
        plot_checkpoint(ckpt.loc[ckpt["run_id"].eq(run_id)], [run_id], out)
        figures.append(str(out.relative_to(output_dir)))
        for filename, title, metrics in COMBINED_GROUPS:
            out = output_dir / "figures" / "by_run" / run_id / filename
            plot_per_run(scalars, run_id, out, title, metrics, args.rolling_window)
            figures.append(str(out.relative_to(output_dir)))
        out = output_dir / "figures" / "by_run" / run_id / "08_reward_composition_format_share_raw.png"
        plot_per_run_reward_composition(composition, run_id, out, args.rolling_window)
        if out.exists():
            figures.append(str(out.relative_to(output_dir)))
    copied = copy_tables(input_dir, output_dir)
    (output_dir / "manifest_clean_plots.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_dir": str(input_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "runs": runs,
                "figures": figures,
                "tables": copied,
                "derived_tables": [
                    "tables/reward_composition_by_step.csv",
                    "tables/reward_composition_latest.csv",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(output_dir, input_dir, figures, selection, copied, args.rolling_window)
    print(f"Wrote {len(figures)} clean continuation figures to {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
