#!/usr/bin/env python3
"""Build clean report-ready figures for the rollout320 official GRPO runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "artifacts" / "reports" / "final-comparison"
OUT_DIR = REPO / "artifacts" / "reports" / "final-figures"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "R0": "#6F768A",
    "R1": "#CC6F47",
    "R2": "#4B9A62",
    "R3": "#BD569B",
    "R4": "#3B6EDB",
    "R5": "#2AA198",
    "R6": "#D97706",
    "reward": "#F0A21A",
    "eval_reward": "#26A6DA",
    "kl": "#8056E5",
    "eval_kl": "#E94B4B",
    "exact": "#1B9E77",
    "format": "#2F6DE0",
    "empty": "#EF4444",
    "no_answer": "#7C3AED",
}

REWARD_SCALE_MAX = {
    "baseline": 10.0,
    "gsm8k_verifiable_simple": 1.2,
    "gsm8k_verifiable_format": 1.8,
}

BASELINE_REWARD_SCALE = 10.0

REWARD_MECHANISMS = {
    "baseline": {
        "native_max": 10.0,
        "components": {
            "match_format_exactly": 3.0,
            "match_format_approximately": 2.5,
            "check_answer": 3.0,
            "check_numbers": 1.5,
        },
        "notes": "Original course reward: strict full-template format, approximate tag format, bracketed answer correctness, and numeric fallback.",
    },
    "gsm8k_verifiable_simple": {
        "native_max": 1.2,
        "components": {
            "gsm8k_simple_numeric": 1.0,
            "gsm8k_simple_format": 0.2,
        },
        "notes": "Numeric GSM8K-style correctness plus a small answer-tag helper.",
    },
    "gsm8k_verifiable_format": {
        "native_max": 1.8,
        "components": {
            "gsm8k_simple_numeric": 1.0,
            "gsm8k_simple_format": 0.2,
            "reasoning_structure_format": 0.6,
        },
        "notes": "Current R1/R4 reward: numeric correctness, answer-tag helper, and explicit reasoning/answer envelope reward.",
    },
}

REWARD_ALIGNMENT_FORMULA = "reward_score_report = reward_score_native / reward_native_max * 10.0"


@dataclass(frozen=True)
class RunSpec:
    key: str
    run_id: str
    branch: str
    label: str
    num_generations: int
    max_steps: int
    reward_mode: str

    @property
    def cloud_dir(self) -> Path:
        return REPO / "artifacts" / "cloud" / self.run_id

    @property
    def sweep_tables(self) -> Path:
        return self.cloud_dir / "artifacts" / "sweep_analysis" / "tables"


RUNS = [
    RunSpec(
        key="R0",
        run_id="baseline-rollout320-full-001",
        branch="R0_baseline_rollout320",
        label="R0 baseline",
        num_generations=2,
        max_steps=3364,
        reward_mode="baseline",
    ),
    RunSpec(
        key="R1",
        run_id="r1-format-rollout320-full-001",
        branch="R1_format_reward_rollout320",
        label="R1 format-aware reward-only",
        num_generations=2,
        max_steps=3364,
        reward_mode="gsm8k_verifiable_format",
    ),
    RunSpec(
        key="R2",
        run_id="r2-k8-beta004-rollout320-full-001",
        branch="R2_k8_beta004_rollout320",
        label="R2 baseline K=8, beta=0.04",
        num_generations=8,
        max_steps=841,
        reward_mode="baseline",
    ),
    RunSpec(
        key="R3",
        run_id="r3-loo-advantage-rollout320-full-001",
        branch="R3_loo_advantage_rollout320",
        label="R3 leave-one-out advantage",
        num_generations=2,
        max_steps=3364,
        reward_mode="baseline",
    ),
    RunSpec(
        key="R4",
        run_id="r4-r12-format-rollout320-lr3e6-001",
        branch="R4_r12_format_lr3e-6_rollout320",
        label="R4 format-aware K=8, lr=3e-6",
        num_generations=8,
        max_steps=841,
        reward_mode="gsm8k_verifiable_format",
    ),
    RunSpec(
        key="R5",
        run_id="r5-lora-r16-rollout320-full-001",
        branch="R5_lora_r16_rollout320",
        label="R5 LoRA rank16 baseline",
        num_generations=2,
        max_steps=3364,
        reward_mode="baseline",
    ),
    RunSpec(
        key="R6",
        run_id="r6-lora-r32-rollout320-full-002",
        branch="R6_lora_r32_rollout320",
        label="R6 LoRA rank32 baseline",
        num_generations=2,
        max_steps=3364,
        reward_mode="baseline",
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.075, 0.965, title, ha="left", va="top", fontsize=13, fontweight="semibold")
    fig.text(0.075, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def style_axes(ax: plt.Axes, percent: bool = False) -> None:
    ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(True, axis="x", color=TOKENS["grid"], linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    if percent:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def save_figure(fig: plt.Figure, stem: str, manifest: list[dict[str, str]], title: str, note: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    svg = FIG_DIR / f"{stem}.svg"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    strip_trailing_whitespace(svg)
    plt.close(fig)
    manifest.append(
        {
            "figure": stem,
            "title": title,
            "png": str(png.relative_to(OUT_DIR)),
            "pdf": str(pdf.relative_to(OUT_DIR)),
            "svg": str(svg.relative_to(OUT_DIR)),
            "note": note,
        }
    )


def strip_trailing_whitespace(path: Path) -> None:
    if path.suffix.lower() != ".svg":
        return
    text = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8")


def wilson_interval_pct(rate_pct: float, total: int = 64, z: float = 1.96) -> tuple[float, float]:
    p = max(0.0, min(1.0, rate_pct / 100.0))
    n = max(1, int(total))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
    return max(0.0, (center - margin) * 100.0), min(100.0, (center + margin) * 100.0)


def load_manifest_specs() -> None:
    manifest = PACKAGE_DIR / "manifest_rollout320_official_comparison.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    seen = {line["key"]: line for line in payload["lines"]}
    for run in RUNS:
        if run.key not in seen:
            raise RuntimeError(f"Missing {run.key} from rollout package manifest")
        line = seen[run.key]
        if line["run_id"] != run.run_id or line["branch"] != run.branch:
            raise RuntimeError(f"Manifest mismatch for {run.key}")


def load_checkpoint_eval() -> pd.DataFrame:
    path = PACKAGE_DIR / "tables" / "checkpoint_eval_rollout_aligned.csv"
    df = pd.read_csv(path)
    df["line"] = df["line"].astype(str)
    selected = {run.key for run in RUNS}
    df = df[df["line"].isin(selected)].copy()
    df["line_label"] = df["line"].map({run.key: run.label for run in RUNS})
    return df


def load_scalar(run: RunSpec) -> pd.DataFrame:
    path = run.sweep_tables / "scalar_pivot.csv"
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[(df["step"].notna()) & (df["step"] >= 1) & (df["step"] <= run.max_steps)].copy()
    df["step"] = df["step"].astype(int)
    df["line"] = run.key
    df["line_label"] = run.label
    df["num_generations"] = run.num_generations
    df["rollouts_seen"] = df["step"] * run.num_generations
    df = normalize_reward_scores_for_report(df, run)
    return df.sort_values(["line", "step"])


def load_scalars() -> dict[str, pd.DataFrame]:
    return {run.key: load_scalar(run) for run in RUNS}


def metric_series(df: pd.DataFrame, metric: str, run: RunSpec, percent: bool = False) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=["rollouts_seen", "value", "smooth"])
    out = df[["rollouts_seen", metric]].copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna()
    if out.empty:
        return pd.DataFrame(columns=["rollouts_seen", "value", "smooth"])
    out = out.groupby("rollouts_seen", as_index=False)[metric].mean()
    out = out.sort_values("rollouts_seen")
    values = out[metric] * (100.0 if percent else 1.0)
    window = max(5, round(320 / run.num_generations))
    min_periods = max(3, window // 5)
    out["value"] = values
    out["smooth"] = values.rolling(window=window, min_periods=min_periods, center=True).mean()
    out["smooth"] = out["smooth"].fillna(values.rolling(window=max(3, min_periods), min_periods=1).mean())
    return out[["rollouts_seen", "value", "smooth"]]


def reward_native_max(run: RunSpec) -> float:
    try:
        return REWARD_SCALE_MAX[run.reward_mode]
    except KeyError as exc:
        raise KeyError(f"Unknown reward mode for scaling: {run.reward_mode}") from exc


def normalize_reward_scores_for_report(df: pd.DataFrame, run: RunSpec) -> pd.DataFrame:
    """Rewrite report-facing reward score columns to the shared baseline 0-10 scale."""
    out = df.copy()
    native_max = reward_native_max(run)
    out["reward_mode"] = run.reward_mode
    out["reward_native_max"] = native_max
    out["reward_score_scale"] = "baseline_0_10"
    for prefix in ("train", "eval"):
        source = f"{prefix}_reward_score"
        if source not in out.columns:
            continue
        raw = pd.to_numeric(out[source], errors="coerce")
        out[f"{prefix}_reward_score_native"] = raw
        out[source] = raw / native_max * BASELINE_REWARD_SCALE
        out[f"{prefix}_reward_score_native_max"] = native_max
        out[f"{prefix}_reward_score_native_pct"] = raw / native_max * 100.0
        out[f"{prefix}_reward_native_max"] = native_max
    return out


def plot_raw_and_smooth(ax: plt.Axes, data: pd.DataFrame, run: RunSpec, label: str | None = None) -> None:
    color = COLORS[run.key]
    ax.plot(data["rollouts_seen"], data["value"], color=color, alpha=0.10, linewidth=0.75)
    ax.plot(data["rollouts_seen"], data["smooth"], color=color, alpha=0.98, linewidth=1.9, label=label or run.label)


def plot_dual_trace(
    ax: plt.Axes,
    data: pd.DataFrame,
    color: str,
    label: str,
    raw_alpha: float = 0.13,
    smooth_width: float = 2.2,
) -> None:
    ax.plot(data["rollouts_seen"], data["value"], color=color, alpha=raw_alpha, linewidth=0.65)
    ax.plot(data["rollouts_seen"], data["smooth"], color=color, alpha=0.98, linewidth=smooth_width, label=label)


def metric_value_at(data: pd.DataFrame, idx: int) -> tuple[float, int]:
    row = data.loc[idx]
    return float(row["smooth"]), int(row["rollouts_seen"])


def annotate_peak_latest(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    prefix: str = "",
    percent: bool = False,
    value_format: str = "{:.2f}",
    xy: tuple[float, float] = (0.98, 0.92),
) -> None:
    if data.empty or data["smooth"].dropna().empty:
        return
    valid = data.dropna(subset=["smooth"]).copy()
    peak_idx = valid["smooth"].idxmax()
    peak_value, peak_rollouts = metric_value_at(valid, peak_idx)
    latest = valid.iloc[-1]
    latest_value = float(latest["smooth"])
    latest_rollouts = int(latest["rollouts_seen"])
    if percent:
        peak_text = f"{peak_value:.1f}%"
        latest_text = f"{latest_value:.1f}%"
    else:
        peak_text = value_format.format(peak_value)
        latest_text = value_format.format(latest_value)
    label = f"{prefix} " if prefix else ""
    text = f"{label}peak   {peak_text} @ {peak_rollouts:,}\n{label}latest {latest_text} @ {latest_rollouts:,}"
    ax.scatter(
        [peak_rollouts],
        [peak_value],
        s=36,
        color=TOKENS["ink"],
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
    )
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color=TOKENS["muted"],
        family="DejaVu Sans Mono",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 2.5},
    )


def scaled_reward_series(df: pd.DataFrame, run: RunSpec) -> pd.DataFrame:
    return metric_series(df, "train_reward_score_native_pct", run)


def aligned_reward_series(df: pd.DataFrame, run: RunSpec) -> pd.DataFrame:
    return metric_series(df, "train_reward_score", run)


def figure_run_health_panel(run: RunSpec, df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    title = f"{run.key} rollout health and answer quality"
    subtitle = "Training-batch diagnostics on the shared rollout axis; annotations use smoothed values."
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True)
    add_header(fig, title, subtitle)
    specs = [
        ("train_official_numeric_exact_rate", "Numeric exact rate", COLORS["exact"], (0.97, 0.88)),
        ("train_format_accuracy", "Format accuracy", COLORS["format"], (0.97, 0.88)),
        ("train_rollout_empty_response_rate", "Empty response rate", COLORS["empty"], (0.97, 0.88)),
        ("train_rollout_extracted_none_rate", "No extracted answer rate", COLORS["no_answer"], (0.97, 0.88)),
    ]
    for ax, (metric, label, color, xy) in zip(axes.ravel(), specs):
        data = metric_series(df, metric, run, percent=True)
        if not data.empty:
            plot_dual_trace(ax, data, color, label)
            annotate_peak_latest(ax, data, percent=True, xy=xy)
        style_axes(ax, percent=True)
        ax.set_title(label, loc="left", fontsize=12, fontweight="semibold", pad=6)
        ax.set_xlim(0, 6800)
        ax.set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Generated rollouts")
    axes[1, 1].set_xlabel("Generated rollouts")
    axes[0, 0].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Rate")
    fig.subplots_adjust(top=0.84, hspace=0.33, wspace=0.16)
    save_figure(
        fig,
        f"per_run/{run.key}_health_answer_quality",
        manifest,
        title,
        "Per-run numeric exact, format, empty response, and no-answer rates.",
    )


def figure_all_health_panel(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "All selected runs: answer quality and format health"
    subtitle = "Smoothed training diagnostics aligned by generated rollouts."
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharex=True)
    add_header(fig, title, subtitle)
    specs = [
        ("train_official_numeric_exact_rate", "Numeric exact rate"),
        ("train_format_accuracy", "Format accuracy"),
        ("train_rollout_empty_response_rate", "Empty response rate"),
        ("train_rollout_extracted_none_rate", "No extracted answer rate"),
    ]
    for ax, (metric, label) in zip(axes.ravel(), specs):
        for run in RUNS:
            data = metric_series(scalars[run.key], metric, run, percent=True)
            if not data.empty:
                ax.plot(data["rollouts_seen"], data["smooth"], color=COLORS[run.key], linewidth=1.9, label=run.label)
        style_axes(ax, percent=True)
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold", pad=6)
        ax.set_xlim(0, 6800)
        ax.set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Generated rollouts")
    axes[1, 1].set_xlabel("Generated rollouts")
    axes[0, 0].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Rate")
    axes[0, 0].legend(loc="best", frameon=True, facecolor="white", edgecolor=TOKENS["axis"])
    fig.subplots_adjust(top=0.85, hspace=0.30, wspace=0.16)
    save_figure(
        fig,
        "01c_all_runs_health_answer_quality",
        manifest,
        title,
        "All selected runs shown together for answer-quality diagnostics.",
    )


def figure_run_reward_kl(run: RunSpec, df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    title = f"{run.key} reward and KL over GRPO training"
    subtitle = "Report reward columns are rewritten to the shared baseline 0-10 scale; solid lines are rollout-aligned moving averages."
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True)
    add_header(fig, title, subtitle)

    reward_specs = [
        ("train_reward_score", "train reward", COLORS["reward"], 0.12),
        ("eval_reward_score", "eval reward", COLORS["eval_reward"], 0.18),
    ]
    for metric, label, color, alpha in reward_specs:
        data = metric_series(df, metric, run)
        if not data.empty:
            plot_dual_trace(axes[0], data, color, label, raw_alpha=alpha)
            annotate_peak_latest(
                axes[0],
                data,
                prefix="train" if "train" in label else "eval",
                value_format="{:.2f}",
                xy=(0.98, 0.88 if "train" in label else 0.62),
            )
    axes[0].set_title("Reward score", loc="left", fontsize=12, fontweight="semibold", pad=6)
    axes[0].set_ylabel("Reward score (baseline 0-10)")
    axes[0].axhline(0, color=TOKENS["axis"], linewidth=0.8)
    style_axes(axes[0])
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"], ncol=2)

    kl_specs = [
        ("train_kl", "train KL", COLORS["kl"], 0.12),
        ("eval_kl", "eval KL", COLORS["eval_kl"], 0.18),
    ]
    for metric, label, color, alpha in kl_specs:
        data = metric_series(df, metric, run)
        if not data.empty:
            plot_dual_trace(axes[1], data, color, label, raw_alpha=alpha)
            annotate_peak_latest(
                axes[1],
                data,
                prefix="train" if "train" in label else "eval",
                value_format="{:.3f}",
                xy=(0.98, 0.88 if "train" in label else 0.62),
            )
    axes[1].set_title("KL to reference", loc="left", fontsize=12, fontweight="semibold", pad=6)
    axes[1].set_ylabel("KL")
    axes[1].set_xlabel("Generated rollouts")
    axes[1].axhline(0, color=TOKENS["axis"], linewidth=0.8)
    style_axes(axes[1])
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"], ncol=2)
    for ax in axes:
        ax.set_xlim(0, 6800)
    fig.subplots_adjust(top=0.86, hspace=0.28)
    save_figure(
        fig,
        f"per_run/{run.key}_reward_and_kl",
        manifest,
        title,
        "Per-run total reward and KL with train/eval traces where available.",
    )


def figure_reward_kl(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "All selected runs: training reward and KL"
    subtitle = "Reward score is rewritten to the shared baseline 0-10 scale; faint traces are step values and solid lines are moving averages."
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    add_header(fig, title, subtitle)
    for ax, metric, ylabel in [
        (axes[0], "train_reward_score", "Mean total reward (baseline 0-10)"),
        (axes[1], "train_kl", "KL to reference"),
    ]:
        for run in RUNS:
            data = metric_series(scalars[run.key], metric, run)
            if not data.empty:
                plot_raw_and_smooth(ax, data, run)
        style_axes(ax)
        ax.set_xlabel("Generated rollouts")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=TOKENS["axis"], linewidth=0.8)
        ax.set_xlim(0, 6800)
    axes[1].set_ylim(-0.05, 1.65)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"], ncol=1)
    fig.subplots_adjust(top=0.82, wspace=0.22)
    save_figure(fig, "01_training_reward_and_kl", manifest, title, "Raw and smoothed training reward/KL traces.")


def figure_scaled_reward(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "Training reward on the unified report scale"
    subtitle = "Same report-facing reward field used in source CSVs: baseline native reward remains unchanged; other modes are rewritten onto 0-10."
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    add_header(fig, title, subtitle)
    for run in RUNS:
        data = aligned_reward_series(scalars[run.key], run)
        if not data.empty:
            ax.plot(data["rollouts_seen"], data["smooth"], color=COLORS[run.key], linewidth=1.9, label=run.label)
    style_axes(ax)
    ax.set_xlabel("Generated rollouts")
    ax.set_ylabel("Aligned reward (baseline 0-10)")
    ax.set_xlim(0, 6800)
    ax.axhline(0, color=TOKENS["axis"], linewidth=0.8)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"], ncol=1)
    fig.subplots_adjust(top=0.80)
    save_figure(fig, "01b_training_reward_aligned_baseline_0_10", manifest, title, "Report-facing reward score on the shared baseline 0-10 reward range.")


def plot_checkpoint_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    show_legend: bool = False,
    show_ci: bool = True,
) -> None:
    for run in RUNS:
        sub = df[df["line"] == run.key].sort_values("rollouts_seen")
        if sub.empty:
            continue
        y = sub[metric].astype(float)
        if show_ci:
            lows, highs = [], []
            for _, row in sub.iterrows():
                if metric == "accuracy" and not pd.isna(row.get("accuracy_ci95_low")):
                    low, high = float(row["accuracy_ci95_low"]), float(row["accuracy_ci95_high"])
                else:
                    low, high = wilson_interval_pct(float(row[metric]), int(row.get("total", 64)))
                lows.append(float(row[metric]) - low)
                highs.append(high - float(row[metric]))
            ax.errorbar(
                sub["rollouts_seen"],
                y,
                yerr=np.vstack([lows, highs]),
                fmt="o-",
                color=COLORS[run.key],
                ecolor=COLORS[run.key],
                elinewidth=1.0,
                capsize=3,
                markersize=4.2,
                linewidth=1.7,
                alpha=0.92,
                label=run.label,
            )
        else:
            ax.plot(
                sub["rollouts_seen"],
                y,
                "o-",
                color=COLORS[run.key],
                markersize=4.0,
                linewidth=1.7,
                label=run.label,
            )
        best_idx = y.idxmax()
        best = sub.loc[best_idx]
        final = sub.sort_values("rollouts_seen").iloc[-1]
        ax.scatter(
            [best["rollouts_seen"]],
            [best[metric]],
            marker="*",
            s=160,
            facecolor=COLORS[run.key],
            edgecolor=TOKENS["ink"],
            linewidth=0.9,
            zorder=6,
        )
        ax.scatter(
            [final["rollouts_seen"]],
            [final[metric]],
            marker="s",
            s=58,
            facecolor=COLORS[run.key],
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
            zorder=6,
        )
    ax.axhline(50, color=TOKENS["ink"], linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.75)
    ax.text(60, 50.8, "50% reference", fontsize=8.5, color=TOKENS["muted"], va="bottom")
    style_axes(ax, percent=True)
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=6)
    ax.set_xlabel("Generated rollouts")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_xlim(0, 6800)
    ax.set_ylim(0, 76)
    if show_legend:
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"], ncol=3)


def figure_checkpoint_exact(df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    title = "Held-out exact accuracy"
    subtitle = "Checkpoint evaluations on 64 held-out GSM8K items; error bars show Wilson 95% intervals, stars mark best checkpoints, squares mark final checkpoints."
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    add_header(fig, title, subtitle)
    plot_checkpoint_metric(ax, df, "accuracy", "Exact-match accuracy", show_legend=True, show_ci=True)
    fig.subplots_adjust(top=0.78)
    save_figure(fig, "02_checkpoint_exact_accuracy_ci", manifest, title, "Exact held-out accuracy with uncertainty and best/final markers.")


def figure_checkpoint_panel(df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    title = "Checkpoint evaluation panel"
    subtitle = "Exact, partial, and format accuracy share the same generated-rollout axis."
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 8.2), sharex=True)
    add_header(fig, title, subtitle)
    specs = [
        ("accuracy", "Exact accuracy"),
        ("partial_accuracy", "Partial numeric accuracy"),
        ("format_accuracy", "Format accuracy"),
    ]
    for idx, (metric, label) in enumerate(specs):
        plot_checkpoint_metric(axes[idx], df, metric, label, show_legend=(idx == 0), show_ci=(metric != "format_accuracy"))
        if idx < 2:
            axes[idx].set_xlabel("")
    fig.subplots_adjust(top=0.88, hspace=0.25)
    save_figure(fig, "03_checkpoint_eval_three_panel", manifest, title, "Three checkpoint metrics aligned by generated rollouts.")


def figure_training_quality(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "Training batch answer-quality rates"
    subtitle = "Rollout-aligned moving averages from training scalars; rates are batch-level diagnostics, not held-out eval."
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharex=True, sharey=True)
    add_header(fig, title, subtitle)
    metrics = [
        ("train_official_numeric_exact_rate", "Exact numeric rate"),
        ("train_numeric_partial_rate", "Partial numeric rate"),
        ("train_format_accuracy", "Format rate"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for run in RUNS:
            data = metric_series(scalars[run.key], metric, run, percent=True)
            if not data.empty:
                ax.plot(data["rollouts_seen"], data["smooth"], color=COLORS[run.key], linewidth=1.9, label=run.label)
        style_axes(ax, percent=True)
        ax.set_title(ylabel, loc="left", fontsize=11, fontweight="semibold", pad=6)
        ax.set_xlabel("Generated rollouts")
        ax.set_xlim(0, 6800)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Training rate (%)")
    axes[0].legend(loc="lower left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"])
    fig.subplots_adjust(top=0.80, wspace=0.17)
    save_figure(fig, "04_training_answer_quality_rates", manifest, title, "Training exact/partial/format rates as smoothed trajectories.")


def figure_reward_components(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "Reward component trajectories"
    subtitle = "Left: selected R4 verifiable components. Right: R0/R2 baseline components. Values are moving averages on each native reward scale."
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    add_header(fig, title, subtitle)

    by_key = {run.key: run for run in RUNS}
    simple = [
        ("train_rewards_gsm8k_simple_numeric", "numeric", "-"),
        ("train_rewards_gsm8k_simple_format", "format", "--"),
        ("train_rewards_reasoning_structure_format", "reasoning format", ":"),
    ]
    for run in [by_key[key] for key in ["R4"] if key in by_key]:
        for metric, comp, ls in simple:
            data = metric_series(scalars[run.key], metric, run)
            if not data.empty:
                axes[0].plot(
                    data["rollouts_seen"],
                    data["smooth"],
                    color=COLORS[run.key],
                    linewidth=1.8,
                    linestyle=ls,
                    label=f"{run.key} {comp}",
                )
    axes[0].set_title("Verifiable reward components", loc="left", fontsize=11, fontweight="semibold", pad=6)
    axes[0].set_ylabel("Component reward")

    baseline_components = [
        ("train_rewards_check_answer", "answer", "-"),
        ("train_rewards_check_numbers", "numbers", ":"),
        ("train_rewards_match_format_approximately", "format approx.", "--"),
        ("train_rewards_match_format_exactly", "format exact", "-."),
    ]
    for run in [by_key["R0"], by_key["R2"]]:
        for metric, comp, ls in baseline_components:
            data = metric_series(scalars[run.key], metric, run)
            if not data.empty:
                axes[1].plot(
                    data["rollouts_seen"],
                    data["smooth"],
                    color=COLORS[run.key],
                    linewidth=1.55,
                    linestyle=ls,
                    label=f"{run.key} {comp}",
                )
    axes[1].set_title("Baseline reward components", loc="left", fontsize=11, fontweight="semibold", pad=6)
    axes[1].set_ylabel("Component reward")

    for idx, ax in enumerate(axes):
        style_axes(ax)
        ax.set_xlabel("Generated rollouts")
        ax.set_xlim(0, 6800)
        ax.axhline(0, color=TOKENS["axis"], linewidth=0.8)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=True,
            facecolor="white",
            edgecolor=TOKENS["axis"],
            ncol=3 if idx == 0 else 2,
        )
    fig.subplots_adjust(top=0.78, bottom=0.28, wspace=0.22)
    save_figure(fig, "05_reward_components", manifest, title, "Reward terms separated by compatible reward mechanism.")


def figure_grpo_diagnostics(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "GRPO training diagnostics"
    subtitle = "Moving averages of variance and rollout health diagnostics on the common generated-rollout axis."
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), sharex=True)
    add_header(fig, title, subtitle)
    metrics = [
        ("train_grpo_frac_reward_zero_std", "Zero-reward-std groups", True),
        ("train_grpo_group_reward_std_mean", "Group reward std.", False),
        ("train_rollout_extracted_none_rate", "Extraction failure rate", True),
        ("train_rollout_overlong_rate_1600", "Overlong response rate", True),
    ]
    for ax, (metric, label, percent) in zip(axes.ravel(), metrics):
        for run in RUNS:
            data = metric_series(scalars[run.key], metric, run, percent=percent)
            if not data.empty:
                ax.plot(data["rollouts_seen"], data["smooth"], color=COLORS[run.key], linewidth=1.8, label=run.label)
        style_axes(ax, percent=percent)
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold", pad=6)
        ax.set_xlabel("Generated rollouts")
        ax.set_xlim(0, 6800)
        if percent:
            ax.set_ylim(-3, 103)
    axes[0, 0].legend(loc="best", frameon=True, facecolor="white", edgecolor=TOKENS["axis"])
    fig.subplots_adjust(top=0.86, hspace=0.30, wspace=0.18)
    save_figure(fig, "06_grpo_training_diagnostics", manifest, title, "Core GRPO variance and rollout health diagnostics.")


def figure_optimization(scalars: dict[str, pd.DataFrame], manifest: list[dict[str, str]]) -> None:
    title = "Optimization traces"
    subtitle = "Smoothed training loss, gradient norm, and actor perplexity; rollout axis is aligned across K=2 and K=8 runs."
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharex=True)
    add_header(fig, title, subtitle)
    metrics = [
        ("train_loss", "Loss"),
        ("train_actor_grad_norm", "Gradient norm"),
        ("train_actor_perplexity", "Actor perplexity"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        for run in RUNS:
            data = metric_series(scalars[run.key], metric, run)
            if not data.empty:
                ax.plot(data["rollouts_seen"], data["smooth"], color=COLORS[run.key], linewidth=1.8, label=run.label)
        style_axes(ax)
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold", pad=6)
        ax.set_xlabel("Generated rollouts")
        ax.set_xlim(0, 6800)
    axes[0].legend(loc="best", frameon=True, facecolor="white", edgecolor=TOKENS["axis"])
    fig.subplots_adjust(top=0.80, wspace=0.18)
    save_figure(fig, "07_optimization_traces", manifest, title, "Optimization scalar trajectories.")


def figure_best_final_bars(df: pd.DataFrame, manifest: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        sub = df[df["line"] == run.key].sort_values("rollouts_seen")
        if sub.empty:
            continue
        best_exact = sub.loc[sub["accuracy"].idxmax()]
        best_partial = sub.loc[sub["partial_accuracy"].idxmax()]
        final = sub.iloc[-1]
        rows.append(
            {
                "line": run.key,
                "label": run.label,
                "best_exact": best_exact["accuracy"],
                "best_exact_rollouts": best_exact["rollouts_seen"],
                "best_partial": best_partial["partial_accuracy"],
                "best_partial_rollouts": best_partial["rollouts_seen"],
                "final_exact": final["accuracy"],
                "final_partial": final["partial_accuracy"],
                "final_format": final["format_accuracy"],
            }
        )
    summary = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "best_final_checkpoint_summary.csv", index=False)

    title = "Best and final checkpoint summary"
    subtitle = "Best checkpoints are selected per metric; final checkpoints are the 6,728-rollout endpoints."
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    add_header(fig, title, subtitle)
    x = np.arange(len(summary))
    width = 0.34
    for ax, left_col, right_col, panel_title in [
        (axes[0], "best_exact", "final_exact", "Exact accuracy"),
        (axes[1], "best_partial", "final_partial", "Partial numeric accuracy"),
    ]:
        colors = [COLORS[k] for k in summary["line"]]
        ax.bar(x - width / 2, summary[left_col], width, color=colors, alpha=0.82, edgecolor=TOKENS["ink"], linewidth=0.6, label="Best")
        ax.bar(x + width / 2, summary[right_col], width, color=colors, alpha=0.28, edgecolor=TOKENS["ink"], linewidth=0.8, hatch="//", label="Final")
        for i, val in enumerate(summary[left_col]):
            ax.text(i - width / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color=TOKENS["ink"])
        for i, val in enumerate(summary[right_col]):
            ax.text(i + width / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color=TOKENS["muted"])
        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="semibold", pad=6)
        ax.set_ylim(0, 78)
        ax.set_ylabel("Held-out accuracy (%)")
        style_axes(ax, percent=True)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["line"].tolist())
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor=TOKENS["axis"])
    fig.subplots_adjust(top=0.78, wspace=0.18)
    save_figure(fig, "08_best_final_checkpoint_summary", manifest, title, "Compact best-versus-final checkpoint comparison.")
    return summary


def figure_contact_sheet(manifest_rows: list[dict[str, str]]) -> None:
    png_paths = [OUT_DIR / row["png"] for row in manifest_rows if row["figure"] != "00_contact_sheet"]
    if not png_paths:
        return
    thumbs = []
    for path in png_paths:
        img = plt.imread(path)
        thumbs.append((path.stem, img))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 3.4)))
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax, (name, img) in zip(axes, thumbs):
        ax.imshow(img)
        ax.set_title(name, fontsize=9, color=TOKENS["ink"], loc="left")
        ax.axis("off")
    for ax in axes[len(thumbs) :]:
        ax.axis("off")
    fig.tight_layout(pad=1.2)
    png = FIG_DIR / "00_contact_sheet.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    manifest_rows.insert(
        0,
        {
            "figure": "00_contact_sheet",
            "title": "Contact sheet",
            "png": str(png.relative_to(OUT_DIR)),
            "pdf": "",
            "svg": "",
            "note": "Quick visual index of all generated report figures.",
        },
    )


def copy_tree_contents(src: Path, dst: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    if not src.exists():
        return copied
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "source": str(path.relative_to(REPO)),
                "package_path": str(target.relative_to(OUT_DIR)),
                "size_bytes": str(path.stat().st_size),
            }
        )
    return copied


def copy_file_if_exists(src: Path, dst: Path) -> dict[str, str] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src.relative_to(REPO)),
        "package_path": str(dst.relative_to(OUT_DIR)),
        "size_bytes": str(src.stat().st_size),
    }


def write_aligned_scalar_bundle(run: RunSpec, dst_dir: Path) -> list[dict[str, str]]:
    source = run.sweep_tables / "scalar_pivot.csv"
    copied: list[dict[str, str]] = []
    if not source.is_file():
        return copied

    raw_target = dst_dir / "scalar_pivot_raw.csv"
    aligned_target = dst_dir / "scalar_pivot.csv"
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_target)
    copied.append(
        {
            "source": str(source.relative_to(REPO)),
            "package_path": str(raw_target.relative_to(OUT_DIR)),
            "size_bytes": str(source.stat().st_size),
            "role": "raw_tensorboard_derived_scalar_pivot",
        }
    )

    aligned = pd.read_csv(source)
    aligned["step"] = pd.to_numeric(aligned["step"], errors="coerce")
    aligned = aligned[aligned["step"].notna()].copy()
    aligned["step"] = aligned["step"].astype(int)
    aligned["line"] = run.key
    aligned["line_label"] = run.label
    aligned["run_id"] = run.run_id
    aligned["branch"] = run.branch
    aligned["num_generations"] = run.num_generations
    aligned["rollouts_seen"] = aligned["step"] * run.num_generations
    aligned = normalize_reward_scores_for_report(aligned, run)
    aligned.to_csv(aligned_target, index=False)
    copied.append(
        {
            "source": str(source.relative_to(REPO)),
            "package_path": str(aligned_target.relative_to(OUT_DIR)),
            "size_bytes": str(aligned_target.stat().st_size),
            "role": "report_primary_scalar_pivot_reward_scores_rewritten_to_baseline_0_10",
        }
    )
    return copied


def collect_omitted_large_sources(run: RunSpec) -> list[dict[str, str]]:
    patterns = [
        "runs/*/tensorboard/events.out.tfevents*",
        "runs/*/tensorboard/observability/events.out.tfevents*",
        "runs/*/artifacts/analysis/scalar_metrics.csv",
        "runs/*/artifacts/analysis/scalar_metrics.json",
        "artifacts/sweep_analysis/tables/scalar_long.csv",
        "artifacts/sweep_analysis/tables/trace_rows_flat.csv",
        "runs/*/checkpoints/*.tar.gz",
        "runs/*/checkpoints/*",
    ]
    rows: list[dict[str, str]] = []
    for pattern in patterns:
        for path in run.cloud_dir.glob(pattern):
            if path.is_file():
                row = {
                    "run": run.key,
                    "source": str(path.relative_to(REPO)),
                    "size_mb": f"{path.stat().st_size / 1024 / 1024:.2f}",
                    "reason": "omitted from Git package because it is large; local raw artifact remains under artifacts/cloud",
                }
                if "checkpoints" not in path.parts:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    row["sha256"] = digest.hexdigest()
                rows.append(row)
    return rows


def copy_web_readable_sources() -> None:
    data_manifest: dict[str, object] = {
        "purpose": "Web-readable GRPO report data bundle. Contains plots plus compact TensorBoard-derived CSV/JSON tables.",
        "selected_runs": [run.key for run in RUNS],
        "rollout_axis": "rollouts_seen = step * num_generations",
        "reward_alignment": {
            "target": "baseline_0_10",
            "formula": REWARD_ALIGNMENT_FORMULA,
            "report_primary_reward_columns_are_rewritten": True,
            "raw_native_values_are_retained_as": [
                "train_reward_score_native",
                "eval_reward_score_native",
                "tensorboard_derived/scalar_pivot_raw.csv",
            ],
            "native_mechanisms": REWARD_MECHANISMS,
        },
        "included_files": [],
        "omitted_large_local_sources": [],
    }
    included: list[dict[str, str]] = []
    omitted: list[dict[str, str]] = []

    for run in RUNS:
        run_dir = OUT_DIR / "data" / run.key
        included.extend(write_aligned_scalar_bundle(run, run_dir / "tensorboard_derived"))
        included_file = copy_file_if_exists(
            run.sweep_tables / "trace_audit_by_call.csv",
            run_dir / "trace" / "trace_audit_by_call.csv",
        )
        if included_file:
            included.append(included_file)

        analysis = run.cloud_dir / "runs" / run.branch / "artifacts" / "analysis"
        for name in [
            "trace_summary.csv",
            "trace_summary.json",
            "trace_rows_sample.json",
            "tensorboard_tags.json",
        ]:
            included_file = copy_file_if_exists(analysis / name, run_dir / "analysis" / name)
            if included_file:
                included.append(included_file)

        for src, rel in [
            (run.cloud_dir / "artifacts" / "reward_k8_pilot_manifest.json", "manifest/reward_k8_pilot_manifest.json"),
            (run.cloud_dir / "runs" / run.branch / "artifacts" / "run_manifest.json", "manifest/run_manifest.json"),
            (run.cloud_dir / "runs" / run.branch / "run_env.txt", "manifest/run_env.txt"),
        ]:
            included_file = copy_file_if_exists(src, run_dir / rel)
            if included_file:
                included.append(included_file)

        included.extend(
            copy_tree_contents(
                run.cloud_dir / "runs" / run.branch / "artifacts" / "checkpoint_eval",
                run_dir / "checkpoint_eval",
            )
        )
        omitted.extend(collect_omitted_large_sources(run))

    data_manifest["included_files"] = included
    data_manifest["omitted_large_local_sources"] = omitted
    (OUT_DIR / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    (OUT_DIR / "reward_alignment_manifest.json").write_text(
        json.dumps(data_manifest["reward_alignment"], indent=2),
        encoding="utf-8",
    )


def write_manifest(manifest_rows: list[dict[str, str]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["figure", "title", "png", "pdf", "svg", "note"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme_lines = [
        "# GRPO rollout320 report figures",
        "",
        "Clean report-ready static figures and compact web-readable data for the selected rollout-aligned comparison runs.",
        "",
        "Lines:",
        "- R0 baseline: K=2, 3,364 steps, 6,728 rollouts.",
        "- R1 format-aware reward-only: K=2, 3,364 steps, 6,728 rollouts.",
        "- R2 baseline K=8: 841 steps, beta=0.04, lr=3e-6, 6,728 rollouts.",
        "- R3 leave-one-out advantage: K=2, 3,364 steps, 6,728 rollouts.",
        "- R4 selected format-aware run: K=8, beta=0.04, lr=3e-6, 841 steps, 6,728 rollouts.",
        "- R5 LoRA rank16 baseline: K=2, 3,364 steps, rank/alpha=16/16, 6,728 rollouts.",
        "- R6 LoRA rank32 baseline: K=2, 3,364 steps, rank/alpha=32/32, 6,728 rollouts.",
        "",
        "All trend plots use `rollouts_seen = step * num_generations` on the x-axis.",
        "Report-facing `train_reward_score` and `eval_reward_score` are rewritten onto the shared baseline 0-10 reward scale.",
        "Original TensorBoard-derived native reward values are retained as `*_reward_score_native` and in each run's `data/<line>/tensorboard_derived/scalar_pivot_raw.csv`.",
        "Checkpoint plots contain exactly the 22 official rollout-aligned eval points.",
        "Compact TensorBoard-derived scalar tables, checkpoint evals, run manifests, run env files, and trace summaries are under `data/`.",
        "Very large local raw sources such as TensorBoard event files, scalar_metrics JSON, flat trace rows, and checkpoint archives are listed in `data_manifest.json` but omitted from Git.",
        "",
        "Figures:",
    ]
    for row in manifest_rows:
        readme_lines.append(f"- `{row['png']}` - {row['title']}. {row['note']}")
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    load_manifest_specs()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint_eval()
    scalars = load_scalars()
    copy_web_readable_sources()

    manifest_rows: list[dict[str, str]] = []
    figure_reward_kl(scalars, manifest_rows)
    figure_scaled_reward(scalars, manifest_rows)
    figure_all_health_panel(scalars, manifest_rows)
    for run in RUNS:
        figure_run_reward_kl(run, scalars[run.key], manifest_rows)
        figure_run_health_panel(run, scalars[run.key], manifest_rows)
    figure_checkpoint_exact(checkpoint, manifest_rows)
    figure_checkpoint_panel(checkpoint, manifest_rows)
    figure_training_quality(scalars, manifest_rows)
    figure_reward_components(scalars, manifest_rows)
    figure_grpo_diagnostics(scalars, manifest_rows)
    figure_optimization(scalars, manifest_rows)
    summary = figure_best_final_bars(checkpoint, manifest_rows)
    checkpoint.to_csv(TABLE_DIR / "checkpoint_eval_rollout_aligned.csv", index=False)
    figure_contact_sheet(manifest_rows)
    write_manifest(manifest_rows)

    archive_base = OUT_DIR.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", OUT_DIR)
    print(f"Wrote report figures to {OUT_DIR}")
    print(f"Wrote zip to {archive_base}.zip")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
