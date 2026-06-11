"""Build plain matplotlib plots for the reward-only GRPO sweep.

The output is deliberately simple: white background, ordinary axes, thin raw
curves, and rolling means only where dense curves would otherwise be unreadable.
Raw data is not dropped; the report folder also copies the full CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]

RUNS = [
    "R1_no_approx",
    "R2_light_format_oldnum",
    "R3_numeric_primary_no_len",
    "R4_numeric_primary_len1200",
    "R5_numeric_primary_answer_only_len1200",
    "R6_numeric_dense_lastnum",
    "R7_numeric_dense_single_answer",
    "R8_numeric_dense_single_answer_short",
    "R9_closed_answer_minimal",
    "R10_numeric_guarded",
    "R11_numeric_guarded_fallback",
]

RUN_LABEL = {
    "R1_no_approx": "R1 no_approx",
    "R2_light_format_oldnum": "R2 light_format_oldnum",
    "R3_numeric_primary_no_len": "R3 numeric_primary_no_len",
    "R4_numeric_primary_len1200": "R4 numeric_primary_len1200",
    "R5_numeric_primary_answer_only_len1200": "R5 answer_only_len1200",
    "R6_numeric_dense_lastnum": "R6 numeric_dense_lastnum",
    "R7_numeric_dense_single_answer": "R7 dense_single_answer",
    "R8_numeric_dense_single_answer_short": "R8 dense_single_answer_short",
    "R9_closed_answer_minimal": "R9 closed_answer_minimal",
    "R10_numeric_guarded": "R10 numeric_guarded",
    "R11_numeric_guarded_fallback": "R11 numeric_guarded_fallback",
}

RUN_COLOR = {
    "R1_no_approx": "#1f77b4",
    "R2_light_format_oldnum": "#ff7f0e",
    "R3_numeric_primary_no_len": "#2ca02c",
    "R4_numeric_primary_len1200": "#9467bd",
    "R5_numeric_primary_answer_only_len1200": "#d62728",
    "R6_numeric_dense_lastnum": "#1f77b4",
    "R7_numeric_dense_single_answer": "#2ca02c",
    "R8_numeric_dense_single_answer_short": "#d62728",
    "R9_closed_answer_minimal": "#9467bd",
    "R10_numeric_guarded": "#17becf",
    "R11_numeric_guarded_fallback": "#bcbd22",
}


def run_label(run_id: str) -> str:
    return RUN_LABEL.get(run_id, run_id)


def run_color(run_id: str) -> str:
    if run_id in RUN_COLOR:
        return RUN_COLOR[run_id]
    for base_id in (
        "R6_numeric_dense_lastnum",
        "R7_numeric_dense_single_answer",
        "R8_numeric_dense_single_answer_short",
        "R9_closed_answer_minimal",
        "R10_numeric_guarded",
        "R11_numeric_guarded_fallback",
    ):
        if base_id in run_id:
            return RUN_COLOR[base_id]
    return DEFAULT_COLORS[abs(hash(run_id)) % len(DEFAULT_COLORS)]


def run_linestyle(run_id: str) -> str:
    return "--" if run_id.startswith("E_") else "-"


METRIC_COLOR = {
    "train_reward_score": "#1f77b4",
    "eval_reward_score": "#ff7f0e",
    "train_kl": "#1f77b4",
    "eval_kl": "#ff7f0e",
    "train_loss": "#1f77b4",
    "eval_loss": "#ff7f0e",
    "train_actor_pg_clipfrac": "#2ca02c",
    "eval_actor_pg_clipfrac": "#d62728",
    "train_grpo_reward_std": "#1f77b4",
    "eval_grpo_reward_std": "#ff7f0e",
    "train_grpo_frac_reward_zero_std": "#2ca02c",
    "eval_grpo_frac_reward_zero_std": "#d62728",
    "train_grpo_advantage_std": "#9467bd",
    "eval_grpo_advantage_std": "#8c564b",
    "rollout_empty_response_rate": "#d62728",
    "rollout_extracted_none_rate": "#ff7f0e",
    "rollout_answer_single_number_rate": "#2ca02c",
    "rollout_no_close_answer_rate": "#d62728",
    "rollout_answer_last_number_exact_rate": "#17becf",
    "rollout_fallback_number_used_rate": "#bcbd22",
    "rollout_fallback_numeric_exact_rate": "#2ca02c",
    "rollout_overlong_rate_1200": "#9467bd",
    "rollout_overlong_rate_1600": "#8c564b",
    "rollout_answer_tag_pair_rate": "#2ca02c",
    "rollout_duplicate_tag_rate": "#7f7f7f",
    "rollout_mean_completion_chars": "#1f77b4",
    "audit_reward_numeric_margin": "#1f77b4",
    "audit_reward_format_leakage": "#ff7f0e",
    "audit_reward_hacking_rate": "#d62728",
    "audit_group_misrank_rate": "#9467bd",
    "audit_dense_wrong_reward_std": "#17becf",
    "audit_fallback_guarded_wrong_reward_std": "#bcbd22",
    "reward_parser_false_negative_rate": "#d62728",
    "reward_numeric_primary_mean": "#1f77b4",
    "reward_numeric_dense_mean": "#1f77b4",
    "reward_answer_hygiene_dense_mean": "#2ca02c",
    "reward_closed_answer_minimal_mean": "#9467bd",
    "reward_numeric_guarded_mean": "#1f77b4",
    "reward_answer_hygiene_guarded_mean": "#2ca02c",
    "reward_numeric_guarded_total_mean": "#17becf",
    "reward_numeric_guarded_fallback_mean": "#1f77b4",
    "reward_answer_hygiene_fallback_mean": "#2ca02c",
    "reward_numeric_guarded_fallback_total_mean": "#bcbd22",
    "reward_format_light_mean": "#ff7f0e",
    "reward_length_penalty_1200_mean": "#d62728",
    "reward_length_penalty_short_mean": "#8c564b",
    "reward_match_format_exactly_mean": "#2ca02c",
    "reward_match_format_approximately_mean": "#9467bd",
    "reward_check_answer_mean": "#8c564b",
    "reward_check_numbers_mean": "#17becf",
    "composition_total_reward_mean": "#1f77b4",
    "composition_numeric_component_mean": "#2ca02c",
    "composition_format_component_mean": "#ff7f0e",
    "composition_length_component_mean": "#d62728",
    "composition_other_component_mean": "#7f7f7f",
    "composition_format_abs_share": "#9467bd",
}

COMPONENTS_BY_MODE = {
    "baseline": ["match_format_exactly", "match_format_approximately", "check_answer", "check_numbers"],
    "no_approx": ["match_format_exactly", "check_answer", "check_numbers"],
    "light_format_oldnum": ["format_strict_light", "answer_tag_light", "check_answer", "check_numbers"],
    "numeric_primary_no_len": ["numeric_primary", "format_strict_light", "answer_tag_light"],
    "numeric_primary_len1200": ["numeric_primary", "format_strict_light", "answer_tag_light", "length_penalty_1200"],
    "numeric_primary_answer_only_len1200": ["numeric_primary", "answer_tag_light", "length_penalty_1200"],
    "numeric_dense_lastnum": ["numeric_dense"],
    "numeric_dense_single_answer": ["numeric_dense", "answer_hygiene_dense"],
    "numeric_dense_single_answer_short": ["numeric_dense", "answer_hygiene_dense", "length_penalty_short"],
    "closed_answer_minimal": ["numeric_dense", "closed_answer_minimal"],
    "numeric_guarded": ["numeric_guarded", "answer_hygiene_guarded"],
    "numeric_guarded_fallback": ["numeric_guarded_fallback", "answer_hygiene_fallback"],
}

NUMERIC_COMPONENTS = {"check_answer", "check_numbers", "numeric_primary", "numeric_dense", "numeric_guarded", "numeric_guarded_fallback"}
FORMAT_COMPONENTS = {
    "match_format_exactly",
    "match_format_approximately",
    "format_strict_light",
    "answer_tag_light",
    "answer_hygiene_dense",
    "closed_answer_minimal",
    "answer_hygiene_guarded",
    "answer_hygiene_fallback",
}
LENGTH_COMPONENTS = {"length_penalty_1200", "length_penalty_short"}


COMBINED_METRIC_GROUPS = [
    (
        "02_reward_score.png",
        "Reward score, rolling mean",
        [
            ("train_reward_score", "train_reward_score", None),
            ("eval_reward_score", "eval_reward_score", None),
        ],
    ),
    (
        "03_kl_loss_clipfrac.png",
        "KL / loss / clip fraction, rolling mean",
        [
            ("train_kl", "train_kl", None),
            ("eval_kl", "eval_kl", None),
            ("train_loss", "train_loss", None),
            ("train_actor_pg_clipfrac", "train_actor_pg_clipfrac", (0, 1)),
        ],
    ),
    (
        "04_grpo_health.png",
        "GRPO group health, rolling mean",
        [
            ("train_grpo_reward_std", "train_grpo_reward_std", None),
            ("train_grpo_frac_reward_zero_std", "train_grpo_frac_reward_zero_std", (0, 1)),
            ("train_grpo_advantage_std", "train_grpo_advantage_std", None),
        ],
    ),
    (
        "05_response_health.png",
        "Response / parser health, rolling mean",
        [
            ("rollout_empty_response_rate", "rollout_empty_response_rate", (0, 1)),
            ("rollout_extracted_none_rate", "rollout_extracted_none_rate", (0, 1)),
            ("reward_parser_false_negative_rate", "reward_parser_false_negative_rate", (0, 1)),
            ("rollout_fallback_number_used_rate", "rollout_fallback_number_used_rate", (0, 1)),
            ("rollout_fallback_numeric_exact_rate", "rollout_fallback_numeric_exact_rate", (0, 1)),
            ("rollout_answer_single_number_rate", "rollout_answer_single_number_rate", (0, 1)),
            ("rollout_no_close_answer_rate", "rollout_no_close_answer_rate", (0, 1)),
            ("rollout_overlong_rate_1600", "rollout_overlong_rate_1600", (0, 1)),
            ("rollout_mean_completion_chars", "rollout_mean_completion_chars", None),
        ],
    ),
    (
        "06_reward_audit.png",
        "Reward audit, rolling mean",
        [
            ("audit_reward_numeric_margin", "audit_reward_numeric_margin", None),
            ("audit_reward_format_leakage", "audit_reward_format_leakage", None),
            ("audit_reward_hacking_rate", "audit_reward_hacking_rate", (0, 1)),
            ("audit_group_misrank_rate", "audit_group_misrank_rate", (0, 1)),
            ("audit_dense_wrong_reward_std", "audit_dense_wrong_reward_std", None),
            ("audit_fallback_guarded_wrong_reward_std", "audit_fallback_guarded_wrong_reward_std", None),
        ],
    ),
    (
        "07_reward_components.png",
        "Reward components, rolling mean",
        [
            ("reward_numeric_primary_mean", "reward_numeric_primary_mean", None),
            ("reward_numeric_dense_mean", "reward_numeric_dense_mean", None),
            ("reward_answer_hygiene_dense_mean", "reward_answer_hygiene_dense_mean", None),
            ("reward_closed_answer_minimal_mean", "reward_closed_answer_minimal_mean", None),
            ("reward_numeric_guarded_mean", "reward_numeric_guarded_mean", None),
            ("reward_answer_hygiene_guarded_mean", "reward_answer_hygiene_guarded_mean", None),
            ("reward_numeric_guarded_total_mean", "reward_numeric_guarded_total_mean", None),
            ("reward_numeric_guarded_fallback_mean", "reward_numeric_guarded_fallback_mean", None),
            ("reward_answer_hygiene_fallback_mean", "reward_answer_hygiene_fallback_mean", None),
            ("reward_numeric_guarded_fallback_total_mean", "reward_numeric_guarded_fallback_total_mean", None),
            ("reward_format_light_mean", "reward_format_light_mean", None),
            ("reward_length_penalty_1200_mean", "reward_length_penalty_1200_mean", None),
            ("reward_length_penalty_short_mean", "reward_length_penalty_short_mean", None),
            ("reward_match_format_exactly_mean", "reward_match_format_exactly_mean", None),
            ("reward_match_format_approximately_mean", "reward_match_format_approximately_mean", None),
        ],
    ),
]

PER_RUN_GROUPS = [
    (
        "02_reward_score_raw.png",
        "Reward score",
        [
            ("train_reward_score", "train_reward_score", None),
            ("eval_reward_score", "eval_reward_score", None),
        ],
    ),
    (
        "03_kl_loss_clipfrac_raw.png",
        "KL / loss / clip fraction",
        [
            ("train_kl", "train_kl", None),
            ("eval_kl", "eval_kl", None),
            ("train_loss", "train_loss", None),
            ("eval_loss", "eval_loss", None),
            ("train_actor_pg_clipfrac", "train_actor_pg_clipfrac", (0, 1)),
            ("eval_actor_pg_clipfrac", "eval_actor_pg_clipfrac", (0, 1)),
        ],
    ),
    (
        "04_grpo_health_raw.png",
        "GRPO group health",
        [
            ("train_grpo_reward_std", "train_grpo_reward_std", None),
            ("eval_grpo_reward_std", "eval_grpo_reward_std", None),
            ("train_grpo_frac_reward_zero_std", "train_grpo_frac_reward_zero_std", (0, 1)),
            ("eval_grpo_frac_reward_zero_std", "eval_grpo_frac_reward_zero_std", (0, 1)),
            ("train_grpo_advantage_std", "train_grpo_advantage_std", None),
            ("eval_grpo_advantage_std", "eval_grpo_advantage_std", None),
        ],
    ),
    (
        "05_response_health_raw.png",
        "Response / parser health",
        [
            ("rollout_empty_response_rate", "rollout_empty_response_rate", (0, 1)),
            ("rollout_extracted_none_rate", "rollout_extracted_none_rate", (0, 1)),
            ("reward_parser_false_negative_rate", "reward_parser_false_negative_rate", (0, 1)),
            ("rollout_fallback_number_used_rate", "rollout_fallback_number_used_rate", (0, 1)),
            ("rollout_fallback_numeric_exact_rate", "rollout_fallback_numeric_exact_rate", (0, 1)),
            ("rollout_answer_single_number_rate", "rollout_answer_single_number_rate", (0, 1)),
            ("rollout_no_close_answer_rate", "rollout_no_close_answer_rate", (0, 1)),
            ("rollout_answer_last_number_exact_rate", "rollout_answer_last_number_exact_rate", (0, 1)),
            ("rollout_answer_tag_pair_rate", "rollout_answer_tag_pair_rate", (0, 1)),
            ("rollout_duplicate_tag_rate", "rollout_duplicate_tag_rate", (0, 1)),
            ("rollout_overlong_rate_1200", "rollout_overlong_rate_1200", (0, 1)),
            ("rollout_overlong_rate_1600", "rollout_overlong_rate_1600", (0, 1)),
            ("rollout_mean_completion_chars", "rollout_mean_completion_chars", None),
        ],
    ),
    (
        "06_reward_audit_raw.png",
        "Reward audit",
        [
            ("audit_reward_numeric_margin", "audit_reward_numeric_margin", None),
            ("audit_reward_format_leakage", "audit_reward_format_leakage", None),
            ("audit_reward_hacking_rate", "audit_reward_hacking_rate", (0, 1)),
            ("audit_group_misrank_rate", "audit_group_misrank_rate", (0, 1)),
            ("audit_dense_wrong_reward_std", "audit_dense_wrong_reward_std", None),
            ("audit_fallback_guarded_wrong_reward_std", "audit_fallback_guarded_wrong_reward_std", None),
        ],
    ),
    (
        "07_reward_components_raw.png",
        "Reward components",
        [
            ("reward_numeric_primary_mean", "reward_numeric_primary_mean", None),
            ("reward_numeric_dense_mean", "reward_numeric_dense_mean", None),
            ("reward_answer_hygiene_dense_mean", "reward_answer_hygiene_dense_mean", None),
            ("reward_closed_answer_minimal_mean", "reward_closed_answer_minimal_mean", None),
            ("reward_numeric_guarded_mean", "reward_numeric_guarded_mean", None),
            ("reward_answer_hygiene_guarded_mean", "reward_answer_hygiene_guarded_mean", None),
            ("reward_numeric_guarded_total_mean", "reward_numeric_guarded_total_mean", None),
            ("reward_numeric_guarded_fallback_mean", "reward_numeric_guarded_fallback_mean", None),
            ("reward_answer_hygiene_fallback_mean", "reward_answer_hygiene_fallback_mean", None),
            ("reward_numeric_guarded_fallback_total_mean", "reward_numeric_guarded_fallback_total_mean", None),
            ("reward_format_light_mean", "reward_format_light_mean", None),
            ("reward_length_penalty_1200_mean", "reward_length_penalty_1200_mean", None),
            ("reward_length_penalty_short_mean", "reward_length_penalty_short_mean", None),
            ("reward_match_format_exactly_mean", "reward_match_format_exactly_mean", None),
            ("reward_match_format_approximately_mean", "reward_match_format_approximately_mean", None),
            ("reward_check_answer_mean", "reward_check_answer_mean", None),
            ("reward_check_numbers_mean", "reward_check_numbers_mean", None),
        ],
    ),
]

TRACE_GROUP = [
    ("numeric_exact_rate", "numeric_exact_rate", (0, 1)),
    ("robust_numeric_exact_rate", "robust_numeric_exact_rate", (0, 1)),
    ("fallback_number_used_rate", "fallback_number_used_rate", (0, 1)),
    ("fallback_numeric_exact_rate", "fallback_numeric_exact_rate", (0, 1)),
    ("parser_false_negative_rate", "parser_false_negative_rate", (0, 1)),
    ("format_accuracy", "format_accuracy", (0, 1)),
    ("answer_single_number_rate", "answer_single_number_rate", (0, 1)),
    ("no_close_answer_rate", "no_close_answer_rate", (0, 1)),
    ("dense_wrong_reward_std", "dense_wrong_reward_std", None),
    ("reward_mean", "reward_mean", None),
    ("reward_numeric_margin", "reward_numeric_margin", None),
    ("reward_hacking_rate", "reward_hacking_rate", (0, 1)),
    ("empty_response_rate", "empty_response_rate", (0, 1)),
    ("extracted_none_rate", "extracted_none_rate", (0, 1)),
    ("overlong_rate_1600", "overlong_rate_1600", (0, 1)),
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
    parser = argparse.ArgumentParser(description="Build clean reward sweep plots.")
    parser.add_argument("--input-dir", default="artifacts/cloud/reward-grid-001")
    parser.add_argument("--output-dir", default="artifacts/reports/reward-grid-001-clean")
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
            "grid.alpha": 0.8,
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


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _read_csv_if_nonempty(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fallback_per_run_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scalar_frames: list[pd.DataFrame] = []
    ckpt_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []

    runs_dir = input_dir / "runs"
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        mode_path = run_dir / "reward_mode.txt"
        reward_mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else run_id
        analysis_dir = run_dir / "artifacts" / "analysis"

        scalar_long = _read_csv_if_nonempty(analysis_dir / "scalar_metrics.csv", low_memory=False)
        if not scalar_long.empty and {"step", "metric", "value"}.issubset(scalar_long.columns):
            scalar_long["step"] = pd.to_numeric(scalar_long["step"], errors="coerce")
            scalar_long["value"] = pd.to_numeric(scalar_long["value"], errors="coerce")
            scalar_wide = (
                scalar_long.pivot_table(index="step", columns="metric", values="value", aggfunc="mean")
                .reset_index()
                .rename_axis(None, axis=1)
            )
            scalar_wide.insert(0, "reward_mode", reward_mode)
            scalar_wide.insert(0, "run_id", run_id)
            scalar_frames.append(scalar_wide)

        ckpt = _read_csv_if_nonempty(analysis_dir / "checkpoint_eval_rows.csv")
        if not ckpt.empty:
            ckpt.insert(0, "reward_mode", reward_mode)
            ckpt.insert(0, "run_id", run_id)
            ckpt_frames.append(ckpt)

        trace = _read_csv_if_nonempty(analysis_dir / "trace_summary.csv", low_memory=False)
        if not trace.empty:
            trace.insert(0, "reward_mode", reward_mode)
            trace.insert(0, "run_id", run_id)
            trace_frames.append(trace)

    scalars = pd.concat(scalar_frames, ignore_index=True, sort=False) if scalar_frames else pd.DataFrame()
    ckpt = pd.concat(ckpt_frames, ignore_index=True, sort=False) if ckpt_frames else pd.DataFrame()
    trace = pd.concat(trace_frames, ignore_index=True, sort=False) if trace_frames else pd.DataFrame()

    if not ckpt.empty:
        best_rows = []
        for (run_id, reward_mode), group in ckpt.groupby(["run_id", "reward_mode"], dropna=False):
            best = group.sort_values(["accuracy", "partial_accuracy"], ascending=False).iloc[0]
            best_rows.append(
                {
                    "run_id": run_id,
                    "reward_mode": reward_mode,
                    "best_step": best.get("step"),
                    "best_accuracy": best.get("accuracy"),
                    "best_partial_accuracy": best.get("partial_accuracy"),
                    "best_format_accuracy": best.get("format_accuracy"),
                    "best_no_close_answer_rate": best.get("no_close_answer_rate"),
                    "best_robust_numeric_exact_rate": best.get("robust_numeric_exact_rate"),
                }
            )
        selection = pd.DataFrame(best_rows)
    else:
        selection = pd.DataFrame()

    return scalars, ckpt, selection, trace


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_dir = input_dir / "artifacts" / "sweep_analysis" / "tables"
    scalars = _read_csv_if_nonempty(table_dir / "scalar_pivot.csv", low_memory=False)
    ckpt = _read_csv_if_nonempty(table_dir / "checkpoint_eval_long.csv")
    selection = _read_csv_if_nonempty(table_dir / "selection_summary.csv")
    trace = _read_csv_if_nonempty(table_dir / "trace_audit_by_call.csv", low_memory=False)

    if scalars.empty or ckpt.empty:
        scalars, ckpt, selection, trace = _fallback_per_run_tables(input_dir)
    if scalars.empty:
        raise SystemExit(f"Missing scalar data in {table_dir} and per-run analysis dirs.")
    if ckpt.empty:
        raise SystemExit(f"Missing checkpoint eval data in {table_dir} and per-run analysis dirs.")

    scalars["step"] = pd.to_numeric(scalars["step"], errors="coerce")
    ckpt = numeric_columns(
        ckpt,
        [
            "step",
            "accuracy",
            "partial_accuracy",
            "format_accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "correct",
            "total",
        ],
    )
    if not trace.empty:
        if "call_index" in trace.columns:
            trace["call_index"] = pd.to_numeric(trace["call_index"], errors="coerce")
        for column in trace.columns:
            if column not in {"run_id", "reward_mode", "dataset_role"}:
                trace[column] = pd.to_numeric(trace[column], errors="coerce")
    return scalars, ckpt, selection, trace


def load_reward_composition(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    table_dir = input_dir / "artifacts" / "sweep_analysis" / "tables"
    trace_rows_path = table_dir / "trace_rows_flat.csv"
    component_names = sorted({name for names in COMPONENTS_BY_MODE.values() for name in names})
    component_cols = [f"component_{name}" for name in component_names]
    base_cols = [
        "run_id",
        "reward_mode",
        "call_index",
        "dataset_role",
        "reward_total",
        "reward_total_recomputed",
    ]
    usecols = set(base_cols + component_cols)
    rows = _read_csv_if_nonempty(trace_rows_path, usecols=lambda column: column in usecols, low_memory=False)
    if rows.empty:
        sample_rows: list[dict[str, Any]] = []
        for sample_path in sorted((input_dir / "runs").glob("*/artifacts/analysis/trace_rows_sample.json")):
            run_id = sample_path.parents[2].name
            mode_path = sample_path.parents[2] / "reward_mode.txt"
            default_mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else run_id
            try:
                raw_rows = json.loads(sample_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for raw in raw_rows:
                flat = {
                    "run_id": run_id,
                    "reward_mode": raw.get("reward_mode") or default_mode,
                    "call_index": raw.get("call_index"),
                    "dataset_role": raw.get("dataset_role"),
                    "reward_total": raw.get("reward_total", 0.0),
                    "reward_total_recomputed": raw.get("reward_total_recomputed", 0.0),
                }
                components = raw.get("reward_components") or {}
                if isinstance(components, dict):
                    for name, value in components.items():
                        flat[f"component_{name}"] = value
                sample_rows.append(flat)
        rows = pd.DataFrame(sample_rows)
    if rows.empty:
        return pd.DataFrame()

    rows["call_index"] = pd.to_numeric(rows["call_index"], errors="coerce")
    for column in component_cols + ["reward_total", "reward_total_recomputed"]:
        if column not in rows.columns:
            rows[column] = 0.0
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)

    rows["total_reward"] = rows["reward_total_recomputed"]
    use_observed_total = rows["total_reward"].abs().eq(0) & rows["reward_total"].abs().gt(0)
    rows.loc[use_observed_total, "total_reward"] = rows.loc[use_observed_total, "reward_total"]

    for target in ("numeric_component", "format_component", "length_component", "other_component"):
        rows[target] = 0.0

    for mode, active_components in COMPONENTS_BY_MODE.items():
        mode_mask = rows["reward_mode"].astype(str).str.lower().eq(mode)
        if not mode_mask.any():
            continue
        for component in active_components:
            column = f"component_{component}"
            if component in NUMERIC_COMPONENTS:
                target = "numeric_component"
            elif component in FORMAT_COMPONENTS:
                target = "format_component"
            elif component in LENGTH_COMPONENTS:
                target = "length_component"
            else:
                target = "other_component"
            rows.loc[mode_mask, target] += rows.loc[mode_mask, column]

    rows["active_component_sum"] = (
        rows["numeric_component"]
        + rows["format_component"]
        + rows["length_component"]
        + rows["other_component"]
    )
    rows["component_sum_error_abs"] = (rows["total_reward"] - rows["active_component_sum"]).abs()
    rows["active_abs_denominator"] = (
        rows["numeric_component"].abs()
        + rows["format_component"].abs()
        + rows["length_component"].abs()
        + rows["other_component"].abs()
    )
    denominator = rows["active_abs_denominator"].replace(0, np.nan)
    rows["numeric_abs_share"] = (rows["numeric_component"].abs() / denominator).fillna(0.0)
    rows["format_abs_share"] = (rows["format_component"].abs() / denominator).fillna(0.0)
    rows["length_abs_share"] = (rows["length_component"].abs() / denominator).fillna(0.0)
    rows["other_abs_share"] = (rows["other_component"].abs() / denominator).fillna(0.0)

    agg_map = {
        "total_reward": "mean",
        "active_component_sum": "mean",
        "component_sum_error_abs": "mean",
        "numeric_component": "mean",
        "format_component": "mean",
        "length_component": "mean",
        "other_component": "mean",
        "numeric_abs_share": "mean",
        "format_abs_share": "mean",
        "length_abs_share": "mean",
        "other_abs_share": "mean",
        "active_abs_denominator": "mean",
    }
    composition = (
        rows.groupby(["run_id", "reward_mode", "call_index"], as_index=False)
        .agg(agg_map | {"dataset_role": "first", "total_reward": "mean"})
        .rename(
            columns={
                "total_reward": "total_reward_mean",
                "active_component_sum": "active_component_sum_mean",
                "component_sum_error_abs": "component_sum_error_abs_mean",
                "numeric_component": "numeric_component_mean",
                "format_component": "format_component_mean",
                "length_component": "length_component_mean",
                "other_component": "other_component_mean",
                "active_abs_denominator": "active_abs_denominator_mean",
            }
        )
        .sort_values(["run_id", "call_index"])
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    composition.to_csv(tables_dir / "reward_composition_by_call.csv", index=False)

    latest_rows = []
    for run_id, sub in composition.groupby("run_id"):
        sub = sub.dropna(subset=["call_index"]).sort_values("call_index")
        if sub.empty:
            continue
        latest_rows.append(sub.iloc[-1].to_dict())
    latest = pd.DataFrame(latest_rows)
    if not latest.empty:
        latest.to_csv(tables_dir / "reward_composition_latest.csv", index=False)
    return composition


def scalar_series(df: pd.DataFrame, run_id: str, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=["step", metric])
    sub = df.loc[df["run_id"].eq(run_id), ["step", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=["step", metric])
    if sub.empty:
        return sub
    sub = sub.groupby("step", as_index=False)[metric].mean().sort_values("step")
    return sub


def trace_series(df: pd.DataFrame, run_id: str, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=["call_index", metric])
    sub = df.loc[df["run_id"].eq(run_id), ["call_index", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=["call_index", metric])
    if sub.empty:
        return sub
    sub = sub.groupby("call_index", as_index=False)[metric].mean().sort_values("call_index")
    return sub


def composition_series(df: pd.DataFrame, run_id: str, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["call_index", metric])
    sub = df.loc[df["run_id"].eq(run_id), ["call_index", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=["call_index", metric])
    if sub.empty:
        return sub
    return sub.groupby("call_index", as_index=False)[metric].mean().sort_values("call_index")


def rolling(values: pd.Series, window: int) -> pd.Series:
    if len(values) < window * 2:
        return values
    return values.rolling(window=window, min_periods=max(4, window // 4), center=True).mean()


def tidy_axis(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None) -> None:
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both")


def save_fig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_checkpoint_eval(ckpt: pd.DataFrame, runs: list[str], out: Path) -> None:
    metrics = [
        ("accuracy", "accuracy (%)"),
        ("partial_accuracy", "partial_accuracy (%)"),
        ("format_accuracy", "format_accuracy (%)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9.8, 7.2), sharex=True)
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
                    color=run_color(run_id),
                    linestyle=run_linestyle(run_id),
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    capsize=2,
                    label=run_label(run_id),
                )
            else:
                ax.plot(
                    sub["step"],
                    sub[metric],
                    color=run_color(run_id),
                    linestyle=run_linestyle(run_id),
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    label=run_label(run_id),
                )
        tidy_axis(ax, ylabel, (0, 100))
    axes[0].set_title("Checkpoint eval")
    axes[-1].set_xlabel("checkpoint step")
    if "step" in ckpt:
        steps = sorted(pd.to_numeric(ckpt["step"], errors="coerce").dropna().unique())
        axes[-1].set_xticks(steps)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, out)


def plot_combined_group(
    scalars: pd.DataFrame,
    runs: list[str],
    out: Path,
    title: str,
    panels: list[tuple[str, str, tuple[float, float] | None]],
    window: int,
) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, max(2.2 * len(panels), 4.8)), sharex=True)
    if len(panels) == 1:
        axes = np.array([axes])
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for run_id in runs:
            sub = scalar_series(scalars, run_id, metric)
            if sub.empty:
                continue
            y = rolling(sub[metric], window)
            ax.plot(
                sub["step"],
                y,
                color=run_color(run_id),
                linestyle=run_linestyle(run_id),
                linewidth=1.2,
                label=run_label(run_id),
            )
        tidy_axis(ax, ylabel, ylim)
    axes[0].set_title(f"{title} (rolling mean, window={window})")
    axes[-1].set_xlabel("scalar step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, out)


def plot_per_run_group(
    scalars: pd.DataFrame,
    run_id: str,
    out: Path,
    title: str,
    panels: list[tuple[str, str, tuple[float, float] | None]],
    window: int,
) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, max(2.05 * len(panels), 4.8)), sharex=True)
    if len(panels) == 1:
        axes = np.array([axes])
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        sub = scalar_series(scalars, run_id, metric)
        if not sub.empty:
            color = METRIC_COLOR.get(metric, "#1f77b4")
            ax.plot(sub["step"], sub[metric], color=color, linewidth=0.45, alpha=0.32, label="raw")
            ax.plot(sub["step"], rolling(sub[metric], window), color=color, linewidth=1.2, label=f"rolling {window}")
        tidy_axis(ax, ylabel, ylim)
    axes[0].set_title(f"{run_label(run_id)}: {title}")
    axes[-1].set_xlabel("scalar step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out)


def plot_per_run_trace(
    trace: pd.DataFrame,
    run_id: str,
    out: Path,
    window: int,
) -> None:
    fig, axes = plt.subplots(len(TRACE_GROUP), 1, figsize=(10.5, 14.0), sharex=True)
    for ax, (metric, ylabel, ylim) in zip(axes, TRACE_GROUP):
        sub = trace_series(trace, run_id, metric)
        if not sub.empty:
            color = METRIC_COLOR.get(f"trace_{metric}", "#1f77b4")
            ax.plot(sub["call_index"], sub[metric], color=color, linewidth=0.45, alpha=0.28, label="raw")
            ax.plot(sub["call_index"], rolling(sub[metric], window), color=color, linewidth=1.2, label=f"rolling {window}")
        tidy_axis(ax, ylabel, ylim)
    axes[0].set_title(f"{run_label(run_id)}: trace audit by call")
    axes[-1].set_xlabel("trace call_index")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out)


def plot_combined_reward_composition(
    composition: pd.DataFrame,
    runs: list[str],
    out: Path,
    window: int,
) -> None:
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
                sub["call_index"],
                rolling(sub[metric], window),
                color=run_color(run_id),
                linestyle=run_linestyle(run_id),
                linewidth=1.2,
                label=run_label(run_id),
            )
        if metric.endswith("_component_mean") or metric == "total_reward_mean":
            ax.axhline(0, color="#444444", linewidth=0.65, alpha=0.65)
        tidy_axis(ax, ylabel, ylim)
    for ax in axes_flat[len(COMPOSITION_PANELS) :]:
        ax.axis("off")
    axes_flat[0].set_title("Reward composition by active reward mode")
    axes_flat[-2].set_xlabel("rollout trace call_index")
    axes_flat[-1].set_xlabel("rollout trace call_index")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.text(
        0.01,
        0.015,
        "Format share = mean(abs(active format/tag component) / (abs(numeric) + abs(format/tag) + abs(length) + abs(other))).",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_fig(fig, out)


def plot_per_run_reward_composition(
    composition: pd.DataFrame,
    run_id: str,
    out: Path,
    window: int,
) -> None:
    if composition.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.2), sharex=True)
    axes_flat = axes.flat
    for ax, (metric, ylabel, ylim) in zip(axes_flat, COMPOSITION_PANELS):
        sub = composition_series(composition, run_id, metric)
        if not sub.empty:
            color = METRIC_COLOR.get(f"composition_{metric}", "#1f77b4")
            ax.plot(sub["call_index"], sub[metric], color=color, linewidth=0.25, alpha=0.12, label="raw")
            ax.plot(sub["call_index"], rolling(sub[metric], window), color=color, linewidth=1.2, label=f"rolling {window}")
        if metric.endswith("_component_mean") or metric == "total_reward_mean":
            ax.axhline(0, color="#444444", linewidth=0.65, alpha=0.65)
        tidy_axis(ax, ylabel, ylim)
    for ax in axes_flat[len(COMPOSITION_PANELS) :]:
        ax.axis("off")
    axes_flat[0].set_title(f"{run_label(run_id)}: active reward composition")
    axes_flat[-2].set_xlabel("rollout trace call_index")
    axes_flat[-1].set_xlabel("rollout trace call_index")
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


def plot_run_checkpoint(ckpt: pd.DataFrame, run_id: str, out: Path) -> None:
    sub = ckpt.loc[ckpt["run_id"].eq(run_id)].sort_values("step")
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for metric, color in [
        ("accuracy", "#1f77b4"),
        ("partial_accuracy", "#2ca02c"),
        ("format_accuracy", "#ff7f0e"),
    ]:
        ax.plot(sub["step"], sub[metric], marker="o", markersize=4, linewidth=1.2, color=color, label=metric)
    tidy_axis(ax, "percent", (0, 100))
    ax.set_title(f"{run_label(run_id)}: checkpoint eval")
    ax.set_xlabel("checkpoint step")
    if not sub.empty:
        ax.set_xticks(sorted(pd.to_numeric(sub["step"], errors="coerce").dropna().unique()))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out)


def build_summary(selection: pd.DataFrame, ckpt: pd.DataFrame, out_dir: Path, runs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id in runs:
        sub = ckpt.loc[ckpt["run_id"].eq(run_id)].copy()
        best = sub.sort_values("accuracy", ascending=False).head(1)
        sel = selection.loc[selection["run_id"].eq(run_id)]
        best_row = best.iloc[0].to_dict() if not best.empty else {}
        sel_row = sel.iloc[0].to_dict() if not sel.empty else {}
        rows.append(
            {
                "run_id": run_id,
                "reward_mode": sel_row.get("reward_mode", ""),
                "best_checkpoint_step": best_row.get("step", ""),
                "best_accuracy": best_row.get("accuracy", ""),
                "best_partial_accuracy": best_row.get("partial_accuracy", ""),
                "best_format_accuracy": best_row.get("format_accuracy", ""),
                "screening_status": sel_row.get("screening_status", ""),
                "elimination_reasons": sel_row.get("elimination_reasons", ""),
                "empty_response_rate": sel_row.get("empty_response_rate", ""),
                "extracted_none_rate": sel_row.get("extracted_none_rate", ""),
                "parser_false_negative_rate": sel_row.get("parser_false_negative_rate", ""),
                "answer_single_number_rate": sel_row.get("answer_single_number_rate", ""),
                "no_close_answer_rate": sel_row.get("no_close_answer_rate", ""),
                "frac_reward_zero_std": sel_row.get("frac_reward_zero_std", ""),
                "dense_wrong_reward_std": sel_row.get("dense_wrong_reward_std", ""),
                "kl": sel_row.get("kl", ""),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tables" / "clean_selection_summary.csv", index=False)
    return summary


def copy_raw_tables(input_dir: Path, output_dir: Path) -> list[str]:
    src_dir = input_dir / "artifacts" / "sweep_analysis" / "tables"
    dst_dir = output_dir / "tables"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    
    def copy_or_keep_existing(path: Path, dst: Path) -> None:
        if dst.exists() and dst.stat().st_size == path.stat().st_size:
            return
        try:
            shutil.copy2(path, dst)
        except OSError:
            if dst.exists() and dst.stat().st_size == path.stat().st_size:
                return
            raise

    for path in src_dir.glob("*.csv"):
        dst = dst_dir / path.name
        copy_or_keep_existing(path, dst)
        copied.append(str(dst.relative_to(output_dir)))
    for path in src_dir.glob("*.json"):
        dst = dst_dir / path.name
        copy_or_keep_existing(path, dst)
        copied.append(str(dst.relative_to(output_dir)))
    return sorted(copied)


def write_readme(output_dir: Path, input_dir: Path, figures: list[str], summary: pd.DataFrame, copied: list[str], window: int) -> None:
    def clean_value(value: Any) -> Any:
        if pd.isna(value):
            return ""
        return value

    lines = [
        f"# {input_dir.name} clean plots",
        "",
        "Plain matplotlib evidence package: white background, ordinary axes, thin raw lines, and rolling means where useful.",
        "",
        "## Paths",
        "",
        f"- Raw fetched run: `{input_dir.resolve()}`",
        f"- Clean plot package: `{output_dir.resolve()}`",
        "- Combined plots: `figures/combined/`",
        "- Per-run plots: `figures/by_run/<run_id>/`",
        "- Full raw tables copied under: `tables/`",
        "",
        "## Plot policy",
        "",
        f"- Per-run timeline plots draw raw curves plus rolling mean, window={window}.",
        f"- Combined timeline plots draw rolling mean, window={window}, because all raw runs on one axis are unreadable.",
        "- No rows are dropped from the evidence package; raw full CSVs are copied into `tables/`.",
        "- Checkpoint eval plots use the actual discrete checkpoint steps present in the CSV.",
        "- Scalar x-axis is TensorBoard scalar step; trace x-axis is rollout trace `call_index`. These are intentionally not forced onto one fake scale.",
        "- Reward composition plots count only the components enabled by each run's `REWARD_MODE`; audit-only component columns are ignored.",
        "- Format share is `abs(format_or_tag_hygiene) / (abs(numeric) + abs(format_or_tag_hygiene) + abs(length) + abs(other))`.",
        "- Derived reward composition tables: `tables/reward_composition_by_call.csv`, `tables/reward_composition_latest.csv`.",
        "",
        "## Selection snapshot",
        "",
        "| run_id | best_step | acc | partial | format | status | elimination_reasons |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.to_dict("records"):
        row = {key: clean_value(value) for key, value in row.items()}
        lines.append(
            "| {run_id} | {best_checkpoint_step} | {best_accuracy:.2f} | {best_partial_accuracy:.2f} | {best_format_accuracy:.2f} | {screening_status} | {elimination_reasons} |".format(
                **row
            )
        )
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure}`")
    lines.extend(["", "## Copied tables", ""])
    for item in copied:
        lines.append(f"- `{item}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    combined_dir = figures_dir / "combined"
    by_run_dir = figures_dir / "by_run"
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    scalars, ckpt, selection, trace = load_inputs(input_dir)
    runs = sorted(
        {
            str(run_id)
            for run_id in pd.concat(
                [
                    scalars.get("run_id", pd.Series(dtype=str)),
                    ckpt.get("run_id", pd.Series(dtype=str)),
                    selection.get("run_id", pd.Series(dtype=str)),
                    trace.get("run_id", pd.Series(dtype=str)),
                ],
                ignore_index=True,
            ).dropna()
            if str(run_id) != "R0_baseline"
        }
    )
    preferred = {run_id: i for i, run_id in enumerate(RUNS)}
    runs.sort(key=lambda run_id: (preferred.get(run_id, 999), run_id))
    copied = copy_raw_tables(input_dir, output_dir)
    summary = build_summary(selection, ckpt, output_dir, runs)
    composition = load_reward_composition(input_dir, output_dir)

    figures: list[str] = []
    out = combined_dir / "01_checkpoint_eval.png"
    plot_checkpoint_eval(ckpt, runs, out)
    figures.append(str(out.relative_to(output_dir)))

    for filename, title, panels in COMBINED_METRIC_GROUPS:
        out = combined_dir / filename
        plot_combined_group(scalars, runs, out, title, panels, args.rolling_window)
        figures.append(str(out.relative_to(output_dir)))
    out = combined_dir / "08_reward_composition_format_share.png"
    plot_combined_reward_composition(composition, runs, out, args.rolling_window)
    if out.exists():
        figures.append(str(out.relative_to(output_dir)))

    for run_id in runs:
        run_fig_dir = by_run_dir / run_id
        out = run_fig_dir / "01_checkpoint_eval.png"
        plot_run_checkpoint(ckpt, run_id, out)
        figures.append(str(out.relative_to(output_dir)))
        for filename, title, panels in PER_RUN_GROUPS:
            out = run_fig_dir / filename
            plot_per_run_group(scalars, run_id, out, title, panels, args.rolling_window)
            figures.append(str(out.relative_to(output_dir)))
        out = run_fig_dir / "08_trace_audit_raw.png"
        plot_per_run_trace(trace, run_id, out, args.rolling_window)
        figures.append(str(out.relative_to(output_dir)))
        out = run_fig_dir / "09_reward_composition_format_share_raw.png"
        plot_per_run_reward_composition(composition, run_id, out, args.rolling_window)
        if out.exists():
            figures.append(str(out.relative_to(output_dir)))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "rolling_window": args.rolling_window,
        "figures": figures,
        "copied_tables": copied,
        "derived_tables": [
            "tables/reward_composition_by_call.csv",
            "tables/reward_composition_latest.csv",
        ],
        "style": "plain matplotlib, raw per-run curves plus rolling mean, combined rolling means",
    }
    (output_dir / "manifest_clean_plots.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(output_dir, input_dir, figures, summary, copied, args.rolling_window)
    print(f"Wrote {len(figures)} clean figures to {figures_dir}")
    print(f"Clean package: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
