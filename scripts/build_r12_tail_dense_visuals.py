"""Build global-step visual supplements for the retained R12 tail512 winner."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


WINNER_RUN = "R12_tail_lr1e-6_beta004_from512"
CANONICAL_RUN = "R12_gsm8k_verifiable_simple"
SOURCE_STEP = 512
FINAL_GLOBAL_STEP = 841

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "gold": "#B8A037",
    "olive": "#71B436",
    "orange": "#CC6F47",
    "pink": "#BD569B",
    "neutral": "#7A828F",
}

STALE_TAIL_OUTPUTS = (
    "figures/dense/01_checkpoint_eval_tail_step.png",
    "figures/dense/02_dense_scalar_performance_tail_step.png",
    "figures/dense/03_dense_reward_components_tail_step.png",
    "figures/dense/04_dense_grpo_response_health_tail_step.png",
    "tables/winner_dense_scalar_grid_32.csv",
)


def apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.labelsize": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "font.family": ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"],
            "legend.fontsize": 8,
            "text.color": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "xtick.labelsize": 8,
            "ytick.color": TOKENS["muted"],
            "ytick.labelsize": 8,
        }
    )


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def as_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def read_scalar_series(path: Path, run_id: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_id") != run_id:
                continue
            step = as_int(row.get("step"))
            value = as_float(row.get("value"))
            metric = row.get("metric") or row.get("tag")
            if step is None or value is None or not metric:
                continue
            series[metric].append((step, value))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return dict(series)


def read_checkpoint_rows(path: Path, run_id: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_id") != run_id:
                continue
            step = as_int(row.get("step"))
            if step is None:
                continue
            parsed: dict[str, float | int | str] = {"step": step}
            for key in (
                "accuracy",
                "partial_accuracy",
                "format_accuracy",
                "robust_numeric_exact_rate",
                "correct",
                "total",
            ):
                value = as_float(row.get(key))
                if value is not None:
                    parsed[key] = value
            rows.append(parsed)
    rows.sort(key=lambda item: int(item["step"]))
    return rows


def merge_winner_path(
    canonical_series: dict[str, list[tuple[int, float]]],
    tail_series: dict[str, list[tuple[int, float]]],
) -> dict[str, list[tuple[int, float]]]:
    """Canonical through source checkpoint, then retained winner continuation."""
    metrics = set(canonical_series) | set(tail_series)
    merged: dict[str, list[tuple[int, float]]] = {}
    for metric in metrics:
        values = [(x, y) for x, y in canonical_series.get(metric, []) if x <= SOURCE_STEP]
        values.extend((x, y) for x, y in tail_series.get(metric, []) if x > SOURCE_STEP)
        values.sort(key=lambda item: item[0])
        merged[metric] = values
    return merged


def merge_checkpoint_rows(
    canonical_rows: list[dict[str, float | int | str]],
    tail_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    merged: list[dict[str, float | int | str]] = []
    for row in canonical_rows:
        if int(row["step"]) <= SOURCE_STEP:
            out = dict(row)
            out["segment"] = "canonical_source"
            merged.append(out)
    for row in tail_rows:
        if int(row["step"]) > SOURCE_STEP:
            out = dict(row)
            out["segment"] = "tail_winner"
            merged.append(out)
    merged.sort(key=lambda item: int(item["step"]))
    return merged


def rolling_xy(values: list[tuple[int, float]], window: int) -> tuple[list[int], list[float]]:
    if not values:
        return [], []
    xs: list[int] = []
    ys: list[float] = []
    buf: list[float] = []
    for x, y in values:
        buf.append(y)
        if len(buf) > window:
            buf.pop(0)
        xs.append(x)
        ys.append(sum(buf) / len(buf))
    return xs, ys


def setup_global_axis(ax) -> None:
    ax.set_xlim(0, FINAL_GLOBAL_STEP)
    ax.set_xticks([0, 128, 256, 384, 512, 640, 768, 841])
    ax.set_xticks([64, 192, 320, 448, 576, 704, 832], minor=True)
    ax.grid(True, which="major", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color=TOKENS["grid"], linewidth=0.45, alpha=0.55)
    ax.axvline(SOURCE_STEP, color=TOKENS["neutral"], linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvline(FINAL_GLOBAL_STEP, color=TOKENS["ink"], linestyle="--", linewidth=1.0, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_series(
    ax,
    series: dict[str, list[tuple[int, float]]],
    metric: str,
    label: str,
    color: str,
    rolling: int = 32,
    scale: float = 1.0,
    raw: bool = True,
) -> None:
    values = series.get(metric, [])
    if not values:
        ax.text(0.02, 0.5, f"missing {metric}", transform=ax.transAxes, va="center", fontsize=9)
        return
    xs = [x for x, _ in values]
    ys = [y * scale for _, y in values]
    if raw and len(values) > 20:
        ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.2)
    elif raw:
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.2, alpha=0.55)
    rx, ry = rolling_xy([(x, y * scale) for x, y in values], rolling)
    ax.plot(rx, ry, color=color, linewidth=2.1, label=label)


def write_checkpoint_csv(rows: list[dict[str, float | int | str]], out: Path) -> None:
    fields = [
        "step",
        "segment",
        "accuracy",
        "partial_accuracy",
        "format_accuracy",
        "robust_numeric_exact_rate",
        "correct",
        "total",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_checkpoint_plot(rows: list[dict[str, float | int | str]], out: Path) -> None:
    xs = [float(row["step"]) for row in rows]
    exact = [float(row.get("accuracy", 0.0)) for row in rows]
    partial = [float(row.get("partial_accuracy", 0.0)) for row in rows]
    robust = [float(row.get("robust_numeric_exact_rate", 0.0)) * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.plot(xs, exact, marker="o", linewidth=2.3, color=TOKENS["blue"], label="checkpoint exact")
    ax.plot(xs, partial, marker="s", linewidth=2.1, color=TOKENS["gold"], label="checkpoint partial")
    ax.plot(xs, robust, marker="^", linewidth=1.8, color=TOKENS["olive"], label="robust exact")
    for x, y in zip(xs, exact):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    setup_global_axis(ax)
    ax.annotate(
        "tail continuation starts at 512",
        (SOURCE_STEP, 51.0),
        textcoords="offset points",
        xytext=(6, 6),
        ha="left",
        fontsize=8,
        color=TOKENS["neutral"],
    )
    if rows:
        final = rows[-1]
        ax.annotate(
            "global 841",
            (float(final["step"]), float(final.get("accuracy", 0.0))),
            textcoords="offset points",
            xytext=(-8, -30),
            ha="right",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_ylim(48, 72)
    ax.set_title("R12 retained winner checkpoint eval, global step 0-841")
    ax.set_xlabel("global training step")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.0,
        -0.18,
        "Path uses canonical R12 full through checkpoint 512, then the retained tail512 winner through checkpoint 841. Only real saved/evaluated checkpoints are plotted.",
        transform=ax.transAxes,
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_accuracy_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=180, sharex=True)
    specs = [
        ("train_numeric_exact_rate", "train exact scalar", TOKENS["blue"]),
        ("train_numeric_partial_rate", "train partial scalar", TOKENS["gold"]),
        ("train_reward_score", "train reward score", TOKENS["olive"]),
        ("eval_reward_score", "eval reward score", TOKENS["pink"]),
    ]
    for ax, (metric, label, color) in zip(axes, specs):
        scale = 100.0 if "rate" in metric else 1.0
        plot_series(ax, series, metric, label, color, rolling=32, scale=scale)
        setup_global_axis(ax)
        ax.set_ylabel("%" if "rate" in metric else "score")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 retained winner dense scalar performance, global step 0-841")
    axes[-1].set_xlabel("global training step")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_reward_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=180, sharex=True)
    specs = [
        ("train_reward_score", "total reward score", TOKENS["blue"]),
        ("train_reward_gsm8k_simple_numeric_mean", "numeric reward component", TOKENS["olive"]),
        ("train_reward_gsm8k_simple_format_mean", "format reward component", TOKENS["gold"]),
        ("train_rewards_sum", "reward sum", TOKENS["pink"]),
    ]
    for ax, (metric, label, color) in zip(axes, specs):
        plot_series(ax, series, metric, label, color, rolling=32)
        setup_global_axis(ax)
        ax.set_ylabel("value")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 retained winner reward and active components, global step 0-841")
    axes[-1].set_xlabel("global training step")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_health_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), dpi=180, sharex=True)
    specs = [
        ("train_kl", "train KL", TOKENS["blue"], 1.0),
        ("train_loss", "train loss", TOKENS["orange"], 1.0),
        ("train_grpo_frac_reward_zero_std", "zero reward std rate", TOKENS["pink"], 100.0),
        ("train_rollout_extracted_none_rate", "extracted-none rate", TOKENS["gold"], 100.0),
        ("train_rollout_answer_single_number_rate", "single-number answer rate", TOKENS["olive"], 100.0),
    ]
    for ax, (metric, label, color, scale) in zip(axes, specs):
        plot_series(ax, series, metric, label, color, rolling=32, scale=scale)
        setup_global_axis(ax)
        ax.set_ylabel("%" if scale == 100.0 else "value")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 retained winner dense GRPO and response health, global step 0-841")
    axes[-1].set_xlabel("global training step")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_32_step_grid(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    metrics = [
        "train_numeric_exact_rate",
        "train_numeric_partial_rate",
        "train_reward_score",
        "train_reward_gsm8k_simple_numeric_mean",
        "train_reward_gsm8k_simple_format_mean",
        "train_kl",
        "train_loss",
        "train_grpo_frac_reward_zero_std",
        "train_rollout_extracted_none_rate",
        "train_rollout_answer_single_number_rate",
    ]
    steps = list(range(0, FINAL_GLOBAL_STEP, 32)) + [FINAL_GLOBAL_STEP]
    rows: list[dict[str, str | int | float]] = []
    for step in steps:
        row: dict[str, str | int | float] = {
            "global_step": step,
            "segment": "canonical_source" if step <= SOURCE_STEP else "tail_winner",
        }
        for metric in metrics:
            values = series.get(metric, [])
            if not values:
                row[metric] = ""
                row[f"{metric}_source_global_step"] = ""
                continue
            nearest = min(values, key=lambda item: abs(item[0] - step))
            row[f"{metric}_source_global_step"] = nearest[0]
            row[metric] = nearest[1]
        rows.append(row)
    fieldnames: list[str] = ["global_step", "segment"]
    for metric in metrics:
        fieldnames.extend([metric, f"{metric}_source_global_step"])
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_stale_tail_outputs(evidence_dir: Path) -> None:
    for rel in STALE_TAIL_OUTPUTS:
        path = evidence_dir / rel
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-clean-dir",
        type=Path,
        default=Path("artifacts/reports/reward-k8-beta004-r12-full-001-clean"),
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("artifacts/reports/r12-full-autotune-tail512-001-clean"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("artifacts/reports/r12-full-autotune-evidence-tail512-001"))
    parser.add_argument("--run-id", default=WINNER_RUN)
    parser.add_argument("--canonical-run-id", default=CANONICAL_RUN)
    args = parser.parse_args()

    apply_chart_style()

    canonical_scalar_path = args.canonical_clean_dir / "tables" / "scalar_long.csv"
    canonical_checkpoint_path = args.canonical_clean_dir / "tables" / "checkpoint_eval_long.csv"
    tail_scalar_path = args.clean_dir / "tables" / "scalar_long.csv"
    tail_checkpoint_path = args.clean_dir / "tables" / "checkpoint_eval_long.csv"
    out_dir = args.evidence_dir / "figures" / "global"
    table_dir = args.evidence_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_tail_outputs(args.evidence_dir)

    canonical_series = read_scalar_series(canonical_scalar_path, args.canonical_run_id)
    tail_series = read_scalar_series(tail_scalar_path, args.run_id)
    winner_global_series = merge_winner_path(canonical_series, tail_series)

    canonical_checkpoints = read_checkpoint_rows(canonical_checkpoint_path, args.canonical_run_id)
    tail_checkpoints = read_checkpoint_rows(tail_checkpoint_path, args.run_id)
    winner_global_checkpoints = merge_checkpoint_rows(canonical_checkpoints, tail_checkpoints)

    write_checkpoint_csv(winner_global_checkpoints, table_dir / "winner_global_checkpoint_eval.csv")
    write_checkpoint_plot(winner_global_checkpoints, out_dir / "01_checkpoint_eval_global_step.png")
    write_accuracy_plot(winner_global_series, out_dir / "02_dense_scalar_performance_global_step.png")
    write_reward_plot(winner_global_series, out_dir / "03_dense_reward_components_global_step.png")
    write_health_plot(winner_global_series, out_dir / "04_dense_grpo_response_health_global_step.png")
    write_32_step_grid(winner_global_series, table_dir / "winner_global_scalar_grid_32.csv")

    print(f"Wrote global-step visuals to {out_dir}")
    print(f"Wrote global 32-step grid to {table_dir / 'winner_global_scalar_grid_32.csv'}")


if __name__ == "__main__":
    main()
