"""Build dense visual supplements for the retained R12 tail512 winner."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


WINNER_RUN = "R12_tail_lr1e-6_beta004_from512"
SOURCE_STEP = 512


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
            tail_step = step - SOURCE_STEP
            if tail_step < 0:
                continue
            series[metric].append((tail_step, value))
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
            parsed: dict[str, float | int | str] = {"step": step, "tail_step": step - SOURCE_STEP}
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
    rows.sort(key=lambda item: int(item["tail_step"]))
    return rows


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


def setup_tail_axis(ax, xmax: int = 352) -> None:
    ax.set_xlim(-4, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(64))
    ax.xaxis.set_minor_locator(MultipleLocator(32))
    ax.grid(True, which="major", color="#e2e2e2", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#efefef", linewidth=0.5)
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
        ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.22)
    elif raw:
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.2, alpha=0.55)
    rx, ry = rolling_xy([(x, y * scale) for x, y in values], rolling)
    ax.plot(rx, ry, color=color, linewidth=2.1, label=label)


def write_checkpoint_plot(rows: list[dict[str, float | int | str]], out: Path) -> None:
    xs = [float(row["tail_step"]) for row in rows]
    exact = [float(row.get("accuracy", 0.0)) for row in rows]
    partial = [float(row.get("partial_accuracy", 0.0)) for row in rows]
    robust = [float(row.get("robust_numeric_exact_rate", 0.0)) * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.plot(xs, exact, marker="o", linewidth=2.3, color="#1f77b4", label="checkpoint exact")
    ax.plot(xs, partial, marker="s", linewidth=2.1, color="#ff7f0e", label="checkpoint partial")
    ax.plot(xs, robust, marker="^", linewidth=1.8, color="#2ca02c", label="robust exact")
    for x, y in zip(xs, exact):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    setup_tail_axis(ax)
    ax.set_ylim(50, 72)
    ax.set_title("R12 tail512 winner checkpoint eval, source-normalized")
    ax.set_xlabel("tail step from source checkpoint 512")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.0,
        -0.18,
        "Only saved/restorable checkpoints are plotted. Minor grid lines mark 32-step spacing; checkpoint eval exists at 0, 64, 128, 192, 256, and 329.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_accuracy_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=180, sharex=True)
    specs = [
        ("train_numeric_exact_rate", "train exact scalar", "#1f77b4"),
        ("train_numeric_partial_rate", "train partial scalar", "#ff7f0e"),
        ("train_reward_score", "train reward score", "#2ca02c"),
        ("eval_reward_score", "eval reward score", "#9467bd"),
    ]
    for ax, (metric, label, color) in zip(axes, specs):
        scale = 100.0 if "rate" in metric else 1.0
        plot_series(ax, series, metric, label, color, rolling=32, scale=scale)
        setup_tail_axis(ax)
        ax.set_ylabel("%" if "rate" in metric else "score")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 tail512 winner dense scalar performance, source-normalized")
    axes[-1].set_xlabel("tail step from source checkpoint 512")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_reward_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=180, sharex=True)
    specs = [
        ("train_reward_score", "total reward score", "#1f77b4"),
        ("train_reward_gsm8k_simple_numeric_mean", "numeric reward component", "#2ca02c"),
        ("train_reward_gsm8k_simple_format_mean", "format reward component", "#ff7f0e"),
        ("train_rewards_sum", "reward sum", "#9467bd"),
    ]
    for ax, (metric, label, color) in zip(axes, specs):
        plot_series(ax, series, metric, label, color, rolling=32)
        setup_tail_axis(ax)
        ax.set_ylabel("value")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 tail512 winner reward and active components")
    axes[-1].set_xlabel("tail step from source checkpoint 512")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_health_plot(series: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), dpi=180, sharex=True)
    specs = [
        ("train_kl", "train KL", "#1f77b4", 1.0),
        ("train_loss", "train loss", "#d62728", 1.0),
        ("train_grpo_frac_reward_zero_std", "zero reward std rate", "#9467bd", 100.0),
        ("train_rollout_extracted_none_rate", "extracted-none rate", "#ff7f0e", 100.0),
        ("train_rollout_answer_single_number_rate", "single-number answer rate", "#2ca02c", 100.0),
    ]
    for ax, (metric, label, color, scale) in zip(axes, specs):
        plot_series(ax, series, metric, label, color, rolling=32, scale=scale)
        setup_tail_axis(ax)
        ax.set_ylabel("%" if scale == 100.0 else "value")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("R12 tail512 winner dense GRPO and response health")
    axes[-1].set_xlabel("tail step from source checkpoint 512")
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
    offsets = list(range(0, 321, 32)) + [329]
    rows: list[dict[str, str | int | float]] = []
    for offset in offsets:
        row: dict[str, str | int | float] = {"tail_step": offset, "global_step": offset + SOURCE_STEP}
        for metric in metrics:
            values = series.get(metric, [])
            if not values:
                row[metric] = ""
                continue
            nearest = min(values, key=lambda item: abs(item[0] - offset))
            row[f"{metric}_source_tail_step"] = nearest[0]
            row[metric] = nearest[1]
        rows.append(row)
    fieldnames: list[str] = ["tail_step", "global_step"]
    for metric in metrics:
        fieldnames.extend([metric, f"{metric}_source_tail_step"])
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=Path("artifacts/reports/r12-full-autotune-tail512-001-clean"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("artifacts/reports/r12-full-autotune-evidence-tail512-001"))
    parser.add_argument("--run-id", default=WINNER_RUN)
    args = parser.parse_args()

    scalar_path = args.clean_dir / "tables" / "scalar_long.csv"
    checkpoint_path = args.clean_dir / "tables" / "checkpoint_eval_long.csv"
    out_dir = args.evidence_dir / "figures" / "dense"
    table_dir = args.evidence_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    series = read_scalar_series(scalar_path, args.run_id)
    checkpoint_rows = read_checkpoint_rows(checkpoint_path, args.run_id)

    write_checkpoint_plot(checkpoint_rows, out_dir / "01_checkpoint_eval_tail_step.png")
    write_accuracy_plot(series, out_dir / "02_dense_scalar_performance_tail_step.png")
    write_reward_plot(series, out_dir / "03_dense_reward_components_tail_step.png")
    write_health_plot(series, out_dir / "04_dense_grpo_response_health_tail_step.png")
    write_32_step_grid(series, table_dir / "winner_dense_scalar_grid_32.csv")

    print(f"Wrote dense visuals to {out_dir}")
    print(f"Wrote 32-step grid to {table_dir / 'winner_dense_scalar_grid_32.csv'}")


if __name__ == "__main__":
    main()
