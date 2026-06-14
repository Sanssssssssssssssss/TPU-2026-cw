"""Build lightweight live preview plots for the active R4 format-aware run."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional in this repo env
    sns = None


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "r4-r12-format-rollout320-lr3e6-001"
BRANCH = "R4_r12_format_lr3e-6_rollout320"
RUN_ROOT = ROOT / "artifacts" / "cloud" / RUN_ID
RUN_DIR = RUN_ROOT / "runs" / BRANCH
TRACE_DIR = RUN_DIR / "artifacts" / "rollout_traces"
OUT_DIR = ROOT / "artifacts" / "reports" / "r4-format-lr3e6-live-preview-001"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": "#5477C4",
    "blue_light": "#A3BEFA",
    "orange": "#CC6F47",
    "olive": "#71B436",
    "pink": "#BD569B",
    "neutral": "#7A828F",
    "neutral_dark": "#464C55",
}


def role_label(raw: object) -> str:
    text = str(raw or "unknown")
    if "train" in text and "val" not in text:
        return "train"
    if "val" in text and "train" not in text:
        return "val"
    return text


def fnum(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_trace_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                comps = row.get("reward_components") or {}
                rows.append(
                    {
                        "trace_call_index": int(row.get("call_index") or 0),
                        "role": role_label(row.get("dataset_role")),
                        "reward_total": fnum(row.get("reward_total")),
                        "reward_recomputed": fnum(row.get("reward_total_recomputed")),
                        "numeric": fnum(comps.get("gsm8k_simple_numeric")),
                        "answer_format": fnum(comps.get("gsm8k_simple_format")),
                        "reasoning_format": fnum(comps.get("reasoning_structure_format")),
                        "format_ok": 1.0 if row.get("format_ok") else 0.0,
                        "numeric_exact": 1.0 if row.get("numeric_exact") else 0.0,
                        "robust_numeric_exact": 1.0 if row.get("robust_numeric_exact") else 0.0,
                        "extracted_none": 1.0 if row.get("extracted_number") in (None, "") else 0.0,
                    }
                )
    return rows


def run_axis_metadata() -> tuple[int, int, int]:
    manifest_path = RUN_ROOT / "artifacts" / "reward_k8_pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_generations = int(manifest.get("num_generations") or 8)
    max_steps = int(manifest.get("max_steps") or 841)
    steps: list[int] = []
    checkpoint_list = RUN_ROOT / "checkpoint_archives.txt"
    if checkpoint_list.exists():
        for line in checkpoint_list.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                step = int(parts[1])
            except ValueError:
                continue
            if 1 < step <= max_steps:
                steps.append(step)
    latest_step = max(steps) if steps else max_steps
    return latest_step, num_generations, latest_step * num_generations


def read_checkpoint_eval_rows() -> list[dict[str, float]]:
    path = RUN_DIR / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    _, num_generations, _ = run_axis_metadata()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            try:
                step = int(float(raw.get("step") or 0))
            except (TypeError, ValueError):
                continue
            item = {"step": float(step), "rollouts_seen": float(step * num_generations)}
            for key in ("accuracy", "partial_accuracy", "format_accuracy", "robust_accuracy"):
                try:
                    item[key] = float(raw.get(key) or 0.0)
                except (TypeError, ValueError):
                    item[key] = 0.0
            rows.append(item)
    return rows


def aggregate_by_call(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["role"]), int(row["trace_call_index"]))].append(row)
    out: list[dict[str, object]] = []
    metrics = [
        "reward_total",
        "reward_recomputed",
        "numeric",
        "answer_format",
        "reasoning_format",
        "format_ok",
        "numeric_exact",
        "robust_numeric_exact",
        "extracted_none",
    ]
    for (role, trace_call), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        row: dict[str, object] = {
            "role": role,
            "trace_call_index": trace_call,
            "rollouts_seen": trace_call,
            "n": len(items),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric) is not None]
            row[metric] = mean(vals) if vals else None
        out.append(row)
    return out


def align_to_latest_checkpoint(rows: list[dict[str, object]], latest_rollouts: int) -> list[dict[str, object]]:
    if not rows:
        return rows
    max_call = max(int(row["trace_call_index"]) for row in rows)
    if max_call <= 0:
        return rows
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["rollouts_seen"] = float(item["trace_call_index"]) / max_call * latest_rollouts
        out.append(item)
    return out


def smooth_rows(rows: list[dict[str, object]], window: int = 96) -> list[dict[str, object]]:
    by_role: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_role[str(row["role"])].append(row)
    metrics = [key for key in rows[0] if key not in {"role", "rollouts_seen", "trace_call_index", "n"}] if rows else []
    smoothed: list[dict[str, object]] = []
    for role, role_rows in by_role.items():
        role_rows.sort(key=lambda row: int(row["rollouts_seen"]))
        half = max(1, window // 2)
        for idx, row in enumerate(role_rows):
            start = max(0, idx - half)
            end = min(len(role_rows), idx + half + 1)
            chunk = role_rows[start:end]
            out = {
                "role": role,
                "trace_call_index": row["trace_call_index"],
                "rollouts_seen": row["rollouts_seen"],
                "n": row["n"],
            }
            for metric in metrics:
                vals = [float(item[metric]) for item in chunk if item.get(metric) is not None]
                out[metric] = mean(vals) if vals else None
            smoothed.append(out)
    return smoothed


def select(rows: list[dict[str, object]], role: str, metric: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in sorted(rows, key=lambda item: float(item["rollouts_seen"])):
        if row["role"] != role or row.get(metric) is None:
            continue
        xs.append(float(row["rollouts_seen"]))
        ys.append(float(row[metric]))
    return xs, ys


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.titleweight": "bold",
        }
    )


def add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=14, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def finish_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.grid(True, axis="y", alpha=0.9)
    ax.grid(True, axis="x", alpha=0.35)


def plot_combined(rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    add_header(
        fig,
        "R4 format-aware training: reward and format are rising together",
        "Smoothed live traces; x-axis is aligned approximately to official rollouts through the latest fetched checkpoint.",
    )

    ax = axes[0]
    for role, color, alpha, style in (("val", COLORS["blue"], 1.0, "-"), ("train", COLORS["neutral"], 0.55, "--")):
        xs, ys = select(rows, role, "reward_total")
        ax.plot(xs, ys, style, color=color, alpha=alpha, label=f"{role} total reward")
    xs, ys = select(rows, "val", "numeric")
    ax.plot(xs, ys, color=COLORS["orange"], linewidth=1.7, label="val numeric reward")
    xs, ys = select(rows, "val", "reasoning_format")
    ax.plot(xs, ys, color=COLORS["olive"], linewidth=1.7, label="val reasoning format reward")
    finish_axes(ax, "", "Reward")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)

    ax = axes[1]
    for metric, label, color, style in (
        ("format_ok", "strict format ok", COLORS["olive"], "-"),
        ("numeric_exact", "numeric exact proxy", COLORS["blue"], "-"),
        ("extracted_none", "extraction failure", COLORS["pink"], "--"),
    ):
        xs, ys = select(rows, "val", metric)
        ax.plot(xs, ys, style, color=color, label=f"val {label}")
    finish_axes(ax, "Generated rollouts", "Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.92)
    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.89])
    out = OUT_DIR / "figures" / "01_r4_live_reward_format_accuracy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_components(rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    add_header(
        fig,
        "R4 format-aware reward components",
        "Validation trace components; x-axis is an approximate live alignment, not raw trace call count.",
    )
    for metric, label, color in (
        ("reward_total", "total reward", COLORS["blue"]),
        ("numeric", "numeric reward", COLORS["orange"]),
        ("answer_format", "answer-tag helper", COLORS["neutral"]),
        ("reasoning_format", "reasoning-structure reward", COLORS["olive"]),
    ):
        xs, ys = select(rows, "val", metric)
        ax.plot(xs, ys, color=color, label=label, linewidth=1.8)
    finish_axes(ax, "Generated rollouts", "Component reward")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.86])
    out = OUT_DIR / "figures" / "02_r4_live_reward_components.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_rates(rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    add_header(
        fig,
        "R4 format-aware trace rates",
        "Smoothed validation traces; numeric exact is a training-trace proxy, not held-out checkpoint accuracy.",
    )
    for metric, label, color, style in (
        ("format_ok", "strict format ok", COLORS["olive"], "-"),
        ("numeric_exact", "numeric exact proxy", COLORS["blue"], "-"),
        ("robust_numeric_exact", "robust numeric exact", COLORS["orange"], "--"),
        ("extracted_none", "extraction failure", COLORS["pink"], ":"),
    ):
        xs, ys = select(rows, "val", metric)
        ax.plot(xs, ys, style, color=color, label=label, linewidth=1.8)
    finish_axes(ax, "Generated rollouts", "Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.86])
    out = OUT_DIR / "figures" / "03_r4_live_format_accuracy_proxy_rates.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_checkpoint_eval(rows: list[dict[str, float]]) -> Path | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: row["rollouts_seen"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    add_header(
        fig,
        "R4 format-aware checkpoint eval",
        "Held-out GSM8K checkpoint evaluation at the official 320-rollout grid.",
    )
    for metric, label, color, style in (
        ("accuracy", "exact", COLORS["blue"], "-"),
        ("partial_accuracy", "partial numeric", COLORS["orange"], "-"),
        ("format_accuracy", "strict format", COLORS["olive"], "-"),
    ):
        ax.plot(
            [row["rollouts_seen"] for row in rows],
            [row[metric] for row in rows],
            style,
            marker="o",
            markersize=3.8,
            color=color,
            linewidth=1.8,
            label=label,
        )
    finish_axes(ax, "Generated rollouts", "Held-out accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.92)
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.86])
    out = OUT_DIR / "figures" / "04_r4_checkpoint_eval_accuracy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_tables(raw_rows: list[dict[str, object]], agg_rows: list[dict[str, object]], smooth: list[dict[str, object]]) -> None:
    tables = OUT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (tables / "r4_live_trace_rows_sample.csv", raw_rows[:2000]),
        (tables / "r4_live_trace_by_call.csv", agg_rows),
        (tables / "r4_live_trace_smoothed.csv", smooth),
    ):
        if not rows:
            continue
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def write_readme(paths: list[Path], smooth: list[dict[str, object]]) -> None:
    latest_step, num_generations, latest_rollouts = run_axis_metadata()
    val_rows = [row for row in smooth if row["role"] == "val"]
    latest = max(val_rows, key=lambda row: float(row["rollouts_seen"])) if val_rows else {}
    early = min(val_rows, key=lambda row: float(row["rollouts_seen"])) if val_rows else {}
    lines = [
        "# R4 format-aware live preview",
        "",
        f"Run: `{RUN_ID}` / `{BRANCH}`",
        "",
        "These are lightweight live trace plots generated without downloading checkpoint archives.",
        f"The x-axis is an approximate live-trace alignment to the latest fetched checkpoint: step {latest_step} x K={num_generations} = {latest_rollouts} official rollouts.",
        "The raw observability `call_index` is not an official rollout count. The numeric-exact rate is a rollout-trace proxy; it is not the held-out checkpoint eval accuracy.",
        "",
        "## Latest smoothed validation snapshot",
        "",
    ]
    for metric in ("reward_total", "numeric", "reasoning_format", "format_ok", "numeric_exact", "extracted_none"):
        if metric in latest and metric in early:
            lines.append(f"- `{metric}`: early {float(early[metric]):.4f} -> latest {float(latest[metric]):.4f}")
    lines += ["", "## Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    raw_rows = read_trace_rows()
    if not raw_rows:
        raise SystemExit(f"No trace rows found under {TRACE_DIR}")
    latest_step, _num_generations, latest_rollouts = run_axis_metadata()
    agg_rows = align_to_latest_checkpoint(aggregate_by_call(raw_rows), latest_rollouts)
    smooth = smooth_rows(agg_rows, window=96)
    write_tables(raw_rows, agg_rows, smooth)
    paths = [plot_combined(smooth), plot_components(smooth), plot_rates(smooth)]
    checkpoint_plot = plot_checkpoint_eval(read_checkpoint_eval_rows())
    if checkpoint_plot is not None:
        paths.append(checkpoint_plot)
    write_readme(paths, smooth)
    print(f"Wrote live preview to {OUT_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
