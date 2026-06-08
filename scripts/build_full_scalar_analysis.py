"""Build no-drop scalar analysis tables and plain figures for a baseline run.

This script is deliberately separate from the presentation report generator:
it focuses on complete experiment records, not summary storytelling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import PIL.JpegImagePlugin  # noqa: F401


DEFAULT_REPORT_DIR = Path("artifacts/reports/course-baseline-001")
DEFAULT_OUT_DIR = DEFAULT_REPORT_DIR / "full_scalar_analysis"

KEY_METRICS = [
    "eval_reward_score",
    "eval_numeric_exact_rate",
    "eval_numeric_partial_rate",
    "eval_format_accuracy",
    "eval_empty_response_rate",
    "eval_extracted_none_rate",
    "eval_has_solution_end_rate",
    "eval_kl",
    "eval_loss",
    "eval_reward_std",
    "eval_frac_reward_zero_std",
    "eval_all_wrong_group_rate",
    "train_reward_score",
    "train_numeric_exact_rate",
    "train_format_accuracy",
    "train_empty_response_rate",
    "train_extracted_none_rate",
    "train_kl",
    "train_loss",
    "train_reward_std",
    "train_frac_reward_zero_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build complete scalar analysis artifacts.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def read_scalar_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "metric": row["metric"],
                        "tag": row["tag"],
                        "step": int(float(row["step"])),
                        "wall_time": float(row["wall_time"]) if row.get("wall_time") else None,
                        "value": float(row["value"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda item: (item["metric"], item["step"]))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row)
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["metric"]].append(row)
    return dict(out)


def peak_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric = grouped(rows)
    summary = []
    for metric in sorted(by_metric):
        points = by_metric[metric]
        values = [float(row["value"]) for row in points]
        max_row = max(points, key=lambda row: float(row["value"]))
        min_row = min(points, key=lambda row: float(row["value"]))
        latest = points[-1]
        peak_value = float(max_row["value"])
        latest_value = float(latest["value"])
        drop_from_peak = peak_value - latest_value
        drop_from_peak_pct = drop_from_peak / abs(peak_value) if not math.isclose(peak_value, 0.0) else None
        summary.append(
            {
                "metric": metric,
                "tag": points[0].get("tag", ""),
                "count": len(points),
                "first_step": points[0]["step"],
                "last_step": latest["step"],
                "latest_value": latest_value,
                "max_step": max_row["step"],
                "max_value": peak_value,
                "min_step": min_row["step"],
                "min_value": float(min_row["value"]),
                "drop_from_peak": drop_from_peak,
                "drop_from_peak_pct": drop_from_peak_pct,
            }
        )
    return summary


def pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = defaultdict(dict)
    metrics = sorted({row["metric"] for row in rows})
    for row in rows:
        by_step[row["step"]][row["metric"]] = row["value"]
    out = []
    for step in sorted(by_step):
        item = {"step": step}
        for metric in metrics:
            item[metric] = by_step[step].get(metric, "")
        out.append(item)
    return out


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    regular = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = Path("C:/Windows/Fonts/segoeuib.ttf")
    mono = Path("C:/Windows/Fonts/consola.ttf")
    if regular.exists():
        return (
            ImageFont.truetype(str(regular), 20),
            ImageFont.truetype(str(bold if bold.exists() else regular), 24),
            ImageFont.truetype(str(mono if mono.exists() else regular), 16),
        )
    return ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()


FONT, FONT_BOLD, FONT_MONO = load_fonts()
INK = (30, 38, 50)
MUTED = (90, 103, 120)
GRID = (220, 226, 234)
BLUE = (37, 99, 158)
ORANGE = (218, 117, 43)
RED = (172, 68, 68)
GREEN = (67, 135, 92)
PURPLE = (110, 90, 170)
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE]


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"))


def nice_range(values: list[float], include_zero: bool = False) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: dict[str, list[tuple[int, float]]],
    xlim: tuple[int, int] | None = None,
    peak_markers: dict[str, tuple[int, float]] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=GRID)
    draw.text((x0 + 10, y0 + 8), title, font=FONT, fill=INK)
    left, right = x0 + 70, x1 - 18
    top, bottom = y0 + 50, y1 - 52
    points = [(x, y) for pts in series.values() for x, y in pts if (xlim is None or xlim[0] <= x <= xlim[1])]
    if not points:
        draw.text((x0 + 25, y0 + 90), "no data", font=FONT, fill=MUTED)
        return
    xmin, xmax = xlim if xlim else (min(x for x, _ in points), max(x for x, _ in points))
    if xmin == xmax:
        xmax += 1
    ymin, ymax = nice_range([y for _, y in points], include_zero=True)
    draw.line((left, bottom, right, bottom), fill=MUTED, width=1)
    draw.line((left, top, left, bottom), fill=MUTED, width=1)
    for i in range(5):
        frac = i / 4
        y = bottom - frac * (bottom - top)
        val = ymin + frac * (ymax - ymin)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((x0 + 7, y - 8), f"{val:.3g}", font=FONT_MONO, fill=MUTED)
    for i in range(5):
        frac = i / 4
        x = left + frac * (right - left)
        val = xmin + frac * (xmax - xmin)
        draw.text((x - 18, bottom + 12), f"{int(val)}", font=FONT_MONO, fill=MUTED)

    def sx(step: float) -> float:
        return left + (step - xmin) / (xmax - xmin) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for idx, (name, pts) in enumerate(series.items()):
        filtered = [(x, y) for x, y in pts if xlim is None or xlim[0] <= x <= xlim[1]]
        if not filtered:
            continue
        scaled = [(sx(x), sy(y)) for x, y in filtered]
        color = PALETTE[idx % len(PALETTE)]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=2)
        for x, y in scaled:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        lx = x0 + 14 + idx * 170
        ly = y1 - 30
        draw.rectangle((lx, ly + 5, lx + 12, ly + 17), fill=color)
        draw.text((lx + 18, ly), name, font=FONT_MONO, fill=INK)
    if peak_markers:
        for label, (step, value) in peak_markers.items():
            if xlim is not None and not (xlim[0] <= step <= xlim[1]):
                continue
            px = sx(step)
            draw.line((px, top, px, bottom), fill=RED, width=1)
            draw.text((px + 3, top + 4), f"peak {step}", font=FONT_MONO, fill=RED)


def plot_small_multiples(
    path: Path,
    title: str,
    panels: list[tuple[str, dict[str, list[tuple[int, float]]], dict[str, tuple[int, float]]]],
    xlim: tuple[int, int] | None = None,
) -> None:
    cols = 2
    panel_w, panel_h = 720, 330
    rows = math.ceil(len(panels) / cols)
    width = 40 + cols * panel_w + (cols - 1) * 24 + 40
    height = 80 + rows * panel_h + (rows - 1) * 22 + 32
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), title, font=FONT_BOLD, fill=INK)
    for idx, (panel_title, series, peaks) in enumerate(panels):
        row, col = divmod(idx, cols)
        x0 = 40 + col * (panel_w + 24)
        y0 = 80 + row * (panel_h + 22)
        draw_panel(draw, (x0, y0, x0 + panel_w, y0 + panel_h), panel_title, series, xlim=xlim, peak_markers=peaks)
    save_image(img, path)


def table_image(path: Path, title: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    widths = [260, 110, 115, 110, 115, 110, 125, 125]
    row_h = 32
    width = 50 + sum(widths) + 50
    height = 80 + row_h * (len(rows) + 1) + 34
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), title, font=FONT_BOLD, fill=INK)
    x0, y0 = 40, 72
    x = x0
    for col, w in zip(columns, widths):
        draw.rectangle((x, y0, x + w, y0 + row_h), fill=(246, 248, 251), outline=GRID)
        draw.text((x + 7, y0 + 8), col, font=FONT_MONO, fill=INK)
        x += w
    y = y0 + row_h
    for i, row in enumerate(rows):
        fill = "white" if i % 2 == 0 else (250, 251, 253)
        x = x0
        for col, w in zip(columns, widths):
            val = row.get(col, "")
            if isinstance(val, float):
                text = f"{val:.4g}"
            else:
                text = str(val)
            if len(text) > 34:
                text = text[:33] + "…"
            draw.rectangle((x, y, x + w, y + row_h), fill=fill, outline=GRID)
            draw.text((x + 7, y + 8), text, font=FONT_MONO, fill=INK)
            x += w
        y += row_h
    save_image(img, path)


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    out_dir = Path(args.output_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    source_long = report_dir / "tables" / "selected_scalar_metrics.csv"
    rows = read_scalar_rows(source_long)
    by_metric = grouped(rows)
    peaks = peak_summary(rows)
    pivot = pivot_rows(rows)
    write_csv(table_dir / "full_scalar_long.csv", rows, ["metric", "tag", "step", "wall_time", "value"])
    write_csv(table_dir / "full_scalar_pivot.csv", pivot)
    write_csv(table_dir / "scalar_peak_summary.csv", peaks)
    write_json(table_dir / "scalar_peak_summary.json", peaks)
    shutil.copy2(source_long, table_dir / "selected_scalar_metrics.source.csv")

    peak_by_metric = {row["metric"]: row for row in peaks}
    key_peak_rows = [peak_by_metric[m] for m in KEY_METRICS if m in peak_by_metric]
    write_csv(table_dir / "key_scalar_peak_summary.csv", key_peak_rows)
    write_json(table_dir / "key_scalar_peak_summary.json", key_peak_rows)

    table_columns = ["metric", "max_step", "max_value", "latest_value", "last_step", "drop_from_peak", "drop_from_peak_pct", "count"]
    table_image(fig_dir / "01_key_scalar_peak_summary_table.png", "Key scalar peak summary", table_columns, key_peak_rows)

    def pts(metric: str) -> list[tuple[int, float]]:
        return [(int(row["step"]), float(row["value"])) for row in by_metric.get(metric, [])]

    def peak(metric: str) -> dict[str, tuple[int, float]]:
        row = peak_by_metric.get(metric)
        return {metric: (int(row["max_step"]), float(row["max_value"]))} if row else {}

    eval_panels = [
        ("eval_reward_score", {"eval_reward_score": pts("eval_reward_score")}, peak("eval_reward_score")),
        ("eval_numeric_exact_rate", {"eval_numeric_exact_rate": pts("eval_numeric_exact_rate")}, peak("eval_numeric_exact_rate")),
        ("eval_format_accuracy", {"eval_format_accuracy": pts("eval_format_accuracy")}, peak("eval_format_accuracy")),
        ("eval_empty_response_rate", {"eval_empty_response_rate": pts("eval_empty_response_rate")}, peak("eval_empty_response_rate")),
        ("eval_extracted_none_rate", {"eval_extracted_none_rate": pts("eval_extracted_none_rate")}, peak("eval_extracted_none_rate")),
        ("eval_kl", {"eval_kl": pts("eval_kl")}, peak("eval_kl")),
    ]
    plot_small_multiples(fig_dir / "02_eval_scalars_full_timeline.png", "Eval scalar full timeline (all eval points)", eval_panels)
    plot_small_multiples(fig_dir / "03_eval_scalars_early_0_1200.png", "Eval scalar early window, steps 0-1200", eval_panels, xlim=(0, 1200))

    train_panels = [
        ("train_reward_score", {"train_reward_score": pts("train_reward_score")}, peak("train_reward_score")),
        ("train_kl", {"train_kl": pts("train_kl")}, peak("train_kl")),
        ("train_reward_std", {"train_reward_std": pts("train_reward_std")}, peak("train_reward_std")),
        ("train_empty_response_rate", {"train_empty_response_rate": pts("train_empty_response_rate")}, peak("train_empty_response_rate")),
        ("train_format_accuracy", {"train_format_accuracy": pts("train_format_accuracy")}, peak("train_format_accuracy")),
        ("train_numeric_exact_rate", {"train_numeric_exact_rate": pts("train_numeric_exact_rate")}, peak("train_numeric_exact_rate")),
    ]
    plot_small_multiples(fig_dir / "04_train_scalars_full_timeline.png", "Train scalar full timeline (all train points)", train_panels)
    plot_small_multiples(fig_dir / "05_train_scalars_early_0_1200.png", "Train scalar early window, steps 0-1200", train_panels, xlim=(0, 1200))

    notes = {
        "created_from": str(source_long),
        "no_drop_tables": [
            "tables/full_scalar_long.csv",
            "tables/full_scalar_pivot.csv",
            "tables/scalar_peak_summary.csv",
        ],
        "checkpoint_eval_limitation": (
            "Checkpoint eval only covers checkpoints that were saved and fetched. "
            "Scalar peaks at early steps identify training/eval signal peaks but are not recoverable model snapshots unless the corresponding checkpoint exists."
        ),
        "headline_peak_corrections": {
            "eval_reward_score_peak": peak_by_metric.get("eval_reward_score"),
            "eval_numeric_exact_rate_peak": peak_by_metric.get("eval_numeric_exact_rate"),
            "eval_format_accuracy_peak": peak_by_metric.get("eval_format_accuracy"),
            "train_reward_score_peak": peak_by_metric.get("train_reward_score"),
        },
    }
    write_json(out_dir / "full_scalar_analysis_manifest.json", notes)
    readme = f"""# Full scalar analysis for `course-baseline-001`

This folder fixes the important limitation in the first summary: checkpoint eval starts at the saved/fetched checkpoints, but the scalar training/eval metrics peak much earlier.

## No-drop tables

- `tables/full_scalar_long.csv`: every selected TensorBoard scalar row, no downsampling.
- `tables/full_scalar_pivot.csv`: one row per step with metric columns.
- `tables/scalar_peak_summary.csv`: max/min/latest for every selected scalar, including peak step.
- `tables/key_scalar_peak_summary.csv`: compact subset for report discussion.

## Key corrections

- `eval_reward_score` peaks at step {peak_by_metric['eval_reward_score']['max_step']} with value {peak_by_metric['eval_reward_score']['max_value']:.6g}; latest is {peak_by_metric['eval_reward_score']['latest_value']:.6g}.
- `eval_numeric_exact_rate` peaks at step {peak_by_metric['eval_numeric_exact_rate']['max_step']} with value {peak_by_metric['eval_numeric_exact_rate']['max_value']:.6g}; latest is {peak_by_metric['eval_numeric_exact_rate']['latest_value']:.6g}.
- `eval_format_accuracy` peaks at step {peak_by_metric['eval_format_accuracy']['max_step']} with value {peak_by_metric['eval_format_accuracy']['max_value']:.6g}; latest is {peak_by_metric['eval_format_accuracy']['latest_value']:.6g}.
- `train_reward_score` peaks at step {peak_by_metric['train_reward_score']['max_step']} with value {peak_by_metric['train_reward_score']['max_value']:.6g}; latest is {peak_by_metric['train_reward_score']['latest_value']:.6g}.

## Interpretation guardrail

Early scalar peaks do not automatically mean the model checkpoint at that exact step can be evaluated. They show when the training/eval signals were highest. A checkpoint-level claim requires a saved and restorable checkpoint at or near that step.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote full scalar analysis to {out_dir}")


if __name__ == "__main__":
    main()
