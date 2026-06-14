"""Build clean per-run and combined visualizations for a reward sweep.

This script is intentionally dependency-light.  It reads the already-fetched
CSV/JSON artifacts produced by ``analyze_reward_sweep.py`` and renders static
PNG charts with Pillow.  The raw tables are copied into the report directory so
the figures remain tied to complete evidence rather than hand-picked summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


RUNS = [
    ("R1_no_approx", "no_approx"),
    ("R2_light_format_oldnum", "light_format_oldnum"),
    ("R3_numeric_primary_no_len", "numeric_primary_no_len"),
    ("R4_numeric_primary_len1200", "numeric_primary_len1200"),
    ("R5_numeric_primary_answer_only_len1200", "numeric_primary_answer_only_len1200"),
]

RUN_COLORS = {
    "R1_no_approx": "#2563eb",
    "R2_light_format_oldnum": "#d97706",
    "R3_numeric_primary_no_len": "#15803d",
    "R4_numeric_primary_len1200": "#7c3aed",
    "R5_numeric_primary_answer_only_len1200": "#db2777",
    "R0_baseline": "#4b5563",
}

SERIES_COLORS = {
    "accuracy": "#2563eb",
    "partial_accuracy": "#15803d",
    "format_accuracy": "#d97706",
    "train_reward_score": "#2563eb",
    "eval_reward_score": "#d97706",
    "train_kl": "#2563eb",
    "eval_kl": "#d97706",
    "train_loss": "#2563eb",
    "eval_loss": "#d97706",
    "train_actor_pg_clipfrac": "#7c3aed",
    "eval_actor_pg_clipfrac": "#db2777",
    "train_grpo_reward_std": "#2563eb",
    "eval_grpo_reward_std": "#d97706",
    "train_grpo_frac_reward_zero_std": "#15803d",
    "eval_grpo_frac_reward_zero_std": "#db2777",
    "train_grpo_advantage_std": "#7c3aed",
    "eval_grpo_advantage_std": "#0f766e",
    "rollout_empty_response_rate": "#dc2626",
    "rollout_extracted_none_rate": "#d97706",
    "rollout_overlong_rate_1200": "#7c3aed",
    "rollout_overlong_rate_1600": "#db2777",
    "rollout_answer_tag_pair_rate": "#15803d",
    "rollout_duplicate_tag_rate": "#0f766e",
    "rollout_answer_multi_number_rate": "#4b5563",
    "rollout_mean_completion_chars": "#2563eb",
    "audit_reward_numeric_margin": "#2563eb",
    "audit_reward_format_leakage": "#d97706",
    "audit_reward_hacking_rate": "#dc2626",
    "audit_group_misrank_rate": "#7c3aed",
    "reward_numeric_primary_mean": "#2563eb",
    "reward_format_light_mean": "#d97706",
    "reward_length_penalty_1200_mean": "#dc2626",
    "reward_match_format_exactly_mean": "#15803d",
    "reward_match_format_approximately_mean": "#7c3aed",
    "reward_check_answer_mean": "#0f766e",
    "reward_check_numbers_mean": "#db2777",
}

RATE_METRICS = {
    "accuracy",
    "partial_accuracy",
    "format_accuracy",
    "train_actor_pg_clipfrac",
    "eval_actor_pg_clipfrac",
    "train_grpo_frac_reward_zero_std",
    "eval_grpo_frac_reward_zero_std",
    "rollout_empty_response_rate",
    "rollout_extracted_none_rate",
    "rollout_overlong_rate_1200",
    "rollout_overlong_rate_1600",
    "rollout_answer_tag_pair_rate",
    "rollout_duplicate_tag_rate",
    "rollout_answer_multi_number_rate",
    "audit_reward_hacking_rate",
    "audit_group_misrank_rate",
}

COMBINED_GROUPS = [
    (
        "01_checkpoint_eval_all_runs.png",
        "Checkpoint evaluation by reward mode",
        "Held-out greedy eval, n=64 per checkpoint. Error bars use precomputed 95% binomial CI for accuracy.",
        [
            ("accuracy", "accuracy (%)", (0, 100)),
            ("partial_accuracy", "partial_accuracy (%)", (0, 100)),
            ("format_accuracy", "format_accuracy (%)", (0, 100)),
        ],
    ),
    (
        "02_reward_score_all_runs.png",
        "Reward score timeline by reward mode",
        "TensorBoard scalar step. Dense lines show 64-step bin means; raw points remain in scalar_long.csv/scalar_pivot.csv.",
        [
            ("train_reward_score", "train reward score", None),
            ("eval_reward_score", "eval reward score", None),
        ],
    ),
    (
        "03_kl_loss_clip_all_runs.png",
        "Policy health timeline by reward mode",
        "KL, loss, and clip fraction from TensorBoard scalars. Dense lines show 64-step bin means.",
        [
            ("train_kl", "train KL", None),
            ("eval_kl", "eval KL", None),
            ("train_actor_pg_clipfrac", "train pg_clipfrac", (0, 1)),
        ],
    ),
    (
        "04_grpo_health_all_runs.png",
        "GRPO group health by reward mode",
        "Reward variance and zero-std diagnostics. Dense lines show 64-step bin means.",
        [
            ("train_grpo_reward_std", "train reward_std", None),
            ("train_grpo_frac_reward_zero_std", "train frac_reward_zero_std", (0, 1)),
            ("train_grpo_advantage_std", "train advantage_std", None),
        ],
    ),
    (
        "05_response_health_all_runs.png",
        "Response and parser health by reward mode",
        "Rates are 0-1. Completion length panel uses characters. Dense lines show 64-step bin means.",
        [
            ("rollout_empty_response_rate", "empty response rate", (0, 1)),
            ("rollout_extracted_none_rate", "extracted None rate", (0, 1)),
            ("rollout_overlong_rate_1600", "overlong rate >1600 chars", (0, 1)),
            ("rollout_mean_completion_chars", "mean completion chars", None),
        ],
    ),
    (
        "06_reward_audit_all_runs.png",
        "Reward audit timeline by reward mode",
        "Checks whether reward prefers numeric correctness over format-only behavior. Dense lines show 64-step bin means.",
        [
            ("audit_reward_numeric_margin", "numeric margin", None),
            ("audit_reward_format_leakage", "format leakage", None),
            ("audit_reward_hacking_rate", "reward hacking rate", (0, 1)),
            ("audit_group_misrank_rate", "group misrank rate", (0, 1)),
        ],
    ),
    (
        "07_reward_components_all_runs.png",
        "Reward component timeline by reward mode",
        "Component means are from the active reward mode, not the baseline reward mirror. Dense lines show 64-step bin means.",
        [
            ("reward_numeric_primary_mean", "numeric primary mean", None),
            ("reward_format_light_mean", "format light mean", None),
            ("reward_length_penalty_1200_mean", "length penalty mean", None),
            ("reward_match_format_exactly_mean", "exact format mean", None),
        ],
    ),
]

PER_RUN_GROUPS = [
    (
        "02_reward_kl.png",
        "Reward and KL timeline",
        "TensorBoard scalar step. Both train and eval are shown when present.",
        [
            ("train_reward_score", "train reward score", None),
            ("eval_reward_score", "eval reward score", None),
            ("train_kl", "train KL", None),
            ("eval_kl", "eval KL", None),
        ],
    ),
    (
        "03_loss_clip.png",
        "Loss and clip fraction timeline",
        "Policy loss and clipped surrogate fraction.",
        [
            ("train_loss", "train loss", None),
            ("eval_loss", "eval loss", None),
            ("train_actor_pg_clipfrac", "train pg_clipfrac", (0, 1)),
            ("eval_actor_pg_clipfrac", "eval pg_clipfrac", (0, 1)),
        ],
    ),
    (
        "04_grpo_group_health.png",
        "GRPO group health timeline",
        "Group variance, zero-std rate, and advantage spread.",
        [
            ("train_grpo_reward_std", "train reward_std", None),
            ("eval_grpo_reward_std", "eval reward_std", None),
            ("train_grpo_frac_reward_zero_std", "train frac_reward_zero_std", (0, 1)),
            ("eval_grpo_frac_reward_zero_std", "eval frac_reward_zero_std", (0, 1)),
            ("train_grpo_advantage_std", "train advantage_std", None),
        ],
    ),
    (
        "05_response_health.png",
        "Response and parser health timeline",
        "Parser and response-shape diagnostics derived from rollout tracing.",
        [
            ("rollout_empty_response_rate", "empty response rate", (0, 1)),
            ("rollout_extracted_none_rate", "extracted None rate", (0, 1)),
            ("rollout_answer_tag_pair_rate", "answer tag pair rate", (0, 1)),
            ("rollout_duplicate_tag_rate", "duplicate tag rate", (0, 1)),
            ("rollout_overlong_rate_1200", "overlong >1200 chars", (0, 1)),
            ("rollout_overlong_rate_1600", "overlong >1600 chars", (0, 1)),
            ("rollout_mean_completion_chars", "mean completion chars", None),
        ],
    ),
    (
        "06_reward_audit.png",
        "Reward audit timeline",
        "Numeric margin, format leakage, hacking, and group misranking diagnostics.",
        [
            ("audit_reward_numeric_margin", "numeric margin", None),
            ("audit_reward_format_leakage", "format leakage", None),
            ("audit_reward_hacking_rate", "reward hacking rate", (0, 1)),
            ("audit_group_misrank_rate", "group misrank rate", (0, 1)),
        ],
    ),
    (
        "07_reward_components.png",
        "Reward component timeline",
        "Per-component means for the active reward mode.",
        [
            ("reward_numeric_primary_mean", "numeric primary mean", None),
            ("reward_format_light_mean", "format light mean", None),
            ("reward_length_penalty_1200_mean", "length penalty mean", None),
            ("reward_match_format_exactly_mean", "exact format mean", None),
            ("reward_match_format_approximately_mean", "approx format mean", None),
            ("reward_check_answer_mean", "check answer mean", None),
            ("reward_check_numbers_mean", "check numbers mean", None),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reward sweep visual report package.")
    parser.add_argument("--input-dir", default="artifacts/cloud/reward-grid-001")
    parser.add_argument("--output-dir", default="artifacts/reports/reward-grid-001")
    parser.add_argument("--baseline-dir", default="artifacts/cloud/course-baseline-001")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def to_int(value: Any) -> int | None:
    value_f = to_float(value)
    if value_f is None:
        return None
    return int(value_f)


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}%"


def short_metric(metric: str) -> str:
    return metric.replace("_", " ")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(26, True)
FONT_SUBTITLE = font(15)
FONT_AXIS = font(13)
FONT_SMALL = font(12)
FONT_LABEL = font(14)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0]


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0, 1]
    if vmin == vmax:
        if vmin == 0:
            return [0, 1]
        pad = abs(vmin) * 0.1 or 1
        vmin -= pad
        vmax += pad
    raw = (vmax - vmin) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw))) if raw else 1
    nice = raw / magnitude
    if nice <= 1:
        step = 1
    elif nice <= 2:
        step = 2
    elif nice <= 5:
        step = 5
    else:
        step = 10
    step *= magnitude
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    cur = start
    guard = 0
    while cur <= end + step * 0.5 and guard < 20:
        ticks.append(cur)
        cur += step
        guard += 1
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_scalar_pivot(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    series: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return series
    wanted = {
        metric
        for group in COMBINED_GROUPS + PER_RUN_GROUPS
        for metric, _label, _yrange in group[3]
    }
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id") or ""
            step = to_float(row.get("step"))
            if not run_id or step is None:
                continue
            for metric in wanted:
                value = to_float(row.get(metric))
                if value is not None:
                    series[run_id][metric].append((step, value))
    return series


def read_baseline_series(path: Path, max_step: int = 768) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if not path.exists():
        return out
    baseline_name_map = {
        "train_kl": "train_kl",
        "eval_kl": "eval_kl",
        "train_loss": "train_loss",
        "eval_loss": "eval_loss",
        "actor_train_pg_clipfrac": "train_actor_pg_clipfrac",
        "actor_eval_pg_clipfrac": "eval_actor_pg_clipfrac",
        "grpo_train_reward_std": "train_grpo_reward_std",
        "grpo_eval_reward_std": "eval_grpo_reward_std",
        "grpo_train_frac_reward_zero_std": "train_grpo_frac_reward_zero_std",
        "grpo_eval_frac_reward_zero_std": "eval_grpo_frac_reward_zero_std",
        "grpo_train_advantage_std": "train_grpo_advantage_std",
        "grpo_eval_advantage_std": "eval_grpo_advantage_std",
        "train_reward_score": "train_reward_score",
        "eval_reward_score": "eval_reward_score",
        "rollout_empty_response_rate": "rollout_empty_response_rate",
        "rollout_extracted_none_rate": "rollout_extracted_none_rate",
        "rollout_overlong_rate_1200": "rollout_overlong_rate_1200",
        "rollout_overlong_rate_1600": "rollout_overlong_rate_1600",
        "rollout_answer_tag_pair_rate": "rollout_answer_tag_pair_rate",
        "rollout_duplicate_tag_rate": "rollout_duplicate_tag_rate",
        "rollout_mean_completion_chars": "rollout_mean_completion_chars",
    }
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = to_int(row.get("step"))
            metric = baseline_name_map.get(row.get("metric") or "")
            value = to_float(row.get("value"))
            if metric and step is not None and step <= max_step and value is not None:
                out[metric].append((float(step), value))
    return out


def read_checkpoint_eval(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(path)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run_id = row.get("run_id") or ""
        step = to_int(row.get("step"))
        if not run_id or step is None:
            continue
        parsed = dict(row)
        for key in [
            "step",
            "accuracy",
            "partial_accuracy",
            "format_accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "correct",
            "total",
        ]:
            parsed[key] = to_float(row.get(key))
        out[run_id].append(parsed)
    for rows_for_run in out.values():
        rows_for_run.sort(key=lambda row: row["step"] or 0)
    return out


def read_baseline_checkpoint(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    parsed = []
    for row in rows:
        step = to_int(row.get("step"))
        if step is None:
            continue
        item = dict(row)
        item["run_id"] = "R0_baseline"
        item["reward_mode"] = "baseline"
        for key in [
            "step",
            "accuracy",
            "partial_accuracy",
            "format_accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "correct",
            "total",
        ]:
            item[key] = to_float(row.get(key))
        parsed.append(item)
    return parsed


def read_trace_audit(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    metrics = [
        "empty_response_rate",
        "extracted_none_rate",
        "format_accuracy",
        "numeric_exact_rate",
        "overlong_rate_1200",
        "overlong_rate_1600",
        "reward_format_leakage",
        "reward_hacking_rate",
        "reward_mean",
        "reward_numeric_margin",
    ]
    for row in read_csv(path):
        run_id = row.get("run_id") or ""
        x = to_float(row.get("call_index"))
        if not run_id or x is None:
            continue
        for metric in metrics:
            value = to_float(row.get(metric))
            if value is not None:
                out[run_id][metric].append((x, value))
    return out


def metric_points(
    series: dict[str, dict[str, list[tuple[float, float]]]],
    run_id: str,
    metric: str,
) -> list[tuple[float, float]]:
    return sorted(series.get(run_id, {}).get(metric, []))


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> int:
    draw.text((36, 26), title, fill="#111827", font=FONT_TITLE)
    y = 62
    for line in wrap_text(draw, subtitle, FONT_SUBTITLE, width - 72):
        draw.text((36, y), line, fill="#4b5563", font=FONT_SUBTITLE)
        y += 20
    return y + 10


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for word in words[1:]:
        candidate = f"{cur} {word}"
        if text_width(draw, candidate, fnt) <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int, max_width: int) -> int:
    cursor_x = x
    cursor_y = y
    for label, color in items:
        item_width = 22 + text_width(draw, label, FONT_SMALL) + 18
        if cursor_x + item_width > x + max_width:
            cursor_x = x
            cursor_y += 22
        draw.line((cursor_x, cursor_y + 8, cursor_x + 16, cursor_y + 8), fill=color, width=3)
        draw.text((cursor_x + 22, cursor_y), label, fill="#374151", font=FONT_SMALL)
        cursor_x += item_width
    return cursor_y + 24


def panel_bounds(values: list[float], forced: tuple[float, float] | None) -> tuple[float, float, list[float]]:
    if forced:
        ticks = nice_ticks(forced[0], forced[1], 5)
        return forced[0], forced[1], ticks
    if not values:
        return 0, 1, [0, 1]
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        pad = abs(vmin) * 0.15 or 1.0
    else:
        pad = (vmax - vmin) * 0.08
    vmin -= pad
    vmax += pad
    ticks = nice_ticks(vmin, vmax, 5)
    return min(ticks), max(ticks), ticks


def bin_points(points: list[tuple[float, float]], bin_size: int | None) -> list[tuple[float, float]]:
    if not bin_size or bin_size <= 1 or len(points) <= bin_size * 3:
        return points
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y in points:
        buckets[int(x // bin_size)].append((x, y))
    binned = []
    for key in sorted(buckets):
        bucket = buckets[key]
        x_mean = sum(item[0] for item in bucket) / len(bucket)
        y_mean = sum(item[1] for item in bucket) / len(bucket)
        binned.append((x_mean, y_mean))
    return binned


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    panel_title: str,
    series_items: list[tuple[str, str, list[tuple[float, float]]]],
    y_range: tuple[float, float] | None = None,
    x_range: tuple[float, float] | None = None,
) -> None:
    x0, y0, x1, y1 = area
    draw.text((x0, y0 - 22), panel_title, fill="#111827", font=FONT_LABEL)
    plot_top = y0 + 2
    plot_bottom = y1 - 28
    plot_left = x0 + 58
    plot_right = x1 - 12
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#d1d5db", width=1)
    all_points = [pt for _label, _color, pts in series_items for pt in pts]
    if not all_points:
        draw.text((plot_left + 10, plot_top + 20), "No scalar data", fill="#6b7280", font=FONT_AXIS)
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin = min(xs) if x_range is None else x_range[0]
    xmax = max(xs) if x_range is None else x_range[1]
    if xmin == xmax:
        xmax = xmin + 1
    ymin, ymax, yticks = panel_bounds(ys, y_range)
    if ymin == ymax:
        ymax = ymin + 1

    def px(x: float) -> int:
        return int(plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left))

    def py(y: float) -> int:
        return int(plot_bottom - (y - ymin) / (ymax - ymin) * (plot_bottom - plot_top))

    for tick in yticks:
        if tick < ymin - 1e-9 or tick > ymax + 1e-9:
            continue
        yy = py(tick)
        draw.line((plot_left, yy, plot_right, yy), fill="#eef2f7", width=1)
        label = fmt_tick(tick)
        draw.text((plot_left - 8 - text_width(draw, label, FONT_SMALL), yy - 7), label, fill="#6b7280", font=FONT_SMALL)

    unique_x = sorted({x for x, _y in all_points})
    xticks = unique_x if len(unique_x) <= 8 else nice_ticks(xmin, xmax, 6)
    for tick in xticks:
        if tick < xmin - 1e-9 or tick > xmax + 1e-9:
            continue
        xx = px(tick)
        draw.line((xx, plot_top, xx, plot_bottom), fill="#f3f4f6", width=1)
        label = fmt_tick(tick)
        draw.text((xx - text_width(draw, label, FONT_SMALL) // 2, plot_bottom + 6), label, fill="#6b7280", font=FONT_SMALL)

    for label, color, points in series_items:
        if len(points) == 1:
            xx, yy = px(points[0][0]), py(points[0][1])
            draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color)
            continue
        draw_all_markers = len(points) <= 30
        last: tuple[int, int] | None = None
        for x, y in points:
            if x < xmin or x > xmax:
                continue
            cur = (px(x), py(y))
            if last is not None:
                draw.line((last[0], last[1], cur[0], cur[1]), fill=color, width=2)
            if draw_all_markers:
                draw.ellipse((cur[0] - 3, cur[1] - 3, cur[0] + 3, cur[1] + 3), fill=color)
            last = cur
        if points and not draw_all_markers:
            lx = px(points[-1][0])
            ly = py(points[-1][1])
            draw.ellipse((lx - 3, ly - 3, lx + 3, ly + 3), fill=color)
    draw.text((plot_right - 42, plot_bottom + 6), "step", fill="#6b7280", font=FONT_SMALL)


def plot_metric_panels(
    out_path: Path,
    title: str,
    subtitle: str,
    panels: list[tuple[str, str, tuple[float, float] | None]],
    panel_series: dict[str, list[tuple[str, str, list[tuple[float, float]]]]],
    legend_items: list[tuple[str, str]],
    width: int = 1500,
    bin_size: int | None = None,
) -> None:
    panel_count = max(1, len(panels))
    panel_height = 230
    height = 115 + panel_count * panel_height + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    start_y = draw_header(draw, title, subtitle, width)
    legend_end = draw_legend(draw, legend_items, 36, start_y, width - 72)
    y = legend_end + 18
    for metric, label, y_range in panels:
        items = [
            (label_item, color, bin_points(points, bin_size))
            for label_item, color, points in panel_series.get(metric, [])
        ]
        draw_line_panel(draw, (36, y + 26, width - 36, y + panel_height), label, items, y_range)
        y += panel_height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def plot_checkpoint_eval(
    out_path: Path,
    title: str,
    subtitle: str,
    rows_by_run: dict[str, list[dict[str, Any]]],
    run_ids: list[str],
    combined: bool,
) -> None:
    panels = [
        ("accuracy", "accuracy (%)", (0, 100)),
        ("partial_accuracy", "partial_accuracy (%)", (0, 100)),
        ("format_accuracy", "format_accuracy (%)", (0, 100)),
    ]
    panel_series: dict[str, list[tuple[str, str, list[tuple[float, float]]]]] = {}
    if combined:
        for metric, _label, _yrange in panels:
            panel_series[metric] = []
            for run_id in run_ids:
                pts = [
                    (float(row["step"]), float(row[metric]))
                    for row in rows_by_run.get(run_id, [])
                    if row.get("step") is not None and row.get(metric) is not None
                ]
                if pts:
                    panel_series[metric].append((run_id, RUN_COLORS.get(run_id, "#111827"), pts))
        legend = [(run_id, RUN_COLORS.get(run_id, "#111827")) for run_id in run_ids if rows_by_run.get(run_id)]
        plot_metric_panels(out_path, title, subtitle, panels, panel_series, legend)
        return

    run_id = run_ids[0]
    for metric, _label, _yrange in panels:
        pts = [
            (float(row["step"]), float(row[metric]))
            for row in rows_by_run.get(run_id, [])
            if row.get("step") is not None and row.get(metric) is not None
        ]
        panel_series[metric] = [(metric, SERIES_COLORS[metric], pts)]
    legend = [(metric, SERIES_COLORS[metric]) for metric, _label, _yrange in panels]
    plot_metric_panels(out_path, title, subtitle, panels, panel_series, legend, width=1350)


def plot_combined_metric_group(
    out_path: Path,
    title: str,
    subtitle: str,
    panel_specs: list[tuple[str, str, tuple[float, float] | None]],
    scalar_series: dict[str, dict[str, list[tuple[float, float]]]],
    run_ids: list[str],
) -> None:
    panel_series: dict[str, list[tuple[str, str, list[tuple[float, float]]]]] = {}
    for metric, _label, _yrange in panel_specs:
        panel_series[metric] = []
        for run_id in run_ids:
            pts = metric_points(scalar_series, run_id, metric)
            if pts:
                panel_series[metric].append((run_id, RUN_COLORS.get(run_id, "#111827"), pts))
    legend = [(run_id, RUN_COLORS.get(run_id, "#111827")) for run_id in run_ids]
    plot_metric_panels(out_path, title, subtitle, panel_specs, panel_series, legend, bin_size=64)


def plot_per_run_metric_group(
    out_path: Path,
    title: str,
    subtitle: str,
    panel_specs: list[tuple[str, str, tuple[float, float] | None]],
    scalar_series: dict[str, dict[str, list[tuple[float, float]]]],
    run_id: str,
) -> None:
    panel_series: dict[str, list[tuple[str, str, list[tuple[float, float]]]]] = {}
    legend: list[tuple[str, str]] = []
    for metric, _label, _yrange in panel_specs:
        pts = metric_points(scalar_series, run_id, metric)
        if pts:
            color = SERIES_COLORS.get(metric, "#2563eb")
            panel_series[metric] = [(metric, color, pts)]
            legend.append((metric, color))
        else:
            panel_series[metric] = []
    plot_metric_panels(out_path, title, subtitle, panel_specs, panel_series, legend, width=1350, bin_size=64)


def plot_trace_audit(
    out_path: Path,
    title: str,
    subtitle: str,
    trace_series: dict[str, dict[str, list[tuple[float, float]]]],
    run_id: str,
) -> None:
    metrics = [
        ("numeric_exact_rate", "trace numeric exact rate", (0, 1)),
        ("format_accuracy", "trace format accuracy", (0, 1)),
        ("reward_mean", "trace reward mean", None),
        ("reward_numeric_margin", "trace numeric margin", None),
        ("reward_hacking_rate", "trace reward hacking rate", (0, 1)),
        ("empty_response_rate", "trace empty response rate", (0, 1)),
        ("extracted_none_rate", "trace extracted None rate", (0, 1)),
        ("overlong_rate_1600", "trace overlong >1600 rate", (0, 1)),
    ]
    color_cycle = ["#2563eb", "#15803d", "#d97706", "#0f766e", "#dc2626", "#7c3aed", "#db2777", "#4b5563"]
    panel_series = {}
    legend = []
    for idx, (metric, _label, _yrange) in enumerate(metrics):
        pts = sorted(trace_series.get(run_id, {}).get(metric, []))
        color = color_cycle[idx % len(color_cycle)]
        panel_series[metric] = [(metric, color, pts)] if pts else []
        if pts:
            legend.append((metric, color))
    plot_metric_panels(out_path, title, subtitle, metrics, panel_series, legend, width=1350, bin_size=64)


def make_table_image(
    out_path: Path,
    title: str,
    subtitle: str,
    columns: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    width: int = 1500,
) -> None:
    row_h = 32
    header_h = 130
    height = header_h + row_h * (len(rows) + 1) + 44
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = draw_header(draw, title, subtitle, width)
    left = 36
    top = max(y + 8, header_h)
    table_w = width - 72
    weights = []
    for key, _label in columns:
        max_len = len(str(key))
        for row in rows:
            max_len = max(max_len, len(str(row.get(key, ""))))
        weights.append(max(10, max_len))
    total = sum(weights)
    widths = [int(table_w * w / total) for w in weights]
    widths[-1] += table_w - sum(widths)

    x = left
    draw.rectangle((left, top, left + table_w, top + row_h), fill="#f3f4f6", outline="#d1d5db")
    for idx, (_key, label) in enumerate(columns):
        draw.text((x + 8, top + 8), label, fill="#111827", font=FONT_SMALL)
        x += widths[idx]
        draw.line((x, top, x, top + row_h * (len(rows) + 1)), fill="#e5e7eb")
    for ridx, row in enumerate(rows):
        yrow = top + row_h * (ridx + 1)
        if ridx % 2:
            draw.rectangle((left, yrow, left + table_w, yrow + row_h), fill="#fafafa")
        x = left
        for cidx, (key, _label) in enumerate(columns):
            text = str(row.get(key, ""))
            if len(text) > 40:
                text = text[:37] + "..."
            draw.text((x + 8, yrow + 8), text, fill="#374151", font=FONT_SMALL)
            x += widths[cidx]
    draw.rectangle((left, top, left + table_w, top + row_h * (len(rows) + 1)), outline="#d1d5db")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def latest_value(series: dict[str, dict[str, list[tuple[float, float]]]], run_id: str, metric: str) -> float | None:
    pts = metric_points(series, run_id, metric)
    if not pts:
        return None
    return pts[-1][1]


def latest_step(series: dict[str, dict[str, list[tuple[float, float]]]], run_id: str) -> int | None:
    steps: list[int] = []
    for pts in series.get(run_id, {}).values():
        if pts:
            steps.append(int(max(p[0] for p in pts)))
    return max(steps) if steps else None


def build_summary_rows(
    selection_rows: list[dict[str, str]],
    ckpt_rows_by_run: dict[str, list[dict[str, Any]]],
    scalar_series: dict[str, dict[str, list[tuple[float, float]]]],
) -> list[dict[str, Any]]:
    selection_by_run = {row.get("run_id"): row for row in selection_rows}
    rows = []
    for run_id, mode in RUNS:
        ckpts = ckpt_rows_by_run.get(run_id, [])
        best = max(ckpts, key=lambda row: row.get("accuracy") or -1, default={})
        sel = selection_by_run.get(run_id, {})
        rows.append(
            {
                "run_id": run_id,
                "reward_mode": mode,
                "best_step": int(best.get("step") or 0) if best else "",
                "best_accuracy": pct(best.get("accuracy")),
                "best_partial": pct(best.get("partial_accuracy")),
                "best_format": pct(best.get("format_accuracy")),
                "latest_scalar_step": latest_step(scalar_series, run_id) or "",
                "latest_reward": fmt_optional(latest_value(scalar_series, run_id, "train_reward_score")),
                "latest_kl": fmt_optional(latest_value(scalar_series, run_id, "train_kl")),
                "empty_rate": fmt_optional(to_float(sel.get("empty_response_rate")), 3),
                "extracted_none_rate": fmt_optional(to_float(sel.get("extracted_none_rate")), 3),
                "zero_std_rate": fmt_optional(to_float(sel.get("frac_reward_zero_std")), 3),
                "status": sel.get("screening_status", ""),
                "elimination_reasons": sel.get("elimination_reasons", ""),
            }
        )
    return rows


def fmt_optional(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def copy_tables(input_dir: Path, output_dir: Path) -> list[str]:
    table_src = input_dir / "artifacts" / "sweep_analysis" / "tables"
    table_dst = output_dir / "tables"
    table_dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in [
        "checkpoint_eval_long.csv",
        "scalar_long.csv",
        "scalar_pivot.csv",
        "selection_summary.csv",
        "selection_summary.json",
        "trace_audit_by_call.csv",
        "trace_rows_flat.csv",
    ]:
        src = table_src / name
        if src.exists():
            dst = table_dst / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def write_readme(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    generated: list[str],
    copied_tables: list[str],
    input_dir: Path,
) -> None:
    lines = [
        "# reward-grid-001 visual report package",
        "",
        "Training has completed and has already been fetched locally. This directory only performs local visualization and evidence packaging; it does not connect to, start, or stop the course TPU.",
        "",
        "## Key Paths",
        "",
        f"- Raw fetched data: `{input_dir.resolve()}`",
        f"- Report package: `{output_dir.resolve()}`",
        "- Per-run figures: `figures/by_run/<run_id>/`",
        "- Combined comparison figures: `figures/combined/`",
        "- Complete table copies: `tables/`",
        "",
        "## Result Summary",
        "",
        "| run_id | reward_mode | best_step | best_accuracy | best_partial | best_format | status | elimination_reasons |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {run_id} | {reward_mode} | {best_step} | {best_accuracy} | {best_partial} | {best_format} | {status} | {elimination_reasons} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Checkpoint-evaluation steps are LoRA checkpoint steps: 256, 512, and 768.",
            "- The scalar timeline x-axis is the TensorBoard logged scalar step; all points are retained in `scalar_pivot.csv`.",
            "- `scalar_long.csv` and `trace_rows_flat.csv` are complete detail tables, not summaries; figures use derived views from these tables.",
            "- All figures are clean information graphics: white background, unified axes, traceable metric keys, and no decorative card or slide styling.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in generated:
        lines.append(f"- `{Path(path).relative_to(output_dir)}`")
    lines.extend(["", "## Table Copies", ""])
    for path in copied_tables:
        lines.append(f"- `{Path(path).relative_to(output_dir)}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    tables_dir = input_dir / "artifacts" / "sweep_analysis" / "tables"
    if not tables_dir.exists():
        raise SystemExit(f"Missing sweep analysis tables: {tables_dir}")

    figures_dir = output_dir / "figures"
    combined_dir = figures_dir / "combined"
    by_run_dir = figures_dir / "by_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_series = read_scalar_pivot(tables_dir / "scalar_pivot.csv")
    baseline_scalars = read_baseline_series(baseline_dir / "artifacts" / "analysis" / "scalar_metrics.csv")
    if baseline_scalars:
        scalar_series["R0_baseline"] = baseline_scalars

    ckpt_rows_by_run = read_checkpoint_eval(tables_dir / "checkpoint_eval_long.csv")
    baseline_ckpts = read_baseline_checkpoint(baseline_dir / "artifacts" / "analysis" / "checkpoint_eval_rows.csv")
    if baseline_ckpts:
        ckpt_rows_by_run["R0_baseline"] = baseline_ckpts

    trace_series = read_trace_audit(tables_dir / "trace_audit_by_call.csv")
    selection_rows = read_csv(tables_dir / "selection_summary.csv")
    summary_rows = build_summary_rows(selection_rows, ckpt_rows_by_run, scalar_series)
    write_csv(output_dir / "tables" / "visual_summary.csv", summary_rows)

    generated: list[str] = []
    all_run_ids = ["R0_baseline"] + [run_id for run_id, _mode in RUNS if run_id in scalar_series or run_id in ckpt_rows_by_run]
    sweep_run_ids = [run_id for run_id, _mode in RUNS]

    out = combined_dir / "01_checkpoint_eval_all_runs.png"
    plot_checkpoint_eval(
        out,
        "Checkpoint evaluation by reward mode",
        "R1-R5 checkpoint eval at 256/512/768. R0 baseline is shown separately to avoid mixing different checkpoint ranges.",
        ckpt_rows_by_run,
        sweep_run_ids,
        combined=True,
    )
    generated.append(str(out))

    if ckpt_rows_by_run.get("R0_baseline"):
        out = combined_dir / "09_baseline_checkpoint_reference.png"
        plot_checkpoint_eval(
            out,
            "R0 baseline checkpoint reference",
            "course-baseline-001 checkpoint eval. This is a reference curve, not part of the reward sweep grid.",
            ckpt_rows_by_run,
            ["R0_baseline"],
            combined=False,
        )
        generated.append(str(out))

    for name, title, subtitle, panels in COMBINED_GROUPS[1:]:
        out = combined_dir / name
        run_ids = all_run_ids if name in {"02_reward_score_all_runs.png", "03_kl_loss_clip_all_runs.png", "04_grpo_health_all_runs.png", "05_response_health_all_runs.png"} else sweep_run_ids
        plot_combined_metric_group(out, title, subtitle, panels, scalar_series, run_ids)
        generated.append(str(out))

    out = combined_dir / "08_selection_summary_table.png"
    make_table_image(
        out,
        "Reward sweep selection summary",
        "Best checkpoint plus guardrail status. Exact values are in tables/selection_summary.csv and tables/visual_summary.csv.",
        [
            ("run_id", "run_id"),
            ("reward_mode", "reward_mode"),
            ("best_step", "best_step"),
            ("best_accuracy", "best_acc"),
            ("best_partial", "best_partial"),
            ("best_format", "best_format"),
            ("empty_rate", "empty"),
            ("extracted_none_rate", "none"),
            ("zero_std_rate", "zero_std"),
            ("status", "status"),
            ("elimination_reasons", "elimination_reasons"),
        ],
        summary_rows,
    )
    generated.append(str(out))

    for run_id, mode in RUNS:
        run_dir = by_run_dir / run_id
        out = run_dir / "01_checkpoint_eval.png"
        plot_checkpoint_eval(
            out,
            f"{run_id}: checkpoint evaluation",
            f"Reward mode: {mode}. Greedy held-out eval, n=64 per checkpoint.",
            ckpt_rows_by_run,
            [run_id],
            combined=False,
        )
        generated.append(str(out))
        for name, title, subtitle, panels in PER_RUN_GROUPS:
            out = run_dir / name
            plot_per_run_metric_group(out, f"{run_id}: {title}", f"Reward mode: {mode}. {subtitle}", panels, scalar_series, run_id)
            generated.append(str(out))
        out = run_dir / "08_trace_audit_by_call.png"
        plot_trace_audit(
            out,
            f"{run_id}: rollout trace audit by call",
            f"Reward mode: {mode}. Uses trace_audit_by_call.csv without dropping call rows.",
            trace_series,
            run_id,
        )
        generated.append(str(out))
        row = [item for item in summary_rows if item["run_id"] == run_id]
        if row:
            out = run_dir / "09_run_summary_table.png"
            make_table_image(
                out,
                f"{run_id}: summary table",
                "Best checkpoint and latest guardrail snapshot.",
                [
                    ("run_id", "run_id"),
                    ("reward_mode", "reward_mode"),
                    ("best_step", "best_step"),
                    ("best_accuracy", "best_acc"),
                    ("best_partial", "best_partial"),
                    ("best_format", "best_format"),
                    ("latest_scalar_step", "latest_scalar_step"),
                    ("latest_reward", "latest_reward"),
                    ("latest_kl", "latest_kl"),
                    ("status", "status"),
                    ("elimination_reasons", "elimination_reasons"),
                ],
                row,
                width=1350,
            )
            generated.append(str(out))

    copied_tables = copy_tables(input_dir, output_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "runs": [{"run_id": run_id, "reward_mode": mode} for run_id, mode in RUNS],
        "figures": [str(Path(path).relative_to(output_dir)) for path in generated],
        "tables": [str(Path(path).relative_to(output_dir)) for path in copied_tables],
        "notes": [
            "Scalar timelines use TensorBoard logged scalar step and keep every row present in scalar_pivot.csv.",
            "Checkpoint eval charts use discrete checkpoint steps 256/512/768.",
            "PNG-only rendering avoids the remote Pillow PDF/JPEG handler failure seen during the TPU-side analysis step.",
        ],
    }
    (output_dir / "manifest_visuals.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(output_dir, summary_rows, generated, copied_tables, input_dir)
    print(f"Wrote {len(generated)} figures to {figures_dir}")
    print(f"Report package: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
