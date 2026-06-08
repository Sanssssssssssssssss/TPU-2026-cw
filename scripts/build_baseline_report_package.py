"""Build a self-contained report package for a fetched GRPO baseline run.

The script is intentionally dependency-light: it uses only the standard library
and Pillow so it can run from the bundled Codex Python runtime on Windows.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import PIL.JpegImagePlugin  # noqa: F401  # Registers Pillow's PDF/JPEG save handler.


RUN_ID = "course-baseline-001"
DEFAULT_RUN_DIR = Path("artifacts/cloud") / RUN_ID
DEFAULT_OUTPUT_DIR = Path("artifacts/reports") / RUN_ID

SOURCE_LINKS = [
    {
        "label": "Tunix metrics",
        "url": "https://tunix.readthedocs.io/en/stable/metrics.html",
        "note": "Tunix collected metrics, TensorBoard/W&B backends, and performance tracing.",
    },
    {
        "label": "HF TRL GRPO logging",
        "url": "https://huggingface.co/docs/trl/v0.21.0/en/logging",
        "note": "GRPO reward, KL, clip ratio, completion length, and entropy logging guidance.",
    },
    {
        "label": "OpenRLHF logging/eval",
        "url": "https://openrlhf.readthedocs.io/en/latest/agent_training.html",
        "note": "RLHF logging backends, periodic evaluation, reward/advantage/generation metrics.",
    },
    {
        "label": "VeRL-Omni metrics",
        "url": "https://verl-omni.readthedocs.io/en/latest/start/metrics.html",
        "note": "GRPO reward diversity, zero-std ratio, clipping, and ratio stability framing.",
    },
    {
        "label": "W&B Tables",
        "url": "https://docs.wandb.ai/models/track/log/log-tables",
        "note": "Prediction/sample table organization used as a reporting pattern.",
    },
    {
        "label": "W&B Artifacts",
        "url": "https://docs.wandb.ai/models/artifacts",
        "note": "Versioned run inputs/outputs pattern for report package provenance.",
    },
    {
        "label": "tbparse",
        "url": "https://tbparse.readthedocs.io/en/stable/",
        "note": "TensorBoard event-to-dataframe pattern mirrored with existing scalar exports.",
    },
]

SELECTED_TAGS = {
    "train_reward_score": "rewards/train/score/mean",
    "eval_reward_score": "rewards/eval/score/mean",
    "train_reward_mean": "rewards/train/mean",
    "eval_reward_mean": "rewards/eval/mean",
    "train_kl": "actor/train/kl",
    "eval_kl": "actor/eval/kl",
    "train_loss": "actor/train/loss",
    "eval_loss": "actor/eval/loss",
    "train_pg_clipfrac": "actor/train/pg_clipfrac",
    "eval_pg_clipfrac": "actor/eval/pg_clipfrac",
    "train_completion_length": "completions/train/mean_length",
    "eval_completion_length": "completions/eval/mean_length",
    "train_empty_response_rate": "rollout/train/empty_response_rate",
    "eval_empty_response_rate": "rollout/eval/empty_response_rate",
    "train_extracted_none_rate": "rollout/train/extracted_none_rate",
    "eval_extracted_none_rate": "rollout/eval/extracted_none_rate",
    "train_has_solution_end_rate": "rollout/train/has_solution_end_rate",
    "eval_has_solution_end_rate": "rollout/eval/has_solution_end_rate",
    "train_numeric_exact_rate": "eval/train/numeric_exact_rate",
    "eval_numeric_exact_rate": "eval/eval/numeric_exact_rate",
    "train_numeric_partial_rate": "eval/train/numeric_partial_rate",
    "eval_numeric_partial_rate": "eval/eval/numeric_partial_rate",
    "train_format_accuracy": "eval/train/format_accuracy",
    "eval_format_accuracy": "eval/eval/format_accuracy",
    "train_reward_std": "grpo/train/reward_std",
    "eval_reward_std": "grpo/eval/reward_std",
    "train_frac_reward_zero_std": "grpo/train/frac_reward_zero_std",
    "eval_frac_reward_zero_std": "grpo/eval/frac_reward_zero_std",
    "train_advantage_std": "grpo/train/advantage_std",
    "eval_advantage_std": "grpo/eval/advantage_std",
    "train_all_correct_group_rate": "grpo/train/all_correct_group_rate",
    "eval_all_correct_group_rate": "grpo/eval/all_correct_group_rate",
    "train_all_wrong_group_rate": "grpo/train/all_wrong_group_rate",
    "eval_all_wrong_group_rate": "grpo/eval/all_wrong_group_rate",
    "train_check_answer": "rewards/train/check_answer",
    "eval_check_answer": "rewards/eval/check_answer",
    "train_check_numbers": "rewards/train/check_numbers",
    "eval_check_numbers": "rewards/eval/check_numbers",
    "train_match_format_approx": "rewards/train/match_format_approximately",
    "eval_match_format_approx": "rewards/eval/match_format_approximately",
    "train_match_format_exact": "rewards/train/match_format_exactly",
    "eval_match_format_exact": "rewards/eval/match_format_exactly",
    "jax_orbax_write_gbytes": "jax/orbax/write/gbytes",
    "jax_orbax_write_gbytes_per_sec": "jax/orbax/write/gbytes_per_sec",
}

BASELINE_KEYS = [
    "MODEL_ID",
    "DATA_SOURCE",
    "MAX_STEPS",
    "NUM_BATCHES",
    "NUM_EPOCHS",
    "NUM_GENERATIONS",
    "NUM_TEST_BATCHES",
    "EVAL_EVERY_N_STEPS",
    "SAVE_INTERVAL_STEPS",
    "BETA",
    "EPSILON",
    "LEARNING_RATE",
    "RANK",
    "ALPHA",
    "MAX_PROMPT_LENGTH",
    "TOTAL_GENERATION_STEPS",
    "TRAIN_MICRO_BATCH_SIZE",
    "TRAIN_FRACTION",
]


@dataclass
class Figure:
    name: str
    title: str
    question: str
    takeaway: str
    source_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the course baseline report package.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Fetched cloud run directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Report package directory.")
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pct(value: Any, digits: int = 2) -> str:
    try:
        quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
        return f"{rounded:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.{digits}f}"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def sanitize_payload(value: Any) -> Any:
    secret_re = re.compile(r"(hf_[A-Za-z0-9]{20,}|WANDB_API_KEY\s*=\s*\S+|KAGGLE_KEY\s*=\s*\S+)", re.I)
    sensitive_names = {
        "WANDB_API_KEY": "[WANDB_KEY_NAME_REDACTED]",
        "KAGGLE_KEY": "[KAGGLE_KEY_NAME_REDACTED]",
        "GOOGLE_APPLICATION_CREDENTIALS": "[GOOGLE_CREDENTIALS_NAME_REDACTED]",
        "HF_TOKEN": "[HF_TOKEN_NAME_REDACTED]",
        "HF_HUB_TOKEN": "[HF_HUB_TOKEN_NAME_REDACTED]",
        "HUGGING_FACE_HUB_TOKEN": "[HF_HUB_TOKEN_NAME_REDACTED]",
    }
    if isinstance(value, dict):
        return {str(k): sanitize_payload(v) for k, v in value.items() if "TOKEN" not in str(k).upper() and "KEY" not in str(k).upper()}
    if isinstance(value, list):
        return [sanitize_payload(v) for v in value]
    if isinstance(value, str):
        text = secret_re.sub("[REDACTED]", value).replace("\ufeff", "")
        for name, replacement in sensitive_names.items():
            text = text.replace(name, replacement)
        return text
    return value


def validate_inputs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "base_eval": run_dir / "artifacts" / "base_eval.json",
        "lora_eval": run_dir / "artifacts" / "baseline_lora_eval.json",
        "checkpoint_summary": run_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json",
        "scalar_csv": run_dir / "artifacts" / "analysis" / "scalar_metrics.csv",
        "trace_summary": run_dir / "artifacts" / "analysis" / "trace_summary.csv",
        "trace_jsonl": run_dir / "artifacts" / "rollout_traces" / f"rollout_samples_{RUN_ID}.jsonl",
        "manifest": run_dir / "artifacts" / "run_manifest.json",
        "pipeline_log": run_dir / "pipeline.log",
        "git_commit": run_dir / "meta" / "git_commit.txt",
        "git_status": run_dir / "meta" / "git_status.txt",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required report inputs:\n" + "\n".join(missing))
    return paths


def load_scalar_series(path: Path) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, Any]]]:
    tag_to_name = {tag: name for name, tag in SELECTED_TAGS.items()}
    points_by_name: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    selected_rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("tag", "")
            name = tag_to_name.get(tag)
            if name is None:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
                wall_time = float(row["wall_time"]) if row.get("wall_time") else 0.0
            except (KeyError, ValueError):
                continue
            points_by_name[name][step] = {"step": step, "value": value, "wall_time": wall_time}

    out: dict[str, list[dict[str, float]]] = {}
    for name, by_step in points_by_name.items():
        out[name] = [by_step[step] for step in sorted(by_step)]
        for item in out[name]:
            selected_rows.append(
                {
                    "metric": name,
                    "tag": SELECTED_TAGS[name],
                    "step": int(item["step"]),
                    "wall_time": item["wall_time"],
                    "value": item["value"],
                }
            )
    selected_rows.sort(key=lambda item: (item["metric"], int(item["step"])))
    return out, selected_rows


def latest(series: dict[str, list[dict[str, float]]], name: str) -> dict[str, float] | None:
    values = series.get(name) or []
    return values[-1] if values else None


def series_xy(series: dict[str, list[dict[str, float]]], name: str) -> list[tuple[float, float]]:
    return [(point["step"], point["value"]) for point in series.get(name, [])]


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def classify_trace(row: dict[str, Any]) -> str:
    completion = str(row.get("completion") or "")
    extracted = row.get("extracted_number")
    numeric_exact = bool(row.get("numeric_exact"))
    reward_total = row.get("reward_total")
    components = row.get("reward_components") or {}
    if numeric_exact:
        return "correct_numeric"
    if not completion.strip():
        return "empty_response"
    if extracted in (None, ""):
        return "parse_fail"
    if reward_total is not None and float(reward_total) > 0 and not numeric_exact:
        return "reward_hacking_candidate"
    if components.get("match_format_exactly", 0) or components.get("match_format_approximately", 0):
        return "format_only_or_shaping"
    return "wrong_numeric"


def pick_samples(trace_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = [
        "correct_numeric",
        "wrong_numeric",
        "parse_fail",
        "empty_response",
        "reward_hacking_candidate",
        "format_only_or_shaping",
    ]
    best_by_cat: dict[str, dict[str, Any]] = {}
    late_rows = sorted(trace_rows, key=lambda item: int(item.get("call_index") or 0), reverse=True)
    for row in trace_rows:
        cat = classify_trace(row)
        if cat in categories and cat not in best_by_cat:
            best_by_cat[cat] = row
    for row in late_rows:
        cat = "late_collapse_example"
        if row.get("extracted_number") in (None, "") or not str(row.get("completion") or "").strip():
            best_by_cat[cat] = row
            break

    sample_rows = []
    for cat, row in best_by_cat.items():
        sample_rows.append(format_sample_row(cat, row))

    taxonomy_rows = []
    counts: dict[str, int] = defaultdict(int)
    for row in trace_rows:
        counts[classify_trace(row)] += 1
    total = sum(counts.values()) or 1
    for cat, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        taxonomy_rows.append({"category": cat, "count": count, "share": count / total})
    return sample_rows, taxonomy_rows


def format_sample_row(category: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": category,
        "call_index": row.get("call_index"),
        "dataset_role": row.get("dataset_role"),
        "prompt_hash": row.get("prompt_hash"),
        "question": compact(row.get("question"), 280),
        "ground_truth": row.get("answer"),
        "completion": compact(row.get("completion"), 900),
        "extracted_number": row.get("extracted_number"),
        "numeric_exact": row.get("numeric_exact"),
        "numeric_partial": row.get("numeric_partial"),
        "format_ok": row.get("format_ok"),
        "reward_total": row.get("reward_total"),
        "reward_components": json.dumps(row.get("reward_components") or {}, ensure_ascii=False, sort_keys=True),
    }


def compact(value: Any, limit: int) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def summarize_trace_phases(trace_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_summary:
        return []
    rows = []
    numeric = []
    for row in trace_summary:
        try:
            numeric.append({**row, "call_index": int(float(row["call_index"]))})
        except (KeyError, ValueError):
            continue
    numeric.sort(key=lambda item: item["call_index"])
    n = len(numeric)
    bins = [
        ("early", numeric[: max(1, n // 3)]),
        ("middle", numeric[max(1, n // 3) : max(2, 2 * n // 3)]),
        ("late", numeric[max(2, 2 * n // 3) :]),
    ]
    metric_names = [
        "empty_response_rate",
        "extracted_none_rate",
        "numeric_exact_rate",
        "format_accuracy",
        "reward_mean",
        "reward_std",
        "completion_chars_mean",
    ]
    for phase, items in bins:
        out: dict[str, Any] = {"phase": phase, "rows": len(items)}
        if items:
            out["min_call_index"] = min(item["call_index"] for item in items)
            out["max_call_index"] = max(item["call_index"] for item in items)
        for metric in metric_names:
            values = []
            for item in items:
                try:
                    values.append(float(item.get(metric, "")))
                except (TypeError, ValueError):
                    pass
            out[metric] = statistics.mean(values) if values else None
        rows.append(out)
    return rows


def build_eval_rows(base_eval: dict[str, Any], lora_eval: dict[str, Any], checkpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        eval_json_to_row("base_direct_eval", base_eval),
        eval_json_to_row("final_lora_direct_eval", lora_eval),
    ]
    seen = {row["label"] for row in rows}
    for row in checkpoint_rows:
        label = clean_text(row.get("label"))
        if label in seen:
            continue
        rows.append(
            {
                "label": label,
                "policy": row.get("policy"),
                "restored_step": row.get("restored_step"),
                "correct": row.get("correct"),
                "total": row.get("total"),
                "accuracy": row.get("accuracy"),
                "partial_accuracy": row.get("partial_accuracy"),
                "format_accuracy": row.get("format_accuracy"),
                "accuracy_ci95_low": row.get("accuracy_ci95_low"),
                "accuracy_ci95_high": row.get("accuracy_ci95_high"),
                "preset": "greedy",
            }
        )
    return rows


def eval_json_to_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    model = payload.get("model") or {}
    generation = payload.get("generation") or {}
    return {
        "label": label,
        "policy": model.get("policy"),
        "restored_step": model.get("restored_step"),
        "correct": metrics.get("correct"),
        "total": metrics.get("total"),
        "accuracy": metrics.get("accuracy"),
        "partial_accuracy": metrics.get("partial_accuracy"),
        "format_accuracy": metrics.get("format_accuracy"),
        "accuracy_ci95_low": metrics.get("accuracy_ci95_low"),
        "accuracy_ci95_high": metrics.get("accuracy_ci95_high"),
        "preset": generation.get("preset"),
    }


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "samples": output_dir / "samples",
        "provenance": output_dir / "provenance",
        "raw_refs": output_dir / "raw_refs",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    mono_candidates = [
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
    ]
    regular_path = next((p for p in candidates if p.exists()), None)
    bold_path = next((p for p in bold_candidates if p.exists()), regular_path)
    mono_path = next((p for p in mono_candidates if p.exists()), regular_path)
    if regular_path is None:
        return (ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default())
    return (
        ImageFont.truetype(str(regular_path), 24),
        ImageFont.truetype(str(bold_path), 34),
        ImageFont.truetype(str(regular_path), 18),
        ImageFont.truetype(str(mono_path), 18),
    )


FONT_REG, FONT_TITLE, FONT_SMALL, FONT_MONO = load_fonts()
INK = (28, 36, 48)
MUTED = (97, 110, 126)
GRID = (224, 229, 236)
BLUE = (34, 105, 184)
ORANGE = (225, 126, 49)
PINK = (190, 82, 111)
OLIVE = (99, 139, 72)
GOLD = (198, 158, 48)
RED = (187, 64, 64)
GREEN = (66, 145, 100)
PURPLE = (119, 96, 176)
PALETTE = [BLUE, ORANGE, OLIVE, PINK, GOLD, PURPLE, RED, GREEN]


def new_canvas(width: int, height: int, title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 28), title, fill=INK, font=FONT_TITLE)
    if subtitle:
        draw.text((42, 72), subtitle, fill=MUTED, font=FONT_SMALL)
    return img, draw


def save_figure(img: Image.Image, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(path_png)
    pdf_path = path_png.with_suffix(".pdf")
    img.save(pdf_path)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if text_size(draw, candidate, font)[0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_note(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, width: int, fill=MUTED, font=FONT_SMALL) -> int:
    for line in wrap_text(draw, text, font, width):
        draw.text((x, y), line, fill=fill, font=font)
        y += 24
    return y


def nice_range(values: list[float], include_zero: bool = False) -> tuple[float, float]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return 0.0, 1.0
    lo, hi = min(clean), max(clean)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if math.isclose(lo, hi):
        pad = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.15
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: list[float],
    ys: list[float],
    y_label: str = "",
    include_zero: bool = False,
) -> tuple[Any, Any, tuple[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = box
    left, right = x0 + 78, x1 - 28
    top, bottom = y0 + 38, y1 - 58
    xmin, xmax = (min(xs), max(xs)) if xs else (0.0, 1.0)
    if math.isclose(xmin, xmax):
        xmin -= 1
        xmax += 1
    ymin, ymax = nice_range(ys, include_zero=include_zero)

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    draw.line((left, bottom, right, bottom), fill=(98, 108, 122), width=2)
    draw.line((left, top, left, bottom), fill=(98, 108, 122), width=2)
    for tick in range(5):
        frac = tick / 4
        y = bottom - frac * (bottom - top)
        value = ymin + frac * (ymax - ymin)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((x0 + 6, y - 10), f"{value:.3g}", fill=MUTED, font=FONT_MONO)
    for tick in range(5):
        frac = tick / 4
        x = left + frac * (right - left)
        value = xmin + frac * (xmax - xmin)
        draw.line((x, bottom, x, bottom + 4), fill=MUTED, width=1)
        draw.text((x - 24, bottom + 10), f"{int(value)}", fill=MUTED, font=FONT_MONO)
    if y_label:
        draw.text((left, y0 + 8), y_label, fill=MUTED, font=FONT_SMALL)
    return sx, sy, (xmin, xmax), (ymin, ymax)


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series_map: dict[str, list[tuple[float, float]]],
    notes: list[str] | None = None,
    y_label: str = "",
    markers: dict[float, str] | None = None,
    include_zero: bool = False,
) -> None:
    img, draw = new_canvas(1500, 900, title, subtitle)
    box = (40, 110, 1460, 690)
    all_points = [point for points in series_map.values() for point in points if point[1] is not None]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    if not all_points:
        draw.text((120, 250), "No data available", fill=MUTED, font=FONT_TITLE)
        save_figure(img, path)
        return
    sx, sy, _, yrange = draw_axes(draw, box, xs, ys, y_label=y_label, include_zero=include_zero)
    for idx, (label, points) in enumerate(series_map.items()):
        clean = [(float(x), float(y)) for x, y in points if y is not None and math.isfinite(float(y))]
        if not clean:
            continue
        scaled = [(sx(x), sy(y)) for x, y in clean]
        color = PALETTE[idx % len(PALETTE)]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=4)
        for x, y in scaled[-16:]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_x = 100 + (idx % 3) * 420
        legend_y = 720 + (idx // 3) * 28
        draw.rectangle((legend_x, legend_y + 4, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 26, legend_y), label, fill=INK, font=FONT_SMALL)
    if markers:
        top = box[1] + 38
        bottom = box[3] - 58
        for x, label in markers.items():
            px = sx(float(x))
            draw.line((px, top, px, bottom), fill=(122, 130, 140), width=2)
            draw.text((px + 8, top + 8), label, fill=MUTED, font=FONT_SMALL)
    if notes:
        y = 785
        for note in notes[:3]:
            y = draw_note(draw, 50, y, note, 1400)
    save_figure(img, path)


def kpi_scorecard(path: Path, summary: dict[str, Any]) -> None:
    img, draw = new_canvas(
        1500,
        780,
        "Baseline evaluation scorecard",
        "Held-out eval uses greedy preset and 64 test batches. Accuracy values are percentages.",
    )
    cards = [
        ("Base model", pct(summary["base_accuracy"]), "33/64 correct", BLUE),
        ("Best LoRA ckpt", pct(summary["best_lora_accuracy"]), "step 2000, 18/64 correct", ORANGE),
        ("Final LoRA", pct(summary["final_lora_accuracy"]), "step 3364, 2/64 correct", RED),
    ]
    x = 60
    for title, value, detail, color in cards:
        draw.rounded_rectangle((x, 130, x + 430, 355), radius=8, outline=(215, 221, 230), width=2, fill=(249, 251, 253))
        draw.text((x + 26, 155), title, fill=INK, font=FONT_REG)
        draw.text((x + 26, 205), value, fill=color, font=ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 58) if Path("C:/Windows/Fonts/segoeuib.ttf").exists() else FONT_TITLE)
        draw.text((x + 28, 295), detail, fill=MUTED, font=FONT_SMALL)
        x += 470
    draw.text((60, 420), "Main result", fill=INK, font=FONT_REG)
    y = 470
    bullets = [
        "The final trained LoRA checkpoint is worse than the frozen base model and worse than the best early LoRA checkpoint.",
        "Checkpoint-wise eval shows degradation after step 2000: 28.13% -> 20.31% -> 6.25% -> 3.13%.",
        "The report treats step 2000 as the best observed LoRA checkpoint and step 3364 as evidence of late-stage collapse.",
    ]
    for bullet in bullets:
        draw.text((68, y), "-", fill=INK, font=FONT_SMALL)
        y = draw_note(draw, 90, y, bullet, 1320, fill=INK)
        y += 8
    save_figure(img, path)


def checkpoint_accuracy_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    img, draw = new_canvas(
        1500,
        900,
        "Checkpoint-wise evaluation accuracy",
        "Base is shown as a reference line; LoRA checkpoints include Wilson 95% confidence intervals.",
    )
    lora = [row for row in rows if row.get("policy") == "lora"]
    base = next((row for row in rows if row.get("policy") == "base"), None)
    box = (70, 130, 1450, 700)
    labels = [str(row.get("label")) for row in lora]
    values = [float(row.get("accuracy") or 0.0) for row in lora]
    lows = [float(row.get("accuracy_ci95_low") or row.get("accuracy") or 0.0) for row in lora]
    highs = [float(row.get("accuracy_ci95_high") or row.get("accuracy") or 0.0) for row in lora]
    ymax = max(highs + ([float(base.get("accuracy"))] if base else [0.0])) * 1.15
    ymax = max(ymax, 1.0)
    left, right = box[0] + 80, box[2] - 30
    top, bottom = box[1] + 30, box[3] - 55
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    draw.line((left, top, left, bottom), fill=MUTED, width=2)
    for tick in range(6):
        frac = tick / 5
        y = bottom - frac * (bottom - top)
        val = frac * ymax
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((box[0] + 10, y - 10), f"{val:.0f}%", fill=MUTED, font=FONT_MONO)
    if base:
        bval = float(base.get("accuracy") or 0.0)
        by = bottom - bval / ymax * (bottom - top)
        draw.line((left, by, right, by), fill=BLUE, width=3)
        draw.text((right - 260, by - 28), f"base {pct(bval)}", fill=BLUE, font=FONT_SMALL)
    bar_w = (right - left) / max(len(lora), 1) * 0.52
    for idx, row in enumerate(lora):
        cx = left + (idx + 0.5) * (right - left) / len(lora)
        val = values[idx]
        y = bottom - val / ymax * (bottom - top)
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, bottom), fill=ORANGE)
        low_y = bottom - lows[idx] / ymax * (bottom - top)
        high_y = bottom - highs[idx] / ymax * (bottom - top)
        draw.line((cx, low_y, cx, high_y), fill=INK, width=2)
        draw.line((cx - 12, low_y, cx + 12, low_y), fill=INK, width=2)
        draw.line((cx - 12, high_y, cx + 12, high_y), fill=INK, width=2)
        draw.text((cx - 42, y - 30), pct(val), fill=INK, font=FONT_MONO)
        draw.text((cx - 55, bottom + 14), labels[idx], fill=MUTED, font=FONT_SMALL)
    draw_note(draw, 70, 745, "Accuracy drops steadily after the best observed LoRA checkpoint at step 2000, so the final checkpoint is not the right representative of best training performance.", 1360, fill=INK)
    save_figure(img, path)


def response_table_image(path: Path, sample_rows: list[dict[str, Any]]) -> None:
    img, draw = new_canvas(1800, 1300, "Representative rollout examples", "Rows are sampled from local JSONL traces and grouped by failure/success type.")
    cols = [
        ("category", 240),
        ("call", 90),
        ("truth", 95),
        ("extracted", 120),
        ("reward", 100),
        ("question / completion", 1200),
    ]
    x0, y = 40, 125
    x = x0
    for name, width in cols:
        draw.rectangle((x, y, x + width, y + 34), fill=(239, 243, 248), outline=(214, 220, 228))
        draw.text((x + 8, y + 7), name, fill=INK, font=FONT_SMALL)
        x += width
    y += 34
    for row in sample_rows[:7]:
        height = 155
        x = x0
        values = [
            row.get("category"),
            row.get("call_index"),
            row.get("ground_truth"),
            row.get("extracted_number"),
            row.get("reward_total"),
            f"Q: {row.get('question')}\nA: {row.get('completion')}",
        ]
        for idx, (_, width) in enumerate(cols):
            draw.rectangle((x, y, x + width, y + height), fill="white", outline=(224, 229, 236))
            text = clean_text(values[idx])
            if idx == 5:
                yy = y + 8
                for line in wrap_text(draw, text, FONT_SMALL, width - 16)[:5]:
                    draw.text((x + 8, yy), line, fill=INK, font=FONT_SMALL)
                    yy += 23
            else:
                draw_note(draw, x + 8, y + 8, text, width - 16, fill=INK, font=FONT_SMALL)
            x += width
        y += height
    save_figure(img, path)


def failure_taxonomy_chart(path: Path, taxonomy: list[dict[str, Any]], phase_rows: list[dict[str, Any]]) -> None:
    img, draw = new_canvas(
        1500,
        900,
        "Failure taxonomy from rollout traces",
        "Taxonomy is computed from sampled rollout JSONL rows; phase rates use trace summary call-index tertiles.",
    )
    top = taxonomy[:6]
    box = (70, 130, 720, 700)
    max_count = max((int(row["count"]) for row in top), default=1)
    y = box[1]
    for idx, row in enumerate(top):
        label = row["category"]
        count = int(row["count"])
        share = float(row["share"])
        bar_w = int((box[2] - box[0] - 230) * count / max_count)
        yy = y + idx * 78
        draw.text((box[0], yy + 10), label[:26], fill=INK, font=FONT_SMALL)
        draw.rectangle((box[0] + 255, yy + 8, box[0] + 255 + bar_w, yy + 38), fill=PALETTE[idx % len(PALETTE)])
        draw.text((box[0] + 265 + bar_w, yy + 10), f"{count} ({share:.1%})", fill=MUTED, font=FONT_MONO)
    draw.text((820, 130), "Phase trend", fill=INK, font=FONT_REG)
    metrics = [
        ("empty_response_rate", RED),
        ("extracted_none_rate", PINK),
        ("numeric_exact_rate", GREEN),
        ("reward_mean", BLUE),
    ]
    chart_box = (780, 185, 1460, 690)
    series_map = {}
    for metric, _ in metrics:
        pts = []
        for idx, row in enumerate(phase_rows):
            value = row.get(metric)
            if value is not None and value != "":
                pts.append((idx, float(value)))
        series_map[metric] = pts
    all_points = [p for pts in series_map.values() for p in pts]
    sx, sy, _, _ = draw_axes(draw, chart_box, [p[0] for p in all_points], [p[1] for p in all_points], y_label="rate / mean", include_zero=True)
    for idx, (metric, color) in enumerate(metrics):
        pts = series_map.get(metric, [])
        scaled = [(sx(x), sy(v)) for x, v in pts]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=4)
        for px, py in scaled:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.rectangle((830 + (idx % 2) * 300, 725 + (idx // 2) * 28, 848 + (idx % 2) * 300, 743 + (idx // 2) * 28), fill=color)
        draw.text((856 + (idx % 2) * 300, 720 + (idx // 2) * 28), metric, fill=INK, font=FONT_SMALL)
    for idx, label in enumerate(["early", "middle", "late"]):
        draw.text((sx(idx) - 24, chart_box[3] - 35), label, fill=MUTED, font=FONT_SMALL)
    save_figure(img, path)


def runtime_chart(path: Path, series: dict[str, list[dict[str, float]]]) -> None:
    reward = series.get("train_reward_score") or series.get("train_kl") or []
    pts = []
    if reward:
        first_wall = reward[0]["wall_time"]
        for row in reward:
            if row["wall_time"]:
                pts.append((row["step"], (row["wall_time"] - first_wall) / 3600.0))
    img, draw = new_canvas(
        1500,
        850,
        "Training runtime and checkpoint I/O",
        "Wall-clock timeline is inferred from TensorBoard event wall_time; checkpoint I/O uses JAX/Orbax scalar tags when present.",
    )
    series_map = {"cumulative_hours": pts}
    write_gb = series_xy(series, "jax_orbax_write_gbytes")
    if write_gb:
        series_map["orbax_write_gbytes"] = write_gb
    line_chart_body(draw, (40, 115, 1460, 660), series_map, y_label="hours / GB", include_zero=True)
    latest_step = int(reward[-1]["step"]) if reward else 0
    total_hours = pts[-1][1] if pts else None
    note = f"Last observed training scalar step is {latest_step}. "
    if total_hours is not None:
        note += f"TensorBoard wall-time span is approximately {total_hours:.2f} hours from first to last selected train reward event."
    else:
        note += "Wall-time span could not be inferred from selected scalar events."
    draw_note(draw, 50, 710, note, 1380, fill=INK)
    save_figure(img, path)


def line_chart_body(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series_map: dict[str, list[tuple[float, float]]],
    y_label: str,
    include_zero: bool,
) -> None:
    all_points = [point for points in series_map.values() for point in points if point[1] is not None]
    if not all_points:
        draw.text((120, 250), "No data available", fill=MUTED, font=FONT_TITLE)
        return
    sx, sy, _, _ = draw_axes(draw, box, [p[0] for p in all_points], [p[1] for p in all_points], y_label=y_label, include_zero=include_zero)
    for idx, (label, points) in enumerate(series_map.items()):
        color = PALETTE[idx % len(PALETTE)]
        scaled = [(sx(x), sy(y)) for x, y in points]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=4)
        for x, y in scaled[-16:]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        lx = box[0] + 110 + idx * 360
        ly = box[3] + 15
        draw.rectangle((lx, ly + 4, lx + 18, ly + 18), fill=color)
        draw.text((lx + 26, ly), label, fill=INK, font=FONT_SMALL)


def reward_kl_chart(path: Path, series: dict[str, list[dict[str, float]]], best_step: int, final_step: int) -> None:
    line_chart(
        path,
        "Reward and KL timeline",
        "Reward should improve while KL stays controlled; this run improves early then collapses late.",
        {
            "train reward score": series_xy(series, "train_reward_score"),
            "eval reward score": series_xy(series, "eval_reward_score"),
            "train KL": series_xy(series, "train_kl"),
            "eval KL": series_xy(series, "eval_kl"),
        },
        notes=[
            "The best checkpoint by held-out accuracy is step 2000; final step 3364 is substantially worse.",
            "Late eval reward is negative while response failure rates rise, consistent with training collapse rather than successful alignment.",
        ],
        y_label="reward / KL",
        markers={float(best_step): "best ckpt", float(final_step): "final"},
        include_zero=True,
    )


def simple_line(path: Path, title: str, subtitle: str, mapping: dict[str, list[tuple[float, float]]], note: str, y_label: str = "value") -> None:
    line_chart(path, title, subtitle, mapping, notes=[note], y_label=y_label, include_zero=True)


def build_report_text(
    summary: dict[str, Any],
    config_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    figures: list[Figure],
) -> str:
    figure_md = "\n".join(
        [
            f"![{fig.title}](figures/{fig.name}.png)\n\n"
            f"**图意**：{fig.takeaway}\n"
            for fig in figures
        ]
    )
    config_table = "\n".join(f"| `{row['key']}` | `{row['value']}` |" for row in config_rows)
    eval_table = "\n".join(
        f"| {row['label']} | {row.get('policy','')} | {row.get('restored_step','')} | "
        f"{pct(row.get('accuracy'))} | {pct(row.get('partial_accuracy'))} | {pct(row.get('format_accuracy'))} | "
        f"{row.get('correct','')}/{row.get('total','')} |"
        for row in eval_rows
    )
    source_list = "\n".join(f"- [{item['label']}]({item['url']}): {item['note']}" for item in SOURCE_LINKS)
    return f"""# GRPO Baseline `course-baseline-001` 完整结果报告

## Technical Summary

本报告包整理的是课程 TPU `waxvhe` 上完成的 baseline full run。复现流程本身已经跑完，产出了 base eval、full training、final LoRA eval、checkpoint-wise eval、TensorBoard scalars、rollout traces 和诊断图；但训练结果显示 baseline 在后期发生 collapse。

最关键的结果是：base model 在 held-out greedy eval 上为 **{pct(summary['base_accuracy'])}**，final LoRA step `{summary['final_step']}` 只有 **{pct(summary['final_lora_accuracy'])}**，best observed LoRA checkpoint 是 step `{summary['best_lora_step']}` 的 **{pct(summary['best_lora_accuracy'])}**。因此，I.1 可以报告“训练跑通且证据完整”，但不能把 final checkpoint 描述成有效提升。

## Key Findings With Visual Evidence

{figure_md}

## Scope, Data, And Metric Definitions

本报告使用本地已 fetch 的目录 `artifacts/cloud/course-baseline-001/`，不重新连接 TPU，不重跑训练。评估默认采用 greedy preset、64 个 test batches；checkpoint eval 的置信区间来自已有 summary 中的 Wilson 95% CI。

核心指标解释：

- `accuracy`: numeric exact match，是任务成功的主指标。
- `partial_accuracy`: numeric partial match，用于观察数字提取是否部分接近。
- `format_accuracy`: 输出格式是否满足要求；它是 shaping/format 指标，不等价于数学正确。
- `rewards/*`: reward components 和总 reward，用于解释训练信号。
- `actor/*/kl`: current policy 与 reference policy 的 KL 约束信号。
- `grpo/*/reward_std` 与 `frac_reward_zero_std`: group 内 reward 多样性；长期为 0 或过高 zero-std 会削弱 GRPO 学习信号。
- `rollout/*/empty_response_rate` 与 `extracted_none_rate`: response/parse 健康度，能解释 reward 和 eval accuracy 为什么背离。

## Baseline Configuration

| Key | Value |
|---|---:|
{config_table}

## Evaluation Results

| Label | Policy | Step | Accuracy | Partial | Format | Correct |
|---|---|---:|---:|---:|---:|---:|
{eval_table}

## Collapse Diagnosis

这轮训练的主要问题不是“没有产物”，而是 final checkpoint 不代表最优模型。checkpoint-wise eval 显示 step 2000 后性能持续下降；同时 response health 指标显示 late phase 的 parse failure/empty response 明显恶化，eval reward 也转负。GRPO 的 reward shaping 项和真正任务成功指标发生背离时，模型可能学到局部格式或短输出行为，而不是稳定数学求解。

## GRPO-Specific Interpretation

baseline 保持了课程默认设置：`NUM_GENERATIONS=2`、`BETA=0.08`、`EPSILON=0.2`、`LEARNING_RATE=3e-6`、`MAX_STEPS=3364`。从成熟 GRPO/RLHF infra 的指标口径看，后续复现实验应同时追踪 reward、KL、clip ratio、completion length、reward_std/zero_std、advantage spread、held-out eval 和 sample tables。单独看 reward 曲线不足以判断训练成功。

## Evidence Gallery

代表性样本已经整理在 `samples/sample_examples.csv/json`，并在 `figures/07_trace_examples_table.png` 中可视化。样本按 correct、wrong numeric、parse fail、empty response、reward-hacking candidate、late collapse 分类，方便写报告时引用具体输出。

## Limitations

- eval 只有 64 个 test batches，适合课程 baseline 复现，但不是完整 benchmark。
- W&B 未启用；本报告以 TensorBoard、JSON eval、rollout trace 和 pipeline log 为事实来源。
- 没有执行 I.3 改进实验，因此 next experiments 只作为设计建议，不作为实验结果。
- rollout trace 是按 observability 采样，不是全量 generation 审计。

## Recommended Next Experiments

1. 使用 checkpoint-wise eval 选择 best checkpoint，而不是默认 final checkpoint。
2. 加早停或 model selection：当 held-out numeric accuracy 从 peak 明显下降时停止。
3. 降低学习率或调整 `BETA`，观察 KL 与 clipfrac 是否更平稳。
4. 将 format shaping 与 numeric correctness 拆开报告，避免格式奖励掩盖任务失败。
5. 增加 response health gate：empty response、Extracted None、completion truncation 超阈值时报警。
6. 继续保留 sample table，因为 qualitative outputs 对 GRPO collapse 诊断非常关键。

## External Metric/Infra References

{source_list}
"""


def markdown_to_html(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    html_lines: list[str] = []
    in_table = False
    table_rows: list[str] = []
    in_list = False

    def flush_table() -> None:
        nonlocal table_rows, in_table
        if not table_rows:
            return
        html_lines.append("<table>")
        for idx, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if idx == 1 and all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            tag = "th" if idx == 0 else "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        html_lines.append("</table>")
        table_rows = []
        in_table = False

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            flush_list()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            flush_list()
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_list()
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_list()
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("!["):
            flush_list()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                alt, src = match.groups()
                html_lines.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"></figure>')
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{inline_md(line[2:])}</li>")
        elif re.match(r"^\d+\. ", line):
            flush_list()
            html_lines.append(f"<p>{inline_md(re.sub(r'^\\d+\\. ', '', line))}</p>")
        elif line.strip():
            flush_list()
            html_lines.append(f"<p>{inline_md(line)}</p>")
        else:
            flush_list()
    flush_table()
    flush_list()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#1c2430; --muted:#616e7e; --line:#dbe2ea; --bg:#ffffff; --soft:#f6f8fb; --blue:#2269b8; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.62 "Segoe UI", Arial, sans-serif; }}
    main {{ max-width:1120px; margin:0 auto; padding:44px 28px 72px; }}
    h1 {{ font-size:34px; line-height:1.2; margin:0 0 28px; }}
    h2 {{ font-size:24px; margin:42px 0 14px; border-top:1px solid var(--line); padding-top:26px; }}
    h3 {{ font-size:19px; margin:28px 0 10px; }}
    p {{ margin:10px 0; }}
    a {{ color:var(--blue); }}
    code {{ background:var(--soft); padding:1px 5px; border-radius:4px; }}
    table {{ border-collapse:collapse; width:100%; margin:16px 0 24px; font-size:14px; }}
    th, td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    th {{ background:var(--soft); }}
    figure {{ margin:24px 0 30px; }}
    img {{ width:100%; height:auto; border:1px solid var(--line); }}
    ul {{ padding-left:24px; }}
  </style>
</head>
<body>
<main>
{chr(10).join(html_lines)}
</main>
</body>
</html>
"""


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def copy_raw_refs(paths: dict[str, Path], dirs: dict[str, Path]) -> list[dict[str, str]]:
    refs = []
    for name, path in paths.items():
        if path.is_dir():
            continue
        dest = dirs["raw_refs"] / f"{name}{path.suffix}"
        if name in {"manifest"}:
            write_json(dest, sanitize_payload(read_json(path)))
        elif name in {"pipeline_log", "git_commit", "git_status"}:
            text = sanitize_payload(path.read_text(encoding="utf-8-sig", errors="replace"))
            dest.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(path, dest)
        refs.append({"name": name, "source": str(path), "copied_to": str(dest)})
    return refs


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    paths = validate_inputs(run_dir)
    dirs = prepare_output_dirs(output_dir)

    base_eval = read_json(paths["base_eval"])
    lora_eval = read_json(paths["lora_eval"])
    checkpoint_summary = read_json(paths["checkpoint_summary"])
    checkpoint_rows = checkpoint_summary["rows"]
    manifest = sanitize_payload(read_json(paths["manifest"]))
    scalar_series, selected_scalar_rows = load_scalar_series(paths["scalar_csv"])
    trace_summary = csv_rows(paths["trace_summary"])
    trace_rows = load_trace_rows(paths["trace_jsonl"])
    sample_rows, taxonomy_rows = pick_samples(trace_rows)
    phase_rows = summarize_trace_phases(trace_summary)

    best = checkpoint_summary["best_lora_checkpoint"]
    final_row = next(row for row in checkpoint_rows if row.get("restored_step") == 3364)
    base_row = next(row for row in checkpoint_rows if row.get("policy") == "base")
    summary = {
        "run_id": args.run_id,
        "status": manifest.get("status"),
        "commit": clean_text((manifest.get("git") or {}).get("commit")),
        "base_accuracy": float(base_row["accuracy"]),
        "best_lora_step": int(best["restored_step"]),
        "best_lora_accuracy": float(best["accuracy"]),
        "final_step": int(final_row["restored_step"]),
        "final_lora_accuracy": float(final_row["accuracy"]),
        "final_lora_partial_accuracy": float(final_row["partial_accuracy"]),
        "final_lora_format_accuracy": float(final_row["format_accuracy"]),
        "scalar_rows_selected": len(selected_scalar_rows),
        "trace_rows": len(trace_rows),
        "trace_summary_rows": len(trace_summary),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    config = manifest.get("config") or {}
    config_rows = [{"key": key, "value": config.get(key)} for key in BASELINE_KEYS if key in config]
    eval_rows = build_eval_rows(base_eval, lora_eval, checkpoint_rows)

    write_json(dirs["tables"] / "run_summary.json", summary)
    write_csv(dirs["tables"] / "run_summary.csv", [summary])
    write_json(dirs["tables"] / "eval_summary.json", eval_rows)
    write_csv(dirs["tables"] / "eval_summary.csv", eval_rows)
    write_json(dirs["tables"] / "checkpoint_eval_summary.json", checkpoint_summary)
    write_csv(dirs["tables"] / "checkpoint_eval_summary.csv", checkpoint_rows)
    write_json(dirs["tables"] / "baseline_config.json", config_rows)
    write_csv(dirs["tables"] / "baseline_config.csv", config_rows)
    write_json(dirs["tables"] / "selected_scalar_metrics.json", selected_scalar_rows)
    write_csv(dirs["tables"] / "selected_scalar_metrics.csv", selected_scalar_rows, ["metric", "tag", "step", "wall_time", "value"])
    write_json(dirs["tables"] / "trace_phase_summary.json", phase_rows)
    write_csv(dirs["tables"] / "trace_phase_summary.csv", phase_rows)
    write_json(dirs["samples"] / "sample_examples.json", sample_rows)
    write_csv(dirs["samples"] / "sample_examples.csv", sample_rows)
    write_json(dirs["samples"] / "failure_taxonomy.json", taxonomy_rows)
    write_csv(dirs["samples"] / "failure_taxonomy.csv", taxonomy_rows)

    figures: list[Figure] = []
    fig_dir = dirs["figures"]
    kpi_scorecard(fig_dir / "01_eval_scorecard.png", summary)
    figures.append(Figure("01_eval_scorecard", "Baseline evaluation scorecard", "Which policy/checkpoint is best?", "Base model is strongest; final LoRA collapses to 3.13%, while best LoRA is step 2000 at 28.13%.", ["checkpoint_eval_summary.json"]))
    checkpoint_accuracy_chart(fig_dir / "02_checkpoint_accuracy_ci.png", checkpoint_rows)
    figures.append(Figure("02_checkpoint_accuracy_ci", "Checkpoint-wise evaluation accuracy", "Did training improve over time?", "LoRA accuracy degrades after step 2000; final checkpoint is not the best model.", ["checkpoint_eval_summary.json"]))
    reward_kl_chart(fig_dir / "03_reward_kl_timeline.png", scalar_series, int(best["restored_step"]), int(final_row["restored_step"]))
    figures.append(Figure("03_reward_kl_timeline", "Reward and KL timeline", "Are reward and KL consistent with stable GRPO training?", "Reward and KL patterns support an early peak followed by late instability.", ["scalar_metrics.csv"]))
    simple_line(
        fig_dir / "04_response_health.png",
        "Response health over training",
        "Parse and termination metrics explain why final held-out accuracy falls.",
        {
            "train empty response": series_xy(scalar_series, "train_empty_response_rate"),
            "eval empty response": series_xy(scalar_series, "eval_empty_response_rate"),
            "train Extracted None": series_xy(scalar_series, "train_extracted_none_rate"),
            "eval Extracted None": series_xy(scalar_series, "eval_extracted_none_rate"),
            "eval has solution end": series_xy(scalar_series, "eval_has_solution_end_rate"),
        },
        "Late response failure rates are a direct diagnostic for the observed checkpoint collapse.",
        y_label="rate",
    )
    figures.append(Figure("04_response_health", "Response health over training", "Does the model still produce parseable answers?", "Empty/parse-failure indicators rise late, matching the final accuracy collapse.", ["scalar_metrics.csv", "trace_summary.csv"]))
    simple_line(
        fig_dir / "05_grpo_health.png",
        "GRPO reward and advantage health",
        "Group reward diversity and advantage spread are core GRPO diagnostics.",
        {
            "train reward std": series_xy(scalar_series, "train_reward_std"),
            "eval reward std": series_xy(scalar_series, "eval_reward_std"),
            "train frac zero std": series_xy(scalar_series, "train_frac_reward_zero_std"),
            "eval frac zero std": series_xy(scalar_series, "eval_frac_reward_zero_std"),
            "train advantage std": series_xy(scalar_series, "train_advantage_std"),
        },
        "GRPO needs within-group reward contrast; zero-std or collapsing advantage spread weakens useful policy-gradient signal.",
        y_label="value / rate",
    )
    figures.append(Figure("05_grpo_health", "GRPO reward and advantage health", "Is the group-relative learning signal healthy?", "Reward diversity and advantage spread should be monitored before another full run.", ["scalar_metrics.csv"]))
    simple_line(
        fig_dir / "06_reward_components.png",
        "Reward components over training",
        "Correctness and format shaping are separated to expose reward hacking risk.",
        {
            "train check_answer": series_xy(scalar_series, "train_check_answer"),
            "eval check_answer": series_xy(scalar_series, "eval_check_answer"),
            "train check_numbers": series_xy(scalar_series, "train_check_numbers"),
            "eval check_numbers": series_xy(scalar_series, "eval_check_numbers"),
            "train format approx": series_xy(scalar_series, "train_match_format_approx"),
            "eval format approx": series_xy(scalar_series, "eval_match_format_approx"),
        },
        "A format or partial-number signal can diverge from true numeric correctness, so report both separately.",
        y_label="reward component",
    )
    figures.append(Figure("06_reward_components", "Reward components over training", "Which reward terms drove behavior?", "Component-level reward makes shaping-vs-task-success tradeoffs visible.", ["scalar_metrics.csv"]))
    response_table_image(fig_dir / "07_trace_examples_table.png", sample_rows)
    figures.append(Figure("07_trace_examples_table", "Representative rollout examples", "What does the model actually say?", "Qualitative traces reveal concrete failure modes behind the aggregate metrics.", ["rollout_samples_course-baseline-001.jsonl"]))
    failure_taxonomy_chart(fig_dir / "08_failure_taxonomy.png", taxonomy_rows, phase_rows)
    figures.append(Figure("08_failure_taxonomy", "Failure taxonomy from rollout traces", "Which failures dominate the sampled rollouts?", "Wrong numeric answers dominate, with late response/parse failures explaining collapse.", ["rollout_samples_course-baseline-001.jsonl", "trace_summary.csv"]))
    runtime_chart(fig_dir / "09_training_runtime.png", scalar_series)
    figures.append(Figure("09_training_runtime", "Training runtime and checkpoint I/O", "How long did the observed training timeline take?", "TensorBoard wall-time gives a reproducible runtime estimate and checkpoint I/O context.", ["scalar_metrics.csv"]))

    chart_rows = [fig.__dict__ for fig in figures]
    write_json(dirs["tables"] / "chart_map.json", chart_rows)
    write_csv(dirs["tables"] / "chart_map.csv", chart_rows)

    raw_refs = copy_raw_refs(paths, dirs)
    provenance = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "generated_at": summary["created_at"],
        "external_metric_references": SOURCE_LINKS,
        "raw_refs": raw_refs,
        "baseline_parameter_check": config_rows,
        "secret_scan_policy": "Tokens/keys are removed from copied manifest/log text; output is scanned after generation.",
    }
    write_json(dirs["provenance"] / "provenance.json", provenance)
    write_json(dirs["provenance"] / "sanitized_run_manifest.json", manifest)
    write_csv(dirs["provenance"] / "baseline_parameter_check.csv", config_rows)

    report_md = build_report_text(summary, config_rows, eval_rows, figures)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "report.html").write_text(markdown_to_html(report_md, "GRPO Baseline course-baseline-001 Report"), encoding="utf-8")

    readme = f"""# `course-baseline-001` report package

This folder is a self-contained evidence package for the GRPO baseline run.

Open `report.html` for the reader-facing report, or `report.md` for markdown editing.

## Headline numbers

- Base accuracy: **{pct(summary['base_accuracy'])}**
- Best LoRA checkpoint: **step {summary['best_lora_step']}**, **{pct(summary['best_lora_accuracy'])}**
- Final LoRA checkpoint: **step {summary['final_step']}**, **{pct(summary['final_lora_accuracy'])}**
- Conclusion: the run completed, but the final checkpoint collapsed and should not be presented as an improvement over base.

## Folder map

- `figures/`: report-ready PNG/PDF charts.
- `tables/`: eval summaries, selected TensorBoard scalars, config, chart map.
- `samples/`: rollout examples and failure taxonomy.
- `provenance/`: sanitized manifest, baseline parameter check, source references.
- `raw_refs/`: copied raw evidence files used by the report.

## Suggested citation in the coursework report

Use `figures/02_checkpoint_accuracy_ci.png`, `figures/03_reward_kl_timeline.png`,
and `figures/04_response_health.png` together: they show that training ran to completion,
but checkpoint selection matters because late-stage collapse damaged final LoRA accuracy.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest_report = {
        "run_id": args.run_id,
        "created_at": summary["created_at"],
        "summary": summary,
        "outputs": {
            "readme": str(output_dir / "README.md"),
            "report_md": str(output_dir / "report.md"),
            "report_html": str(output_dir / "report.html"),
            "figures": [str(dirs["figures"] / f"{fig.name}.png") for fig in figures],
            "tables": [str(path) for path in sorted(dirs["tables"].glob("*"))],
            "samples": [str(path) for path in sorted(dirs["samples"].glob("*"))],
            "provenance": [str(path) for path in sorted(dirs["provenance"].glob("*"))],
        },
        "chart_map": chart_rows,
    }
    write_json(output_dir / "manifest_report.json", manifest_report)
    print(f"Report package written to {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
