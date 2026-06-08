"""Build report-ready diagnostics from a GRPO run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCALAR_CANDIDATES = {
    "train_score_mean": ["rewards/train/score/mean"],
    "eval_score_mean": ["rewards/eval/score/mean"],
    "train_kl": ["actor/train/kl"],
    "eval_kl": ["actor/eval/kl"],
    "train_loss": ["actor/train/loss"],
    "eval_loss": ["actor/eval/loss"],
    "train_pg_clipfrac": ["actor/train/pg_clipfrac", "actor/train/clipfrac"],
    "empty_response_rate": [
        "rollout/train/empty_response_rate",
        "rollout/empty_response_rate",
        "observability/rollout/empty_response_rate",
    ],
    "extracted_none_rate": [
        "rollout/train/extracted_none_rate",
        "rollout/extracted_none_rate",
        "observability/rollout/extracted_none_rate",
    ],
    "numeric_exact_rate": ["eval/train/numeric_exact_rate", "eval/numeric_exact_rate"],
    "format_accuracy": ["eval/train/format_accuracy", "eval/format_accuracy"],
    "reward_std": ["grpo/train/reward_std", "grpo/reward_std"],
    "frac_reward_zero_std": ["grpo/train/frac_reward_zero_std", "grpo/frac_reward_zero_std"],
    "advantage_std": ["grpo/train/advantage_std", "grpo/advantage_std"],
    "completion_chars": ["rollout/train/mean_completion_chars", "rollout/mean_completion_chars"],
    "completion_length": ["completions/train/mean_length", "completions/mean_length"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze GRPO run scalars, traces, and checkpoint evals.")
    ap.add_argument("--run-dir", default=None, help="Run directory containing tensorboard/ and artifacts/.")
    ap.add_argument("--tensorboard-dir", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--checkpoint-summary", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return ap.parse_args()


def event_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted({p.parent for p in log_dir.rglob("events.out.tfevents*")})


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"TensorBoard reader unavailable: {exc}")
        return None
    return event_accumulator


def read_tensorboard_scalars(log_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    event_accumulator = load_event_accumulator()
    if event_accumulator is None:
        return [], []

    rows: list[dict[str, Any]] = []
    tags_seen: set[str] = set()
    for directory in event_dirs(log_dir):
        try:
            ea = event_accumulator.EventAccumulator(str(directory), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception as exc:
            print(f"Skipping TensorBoard directory {directory}: {exc}")
            continue
        for tag in sorted(ea.Tags().get("scalars", [])):
            tags_seen.add(tag)
            canonical = canonical_metric_name(tag)
            for event in ea.Scalars(tag):
                rows.append(
                    {
                        "source": "tensorboard",
                        "metric": canonical,
                        "tag": tag,
                        "step": int(event.step),
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                    }
                )
    return rows, sorted(tags_seen)


def canonical_metric_name(tag: str) -> str:
    for name, candidates in SCALAR_CANDIDATES.items():
        for candidate in candidates:
            if tag == candidate or tag.endswith("/" + candidate) or tag.endswith(candidate):
                return name
    normalized = tag.replace("/", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def read_traces(trace_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not trace_dir.exists():
        return [], []

    rows = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_file"] = str(path)
                rows.append(row)

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row.get("call_index") or 0), str(row.get("dataset_role") or "unknown"))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (call_index, dataset_role), items in sorted(grouped.items()):
        n = len(items)
        empty = sum(1 for item in items if not str(item.get("completion") or "").strip())
        extracted_none = sum(1 for item in items if item.get("extracted_number") in (None, ""))
        exact = sum(1 for item in items if bool(item.get("numeric_exact")))
        fmt = sum(1 for item in items if bool(item.get("format_ok")))
        reward_values = [item.get("reward_total") for item in items if item.get("reward_total") is not None]
        completion_chars = [int(item.get("completion_chars") or 0) for item in items]
        summary.append(
            {
                "call_index": call_index,
                "dataset_role": dataset_role,
                "sampled_rows": n,
                "empty_response_rate": ratio(empty, n),
                "extracted_none_rate": ratio(extracted_none, n),
                "numeric_exact_rate": ratio(exact, n),
                "format_accuracy": ratio(fmt, n),
                "reward_mean": mean(reward_values),
                "reward_std": std(reward_values),
                "completion_chars_mean": mean(completion_chars),
            }
        )
    return rows, summary


def read_checkpoint_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read checkpoint summary {path}: {exc}")
        return []
    return list(payload.get("rows") or [])


def ratio(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def std(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / len(clean))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names = set()
        for row in rows:
            names.update(row)
        fieldnames = sorted(names)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    print(f"Wrote {path}")


def series_from_rows(rows: list[dict[str, Any]], *metric_names: str) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, dict[int, float]] = {name: {} for name in metric_names}
    for row in rows:
        metric = row.get("metric")
        if metric not in out:
            continue
        out[metric][int(row["step"])] = float(row["value"])
    return {name: sorted(points.items()) for name, points in out.items() if points}


def render_dashboard(
    output_dir: Path,
    scalar_rows: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    formats: list[str],
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Pillow unavailable, skipping diagnostics plot: {exc}")
        return

    width, height = 1500, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((28, 18), "GRPO training diagnostics", fill=(20, 20, 20))
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (255, 127, 14),
        (148, 103, 189),
    ]

    panels = [
        ((24, 56, 730, 380), "Reward score mean", series_from_rows(scalar_rows, "train_score_mean", "eval_score_mean")),
        ((770, 56, 1476, 380), "KL", series_from_rows(scalar_rows, "train_kl", "eval_kl")),
        (
            (24, 420, 730, 744),
            "Response failures",
            merge_trace_series(
                series_from_rows(scalar_rows, "empty_response_rate", "extracted_none_rate"),
                trace_summary,
                ["empty_response_rate", "extracted_none_rate"],
            ),
        ),
        (
            (770, 420, 1476, 744),
            "GRPO health",
            series_from_rows(scalar_rows, "reward_std", "frac_reward_zero_std", "advantage_std"),
        ),
        (
            (24, 784, 730, 1128),
            "Checkpoint accuracy",
            checkpoint_series(checkpoint_rows),
        ),
        (
            (770, 784, 1476, 1128),
            "Completion length / clip",
            series_from_rows(scalar_rows, "completion_length", "completion_chars", "train_pg_clipfrac"),
        ),
    ]

    for idx, (box, title, grouped) in enumerate(panels):
        draw_panel(draw, box, title, grouped, colors[idx:] + colors[:idx])

    output_dir.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        path = output_dir / "grpo_diagnostics.png"
        img.save(path)
        print(f"Wrote {path}")
    if "pdf" in formats:
        path = output_dir / "grpo_diagnostics.pdf"
        try:
            img.save(path)
            print(f"Wrote {path}")
        except Exception as exc:
            print(f"Could not write PDF plot {path}; PNG is still available: {exc}")


def merge_trace_series(
    scalar_series: dict[str, list[tuple[int, float]]],
    trace_summary: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, list[tuple[int, float]]]:
    out = dict(scalar_series)
    for name in metric_names:
        if name in out:
            continue
        points = [
            (int(row["call_index"]), float(row[name]))
            for row in trace_summary
            if row.get(name) is not None
        ]
        if points:
            out[f"trace_{name}"] = points
    return out


def checkpoint_series(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    points = []
    base = []
    for row in rows:
        step = row.get("step")
        if step is None:
            base.append((0, float(row.get("accuracy") or 0.0)))
        else:
            points.append((int(step), float(row.get("accuracy") or 0.0)))
    out = {}
    if points:
        out["lora_accuracy"] = sorted(points)
    if base and points:
        x0 = min(step for step, _ in points)
        x1 = max(step for step, _ in points)
        out["base_accuracy"] = [(x0, base[0][1]), (x1, base[0][1])]
    elif base:
        out["base_accuracy"] = base
    return out


def draw_panel(draw, box, title: str, grouped: dict[str, list[tuple[int, float]]], colors: list[tuple[int, int, int]]):
    x0, y0, x1, y1 = box
    left, right = x0 + 62, x1 - 20
    top, bottom = y0 + 42, y1 - 44
    draw.rectangle(box, outline=(220, 220, 220))
    draw.text((x0 + 12, y0 + 12), title, fill=(20, 20, 20))
    draw.line((left, top, left, bottom, right, bottom), fill=(80, 80, 80), width=2)

    all_points = [point for points in grouped.values() for point in points]
    if not all_points:
        draw.text((left + 20, top + 50), "No data", fill=(130, 130, 130))
        return

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = nice_range(ys)
    if xmin == xmax:
        xmin -= 1
        xmax += 1

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    for tick in range(4):
        frac = tick / 3
        y = bottom - frac * (bottom - top)
        value = ymin + frac * (ymax - ymin)
        draw.line((left - 4, y, left, y), fill=(90, 90, 90))
        draw.text((x0 + 8, y - 7), f"{value:.3g}", fill=(90, 90, 90))

    legend_y = top
    for idx, (name, points) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        scaled = [(sx(step), sy(value)) for step, value in sorted(points)]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=3)
        for x, y in scaled[-12:]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.rectangle((right - 172, legend_y, right - 160, legend_y + 12), fill=color)
        draw.text((right - 154, legend_y - 2), name[:26], fill=(50, 50, 50))
        legend_y += 18


def nice_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    tensorboard_dir = Path(args.tensorboard_dir).expanduser() if args.tensorboard_dir else None
    trace_dir = Path(args.trace_dir).expanduser() if args.trace_dir else None
    checkpoint_summary = Path(args.checkpoint_summary).expanduser() if args.checkpoint_summary else None

    if run_dir is not None:
        tensorboard_dir = tensorboard_dir or run_dir / "tensorboard"
        trace_dir = trace_dir or run_dir / "artifacts" / "rollout_traces"
        checkpoint_summary = checkpoint_summary or run_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json"
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "artifacts" / "analysis"
    else:
        tensorboard_dir = tensorboard_dir or Path("artifacts/tensorboard")
        trace_dir = trace_dir or Path("artifacts/observability/rollout_traces")
        output_dir = Path(args.output_dir or "artifacts/analysis").expanduser()

    scalar_rows, tags = read_tensorboard_scalars(tensorboard_dir)
    trace_rows, trace_summary = read_traces(trace_dir)
    checkpoint_rows = read_checkpoint_summary(checkpoint_summary)

    write_json(output_dir / "tensorboard_tags.json", tags)
    write_json(output_dir / "scalar_metrics.json", scalar_rows)
    write_csv(output_dir / "scalar_metrics.csv", scalar_rows, ["source", "metric", "tag", "step", "wall_time", "value"])
    write_json(output_dir / "trace_rows_sample.json", trace_rows[:500])
    write_json(output_dir / "trace_summary.json", trace_summary)
    write_csv(output_dir / "trace_summary.csv", trace_summary)
    if checkpoint_rows:
        write_json(output_dir / "checkpoint_eval_rows.json", checkpoint_rows)
        write_csv(output_dir / "checkpoint_eval_rows.csv", checkpoint_rows)
    render_dashboard(output_dir, scalar_rows, trace_summary, checkpoint_rows, args.formats)


if __name__ == "__main__":
    main()
