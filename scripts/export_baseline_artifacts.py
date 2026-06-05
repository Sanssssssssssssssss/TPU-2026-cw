"""Export baseline GRPO report artifacts from TensorBoard, W&B, and eval JSON.

Examples:
    python export_baseline_artifacts.py \
        --eval-json artifacts/base_eval.json \
        --eval-json artifacts/baseline_lora_eval.json

    python export_baseline_artifacts.py \
        --tensorboard-dir /tmp/content/tmp/tensorboard/grpo \
        --wandb-run entity/project/run_id
"""

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_TENSORBOARD_DIR = "/tmp/content/tmp/tensorboard/grpo"
DEFAULT_OUTPUT_DIR = "artifacts/baseline"
DEFAULT_REWARD_TAG = "rewards/train/score/mean"
DEFAULT_KL_TAG = "actor/train/kl"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export I.1 baseline tables and reward/KL plots."
    )
    ap.add_argument("--tensorboard-dir", default=DEFAULT_TENSORBOARD_DIR)
    ap.add_argument(
        "--wandb-run",
        default=None,
        help="Optional W&B run path in entity/project/run_id form.",
    )
    ap.add_argument("--reward-tag", default=DEFAULT_REWARD_TAG)
    ap.add_argument("--kl-tag", default=DEFAULT_KL_TAG)
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--eval-json",
        action="append",
        default=[],
        help="Evaluation JSON written by evaluate.py. May be passed multiple times.",
    )
    ap.add_argument(
        "--list-tags",
        action="store_true",
        help="Print TensorBoard scalar tags and exit.",
    )
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf"],
        help="Plot formats to write.",
    )
    return ap.parse_args()


def event_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted({p.parent for p in log_dir.rglob("events.out.tfevents*")})


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"TensorBoard event reader is unavailable: {exc}")
        return None
    return event_accumulator


def scalar_tags_from_tensorboard(log_dir: Path) -> list[str]:
    event_accumulator = load_event_accumulator()
    if event_accumulator is None:
        return []

    tags = set()
    for directory in event_dirs(log_dir):
        try:
            ea = event_accumulator.EventAccumulator(str(directory), size_guidance={"scalars": 0})
            ea.Reload()
            tags.update(ea.Tags().get("scalars", []))
        except Exception as exc:
            print(f"Skipping TensorBoard directory {directory}: {exc}")
    return sorted(tags)


def resolve_tag(requested: str, available: list[str]) -> str | None:
    if requested in available:
        return requested

    suffix_matches = [
        tag for tag in available
        if tag.endswith("/" + requested) or tag.endswith(requested)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    lower = requested.lower()
    lower_matches = [tag for tag in available if tag.lower() == lower]
    if len(lower_matches) == 1:
        return lower_matches[0]

    return None


def read_tensorboard_scalars(log_dir: Path, requested_tags: dict[str, str], run_name: str) -> list[dict]:
    event_accumulator = load_event_accumulator()
    if event_accumulator is None:
        return []

    directories = event_dirs(log_dir)
    if not directories:
        print(f"No TensorBoard event files found under {log_dir}")
        return []

    available = scalar_tags_from_tensorboard(log_dir)
    resolved = {}
    for metric, tag in requested_tags.items():
        actual = resolve_tag(tag, available)
        if actual is None:
            print(f"TensorBoard tag not found for {metric}: requested '{tag}'")
        else:
            resolved[metric] = actual

    rows = []
    for directory in directories:
        try:
            ea = event_accumulator.EventAccumulator(str(directory), size_guidance={"scalars": 0})
            ea.Reload()
            tags_here = set(ea.Tags().get("scalars", []))
        except Exception as exc:
            print(f"Skipping TensorBoard directory {directory}: {exc}")
            continue

        for metric, tag in resolved.items():
            if tag not in tags_here:
                continue
            for event in ea.Scalars(tag):
                rows.append({
                    "source": "tensorboard",
                    "run": run_name,
                    "metric": metric,
                    "tag": tag,
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                })
    return rows


def read_wandb_scalars(run_path: str, requested_tags: dict[str, str], run_name: str) -> list[dict]:
    try:
        import wandb
    except Exception as exc:
        print(f"W&B is unavailable: {exc}")
        return []

    rows = []
    try:
        run = wandb.Api().run(run_path)
    except Exception as exc:
        print(f"Could not open W&B run {run_path}: {exc}")
        return rows

    for metric, tag in requested_tags.items():
        try:
            history = run.scan_history(keys=["_step", tag])
            for item in history:
                step = item.get("_step")
                if step is None or tag not in item or item[tag] is None:
                    continue
                rows.append({
                    "source": "wandb",
                    "run": run_name,
                    "metric": metric,
                    "tag": tag,
                    "step": int(step),
                    "wall_time": None,
                    "value": float(item[tag]),
                })
        except Exception as exc:
            print(f"Could not read W&B history key {tag} for {run_path}: {exc}")
    return rows


def load_eval_rows(paths: list[str]) -> list[dict]:
    rows = []
    for raw in paths:
        path = Path(raw).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping eval JSON {path}: {exc}")
            continue

        model = payload.get("model", {})
        generation = payload.get("generation", {})
        metrics = payload.get("metrics", {})
        rows.append({
            "file": str(path),
            "policy": model.get("policy"),
            "checkpoint_restored": model.get("checkpoint_restored"),
            "restored_step": model.get("restored_step"),
            "preset": generation.get("preset"),
            "num_passes": generation.get("num_passes"),
            "correct": metrics.get("correct"),
            "total": metrics.get("total"),
            "accuracy": metrics.get("accuracy"),
            "partial_accuracy": metrics.get("partial_accuracy"),
            "format_accuracy": metrics.get("format_accuracy"),
        })
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    print(f"Wrote {path}")


def group_points(rows: list[dict], metric: str) -> dict[str, list[tuple[int, float]]]:
    grouped = {}
    for row in rows:
        if row["metric"] != metric:
            continue
        key = f"{row['run']} ({row['source']})"
        grouped.setdefault(key, []).append((row["step"], row["value"]))

    for key, points in grouped.items():
        dedup = {}
        for step, value in points:
            dedup[step] = value
        grouped[key] = sorted(dedup.items())
    return grouped


def nice_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_panel(draw, box, title: str, grouped: dict[str, list[tuple[int, float]]], colors):
    x0, y0, x1, y1 = box
    left = x0 + 58
    right = x1 - 18
    top = y0 + 34
    bottom = y1 - 42

    draw.text((x0 + 12, y0 + 8), title, fill=(20, 20, 20))
    draw.line((left, top, left, bottom, right, bottom), fill=(80, 80, 80), width=2)

    all_points = [point for points in grouped.values() for point in points]
    if not all_points:
        draw.text((left + 20, top + 40), "No scalar data found", fill=(120, 120, 120))
        return

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = nice_range(ys)
    if xmin == xmax:
        xmin -= 1
        xmax += 1

    def scale_x(x):
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def scale_y(y):
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    for tick in range(5):
        frac = tick / 4
        x = left + frac * (right - left)
        y = bottom - frac * (bottom - top)
        step_label = str(int(round(xmin + frac * (xmax - xmin))))
        value_label = f"{ymin + frac * (ymax - ymin):.3g}"
        draw.line((x, bottom, x, bottom + 4), fill=(80, 80, 80))
        draw.text((x - 16, bottom + 8), step_label, fill=(80, 80, 80))
        draw.line((left - 4, y, left, y), fill=(80, 80, 80))
        draw.text((x0 + 6, y - 7), value_label, fill=(80, 80, 80))

    legend_y = top
    for idx, (name, points) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        if len(points) == 1:
            x = scale_x(points[0][0])
            y = scale_y(points[0][1])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            scaled = [(scale_x(x), scale_y(y)) for x, y in points]
            draw.line(scaled, fill=color, width=3)
        draw.rectangle((right - 150, legend_y, right - 138, legend_y + 12), fill=color)
        draw.text((right - 132, legend_y - 2), name[:24], fill=(40, 40, 40))
        legend_y += 18


def render_plot(rows: list[dict], output_dir: Path, formats: list[str]) -> None:
    try:
        from PIL import Image, ImageDraw, JpegImagePlugin  # noqa: F401
        Image.init()
    except Exception as exc:
        print(f"Pillow is unavailable, skipping plots: {exc}")
        return

    width, height = 1200, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (255, 127, 14),
        (148, 103, 189),
    ]

    reward = group_points(rows, "mean_reward")
    kl = group_points(rows, "kl")
    draw_panel(draw, (20, 20, width - 20, 365), "Mean reward vs GRPO step", reward, colors)
    draw_panel(draw, (20, 390, width - 20, height - 20), "KL vs GRPO step", kl, colors)

    output_dir.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        path = output_dir / "baseline_curves.png"
        img.save(path)
        print(f"Wrote {path}")
    if "pdf" in formats:
        path = output_dir / "baseline_curves.pdf"
        img.save(path)
        print(f"Wrote {path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    requested_tags = {
        "mean_reward": args.reward_tag,
        "kl": args.kl_tag,
    }

    tb_dir = Path(args.tensorboard_dir).expanduser()
    if args.list_tags:
        for tag in scalar_tags_from_tensorboard(tb_dir):
            print(tag)
        return

    metric_rows = read_tensorboard_scalars(tb_dir, requested_tags, args.run_name)
    if args.wandb_run:
        metric_rows.extend(read_wandb_scalars(args.wandb_run, requested_tags, args.run_name))

    eval_rows = load_eval_rows(args.eval_json)

    write_json(output_dir / "scalar_metrics.json", metric_rows)
    write_csv(
        output_dir / "scalar_metrics.csv",
        metric_rows,
        ["source", "run", "metric", "tag", "step", "wall_time", "value"],
    )
    write_json(output_dir / "eval_summary.json", eval_rows)
    write_csv(
        output_dir / "eval_summary.csv",
        eval_rows,
        [
            "file",
            "policy",
            "checkpoint_restored",
            "restored_step",
            "preset",
            "num_passes",
            "correct",
            "total",
            "accuracy",
            "partial_accuracy",
            "format_accuracy",
        ],
    )
    render_plot(metric_rows, output_dir, args.formats)


if __name__ == "__main__":
    main()
