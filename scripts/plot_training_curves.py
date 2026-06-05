"""Plot current GRPO train/eval TensorBoard curves.

This is intentionally lightweight so it can run on the TPU VM inside the Tunix
venv without depending on matplotlib. It reads TensorBoard scalar events and
writes a PNG/PDF comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCORE_TAGS = {
    "train score mean": "rewards/train/score/mean",
    "eval score mean": "rewards/eval/score/mean",
}

KL_TAGS = {
    "train KL": "actor/train/kl",
    "eval KL": "actor/eval/kl",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot current GRPO train/eval curves.")
    ap.add_argument("--tensorboard-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return ap.parse_args()


def event_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted({p.parent for p in log_dir.rglob("events.out.tfevents*")})


def read_scalars(log_dir: Path, tags: dict[str, str]) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: dict[str, dict[int, float]] = {name: {} for name in tags}
    for directory in event_dirs(log_dir):
        acc = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for name, tag in tags.items():
            if tag not in available:
                continue
            for event in acc.Scalars(tag):
                series[name][int(event.step)] = float(event.value)
    return {name: sorted(points.items()) for name, points in series.items()}


def nice_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_panel(draw, box, title: str, series: dict[str, list[tuple[int, float]]], colors):
    x0, y0, x1, y1 = box
    left = x0 + 72
    right = x1 - 28
    top = y0 + 42
    bottom = y1 - 52

    draw.text((x0 + 16, y0 + 12), title, fill=(18, 22, 28))
    draw.line((left, top, left, bottom, right, bottom), fill=(72, 78, 86), width=2)

    all_points = [point for points in series.values() for point in points]
    if not all_points:
        draw.text((left + 16, top + 40), "No scalar data found", fill=(120, 120, 120))
        return

    xmin = min(step for step, _ in all_points)
    xmax = max(step for step, _ in all_points)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    ymin, ymax = nice_range([value for _, value in all_points])

    def sx(step: int) -> float:
        return left + (step - xmin) / (xmax - xmin) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for tick in range(5):
        frac = tick / 4
        x = left + frac * (right - left)
        y = bottom - frac * (bottom - top)
        step_label = str(int(round(xmin + frac * (xmax - xmin))))
        value_label = f"{ymin + frac * (ymax - ymin):.3g}"
        draw.line((x, top, x, bottom), fill=(232, 235, 239), width=1)
        draw.line((left, y, right, y), fill=(232, 235, 239), width=1)
        draw.text((x - 18, bottom + 12), step_label, fill=(74, 80, 88))
        draw.text((x0 + 10, y - 7), value_label, fill=(74, 80, 88))

    legend_x = right - 220
    legend_y = top + 6
    for idx, (name, points) in enumerate(series.items()):
        if not points:
            continue
        color = colors[idx % len(colors)]
        scaled = [(sx(step), sy(value)) for step, value in points]
        if len(scaled) == 1:
            x, y = scaled[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.line(scaled, fill=color, width=4)
        latest_step, latest_value = points[-1]
        label = f"{name}: {latest_value:.3g} @ {latest_step}"
        draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=color)
        draw.text((legend_x + 22, legend_y - 2), label, fill=(34, 38, 44))
        legend_y += 24


def write_rows(output_dir: Path, run_name: str, grouped: dict[str, list[tuple[int, float]]]) -> None:
    rows = []
    for metric, points in grouped.items():
        for step, value in points:
            rows.append({"run": run_name, "metric": metric, "step": step, "value": value})

    with (output_dir / "current_training_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "metric", "step", "value"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "current_training_curves.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render(output_dir: Path, score: dict[str, list[tuple[int, float]]], kl: dict[str, list[tuple[int, float]]], formats: list[str]) -> None:
    from PIL import Image, ImageDraw, JpegImagePlugin  # noqa: F401

    Image.init()
    width, height = 1280, 820
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    colors = [(20, 103, 178), (211, 74, 61), (35, 138, 91), (232, 139, 38)]

    draw_panel(draw, (24, 24, width - 24, 395), "Score mean: train vs eval", score, colors)
    draw_panel(draw, (24, 425, width - 24, height - 24), "KL: train vs eval", kl, colors)

    output_dir.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        path = output_dir / "current_training_curves.png"
        image.save(path)
        print(f"Wrote {path}")
    if "pdf" in formats:
        path = output_dir / "current_training_curves.pdf"
        image.save(path)
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    tb_dir = Path(args.tensorboard_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    score = read_scalars(tb_dir, SCORE_TAGS)
    kl = read_scalars(tb_dir, KL_TAGS)
    combined = {}
    combined.update(score)
    combined.update(kl)
    write_rows(output_dir, args.run_name, combined)
    render(output_dir, score, kl, args.formats)


if __name__ == "__main__":
    main()
