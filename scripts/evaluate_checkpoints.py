"""Evaluate base and multiple LoRA checkpoints, then summarize results.

This script intentionally shells out to evaluate.py for each checkpoint. That
keeps restore logic in one place and makes each eval JSON independently useful
for the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate base and LoRA checkpoints on GSM8K.")
    ap.add_argument("--ckpt-dir", required=True, help="Actor checkpoint root containing numeric step dirs.")
    ap.add_argument(
        "--steps",
        nargs="+",
        default=["auto"],
        help="Checkpoint steps to evaluate, or 'auto' to discover numeric directories.",
    )
    ap.add_argument("--include-base", action="store_true", help="Also run base model evaluation.")
    ap.add_argument("--preset", default="greedy", choices=["greedy", "standard", "liberal"])
    ap.add_argument("--source", default=None, choices=["tfds", "kaggle"])
    ap.add_argument("--num-passes", type=int, default=1)
    ap.add_argument(
        "--num-test-batches",
        type=int,
        default=None,
        help="Override NUM_TEST_BATCHES for this script's subprocesses.",
    )
    ap.add_argument("--output-dir", default="artifacts/checkpoint_eval")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def discover_steps(ckpt_dir: Path) -> list[int]:
    steps = []
    if not ckpt_dir.exists():
        return steps
    for child in ckpt_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return sorted(set(steps))


def resolve_steps(raw_steps: list[str], ckpt_dir: Path) -> list[int]:
    if len(raw_steps) == 1 and raw_steps[0].lower() == "auto":
        return discover_steps(ckpt_dir)
    steps = []
    for raw in raw_steps:
        if raw.lower() == "auto":
            steps.extend(discover_steps(ckpt_dir))
        else:
            steps.append(int(raw))
    return sorted(set(steps))


def run_eval(
    *,
    output_json: Path,
    preset: str,
    source: str | None,
    num_passes: int,
    env: dict[str, str],
    ckpt_dir: Path | None = None,
    step: int | None = None,
    dry_run: bool = False,
) -> int:
    cmd = [sys.executable, "-u", "evaluate.py", "--preset", preset, "--num-passes", str(num_passes)]
    if source:
        cmd += ["--source", source]
    if ckpt_dir is None:
        cmd += ["--no-restore"]
    else:
        cmd += ["--ckpt-dir", str(ckpt_dir), "--step", str(step)]
    cmd += ["--output-json", str(output_json)]

    print(" ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env)
    return result.returncode


def load_eval(path: Path, *, label: str, step: int | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    total = int(metrics.get("total") or 0)
    correct = int(metrics.get("correct") or 0)
    lo, hi = wilson_ci(correct, total)
    return {
        "label": label,
        "step": step,
        "file": str(path),
        "policy": payload.get("model", {}).get("policy"),
        "restored_step": payload.get("model", {}).get("restored_step"),
        "correct": correct,
        "total": total,
        "accuracy": float(metrics.get("accuracy") or 0.0),
        "accuracy_ci95_low": lo,
        "accuracy_ci95_high": hi,
        "partial_accuracy": float(metrics.get("partial_accuracy") or 0.0),
        "format_accuracy": float(metrics.get("format_accuracy") or 0.0),
    }


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = correct / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - margin) / denom * 100.0, (centre + margin) / denom * 100.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "step",
        "policy",
        "restored_step",
        "correct",
        "total",
        "accuracy",
        "accuracy_ci95_low",
        "accuracy_ci95_high",
        "partial_accuracy",
        "format_accuracy",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    print(f"Wrote {path}")


def render_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Pillow unavailable, skipping checkpoint plot: {exc}")
        return

    if not rows:
        return

    width, height = 1000, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin_left, margin_right, margin_top, margin_bottom = 72, 32, 52, 74
    left, right = margin_left, width - margin_right
    top, bottom = margin_top, height - margin_bottom
    draw.text((24, 18), "Checkpoint GSM8K accuracy", fill=(20, 20, 20))
    draw.line((left, top, left, bottom, right, bottom), fill=(70, 70, 70), width=2)

    lora_rows = [row for row in rows if row.get("step") is not None]
    base_rows = [row for row in rows if row.get("step") is None]
    xs = [int(row["step"]) for row in lora_rows] or [0]
    ys = [float(row["accuracy"]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    ymin = max(0.0, min(ys) - 5.0)
    ymax = min(100.0, max(ys) + 5.0)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    for tick in range(5):
        frac = tick / 4
        y = bottom - frac * (bottom - top)
        value = ymin + frac * (ymax - ymin)
        draw.line((left - 4, y, left, y), fill=(70, 70, 70))
        draw.text((12, y - 7), f"{value:.1f}%", fill=(70, 70, 70))

    if base_rows:
        base_acc = float(base_rows[0]["accuracy"])
        y = sy(base_acc)
        draw.line((left, y, right, y), fill=(120, 120, 120), width=2)
        draw.text((right - 180, y - 18), f"base {base_acc:.2f}%", fill=(80, 80, 80))

    points = [(sx(int(row["step"])), sy(float(row["accuracy"]))) for row in lora_rows]
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=3)
    for row, (x, y) in zip(lora_rows, points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(31, 119, 180))
        draw.text((x - 18, bottom + 14), str(row["step"]), fill=(70, 70, 70))
        draw.text((x - 20, y - 24), f"{row['accuracy']:.1f}", fill=(31, 119, 180))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"Wrote {path}")
    pdf = path.with_suffix(".pdf")
    img.save(pdf)
    print(f"Wrote {pdf}")


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = resolve_steps(args.steps, ckpt_dir)
    if not steps:
        print(f"No checkpoint steps found under {ckpt_dir}")

    env = dict(os.environ)
    if args.num_test_batches is not None:
        env["NUM_TEST_BATCHES"] = str(args.num_test_batches)

    eval_files: list[tuple[str, int | None, Path]] = []
    failures: list[dict[str, Any]] = []

    if args.include_base:
        path = output_dir / "base_eval.json"
        eval_files.append(("base", None, path))
        if not (args.skip_existing and path.exists()):
            code = run_eval(
                output_json=path,
                preset=args.preset,
                source=args.source,
                num_passes=args.num_passes,
                env=env,
                dry_run=args.dry_run,
            )
            if code != 0:
                failures.append({"label": "base", "step": None, "exit_code": code})
                if not args.continue_on_error:
                    return code

    for step in steps:
        path = output_dir / f"checkpoint_{step}_eval.json"
        eval_files.append((f"ckpt-{step}", step, path))
        if args.skip_existing and path.exists():
            continue
        code = run_eval(
            output_json=path,
            preset=args.preset,
            source=args.source,
            num_passes=args.num_passes,
            env=env,
            ckpt_dir=ckpt_dir,
            step=step,
            dry_run=args.dry_run,
        )
        if code != 0:
            failures.append({"label": f"ckpt-{step}", "step": step, "exit_code": code})
            if not args.continue_on_error:
                return code

    if args.dry_run:
        return 0

    rows = []
    for label, step, path in eval_files:
        if path.exists():
            rows.append(load_eval(path, label=label, step=step))

    lora_rows = [row for row in rows if row.get("step") is not None]
    best = None
    if lora_rows:
        best = max(
            lora_rows,
            key=lambda row: (
                float(row["accuracy"]),
                float(row["partial_accuracy"]),
                float(row["format_accuracy"]),
                int(row["step"]),
            ),
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ckpt_dir": str(ckpt_dir),
        "preset": args.preset,
        "num_passes": args.num_passes,
        "num_test_batches": args.num_test_batches or os.environ.get("NUM_TEST_BATCHES"),
        "steps": steps,
        "rows": rows,
        "best_lora_checkpoint": best,
        "failures": failures,
    }
    write_json(output_dir / "checkpoint_eval_summary.json", summary)
    write_csv(output_dir / "checkpoint_eval_summary.csv", rows)
    render_plot(output_dir / "checkpoint_accuracy.png", rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
