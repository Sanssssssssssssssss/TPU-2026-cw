"""Shared helpers for final rollout-aligned GRPO report packages."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROLLOUTS = [
    320,
    640,
    960,
    1280,
    1600,
    1920,
    2240,
    2560,
    2880,
    3200,
    3520,
    3840,
    4160,
    4480,
    4800,
    5120,
    5440,
    5760,
    6080,
    6400,
    6720,
    6728,
]

EXPECTED_K2_STEPS = [
    160,
    320,
    480,
    640,
    800,
    960,
    1120,
    1280,
    1440,
    1600,
    1760,
    1920,
    2080,
    2240,
    2400,
    2560,
    2720,
    2880,
    3040,
    3200,
    3360,
    3364,
]

EXPECTED_K8_STEPS = [
    40,
    80,
    120,
    160,
    200,
    240,
    280,
    320,
    360,
    400,
    440,
    480,
    520,
    560,
    600,
    640,
    680,
    720,
    760,
    800,
    840,
    841,
]


@dataclass(frozen=True)
class OfficialRun:
    key: str
    run_id: str
    branch: str
    legend: str
    reward_mode: str
    num_generations: int
    max_steps: int
    checkpoint_steps: list[int]
    learning_rate: str
    beta: str
    rank: str
    alpha: str
    advantage_estimator: str = "grpo"

    @property
    def final_step(self) -> int:
        return self.checkpoint_steps[-1]


RUNS: list[OfficialRun] = []

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

SERIES_COLORS: dict[str, str] = {}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                pass
        return out
    return [int(part) for part in str(value).replace(",", " ").split() if part.strip().lstrip("-").isdigit()]


def metric_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"TensorBoard scalar reader unavailable: {exc}")
        return None
    return event_accumulator


def canonical_metric_name(tag: str) -> str:
    if tag == "rewards/train/score/mean":
        return "train_reward_score"
    if tag == "rewards/eval/score/mean":
        return "eval_reward_score"
    if tag == "actor/train/kl":
        return "train_kl"
    if tag == "actor/eval/kl":
        return "eval_kl"
    if tag == "actor/train/loss":
        return "train_loss"
    if tag == "actor/eval/loss":
        return "eval_loss"
    parts = tag.split("/")
    if len(parts) >= 3 and parts[1] in {"train", "eval"}:
        namespace, role = parts[0], parts[1]
        name = "_".join(parts[2:])
        if namespace == "rollout":
            return f"{role}_rollout_{name}"
        if namespace == "reward":
            return f"{role}_reward_{name}"
        if namespace == "grpo":
            return f"{role}_grpo_{name}"
        return f"{role}_{namespace}_{name}"
    return tag.replace("/", "_").replace("-", "_").strip("_")


def read_tensorboard_scalars(tensorboard_dir: Path, run: OfficialRun) -> list[dict[str, Any]]:
    event_accumulator = load_event_accumulator()
    if event_accumulator is None or not tensorboard_dir.exists():
        return read_scalar_pivot_fallback(tensorboard_dir, run)
    rows: list[dict[str, Any]] = []
    event_dirs = sorted({path.parent for path in tensorboard_dir.rglob("events.out.tfevents*")})
    for directory in event_dirs:
        try:
            acc = event_accumulator.EventAccumulator(str(directory), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:
            print(f"Skipping TensorBoard directory {directory}: {exc}")
            continue
        for tag in sorted(acc.Tags().get("scalars", [])):
            metric = canonical_metric_name(tag)
            for event in acc.Scalars(tag):
                step = int(event.step)
                rows.append(
                    {
                        "line": run.key,
                        "run_id": run.run_id,
                        "branch": run.branch,
                        "legend": run.legend,
                        "metric": metric,
                        "tag": tag,
                        "step": step,
                        "rollouts_seen": step * run.num_generations,
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                    }
                )
    rows.sort(key=lambda row: (row["line"], row["metric"], row["step"], row["tag"]))
    return rows


def read_scalar_pivot_fallback(tensorboard_dir: Path, run: OfficialRun) -> list[dict[str, Any]]:
    """Read compact sweep scalar tables when TensorBoard event parsing is unavailable."""
    try:
        run_dir = tensorboard_dir.parent.parent.parent
    except IndexError:
        return []
    pivot_path = run_dir / "artifacts" / "sweep_analysis" / "tables" / "scalar_pivot.csv"
    pivot_rows = read_csv_rows(pivot_path)
    if not pivot_rows:
        return []
    metadata = {"step", "line", "line_label", "run_id", "branch", "num_generations", "rollouts_seen"}
    rows: list[dict[str, Any]] = []
    for raw in pivot_rows:
        try:
            step = int(float(raw.get("step") or 0))
        except (TypeError, ValueError):
            continue
        for metric, value in raw.items():
            if metric in metadata:
                continue
            number = metric_float(value)
            if number is None:
                continue
            rows.append(
                {
                    "line": run.key,
                    "run_id": run.run_id,
                    "branch": run.branch,
                    "legend": run.legend,
                    "metric": metric,
                    "tag": f"scalar_pivot/{metric}",
                    "step": step,
                    "rollouts_seen": step * run.num_generations,
                    "wall_time": "",
                    "value": number,
                }
            )
    rows.sort(key=lambda row: (row["line"], row["metric"], row["step"], row["tag"]))
    return rows


def checkpoint_archive_count(run_dir: Path) -> int:
    archive_dir = run_dir / "checkpoint_archives"
    if archive_dir.exists():
        return len(list(archive_dir.glob("*.tar.gz")))
    manifest = run_dir / "checkpoint_archives.txt"
    if manifest.is_file():
        return sum(1 for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    return 0


def checkpoint_exists(run_dir: Path, run: OfficialRun) -> bool:
    local_ckpt = run_dir / "runs" / run.branch / "ckpts" / "actor" / str(run.final_step)
    if local_ckpt.exists():
        return True
    archive_dir = run_dir / "checkpoint_archives"
    if archive_dir.exists():
        suffix = f"-{run.final_step}.tar.gz"
        return any(path.name.endswith(suffix) and run.branch in path.name for path in archive_dir.glob("*.tar.gz"))
    return False


def verify_one(run_dir: Path, run: OfficialRun) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    branch_dir = run_dir / "runs" / run.branch
    manifest_path = run_dir / "artifacts" / "reward_k8_pilot_manifest.json"
    env_path = branch_dir / "run_env.txt"
    eval_csv = branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.csv"
    eval_json = branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json"

    for path, label in [
        (run_dir / "pipeline.log", "pipeline log"),
        (manifest_path, "K8 pilot manifest"),
        (env_path, "run env"),
        (branch_dir / "train.log", "train log"),
        (eval_csv, "checkpoint eval CSV"),
        (eval_json, "checkpoint eval JSON"),
    ]:
        if not path.is_file():
            errors.append(f"missing {label}: {path}")

    if not any((branch_dir / "tensorboard").rglob("events.out.tfevents*")):
        errors.append(f"missing TensorBoard events: {branch_dir / 'tensorboard'}")
    if not any((branch_dir / "artifacts" / "rollout_traces").glob("*.jsonl")):
        errors.append(f"missing rollout traces: {branch_dir / 'artifacts' / 'rollout_traces'}")
    if not any((branch_dir / "artifacts" / "checkpoint_eval_examples").glob("*.jsonl")):
        warnings.append(f"missing checkpoint eval examples JSONL: {branch_dir / 'artifacts' / 'checkpoint_eval_examples'}")

    manifest = read_json(manifest_path)
    env = parse_env(env_path)
    source_checkpoint = manifest.get("source_checkpoint")
    if source_checkpoint not in (None, {}, "null", ""):
        errors.append(f"source_checkpoint is not null: {source_checkpoint!r}")
    source_keys = sorted(key for key in env if key.startswith("K8_SOURCE_"))
    if source_keys:
        errors.append(f"source checkpoint env keys present: {source_keys}")

    expected_env = {
        "REWARD_MODE": run.reward_mode,
        "NUM_GENERATIONS": str(run.num_generations),
        "MAX_STEPS": str(run.max_steps),
        "LEARNING_RATE": run.learning_rate,
        "BETA": run.beta,
        "RANK": run.rank,
        "ALPHA": run.alpha,
        "MAX_TO_KEEP": "30",
        "K8_ROLLOUT_CHECKPOINT_INTERVAL": "320",
    }
    if run.advantage_estimator != "grpo":
        expected_env["GRPO_ADVANTAGE_ESTIMATOR"] = run.advantage_estimator
    manifest_key_by_env = {
        "MAX_STEPS": "max_steps",
        "LEARNING_RATE": "learning_rate",
        "BETA": "beta",
        "RANK": "rank",
        "ALPHA": "alpha",
        "MAX_TO_KEEP": "max_to_keep",
        "K8_ROLLOUT_CHECKPOINT_INTERVAL": "rollout_checkpoint_interval",
        "GRPO_ADVANTAGE_ESTIMATOR": "advantage_estimator",
    }
    for key, expected in expected_env.items():
        manifest_key = manifest_key_by_env.get(key, key.lower())
        actual = env.get(key) or str(manifest.get(manifest_key, ""))
        if actual != expected:
            errors.append(f"{run.run_id} {key} expected {expected}, got {actual or '<missing>'}")

    checkpoint_steps = parse_int_list(manifest.get("checkpoint_steps") or manifest.get("checkpoint_eval_steps"))
    if checkpoint_steps != run.checkpoint_steps:
        errors.append(f"checkpoint steps mismatch: expected {run.checkpoint_steps}, got {checkpoint_steps}")
    checkpoint_rollouts = parse_int_list(manifest.get("checkpoint_rollouts") or env.get("K8_CHECKPOINT_ROLLOUTS"))
    if checkpoint_rollouts != EXPECTED_ROLLOUTS:
        errors.append(f"checkpoint rollouts mismatch: expected {EXPECTED_ROLLOUTS}, got {checkpoint_rollouts}")

    archive_count = checkpoint_archive_count(run_dir)
    if archive_count < 22:
        errors.append(f"checkpoint archive count expected >=22, got {archive_count}")
    if not checkpoint_exists(run_dir, run):
        errors.append(f"missing final checkpoint {run.final_step}")

    eval_rows_raw = read_csv_rows(eval_csv)
    if len(eval_rows_raw) != 22:
        errors.append(f"checkpoint eval CSV expected 22 rows, got {len(eval_rows_raw)}")

    eval_steps = []
    checkpoint_rows: list[dict[str, Any]] = []
    for raw in eval_rows_raw:
        step = int(float(raw.get("step") or 0))
        eval_steps.append(step)
        row: dict[str, Any] = {
            "line": run.key,
            "run_id": run.run_id,
            "branch": run.branch,
            "legend": run.legend,
            "reward_mode": run.reward_mode,
            "num_generations": run.num_generations,
            "step": step,
            "rollouts_seen": step * run.num_generations,
        }
        for key, value in raw.items():
            if key not in row:
                row[key] = value
        checkpoint_rows.append(row)
    if eval_steps and eval_steps != run.checkpoint_steps:
        errors.append(f"checkpoint eval row steps mismatch: expected {run.checkpoint_steps}, got {eval_steps}")
    row_rollouts = [row["rollouts_seen"] for row in checkpoint_rows]
    if row_rollouts and row_rollouts != EXPECTED_ROLLOUTS:
        errors.append(f"rollouts_seen mismatch: expected {EXPECTED_ROLLOUTS}, got {row_rollouts}")

    scalars = read_tensorboard_scalars(branch_dir / "tensorboard", run)
    status = {
        "line": run.key,
        "run_id": run.run_id,
        "branch": run.branch,
        "raw_dir": str(run_dir),
        "manifest": str(manifest_path),
        "run_env": str(env_path),
        "checkpoint_eval_csv": str(eval_csv),
        "checkpoint_eval_rows": len(eval_rows_raw),
        "checkpoint_archive_count": archive_count,
        "final_checkpoint_step": run.final_step,
        "final_checkpoint_exists": checkpoint_exists(run_dir, run),
        "scalar_rows": len(scalars),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    return status, checkpoint_rows, scalars


def setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.95,
            "font.family": ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )
    return plt


def add_header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))


def plot_checkpoint_metric(rows: list[dict[str, Any]], metric: str, title: str, subtitle: str, path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    add_header(fig, title, subtitle)
    for run in RUNS:
        series = [row for row in rows if row["line"] == run.key]
        points = []
        for row in series:
            value = metric_float(row.get(metric))
            if value is not None:
                points.append((int(row["rollouts_seen"]), value))
        if not points:
            continue
        points.sort()
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=SERIES_COLORS[run.key],
            label=run.legend,
        )
    ax.set_xlabel("Rollouts seen")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlim(0, 6900)
    ax.set_xticks(EXPECTED_ROLLOUTS[::2] + [6728])
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", frameon=False, ncols=1)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.98, bottom=0.18)
    save_figure(fig, path)
    plt.close(fig)


def plot_scalar_metric(rows: list[dict[str, Any]], metric: str, title: str, subtitle: str, path: Path, ylim: tuple[float, float] | None = None) -> bool:
    selected = [row for row in rows if row.get("metric") == metric and metric_float(row.get("value")) is not None]
    if not selected:
        return False
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    add_header(fig, title, subtitle)
    for run in RUNS:
        points = [
            (int(row["rollouts_seen"]), float(row["value"]))
            for row in selected
            if row.get("line") == run.key and int(row.get("rollouts_seen") or 0) <= 6728
        ]
        if not points:
            continue
        points.sort()
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            linewidth=1.2,
            color=SERIES_COLORS[run.key],
            label=run.legend,
        )
    ax.set_xlabel("Rollouts seen")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlim(0, 6900)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.98, bottom=0.14)
    save_figure(fig, path)
    plt.close(fig)
    return True


def build_clean_report(run: OfficialRun, report_dir: Path, status: dict[str, Any], ckpt_rows: list[dict[str, Any]], scalar_rows: list[dict[str, Any]]) -> None:
    if report_dir.exists():
        shutil.rmtree(report_dir)
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "tables" / "checkpoint_eval_rollout.csv", ckpt_rows)
    write_csv(report_dir / "tables" / "scalar_long.csv", scalar_rows)
    plot_checkpoint_metric(ckpt_rows, "accuracy", f"{run.legend}: checkpoint exact accuracy", "GSM8K checkpoint eval at aligned rollout points.", report_dir / "figures" / "01_checkpoint_exact_accuracy_by_rollouts.png")
    plot_checkpoint_metric(ckpt_rows, "partial_accuracy", f"{run.legend}: checkpoint partial accuracy", "GSM8K partial accuracy at aligned rollout points.", report_dir / "figures" / "02_checkpoint_partial_accuracy_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_reward_score", f"{run.legend}: train reward", "TensorBoard scalar transformed to rollouts_seen = step x NUM_GENERATIONS.", report_dir / "figures" / "03_train_reward_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_rollout_extracted_none_rate", f"{run.legend}: extracted-none rate", "Rollout parser health from TensorBoard observability scalars.", report_dir / "figures" / "04_extracted_none_by_rollouts.png", ylim=(0, 1))
    (report_dir / "manifest_clean_plots.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run": status,
                "tables": ["tables/checkpoint_eval_rollout.csv", "tables/scalar_long.csv"],
                "figures": sorted(str(path.relative_to(report_dir)) for path in (report_dir / "figures").glob("*.png")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "README.md").write_text(
        f"# {run.legend} clean rollout320 report\n\n"
        "This report is generated from the fetched local TPU artifacts. "
        "Checkpoint plots use `rollouts_seen = step x NUM_GENERATIONS`.\n",
        encoding="utf-8",
    )


def copy_ref(src: Path, dst: Path) -> dict[str, str] | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "output": str(dst)}
