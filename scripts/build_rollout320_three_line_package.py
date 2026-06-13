"""Verify and package the rollout-aligned official GRPO three-line rerun."""

from __future__ import annotations

import argparse
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

    @property
    def final_step(self) -> int:
        return self.checkpoint_steps[-1]


RUNS = [
    OfficialRun(
        key="R0",
        run_id="baseline-rollout320-full-001",
        branch="R0_baseline_rollout320",
        legend="R0 baseline dense32-rollout",
        reward_mode="baseline",
        num_generations=2,
        max_steps=3364,
        checkpoint_steps=EXPECTED_K2_STEPS,
        learning_rate="3e-6",
        beta="0.08",
        rank="64",
        alpha="64",
    ),
    OfficialRun(
        key="R1",
        run_id="r1-reward-only-rollout320-full-001",
        branch="R1_reward_only_rollout320",
        legend="R1 reward-only dense32-rollout",
        reward_mode="gsm8k_verifiable_simple",
        num_generations=2,
        max_steps=3364,
        checkpoint_steps=EXPECTED_K2_STEPS,
        learning_rate="3e-6",
        beta="0.08",
        rank="64",
        alpha="64",
    ),
    OfficialRun(
        key="R4",
        run_id="r4-r12-full-rollout320-lr1e6-001",
        branch="R4_r12_full_lr1e-6_rollout320",
        legend="R4 R12 full-from-zero lr1e-6 dense32-rollout",
        reward_mode="gsm8k_verifiable_simple",
        num_generations=8,
        max_steps=841,
        checkpoint_steps=EXPECTED_K8_STEPS,
        learning_rate="1e-6",
        beta="0.04",
        rank="64",
        alpha="64",
    ),
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

SERIES_COLORS = {
    "R0": "#5477C4",
    "R1": "#CC6F47",
    "R4": "#71B436",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-root", type=Path, default=Path("artifacts/cloud"))
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/grpo-rollout320-three-line-evidence-001"),
    )
    parser.add_argument("--allow-missing", action="store_true", help="Write a partial verification report instead of failing on missing runs.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create the final .zip package.")
    return parser.parse_args()


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
        return []
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
    manifest_key_by_env = {
        "MAX_STEPS": "max_steps",
        "LEARNING_RATE": "learning_rate",
        "BETA": "beta",
        "RANK": "rank",
        "ALPHA": "alpha",
        "MAX_TO_KEEP": "max_to_keep",
        "K8_ROLLOUT_CHECKPOINT_INTERVAL": "rollout_checkpoint_interval",
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
            "font.family": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
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


def build_final_package(output_dir: Path, statuses: list[dict[str, Any]], ckpt_rows: list[dict[str, Any]], scalar_rows: list[dict[str, Any]], cloud_root: Path, reports_root: Path, make_zip: bool) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tables" / "checkpoint_eval_rollout_aligned.csv", ckpt_rows)
    write_csv(output_dir / "tables" / "scalar_long_rollout_aligned.csv", scalar_rows)
    write_csv(output_dir / "tables" / "verification_summary.csv", statuses)

    plot_checkpoint_metric(ckpt_rows, "accuracy", "Checkpoint exact accuracy by aligned rollouts", "Official three-line rerun only; 22 eval points from 320 to 6728 rollouts.", output_dir / "figures" / "01_checkpoint_exact_accuracy_by_rollouts.png")
    plot_checkpoint_metric(ckpt_rows, "partial_accuracy", "Checkpoint partial accuracy by aligned rollouts", "Official three-line rerun only; x-axis is rollout budget, not raw training step.", output_dir / "figures" / "02_checkpoint_partial_accuracy_by_rollouts.png")
    plot_checkpoint_metric(ckpt_rows, "format_accuracy", "Checkpoint format accuracy by aligned rollouts", "Format accuracy from checkpoint eval CSV at the same rollout points.", output_dir / "figures" / "03_checkpoint_format_accuracy_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_reward_score", "Train reward by aligned rollouts", "TensorBoard scalar rows retained in tables/scalar_long_rollout_aligned.csv.", output_dir / "figures" / "04_train_reward_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_kl", "Train KL by aligned rollouts", "Policy KL scalar transformed with rollouts_seen = step x NUM_GENERATIONS.", output_dir / "figures" / "05_train_kl_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_loss", "Train loss by aligned rollouts", "Actor loss scalar transformed with rollouts_seen = step x NUM_GENERATIONS.", output_dir / "figures" / "06_train_loss_by_rollouts.png")
    plot_scalar_metric(scalar_rows, "train_rollout_extracted_none_rate", "Extracted-none rate by aligned rollouts", "Parser health scalar; lower is better.", output_dir / "figures" / "07_extracted_none_by_rollouts.png", ylim=(0, 1))
    plot_scalar_metric(scalar_rows, "train_rollout_empty_response_rate", "Empty response rate by aligned rollouts", "Response health scalar; lower is better.", output_dir / "figures" / "08_empty_response_by_rollouts.png", ylim=(0, 1))

    copied: list[dict[str, str]] = []
    for run in RUNS:
        run_dir = cloud_root / run.run_id
        branch_dir = run_dir / "runs" / run.branch
        report_dir = reports_root / f"{run.run_id}-clean"
        for src, dst in [
            (run_dir / "pipeline.log", output_dir / "raw_refs" / run.key / "pipeline.log"),
            (run_dir / "checkpoint_archives.txt", output_dir / "raw_refs" / run.key / "checkpoint_archives.txt"),
            (run_dir / "artifacts" / "reward_k8_pilot_manifest.json", output_dir / "raw_refs" / run.key / "reward_k8_pilot_manifest.json"),
            (run_dir / "artifacts" / "reward_k8_pilot_manifest.json.pre_repair", output_dir / "raw_refs" / run.key / "reward_k8_pilot_manifest.json.pre_repair"),
            (branch_dir / "run_env.txt", output_dir / "raw_refs" / run.key / "run_env.txt"),
            (branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.csv", output_dir / "raw_refs" / run.key / "checkpoint_eval_summary.csv"),
            (branch_dir / "artifacts" / "checkpoint_eval" / "checkpoint_eval_summary.json", output_dir / "raw_refs" / run.key / "checkpoint_eval_summary.json"),
            (report_dir / "manifest_clean_plots.json", output_dir / "raw_refs" / run.key / "manifest_clean_plots.json"),
        ]:
            item = copy_ref(src, dst)
            if item:
                copied.append(item)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Official rollout-aligned GRPO three-line evidence package.",
        "rollout_axis": "rollouts_seen = step * num_generations",
        "expected_rollouts": EXPECTED_ROLLOUTS,
        "lines": [
            {
                "key": run.key,
                "run_id": run.run_id,
                "branch": run.branch,
                "legend": run.legend,
                "num_generations": run.num_generations,
                "max_steps": run.max_steps,
                "checkpoint_steps": run.checkpoint_steps,
                "source_checkpoint": None,
            }
            for run in RUNS
        ],
        "verification": statuses,
        "copied_files": copied,
    }
    (output_dir / "manifest_rollout320_three_line.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# GRPO Rollout320 Three-Line Evidence\n\n"
        "This package contains only the three official rollout-aligned lines currently in scope: R0 baseline, R1 reward-only, and R4 R12 full-from-zero lr1e-6. "
        "The x-axis in all comparison figures is `rollouts_seen`, not raw training step. Old canonical runs, reward-only continuation, and tail winner runs are excluded from the main plots.\n",
        encoding="utf-8",
    )
    if make_zip:
        archive_base = output_dir.with_suffix("")
        shutil.make_archive(str(archive_base), "zip", output_dir)


def main() -> int:
    args = parse_args()
    statuses: list[dict[str, Any]] = []
    all_ckpt_rows: list[dict[str, Any]] = []
    all_scalar_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for run in RUNS:
        run_dir = args.cloud_root / run.run_id
        if not run_dir.exists():
            status = {
                "line": run.key,
                "run_id": run.run_id,
                "branch": run.branch,
                "raw_dir": str(run_dir),
                "passed": False,
                "errors": [f"missing fetched run directory: {run_dir}"],
                "warnings": [],
            }
            statuses.append(status)
            missing.append(run.run_id)
            continue
        status, ckpt_rows, scalar_rows = verify_one(run_dir, run)
        statuses.append(status)
        all_ckpt_rows.extend(ckpt_rows)
        all_scalar_rows.extend(scalar_rows)
        if status["passed"]:
            build_clean_report(run, args.reports_root / f"{run.run_id}-clean", status, ckpt_rows, scalar_rows)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    verification_path = args.output_dir.parent / "grpo-rollout320-three-line-verification.json"
    verification_path.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "runs": statuses}, indent=2), encoding="utf-8")

    failed = [status for status in statuses if not status.get("passed")]
    if failed:
        print(f"Wrote partial verification report to {verification_path}")
        for status in failed:
            print(f"{status['run_id']}: FAILED")
            for error in status.get("errors", []):
                print(f"  - {error}")
        failed_missing_only = [
            status
            for status in failed
            if status.get("errors") == [f"missing fetched run directory: {args.cloud_root / status['run_id']}"]
        ]
        if args.allow_missing and len(failed_missing_only) == len(failed):
            return 0
        return 1

    build_final_package(
        args.output_dir,
        statuses,
        all_ckpt_rows,
        all_scalar_rows,
        args.cloud_root,
        args.reports_root,
        make_zip=not args.no_zip,
    )
    print(f"Wrote rollout320 three-line package to {args.output_dir}")
    if not args.no_zip:
        print(f"Wrote zip to {args.output_dir}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
