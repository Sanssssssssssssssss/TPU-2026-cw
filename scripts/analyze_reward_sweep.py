"""Build no-drop analysis tables and clean figures for a reward-only GRPO sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SWEEP_RUNS = [
    ("R1_no_approx", "no_approx"),
    ("R2_light_format_oldnum", "light_format_oldnum"),
    ("R3_numeric_primary_no_len", "numeric_primary_no_len"),
    ("R4_numeric_primary_len1200", "numeric_primary_len1200"),
    ("R5_numeric_primary_answer_only_len1200", "numeric_primary_answer_only_len1200"),
    ("R9_closed_answer_minimal", "closed_answer_minimal"),
    ("R10_numeric_guarded", "numeric_guarded"),
    ("R11_numeric_guarded_fallback", "numeric_guarded_fallback"),
    ("R12_gsm8k_verifiable_simple", "gsm8k_verifiable_simple"),
]

ROLE_WORDS = {"train", "eval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a reward-mode GRPO sweep.")
    parser.add_argument("--input-dir", required=True, help="Fetched or remote sweep run directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline-dir",
        default="artifacts/cloud/course-baseline-001",
        help="Optional baseline run directory used as R0 control for steps 0-768.",
    )
    parser.add_argument("--max-control-step", type=int, default=768)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return parser.parse_args()


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"TensorBoard reader unavailable: {exc}")
        return None
    return event_accumulator


def event_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted({p.parent for p in log_dir.rglob("events.out.tfevents*")})


def canonical_metric_name(tag: str) -> str:
    parts = tag.split("/")
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
    if len(parts) >= 3 and parts[1] in ROLE_WORDS:
        namespace, role = parts[0], parts[1]
        name = "_".join(parts[2:])
        if namespace == "eval":
            return f"{role}_{name}"
        if namespace == "rollout":
            return f"{role}_rollout_{name}"
        if namespace == "grpo":
            return f"{role}_grpo_{name}"
        if namespace == "reward":
            return f"{role}_reward_{name}"
        if namespace == "audit":
            return f"{role}_audit_{name}"
        return f"{role}_{namespace}_{name}"
    return tag.replace("/", "_").replace("-", "_").strip("_")


def read_tensorboard_scalars(log_dir: Path, run_id: str, reward_mode: str, max_step: int | None = None) -> list[dict[str, Any]]:
    event_accumulator = load_event_accumulator()
    if event_accumulator is None:
        return []

    rows: list[dict[str, Any]] = []
    for directory in event_dirs(log_dir):
        try:
            ea = event_accumulator.EventAccumulator(str(directory), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception as exc:
            print(f"Skipping TensorBoard directory {directory}: {exc}")
            continue
        for tag in sorted(ea.Tags().get("scalars", [])):
            metric = canonical_metric_name(tag)
            for event in ea.Scalars(tag):
                step = int(event.step)
                if max_step is not None and step > max_step:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "reward_mode": reward_mode,
                        "source": "tensorboard",
                        "metric": metric,
                        "tag": tag,
                        "step": step,
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                    }
                )
    rows.sort(key=lambda row: (row["run_id"], row["metric"], row["step"], row["tag"]))
    return rows


def run_dirs(input_dir: Path, baseline_dir: Path | None, max_control_step: int) -> list[dict[str, Any]]:
    out = []
    if baseline_dir is not None and baseline_dir.exists():
        out.append(
            {
                "run_id": "R0_baseline",
                "reward_mode": "baseline",
                "run_dir": baseline_dir,
                "tensorboard_dir": baseline_dir / "tensorboard",
                "trace_dir": baseline_dir / "artifacts" / "rollout_traces",
                "checkpoint_eval_dir": baseline_dir / "artifacts" / "checkpoint_eval",
                "max_step": max_control_step,
            }
        )
    manifest_runs: list[tuple[str, str]] = []
    for manifest_name in (
        "reward_k8_pilot_manifest.json",
        "reward_r10_manifest.json",
        "reward_r9_manifest.json",
        "reward_dense_manifest.json",
        "reward_sweep_manifest.json",
    ):
        manifest = read_json(input_dir / "artifacts" / manifest_name)
        if not manifest:
            continue
        raw_runs = manifest.get("runs") or manifest.get("reward_runs") or []
        for item in raw_runs:
            if isinstance(item, dict):
                run_id = item.get("run_id")
                mode = item.get("reward_mode")
                if run_id and mode:
                    manifest_runs.append((str(run_id), str(mode)))
        if manifest_runs:
            break
    configured_runs = manifest_runs or SWEEP_RUNS
    for run_id, mode in configured_runs:
        child = input_dir / "runs" / run_id
        out.append(
            {
                "run_id": run_id,
                "reward_mode": mode,
                "run_dir": child,
                "tensorboard_dir": child / "tensorboard",
                "trace_dir": child / "artifacts" / "rollout_traces",
                "checkpoint_eval_dir": child / "artifacts" / "checkpoint_eval",
                "max_step": None,
            }
        )
    return out


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_traces(trace_dir: Path, run_id: str, reward_mode: str) -> list[dict[str, Any]]:
    rows = []
    if not trace_dir.exists():
        return rows
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                components = row.get("reward_components") or {}
                flat = {
                    "run_id": run_id,
                    "reward_mode": reward_mode,
                    "file": str(path),
                    "call_index": row.get("call_index"),
                    "dataset_role": row.get("dataset_role"),
                    "group_index": row.get("group_index"),
                    "generation_index": row.get("generation_index"),
                    "prompt_hash": row.get("prompt_hash"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "completion": row.get("completion"),
                    "completion_chars": row.get("completion_chars"),
                    "reward_total": row.get("reward_total"),
                    "reward_total_recomputed": row.get("reward_total_recomputed"),
                    "advantage": row.get("advantage"),
                    "formatted_answer": row.get("formatted_answer"),
                    "extracted_number": row.get("extracted_number"),
                    "official_extracted_number": row.get("official_extracted_number"),
                    "robust_extracted_number": row.get("robust_extracted_number"),
                    "numeric_exact": row.get("numeric_exact"),
                    "numeric_partial": row.get("numeric_partial"),
                    "official_numeric_exact": row.get("official_numeric_exact"),
                    "official_numeric_partial": row.get("official_numeric_partial"),
                    "robust_numeric_exact": row.get("robust_numeric_exact"),
                    "robust_numeric_partial": row.get("robust_numeric_partial"),
                    "parser_false_negative": row.get("parser_false_negative"),
                    "format_ok": row.get("format_ok"),
                    "answer_tag_pair_ok": row.get("answer_tag_pair_ok"),
                    "duplicate_or_broken_answer_tag": row.get("duplicate_or_broken_answer_tag"),
                    "overlong_1200": row.get("overlong_1200"),
                    "overlong_1600": row.get("overlong_1600"),
                    "answer_multi_number": row.get("answer_multi_number"),
                    "answer_single_number": row.get("answer_single_number"),
                    "robust_answer_number_count": row.get("robust_answer_number_count"),
                    "no_close_answer": row.get("no_close_answer"),
                }
                for key, value in components.items():
                    flat[f"component_{key}"] = value
                rows.append(flat)
    return rows


def read_checkpoint_rows(eval_dir: Path, run_id: str, reward_mode: str) -> list[dict[str, Any]]:
    summary = read_json(eval_dir / "checkpoint_eval_summary.json")
    if not summary:
        return []
    rows = []
    for row in summary.get("rows") or []:
        rows.append(
            {
                "run_id": run_id,
                "reward_mode": reward_mode,
                "label": row.get("label"),
                "step": row.get("step"),
                "policy": row.get("policy"),
                "restored_step": row.get("restored_step"),
                "correct": row.get("correct"),
                "total": row.get("total"),
                "accuracy": row.get("accuracy"),
                "accuracy_ci95_low": row.get("accuracy_ci95_low"),
                "accuracy_ci95_high": row.get("accuracy_ci95_high"),
                "partial_accuracy": row.get("partial_accuracy"),
                "format_accuracy": row.get("format_accuracy"),
                "no_close_answer_rate": row.get("no_close_answer_rate"),
                "robust_numeric_exact_rate": row.get("robust_numeric_exact_rate"),
                "text_after_close_rate": row.get("text_after_close_rate"),
                "file": row.get("file"),
            }
        )
    return rows


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def float_value(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def ratio(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if float_value(v) is not None]
    return sum(clean) / len(clean) if clean else None


def std(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if float_value(v) is not None]
    if not clean:
        return None
    avg = sum(clean) / len(clean)
    return math.sqrt(sum((value - avg) ** 2 for value in clean) / len(clean))


def trace_audit_summary(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        try:
            call_index = int(row.get("call_index") or 0)
        except Exception:
            call_index = 0
        grouped[(row["run_id"], row["reward_mode"], str(row.get("dataset_role") or "unknown"), call_index)].append(row)

    out = []
    for (run_id, reward_mode, dataset_role, call_index), rows in sorted(grouped.items()):
        reward_pairs = [(float_value(row.get("reward_total")), row) for row in rows]
        reward_pairs = [(reward, row) for reward, row in reward_pairs if reward is not None]
        rewards_clean = [reward for reward, _ in reward_pairs]
        exact = [bool_value(row.get("numeric_exact")) for row in rows]
        fmt = [bool_value(row.get("format_ok")) for row in rows]
        robust_exact = [bool_value(row.get("robust_numeric_exact")) for row in rows]
        fallback_used = [bool_value(row.get("fallback_number_used")) for row in rows]
        fallback_exact = [bool_value(row.get("fallback_numeric_exact")) for row in rows]
        parser_false_negative = [bool_value(row.get("parser_false_negative")) for row in rows]
        answer_single_number = [bool_value(row.get("answer_single_number")) for row in rows]
        no_close_answer = [bool_value(row.get("no_close_answer")) for row in rows]
        empty = [not str(row.get("completion") or "").strip() for row in rows]
        extracted_none = [row.get("extracted_number") in (None, "") for row in rows]
        overlong_1200 = [bool_value(row.get("overlong_1200")) or int(row.get("completion_chars") or 0) > 1200 for row in rows]
        overlong_1600 = [bool_value(row.get("overlong_1600")) or int(row.get("completion_chars") or 0) > 1600 for row in rows]
        correct_rewards = [reward for reward, row in reward_pairs if bool_value(row.get("numeric_exact"))]
        wrong_rewards = [reward for reward, row in reward_pairs if not bool_value(row.get("numeric_exact"))]
        formatted_wrong_rewards = [
            reward
            for reward, row in reward_pairs
            if bool_value(row.get("format_ok")) and not bool_value(row.get("numeric_exact"))
        ]
        dense_wrong_rewards = [
            float_value(row.get("component_numeric_dense"))
            for row in rows
            if not bool_value(row.get("robust_numeric_exact"))
            and float_value(row.get("component_numeric_dense")) is not None
        ]
        guarded_wrong_rewards = [
            float_value(row.get("component_numeric_guarded"))
            for row in rows
            if not bool_value(row.get("robust_numeric_exact"))
            and float_value(row.get("component_numeric_guarded")) is not None
        ]
        fallback_guarded_wrong_rewards = [
            float_value(row.get("component_numeric_guarded_fallback"))
            for row in rows
            if not bool_value(row.get("fallback_numeric_exact"))
            and float_value(row.get("component_numeric_guarded_fallback")) is not None
        ]
        out.append(
            {
                "run_id": run_id,
                "reward_mode": reward_mode,
                "dataset_role": dataset_role,
                "call_index": call_index,
                "rows": len(rows),
                "reward_mean": mean(rewards_clean),
                "numeric_exact_rate": ratio(sum(exact), len(exact)),
                "robust_numeric_exact_rate": ratio(sum(robust_exact), len(robust_exact)),
                "fallback_number_used_rate": ratio(sum(fallback_used), len(fallback_used)),
                "fallback_numeric_exact_rate": ratio(sum(fallback_exact), len(fallback_exact)),
                "parser_false_negative_rate": ratio(sum(parser_false_negative), len(parser_false_negative)),
                "format_accuracy": ratio(sum(fmt), len(fmt)),
                "empty_response_rate": ratio(sum(empty), len(empty)),
                "extracted_none_rate": ratio(sum(extracted_none), len(extracted_none)),
                "answer_single_number_rate": ratio(sum(answer_single_number), len(answer_single_number)),
                "no_close_answer_rate": ratio(sum(no_close_answer), len(no_close_answer)),
                "overlong_rate_1200": ratio(sum(overlong_1200), len(overlong_1200)),
                "overlong_rate_1600": ratio(sum(overlong_1600), len(overlong_1600)),
                "dense_wrong_reward_std": std(dense_wrong_rewards),
                "guarded_wrong_reward_std": std(guarded_wrong_rewards),
                "fallback_guarded_wrong_reward_std": std(fallback_guarded_wrong_rewards),
                "reward_numeric_margin": (
                    mean(correct_rewards) - mean(wrong_rewards)
                    if mean(correct_rewards) is not None and mean(wrong_rewards) is not None
                    else None
                ),
                "reward_format_leakage": mean(formatted_wrong_rewards),
                "reward_hacking_rate": ratio(
                    sum(
                        1
                        for reward, row in reward_pairs
                        if not bool_value(row.get("numeric_exact")) and reward >= 3.0
                    ),
                    len(rows),
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def pivot_scalar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = sorted({row["metric"] for row in rows})
    grouped: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = (row["run_id"], row["reward_mode"], int(row["step"]))
        grouped[key][row["metric"]] = row["value"]
    out = []
    for (run_id, reward_mode, step), values in sorted(grouped.items()):
        item = {"run_id": run_id, "reward_mode": reward_mode, "step": step}
        for metric in metrics:
            item[metric] = values.get(metric, "")
        out.append(item)
    return out


def latest_metric(rows: list[dict[str, Any]], run_id: str, candidates: list[str], max_step: int = 768) -> float | None:
    options = [
        row
        for row in rows
        if row["run_id"] == run_id and row["metric"] in candidates and int(row["step"]) <= max_step
    ]
    if not options:
        return None
    options.sort(key=lambda row: int(row["step"]))
    return float(options[-1]["value"])


def sweep_selection_summary(scalar_rows: list[dict[str, Any]], trace_summary: list[dict[str, Any]], ckpt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    modes_by_run = {}
    for row in scalar_rows + trace_summary + ckpt_rows:
        run_id = row.get("run_id")
        mode = row.get("reward_mode")
        if run_id and mode:
            modes_by_run[str(run_id)] = str(mode)
    run_ids = sorted(modes_by_run)
    if "R0_baseline" in run_ids:
        run_ids.remove("R0_baseline")
        run_ids = ["R0_baseline"] + run_ids
    latest_trace: dict[str, dict[str, Any]] = {}
    for row in trace_summary:
        if row.get("dataset_role") not in {"eval", "unknown"}:
            continue
        current = latest_trace.get(row["run_id"])
        if current is None or int(row.get("call_index") or 0) > int(current.get("call_index") or 0):
            latest_trace[row["run_id"]] = row

    ckpt_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ckpt_rows:
        ckpt_by_run[row["run_id"]].append(row)

    for run_id in run_ids:
        mode = modes_by_run.get(run_id, "baseline" if run_id == "R0_baseline" else "")
        best_ckpt = None
        if ckpt_by_run.get(run_id):
            best_ckpt = max(
                ckpt_by_run[run_id],
                key=lambda row: (
                    float(row.get("accuracy") or 0),
                    float(row.get("partial_accuracy") or 0),
                    float(row.get("format_accuracy") or 0),
                    int(row.get("step") or -1),
                ),
            )
        trace = latest_trace.get(run_id, {})
        leakage = latest_metric(
            scalar_rows,
            run_id,
            ["eval_audit_reward_format_leakage", "train_audit_reward_format_leakage"],
        )
        hacking = latest_metric(
            scalar_rows,
            run_id,
            ["eval_audit_reward_hacking_rate", "train_audit_reward_hacking_rate"],
        )
        margin = latest_metric(
            scalar_rows,
            run_id,
            ["eval_audit_reward_numeric_margin", "train_audit_reward_numeric_margin"],
        )
        empty = latest_metric(
            scalar_rows,
            run_id,
            ["eval_rollout_empty_response_rate", "train_rollout_empty_response_rate", "eval_empty_response_rate"],
        )
        extracted_none = latest_metric(
            scalar_rows,
            run_id,
            ["eval_rollout_extracted_none_rate", "train_rollout_extracted_none_rate", "eval_extracted_none_rate"],
        )
        overlong = latest_metric(
            scalar_rows,
            run_id,
            ["eval_rollout_overlong_rate_1600", "train_rollout_overlong_rate_1600"],
        )
        zero_std = latest_metric(
            scalar_rows,
            run_id,
            ["eval_grpo_frac_reward_zero_std", "train_grpo_frac_reward_zero_std", "eval_frac_reward_zero_std"],
        )
        parser_false_negative = latest_metric(
            scalar_rows,
            run_id,
            [
                "reward_parser_false_negative_rate",
                "eval_reward_parser_false_negative_rate",
                "train_reward_parser_false_negative_rate",
            ],
        )
        answer_single_number = latest_metric(
            scalar_rows,
            run_id,
            [
                "rollout_answer_single_number_rate",
                "eval_rollout_answer_single_number_rate",
                "train_rollout_answer_single_number_rate",
            ],
        )
        no_close_answer = latest_metric(
            scalar_rows,
            run_id,
            ["rollout_no_close_answer_rate", "eval_rollout_no_close_answer_rate", "train_rollout_no_close_answer_rate"],
        )
        dense_wrong_std = latest_metric(
            scalar_rows,
            run_id,
            ["audit_dense_wrong_reward_std", "eval_audit_dense_wrong_reward_std", "train_audit_dense_wrong_reward_std"],
        )
        kl = latest_metric(scalar_rows, run_id, ["eval_kl", "train_kl"])
        row = {
            "run_id": run_id,
            "reward_mode": mode,
            "best_checkpoint_step": best_ckpt.get("step") if best_ckpt else "",
            "best_checkpoint_accuracy": best_ckpt.get("accuracy") if best_ckpt else "",
            "best_checkpoint_partial_accuracy": best_ckpt.get("partial_accuracy") if best_ckpt else "",
            "best_checkpoint_format_accuracy": best_ckpt.get("format_accuracy") if best_ckpt else "",
            "reward_numeric_margin": margin if margin is not None else trace.get("reward_numeric_margin"),
            "reward_format_leakage": leakage if leakage is not None else trace.get("reward_format_leakage"),
            "reward_hacking_rate": hacking if hacking is not None else trace.get("reward_hacking_rate"),
            "empty_response_rate": empty if empty is not None else trace.get("empty_response_rate"),
            "extracted_none_rate": extracted_none if extracted_none is not None else trace.get("extracted_none_rate"),
            "parser_false_negative_rate": (
                parser_false_negative if parser_false_negative is not None else trace.get("parser_false_negative_rate")
            ),
            "answer_single_number_rate": (
                answer_single_number if answer_single_number is not None else trace.get("answer_single_number_rate")
            ),
            "no_close_answer_rate": no_close_answer if no_close_answer is not None else trace.get("no_close_answer_rate"),
            "overlong_rate_1600": overlong if overlong is not None else trace.get("overlong_rate_1600"),
            "dense_wrong_reward_std": dense_wrong_std if dense_wrong_std is not None else trace.get("dense_wrong_reward_std"),
            "frac_reward_zero_std": zero_std,
            "kl": kl,
        }
        elimination_reasons = []
        if float_value(row["reward_format_leakage"]) is not None and float(row["reward_format_leakage"]) > 1.0:
            elimination_reasons.append("reward_format_leakage>1.0")
        if float_value(row["empty_response_rate"]) is not None and float(row["empty_response_rate"]) > 0.15:
            elimination_reasons.append("empty_response_rate>0.15")
        if float_value(row["extracted_none_rate"]) is not None and float(row["extracted_none_rate"]) > 0.35:
            elimination_reasons.append("extracted_none_rate>0.35")
        if float_value(row["frac_reward_zero_std"]) is not None and float(row["frac_reward_zero_std"]) > 0.65:
            elimination_reasons.append("frac_reward_zero_std>0.65")
        if float_value(row["answer_single_number_rate"]) is not None and float(row["answer_single_number_rate"]) < 0.8:
            elimination_reasons.append("answer_single_number_rate<0.8")
        if float_value(row["kl"]) is not None and float(row["kl"]) > 0.8:
            elimination_reasons.append("kl>0.8")
        row["elimination_reasons"] = ";".join(elimination_reasons)
        row["screening_status"] = "eliminate" if elimination_reasons else "candidate"
        out.append(row)
    return out


def load_fonts():
    try:
        from PIL import ImageFont

        regular = Path("C:/Windows/Fonts/segoeui.ttf")
        mono = Path("C:/Windows/Fonts/consola.ttf")
        if regular.exists():
            return ImageFont.truetype(str(regular), 18), ImageFont.truetype(str(mono if mono.exists() else regular), 14)
    except Exception:
        pass
    from PIL import ImageFont

    return ImageFont.load_default(), ImageFont.load_default()


def save_plot(img, path: Path, formats: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        img.save(path.with_suffix(".png"))
        print(f"Wrote {path.with_suffix('.png')}")
    if "pdf" in formats:
        img.save(path.with_suffix(".pdf"))
        print(f"Wrote {path.with_suffix('.pdf')}")


def metric_series(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["metric"] not in metric_names:
            continue
        out[row["run_id"]][int(row["step"])] = float(row["value"])
    return {run_id: sorted(points.items()) for run_id, points in out.items()}


def nice_range(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return 0.0, 1.0
    lo, hi = min(clean), max(clean)
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_small_multiples(path: Path, title: str, panels: list[tuple[str, dict[str, list[tuple[int, float]]]]], formats: list[str]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Pillow unavailable; skipping plot {path}: {exc}")
        return

    font, mono = load_fonts()
    colors = {
        "R0_baseline": (90, 90, 90),
        "R1_no_approx": (37, 99, 158),
        "R2_light_format_oldnum": (218, 117, 43),
        "R3_numeric_primary_no_len": (67, 135, 92),
        "R4_numeric_primary_len1200": (172, 68, 68),
        "R5_numeric_primary_answer_only_len1200": (110, 90, 170),
    }
    cols = 2
    panel_w, panel_h = 720, 320
    panel_rows = math.ceil(len(panels) / cols)
    width = 40 + cols * panel_w + 24 + 40
    height = 72 + panel_rows * panel_h + (panel_rows - 1) * 24 + 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 22), title, font=font, fill=(20, 28, 36))

    for idx, (panel_title, series) in enumerate(panels):
        row, col = divmod(idx, cols)
        x0 = 40 + col * (panel_w + 24)
        y0 = 72 + row * (panel_h + 24)
        x1, y1 = x0 + panel_w, y0 + panel_h
        left, right = x0 + 70, x1 - 22
        top, bottom = y0 + 44, y1 - 58
        draw.rectangle((x0, y0, x1, y1), outline=(220, 226, 234))
        draw.text((x0 + 12, y0 + 12), panel_title, font=font, fill=(20, 28, 36))
        points = [(step, value) for pts in series.values() for step, value in pts]
        if not points:
            draw.text((left, top + 40), "no data", font=font, fill=(120, 130, 140))
            continue
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        if xmin == xmax:
            xmax += 1
        ymin, ymax = nice_range([y for _, y in points])

        def sx(step: int) -> float:
            return left + (step - xmin) / (xmax - xmin) * (right - left)

        def sy(value: float) -> float:
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        for tick in range(5):
            frac = tick / 4
            y = bottom - frac * (bottom - top)
            val = ymin + frac * (ymax - ymin)
            draw.line((left, y, right, y), fill=(232, 236, 241))
            draw.text((x0 + 8, y - 8), f"{val:.3g}", font=mono, fill=(84, 96, 110))
        draw.line((left, top, left, bottom, right, bottom), fill=(100, 110, 122))

        legend_x, legend_y = x0 + 12, y1 - 36
        for run_id, pts in series.items():
            color = colors.get(run_id, (50, 50, 50))
            scaled = [(sx(step), sy(value)) for step, value in pts]
            if len(scaled) > 1:
                draw.line(scaled, fill=color, width=2)
            for x, y in scaled:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            draw.rectangle((legend_x, legend_y + 4, legend_x + 10, legend_y + 14), fill=color)
            draw.text((legend_x + 14, legend_y), run_id.replace("_", " "), font=mono, fill=(30, 38, 50))
            legend_x += 178
            if legend_x > x1 - 150:
                legend_x = x0 + 12
                legend_y += 18

    save_plot(img, path, formats)


def render_figures(output_dir: Path, scalar_rows: list[dict[str, Any]], ckpt_rows: list[dict[str, Any]], formats: list[str]) -> None:
    fig_dir = output_dir / "figures"
    draw_small_multiples(
        fig_dir / "01_audit_metrics",
        "Reward sweep audit metrics",
        [
            ("reward_numeric_margin", metric_series(scalar_rows, ["eval_audit_reward_numeric_margin", "train_audit_reward_numeric_margin"])),
            ("reward_format_leakage", metric_series(scalar_rows, ["eval_audit_reward_format_leakage", "train_audit_reward_format_leakage"])),
            ("reward_hacking_rate", metric_series(scalar_rows, ["eval_audit_reward_hacking_rate", "train_audit_reward_hacking_rate"])),
            ("group_misrank_rate", metric_series(scalar_rows, ["eval_audit_group_misrank_rate", "train_audit_group_misrank_rate"])),
        ],
        formats,
    )
    draw_small_multiples(
        fig_dir / "02_response_and_grpo_health",
        "Reward sweep response and GRPO health",
        [
            ("empty_response_rate", metric_series(scalar_rows, ["eval_rollout_empty_response_rate", "train_rollout_empty_response_rate", "eval_empty_response_rate"])),
            ("extracted_none_rate", metric_series(scalar_rows, ["eval_rollout_extracted_none_rate", "train_rollout_extracted_none_rate", "eval_extracted_none_rate"])),
            ("overlong_rate_1600", metric_series(scalar_rows, ["eval_rollout_overlong_rate_1600", "train_rollout_overlong_rate_1600"])),
            ("frac_reward_zero_std", metric_series(scalar_rows, ["eval_grpo_frac_reward_zero_std", "train_grpo_frac_reward_zero_std", "eval_frac_reward_zero_std"])),
        ],
        formats,
    )
    draw_small_multiples(
        fig_dir / "03_reward_components",
        "Reward sweep component means",
        [
            ("numeric_primary_mean", metric_series(scalar_rows, ["eval_reward_numeric_primary_mean", "train_reward_numeric_primary_mean"])),
            ("numeric_dense_mean", metric_series(scalar_rows, ["eval_reward_numeric_dense_mean", "train_reward_numeric_dense_mean"])),
            ("closed_answer_minimal_mean", metric_series(scalar_rows, ["eval_reward_closed_answer_minimal_mean", "train_reward_closed_answer_minimal_mean"])),
            ("numeric_guarded_mean", metric_series(scalar_rows, ["eval_reward_numeric_guarded_mean", "train_reward_numeric_guarded_mean"])),
            ("answer_hygiene_guarded_mean", metric_series(scalar_rows, ["eval_reward_answer_hygiene_guarded_mean", "train_reward_answer_hygiene_guarded_mean"])),
            ("numeric_guarded_total_mean", metric_series(scalar_rows, ["eval_reward_numeric_guarded_total_mean", "train_reward_numeric_guarded_total_mean"])),
            ("numeric_guarded_fallback_mean", metric_series(scalar_rows, ["eval_reward_numeric_guarded_fallback_mean", "train_reward_numeric_guarded_fallback_mean"])),
            ("answer_hygiene_fallback_mean", metric_series(scalar_rows, ["eval_reward_answer_hygiene_fallback_mean", "train_reward_answer_hygiene_fallback_mean"])),
            ("numeric_guarded_fallback_total_mean", metric_series(scalar_rows, ["eval_reward_numeric_guarded_fallback_total_mean", "train_reward_numeric_guarded_fallback_total_mean"])),
            ("gsm8k_simple_numeric_mean", metric_series(scalar_rows, ["eval_reward_gsm8k_simple_numeric_mean", "train_reward_gsm8k_simple_numeric_mean"])),
            ("gsm8k_simple_format_mean", metric_series(scalar_rows, ["eval_reward_gsm8k_simple_format_mean", "train_reward_gsm8k_simple_format_mean"])),
            ("format_light_mean", metric_series(scalar_rows, ["eval_reward_format_light_mean", "train_reward_format_light_mean"])),
            ("length_penalty_mean", metric_series(scalar_rows, ["eval_reward_length_penalty_mean", "train_reward_length_penalty_mean"])),
            ("approx_format_mean", metric_series(scalar_rows, ["eval_reward_match_format_approximately_mean", "train_reward_match_format_approximately_mean"])),
        ],
        formats,
    )

    ckpt_scalar_rows = []
    for row in ckpt_rows:
        if row.get("step") in (None, ""):
            continue
        for metric in ("accuracy", "partial_accuracy", "format_accuracy"):
            value = float_value(row.get(metric))
            if value is None:
                continue
            ckpt_scalar_rows.append(
                {
                    "run_id": row["run_id"],
                    "metric": f"checkpoint_{metric}",
                    "step": int(row["step"]),
                    "value": value,
                }
            )
    draw_small_multiples(
        fig_dir / "04_checkpoint_eval",
        "Reward sweep checkpoint evaluations",
        [
            ("checkpoint_accuracy", metric_series(ckpt_scalar_rows, ["checkpoint_accuracy"])),
            ("checkpoint_partial_accuracy", metric_series(ckpt_scalar_rows, ["checkpoint_partial_accuracy"])),
            ("checkpoint_format_accuracy", metric_series(ckpt_scalar_rows, ["checkpoint_format_accuracy"])),
        ],
        formats,
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    baseline_dir = Path(args.baseline_dir).expanduser() if args.baseline_dir else None
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    scalar_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    ckpt_rows: list[dict[str, Any]] = []
    manifest_runs = []
    for item in run_dirs(input_dir, baseline_dir, args.max_control_step):
        scalar_rows.extend(
            read_tensorboard_scalars(
                item["tensorboard_dir"],
                item["run_id"],
                item["reward_mode"],
                max_step=item["max_step"],
            )
        )
        trace_rows.extend(read_traces(item["trace_dir"], item["run_id"], item["reward_mode"]))
        ckpt_rows.extend(read_checkpoint_rows(item["checkpoint_eval_dir"], item["run_id"], item["reward_mode"]))
        manifest_runs.append(
            {
                "run_id": item["run_id"],
                "reward_mode": item["reward_mode"],
                "run_dir": str(item["run_dir"]),
                "tensorboard_dir_exists": item["tensorboard_dir"].exists(),
                "trace_dir_exists": item["trace_dir"].exists(),
                "checkpoint_eval_dir_exists": item["checkpoint_eval_dir"].exists(),
            }
        )

    trace_summary = trace_audit_summary(trace_rows)
    scalar_pivot = pivot_scalar_rows(scalar_rows)
    selection_rows = sweep_selection_summary(scalar_rows, trace_summary, ckpt_rows)

    write_csv(table_dir / "scalar_long.csv", scalar_rows)
    write_csv(table_dir / "scalar_pivot.csv", scalar_pivot)
    write_csv(table_dir / "trace_rows_flat.csv", trace_rows)
    write_csv(table_dir / "trace_audit_by_call.csv", trace_summary)
    write_csv(table_dir / "checkpoint_eval_long.csv", ckpt_rows)
    write_csv(table_dir / "selection_summary.csv", selection_rows)
    write_json(table_dir / "selection_summary.json", {"rows": selection_rows})

    render_figures(output_dir, scalar_rows, ckpt_rows, args.formats)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "max_control_step": args.max_control_step,
        "runs": manifest_runs,
        "tables": [
            "tables/scalar_long.csv",
            "tables/scalar_pivot.csv",
            "tables/trace_rows_flat.csv",
            "tables/trace_audit_by_call.csv",
            "tables/checkpoint_eval_long.csv",
            "tables/selection_summary.csv",
        ],
        "figures": [
            "figures/01_audit_metrics.png",
            "figures/02_response_and_grpo_health.png",
            "figures/03_reward_components.png",
            "figures/04_checkpoint_eval.png",
        ],
        "notes": [
            "R0 baseline is included only for scalar/trace control where local artifacts exist.",
            "Checkpoint-level comparisons are limited to checkpoints that were actually saved and evaluated.",
            "Selection summary is a screening aid; final mainline requires a second longer finalist run.",
        ],
    }
    write_json(output_dir / "manifest_report.json", manifest)

    readme = """# Reward Grid 001

This folder is the reward-only GRPO sweep analysis package.

The source-of-truth tables are in `tables/`. Figures are intentionally plain
and table-backed: they summarize scalar, trace, and checkpoint evidence without
dropping the underlying rows.

Primary screening table: `tables/selection_summary.csv`.
Complete scalar table: `tables/scalar_long.csv`.
Complete flattened trace table: `tables/trace_rows_flat.csv`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote reward sweep report package to {output_dir}")


if __name__ == "__main__":
    main()
