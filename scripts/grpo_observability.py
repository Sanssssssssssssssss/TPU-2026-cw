"""Observability helpers for GRPO training.

The training loop is owned by Tunix, so this module keeps instrumentation
lightweight: numeric diagnostics are returned through Tunix ``metric_fns``;
sampled rollout rows are written to JSONL and, when available, W&B Tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rewards import REWARD_MODE, reward_components_for_mode, reward_diagnostics_for_observability

REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _repeat_to_len(values: Iterable[Any], target_len: int) -> list[Any]:
    items = list(values)
    if len(items) == target_len:
        return items
    if not items:
        return [None] * target_len
    if target_len % len(items) == 0:
        repeats = target_len // len(items)
        return [item for item in items for _ in range(repeats)]
    if len(items) > target_len:
        return items[:target_len]
    return items + [items[-1]] * (target_len - len(items))


def _float_array(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)
    return arr.reshape(-1)


def _resize_metric_array(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return values
    if values.size == size:
        return values
    if values.size == 0:
        return np.zeros(size, dtype=np.float32)
    return np.resize(values, size).astype(np.float32)


def _ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _git_text(args: list[str]) -> str | None:
    repo = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_env_or_text(env_name: str, args: list[str]) -> str | None:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value.strip()
    return _git_text(args)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def collect_config_snapshot(config_module: Any) -> dict[str, Any]:
    """Collect public config constants without secrets."""
    out: dict[str, Any] = {}
    for name in dir(config_module):
        if not name.isupper():
            continue
        if "TOKEN" in name or "KEY" in name or "SECRET" in name:
            continue
        value = getattr(config_module, name)
        if callable(value):
            continue
        out[name] = _jsonable(value)
    return out


@dataclass
class GRPOObservability:
    run_id: str
    output_dir: Path
    tensorboard_dir: Path
    num_generations: int
    config_snapshot: dict[str, Any]
    trace_every_n_steps: int = 64
    trace_max_rows: int = 32
    trace_max_chars: int = 4000
    enable_wandb: bool = True
    enable_tensorboard: bool = True
    enable_alerts: bool = True
    early_stop: bool = False
    alert_cooldown_calls: int = 8
    trace_dir: Path = field(init=False)
    alerts_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _call_index: int = field(default=0, init=False)
    _last_alert_call: dict[str, int] = field(default_factory=dict, init=False)
    _wandb: Any = field(default=None, init=False)
    _wandb_run: Any = field(default=None, init=False)
    _tb_writer: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self.tensorboard_dir = Path(self.tensorboard_dir).expanduser()
        self.trace_dir = Path(os.environ.get("OBS_TRACE_DIR", self.output_dir / "rollout_traces")).expanduser()
        self.alerts_path = self.output_dir / "alerts.jsonl"
        self.manifest_path = Path(
            os.environ.get("OBS_RUN_MANIFEST", self.output_dir / "run_manifest.json")
        ).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        *,
        run_id: str | None,
        tensorboard_dir: str,
        num_generations: int,
        config_snapshot: dict[str, Any],
    ) -> "GRPOObservability":
        resolved_run_id = run_id or os.environ.get("RUN_ID") or "local-grpo"
        output_dir = Path(os.environ.get("OBS_OUTPUT_DIR", "artifacts/observability"))
        return cls(
            run_id=resolved_run_id,
            output_dir=output_dir,
            tensorboard_dir=Path(tensorboard_dir),
            num_generations=num_generations,
            config_snapshot=config_snapshot,
            trace_every_n_steps=_env_int("OBS_TRACE_EVERY_N_STEPS", 64),
            trace_max_rows=_env_int("OBS_TRACE_MAX_ROWS", 32),
            trace_max_chars=_env_int("OBS_TRACE_MAX_CHARS", 4000),
            enable_wandb=_env_bool("OBS_ENABLE_WANDB", True),
            enable_tensorboard=_env_bool("OBS_ENABLE_TENSORBOARD", True),
            enable_alerts=_env_bool("OBS_ENABLE_ALERTS", True),
            early_stop=_env_bool("OBS_EARLY_STOP", False),
            alert_cooldown_calls=_env_int("OBS_ALERT_COOLDOWN_CALLS", 8),
        )

    def start_wandb(self, *, project: str, entity: str | None) -> Any:
        if not self.enable_wandb or not os.environ.get("WANDB_API_KEY"):
            print("Observability: W&B disabled or WANDB_API_KEY missing.")
            return None
        try:
            import wandb
        except Exception as exc:
            print(f"Observability: W&B import failed: {exc}")
            return None

        self._wandb = wandb
        if wandb.run is not None:
            self._wandb_run = wandb.run
            return self._wandb_run

        kwargs: dict[str, Any] = {
            "project": project,
            "id": self.run_id,
            "resume": "allow",
            "config": self.config_snapshot,
            "tags": ["grpo", "tpu", "observability"],
        }
        if entity:
            kwargs["entity"] = entity
        try:
            self._wandb_run = wandb.init(**kwargs)
            return self._wandb_run
        except Exception as exc:
            print(f"Observability: W&B init failed, continuing without W&B: {exc}")
            self._wandb = None
            self._wandb_run = None
            return None

    def start_tensorboard(self) -> None:
        if not self.enable_tensorboard:
            return
        try:
            from tensorboardX import SummaryWriter
        except Exception as exc:
            print(f"Observability: tensorboardX import failed: {exc}")
            return
        custom_dir = self.tensorboard_dir / "observability"
        custom_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(custom_dir))

    def write_manifest(self, *, status: str = "running", extra: dict[str, Any] | None = None) -> None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "status": status,
            "git": {
                "commit": _git_env_or_text("GIT_COMMIT", ["rev-parse", "HEAD"]),
                "status_short": _git_env_or_text("GIT_STATUS_SHORT", ["status", "--short"]),
            },
            "paths": {
                "output_dir": str(self.output_dir),
                "trace_dir": str(self.trace_dir),
                "tensorboard_dir": str(self.tensorboard_dir),
                "manifest_path": str(self.manifest_path),
            },
            "observability": {
                "trace_every_n_steps": self.trace_every_n_steps,
                "trace_max_rows": self.trace_max_rows,
                "enable_wandb": bool(self._wandb_run),
                "enable_tensorboard": bool(self._tb_writer),
                "enable_alerts": self.enable_alerts,
                "early_stop": self.early_stop,
            },
            "reward": {
                "mode": REWARD_MODE,
                "components_for_mode": reward_components_for_mode(REWARD_MODE),
            },
            "config": self.config_snapshot,
        }
        if self._wandb_run is not None:
            payload["wandb"] = {
                "project": getattr(self._wandb_run, "project", None),
                "entity": getattr(self._wandb_run, "entity", None),
                "id": getattr(self._wandb_run, "id", None),
                "url": getattr(self._wandb_run, "url", None),
            }
        if extra:
            payload["extra"] = _jsonable(extra)

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def metric_fn(self, prompts, completions, rewards, advantages, **kwargs):
        completions_list = [_safe_text(x) for x in _as_list(completions)]
        n = len(completions_list)
        answers = _repeat_to_len([_safe_text(x) if x is not None else None for x in _as_list(kwargs.get("answer"))], n)
        questions = _repeat_to_len([_safe_text(x) for x in _as_list(kwargs.get("question"))], n)
        prompts_list = _repeat_to_len([_safe_text(x) for x in _as_list(prompts)], n)
        trajectory_ids = _repeat_to_len([_safe_text(x) for x in _as_list(kwargs.get("trajectory_ids"))], n)
        dataset_roles = _repeat_to_len([_safe_text(x) for x in _as_list(kwargs.get("dataset_role"))], n)
        dataset_role = next((x for x in dataset_roles if x), "unknown")

        rewards_arr = _float_array(rewards)
        advantages_arr = _float_array(advantages)
        rewards_arr = _resize_metric_array(rewards_arr, n)
        advantages_arr = _resize_metric_array(advantages_arr, n)

        diagnostics = reward_diagnostics_for_observability(completions_list, answers, REWARD_MODE)
        empty = np.asarray([not c.strip() for c in completions_list], dtype=np.float32)
        extracted_none = np.asarray([d["extracted_number"] is None for d in diagnostics], dtype=np.float32)
        exact = np.asarray([d["numeric_exact"] for d in diagnostics], dtype=np.float32)
        partial = np.asarray([d["numeric_partial"] for d in diagnostics], dtype=np.float32)
        format_ok = np.asarray([d["format_ok"] for d in diagnostics], dtype=np.float32)
        answer_tag_pair_ok = np.asarray([d["answer_tag_pair_ok"] for d in diagnostics], dtype=np.float32)
        duplicate_tag = np.asarray([d["duplicate_or_broken_answer_tag"] for d in diagnostics], dtype=np.float32)
        overlong_1200 = np.asarray([d["overlong_1200"] for d in diagnostics], dtype=np.float32)
        overlong_1600 = np.asarray([d["overlong_1600"] for d in diagnostics], dtype=np.float32)
        answer_multi_number = np.asarray([d["answer_multi_number"] for d in diagnostics], dtype=np.float32)
        chars = np.asarray([len(c) for c in completions_list], dtype=np.float32)
        words = np.asarray([len(c.split()) for c in completions_list], dtype=np.float32)
        has_solution_end = np.asarray(["</answer>" in c for c in completions_list], dtype=np.float32)

        metric_values: dict[str, float] = {
            "rollout/empty_response_rate": float(empty.mean()) if n else 0.0,
            "rollout/extracted_none_rate": float(extracted_none.mean()) if n else 0.0,
            "rollout/mean_completion_chars": float(chars.mean()) if n else 0.0,
            "rollout/max_completion_chars": float(chars.max()) if n else 0.0,
            "rollout/mean_completion_words": float(words.mean()) if n else 0.0,
            "rollout/has_solution_end_rate": float(has_solution_end.mean()) if n else 0.0,
            "rollout/answer_tag_pair_rate": float(answer_tag_pair_ok.mean()) if n else 0.0,
            "rollout/duplicate_tag_rate": float(duplicate_tag.mean()) if n else 0.0,
            "rollout/overlong_rate_1200": float(overlong_1200.mean()) if n else 0.0,
            "rollout/overlong_rate_1600": float(overlong_1600.mean()) if n else 0.0,
            "rollout/answer_multi_number_rate": float(answer_multi_number.mean()) if n else 0.0,
            "eval/numeric_exact_rate": float(exact.mean()) if n else 0.0,
            "eval/numeric_partial_rate": float(partial.mean()) if n else 0.0,
            "eval/format_accuracy": float(format_ok.mean()) if n else 0.0,
        }

        if rewards_arr.size:
            metric_values.update(
                {
                    "grpo/reward_mean": float(rewards_arr.mean()),
                    "grpo/reward_std": float(rewards_arr.std()),
                    "grpo/reward_min": float(rewards_arr.min()),
                    "grpo/reward_max": float(rewards_arr.max()),
                }
            )
        if advantages_arr.size:
            metric_values.update(
                {
                    "grpo/advantage_mean": float(advantages_arr.mean()),
                    "grpo/advantage_std": float(advantages_arr.std()),
                    "grpo/advantage_min": float(advantages_arr.min()),
                    "grpo/advantage_max": float(advantages_arr.max()),
                }
            )

        for component in (
            "match_format_exactly",
            "match_format_approximately",
            "check_answer",
            "check_numbers",
            "format_strict_light",
            "answer_tag_light",
            "numeric_primary",
            "length_penalty_1200",
        ):
            values = np.asarray([d[component] for d in diagnostics], dtype=np.float32)
            metric_values[f"reward/{component}_mean"] = float(values.mean()) if values.size else 0.0
        metric_values["reward/format_light_mean"] = float(
            np.asarray([d["format_light_total"] for d in diagnostics], dtype=np.float32).mean()
        ) if n else 0.0
        metric_values["reward/length_penalty_mean"] = metric_values["reward/length_penalty_1200_mean"]

        if rewards_arr.size and n:
            correct_rewards = rewards_arr[exact.astype(bool)]
            wrong_mask = ~exact.astype(bool)
            wrong_rewards = rewards_arr[wrong_mask]
            formatted_wrong_rewards = rewards_arr[(format_ok.astype(bool)) & wrong_mask]
            metric_values.update(
                {
                    "audit/reward_numeric_margin": float(correct_rewards.mean() - wrong_rewards.mean())
                    if correct_rewards.size and wrong_rewards.size
                    else 0.0,
                    "audit/reward_format_leakage": float(formatted_wrong_rewards.mean())
                    if formatted_wrong_rewards.size
                    else 0.0,
                    "audit/reward_hacking_rate": float(((wrong_mask) & (rewards_arr >= 3.0)).mean()),
                    "audit/numeric_correct_count": float(correct_rewards.size),
                    "audit/numeric_wrong_count": float(wrong_rewards.size),
                    "audit/formatted_wrong_count": float(formatted_wrong_rewards.size),
                }
            )

        if self.num_generations > 1 and n >= self.num_generations:
            usable = (n // self.num_generations) * self.num_generations
            reward_groups = rewards_arr[:usable].reshape(-1, self.num_generations)
            exact_groups = exact[:usable].reshape(-1, self.num_generations)
            group_std = reward_groups.std(axis=1)
            misrank_groups = 0
            comparable_groups = 0
            for reward_group, exact_group in zip(reward_groups, exact_groups):
                correct_mask = exact_group.astype(bool)
                wrong_mask = ~correct_mask
                if not correct_mask.any() or not wrong_mask.any():
                    continue
                comparable_groups += 1
                if reward_group[wrong_mask].max() >= reward_group[correct_mask].max():
                    misrank_groups += 1
            metric_values.update(
                {
                    "grpo/group_reward_std_mean": float(group_std.mean()),
                    "grpo/frac_reward_zero_std": float((group_std <= 1e-8).mean()),
                    "grpo/all_correct_group_rate": float((exact_groups.sum(axis=1) == self.num_generations).mean()),
                    "grpo/all_wrong_group_rate": float((exact_groups.sum(axis=1) == 0).mean()),
                    "audit/group_misrank_rate": _ratio(misrank_groups, comparable_groups),
                    "audit/group_misrank_count": float(misrank_groups),
                    "audit/group_misrank_comparable_groups": float(comparable_groups),
                }
            )

        with self._lock:
            self._call_index += 1
            call_index = self._call_index

        self._write_tensorboard(call_index, metric_values, rewards_arr, advantages_arr, chars)
        self._maybe_trace(
            call_index,
            dataset_role,
            prompts_list,
            questions,
            answers,
            completions_list,
            trajectory_ids,
            diagnostics,
            rewards_arr,
            advantages_arr,
        )
        self._maybe_alert(call_index, metric_values)

        return {name: (value, np.mean) for name, value in metric_values.items()}

    def _write_tensorboard(
        self,
        call_index: int,
        metric_values: dict[str, float],
        rewards: np.ndarray,
        advantages: np.ndarray,
        completion_chars: np.ndarray,
    ) -> None:
        if self._tb_writer is None:
            return
        try:
            for name, value in metric_values.items():
                self._tb_writer.add_scalar(name, value, call_index)
            if rewards.size:
                self._tb_writer.add_histogram("grpo/reward_distribution", rewards, call_index)
            if advantages.size:
                self._tb_writer.add_histogram("grpo/advantage_distribution", advantages, call_index)
            if completion_chars.size:
                self._tb_writer.add_histogram("rollout/completion_chars_distribution", completion_chars, call_index)
            self._tb_writer.flush()
        except Exception as exc:
            print(f"Observability: TensorBoard write failed: {exc}")

    def _maybe_trace(
        self,
        call_index: int,
        dataset_role: str,
        prompts: list[str],
        questions: list[str],
        answers: list[Any],
        completions: list[str],
        trajectory_ids: list[str],
        diagnostics: list[dict[str, Any]],
        rewards: np.ndarray,
        advantages: np.ndarray,
    ) -> None:
        if self.trace_every_n_steps <= 0:
            return
        if (call_index - 1) % self.trace_every_n_steps != 0:
            return

        rows = []
        limit = min(len(completions), self.trace_max_rows)
        for i in range(limit):
            group_index = i // max(self.num_generations, 1)
            generation_index = i % max(self.num_generations, 1)
            completion = completions[i]
            if len(completion) > self.trace_max_chars:
                completion = completion[: self.trace_max_chars] + "\n...[truncated]"
            rows.append(
                {
                    "call_index": call_index,
                    "dataset_role": dataset_role,
                    "row_index": i,
                    "group_index": group_index,
                    "generation_index": generation_index,
                    "trajectory_id": trajectory_ids[i],
                    "prompt_hash": _hash_text(prompts[i]),
                    "question": questions[i],
                    "answer": answers[i],
                    "completion": completion,
                    "completion_chars": len(completions[i]),
                    "reward_total": float(rewards[i]) if i < rewards.size else None,
                    "advantage": float(advantages[i]) if i < advantages.size else None,
                    "reward_components": {
                        "match_format_exactly": diagnostics[i]["match_format_exactly"],
                        "match_format_approximately": diagnostics[i]["match_format_approximately"],
                        "check_answer": diagnostics[i]["check_answer"],
                        "check_numbers": diagnostics[i]["check_numbers"],
                        "format_strict_light": diagnostics[i]["format_strict_light"],
                        "answer_tag_light": diagnostics[i]["answer_tag_light"],
                        "numeric_primary": diagnostics[i]["numeric_primary"],
                        "length_penalty_1200": diagnostics[i]["length_penalty_1200"],
                    },
                    "reward_mode": diagnostics[i]["reward_mode"],
                    "reward_total_recomputed": diagnostics[i]["reward_total_recomputed"],
                    "formatted_answer": diagnostics[i]["formatted_answer"],
                    "extracted_number": diagnostics[i]["extracted_number"],
                    "numeric_exact": diagnostics[i]["numeric_exact"],
                    "numeric_partial": diagnostics[i]["numeric_partial"],
                    "format_ok": diagnostics[i]["format_ok"],
                    "answer_tag_pair_ok": diagnostics[i]["answer_tag_pair_ok"],
                    "duplicate_or_broken_answer_tag": diagnostics[i]["duplicate_or_broken_answer_tag"],
                    "overlong_1200": diagnostics[i]["overlong_1200"],
                    "overlong_1600": diagnostics[i]["overlong_1600"],
                    "answer_multi_number": diagnostics[i]["answer_multi_number"],
                }
            )

        path = self.trace_dir / f"rollout_samples_{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")

        self._log_wandb_table(call_index, rows)

    def _log_wandb_table(self, call_index: int, rows: list[dict[str, Any]]) -> None:
        if self._wandb is None or self._wandb_run is None or not rows:
            return
        try:
            columns = [
                "call_index",
                "dataset_role",
                "group_index",
                "generation_index",
                "prompt_hash",
                "question",
                "answer",
                "completion",
                "reward_total",
                "reward_total_recomputed",
                "advantage",
                "extracted_number",
                "numeric_exact",
                "numeric_partial",
                "format_ok",
                "answer_tag_pair_ok",
                "duplicate_or_broken_answer_tag",
                "overlong_1600",
            ]
            table_rows = [[row.get(col) for col in columns] for row in rows]
            table = self._wandb.Table(columns=columns, data=table_rows)
            self._wandb_run.log({"rollout_samples": table, "obs_trace_call": call_index})
        except Exception as exc:
            print(f"Observability: W&B table write failed: {exc}")

    def _maybe_alert(self, call_index: int, metric_values: dict[str, float]) -> None:
        if not self.enable_alerts:
            return
        checks = {
            "empty_response_spike": (
                metric_values.get("rollout/empty_response_rate", 0.0),
                _env_float("OBS_ALERT_EMPTY_RESPONSE_RATE", 0.30),
                "Empty response rate is high",
            ),
            "extracted_none_spike": (
                metric_values.get("rollout/extracted_none_rate", 0.0),
                _env_float("OBS_ALERT_EXTRACTED_NONE_RATE", 0.50),
                "Answer extraction failure rate is high",
            ),
            "zero_reward_std_spike": (
                metric_values.get("grpo/frac_reward_zero_std", 0.0),
                _env_float("OBS_ALERT_ZERO_REWARD_STD_RATE", 0.90),
                "Most GRPO groups have zero reward variance",
            ),
        }
        for key, (value, threshold, text) in checks.items():
            if value < threshold:
                continue
            last = self._last_alert_call.get(key, -10**9)
            if call_index - last < self.alert_cooldown_calls:
                continue
            self._last_alert_call[key] = call_index
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "call_index": call_index,
                "key": key,
                "value": float(value),
                "threshold": float(threshold),
                "text": text,
            }
            with self.alerts_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
            print(f"OBS ALERT {key}: value={value:.4f} threshold={threshold:.4f} - {text}")
            if self._wandb is not None and self._wandb_run is not None:
                try:
                    self._wandb_run.alert(title=f"GRPO {key}", text=f"{text}: {value:.4f} >= {threshold:.4f}")
                except Exception:
                    try:
                        self._wandb.alert(title=f"GRPO {key}", text=f"{text}: {value:.4f} >= {threshold:.4f}")
                    except Exception as exc:
                        print(f"Observability: W&B alert failed: {exc}")
            if self.early_stop:
                raise RuntimeError(f"OBS_EARLY_STOP triggered by {key}: {value:.4f} >= {threshold:.4f}")

    def finish(self, *, status: str, extra: dict[str, Any] | None = None) -> None:
        self.write_manifest(status=status, extra=extra)
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception as exc:
                print(f"Observability: TensorBoard close failed: {exc}")
        self._log_wandb_artifact(status)
        if self._wandb is not None and self._wandb_run is not None:
            try:
                self._wandb_run.summary["observability_status"] = status
                self._wandb_run.finish()
            except Exception as exc:
                print(f"Observability: W&B finish failed: {exc}")

    def _log_wandb_artifact(self, status: str) -> None:
        if self._wandb is None or self._wandb_run is None:
            return
        try:
            artifact = self._wandb.Artifact(
                name=f"{self.run_id}-observability",
                type="run-observability",
                metadata={"status": status},
            )
            if self.manifest_path.exists():
                artifact.add_file(str(self.manifest_path), name="run_manifest.json")
            if self.alerts_path.exists():
                artifact.add_file(str(self.alerts_path), name="alerts.jsonl")
            if self.trace_dir.exists():
                artifact.add_dir(str(self.trace_dir), name="rollout_traces")
            self._wandb_run.log_artifact(artifact)
        except Exception as exc:
            print(f"Observability: W&B artifact logging failed: {exc}")
