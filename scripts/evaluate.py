"""Standalone evaluation of a base or LoRA policy on the GSM8K test set.

Reports:
  * accuracy           - exact numeric match
  * partial_accuracy   - answer within 10% of ground truth
  * format_accuracy    - fraction of completions whose template parses

Examples:
    python evaluate.py --no-restore --preset greedy --output-json artifacts/base_eval.json
    python evaluate.py --ckpt-dir /tmp/content/ckpts/actor --step 0 --preset greedy \
        --output-json artifacts/baseline_lora_eval.json
"""

import argparse
from collections import Counter
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PRESET_CHOICES = ("greedy", "standard", "liberal")
SOURCE_CHOICES = ("tfds", "kaggle")
DEFAULT_CKPT_ROOT = "~/tpu-2026/ckpts_backup/actor"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
FAILURE_TYPES = (
    "correct",
    "format_fail_only",
    "missing_answer_open",
    "missing_answer_close",
    "no_number",
    "wrong_number",
    "multiple_numbers",
    "trailing_text_after_close",
    "empty_completion",
)


def generate(
    question,
    sampler,
    eos_tokens,
    template,
    system_prompt,
    generation_steps,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    seed=None,
):
    if isinstance(question, str):
        batch = [template.format(system_prompt=system_prompt, question=question)]
    else:
        batch = [template.format(system_prompt=system_prompt, question=q) for q in question]

    out = sampler(
        input_strings=batch,
        max_generation_steps=generation_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=eos_tokens,
    )
    return out.text[0] if isinstance(question, str) else out.text


def evaluate(
    dataset,
    sampler,
    eos_tokens,
    template,
    system_prompt,
    match_format,
    match_numbers,
    tqdm,
    generation_steps,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    diagnostics_fn: Callable[[list[str], list[Any], str | None], list[dict[str, Any]]] | None = None,
):
    corr = partially_corr = corr_format = total = 0
    example_rows: list[dict[str, Any]] = []

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]
        per_q = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(
                questions,
                sampler,
                eos_tokens,
                template,
                system_prompt,
                generation_steps,
                temperature,
                top_k,
                top_p,
                seed=p,
            )
            for i, r in enumerate(responses):
                per_q[i].append(r)
            if diagnostics_fn is not None:
                diagnostics = diagnostics_fn(responses, answers, "numeric_dense_single_answer")
                for i, (question, answer, response, diag) in enumerate(
                    zip(questions, answers, responses, diagnostics)
                ):
                    example_rows.append(build_example_record(question, answer, response, diag, i, p))

        for responses, ans in zip(per_q, answers):
            got_corr = got_partial = got_format = False
            for r in responses:
                ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
                try:
                    if float(ext.strip()) == float(ans.strip()):
                        got_corr = True
                    ratio = float(ext.strip()) / float(ans.strip())
                    if 0.9 <= ratio <= 1.1:
                        got_partial = True
                except Exception:
                    pass
                if match_format.search(r) is not None:
                    got_format = True
                if got_corr and got_partial and got_format:
                    break

            corr += int(got_corr)
            partially_corr += int(got_partial)
            corr_format += int(got_format)
            total += 1
            if total % 10 == 0:
                print(
                    f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                    f"partial={partially_corr/total*100:.2f}% fmt={corr_format/total*100:.2f}%"
                )

    return (
        corr,
        total,
        corr / total * 100,
        partially_corr / total * 100,
        corr_format / total * 100,
        example_rows,
    )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _text_after_answer_close(completion: str) -> str:
    index = completion.find(ANSWER_CLOSE)
    if index < 0:
        return ""
    return completion[index + len(ANSWER_CLOSE) :].strip()


def classify_failure(completion: str, diagnostics: dict[str, Any]) -> str:
    """Classify why a completion did or did not pass the official eval."""
    text = _safe_text(completion)
    has_open = ANSWER_OPEN in text
    has_close = ANSWER_CLOSE in text
    text_after_close = _text_after_answer_close(text)
    official_exact = bool(diagnostics.get("official_numeric_exact"))
    format_ok = bool(diagnostics.get("format_ok"))
    answer_number_count = int(diagnostics.get("robust_answer_number_count") or 0)

    if not text.strip():
        return "empty_completion"
    if official_exact and format_ok:
        return "correct"
    if not has_open:
        return "missing_answer_open"
    if has_open and not has_close:
        return "missing_answer_close"
    if diagnostics.get("official_extracted_number") is None:
        return "no_number"
    if answer_number_count > 1:
        return "multiple_numbers"
    if text_after_close:
        return "trailing_text_after_close"
    if official_exact and not format_ok:
        return "format_fail_only"
    return "wrong_number"


def build_example_record(
    question: Any,
    gold_answer: Any,
    completion: Any,
    diagnostics: dict[str, Any],
    example_index: int,
    pass_index: int,
) -> dict[str, Any]:
    completion_text = _safe_text(completion)
    has_open = ANSWER_OPEN in completion_text
    has_close = ANSWER_CLOSE in completion_text
    text_after_close = _text_after_answer_close(completion_text)
    failure_type = classify_failure(completion_text, diagnostics)
    return {
        "example_index": example_index,
        "pass_index": pass_index,
        "question": _safe_text(question),
        "gold_answer": _safe_text(gold_answer),
        "completion": completion_text,
        "official_extracted_number": diagnostics.get("official_extracted_number"),
        "robust_extracted_number": diagnostics.get("robust_extracted_number"),
        "has_answer_open": has_open,
        "has_answer_close": has_close,
        "has <answer>": has_open,
        "has </answer>": has_close,
        "text_after_answer_close": text_after_close,
        "numeric_exact": bool(diagnostics.get("official_numeric_exact")),
        "partial_numeric": bool(diagnostics.get("official_numeric_partial")),
        "robust_numeric_exact": bool(diagnostics.get("robust_numeric_exact")),
        "robust_partial_numeric": bool(diagnostics.get("robust_numeric_partial")),
        "format_ok": bool(diagnostics.get("format_ok")),
        "failure_type": failure_type,
        "answer_number_count": diagnostics.get("robust_answer_number_count"),
        "parser_false_negative": bool(diagnostics.get("parser_false_negative")),
    }


def summarize_examples(example_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(example_rows)
    failure_counts = Counter(row.get("failure_type", "unknown") for row in example_rows)
    for name in FAILURE_TYPES:
        failure_counts.setdefault(name, 0)
    return {
        "failure_counts": dict(sorted(failure_counts.items())),
        "no_close_answer_rate": (
            sum(bool(row.get("has_answer_open")) and not bool(row.get("has_answer_close")) for row in example_rows)
            / total
            if total
            else 0.0
        ),
        "text_after_close_rate": (
            sum(bool(_safe_text(row.get("text_after_answer_close")).strip()) for row in example_rows) / total
            if total
            else 0.0
        ),
        "robust_numeric_exact_rate": (
            sum(bool(row.get("robust_numeric_exact")) for row in example_rows) / total if total else 0.0
        ),
    }


def git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def write_json(path: str, payload: dict) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evaluation JSON: {out}")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote evaluation examples JSONL: {out}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate the base Gemma policy or a restored LoRA checkpoint on GSM8K."
    )
    ap.add_argument("--preset", default="greedy", choices=PRESET_CHOICES)
    ap.add_argument(
        "--source",
        default=None,
        choices=SOURCE_CHOICES,
        help="Dataset source. Defaults to DATA_SOURCE from config.py.",
    )
    ap.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_ROOT,
        help="Directory containing per-step actor checkpoint subdirectories.",
    )
    ap.add_argument("--step", type=int, default=0, help="Checkpoint step to restore. Use 0 for latest.")
    ap.add_argument(
        "--no-restore",
        action="store_true",
        help="Evaluate the base model by leaving LoRA adapters at their initial zero update.",
    )
    ap.add_argument(
        "--num-passes",
        type=int,
        default=1,
        help="Number of generations per prompt for pass@N-style evaluation.",
    )
    ap.add_argument("--output-json", default=None, help="Optional path for machine-readable results.")
    ap.add_argument(
        "--output-examples-jsonl",
        default=None,
        help="Optional JSONL path containing one row per generated completion with parser diagnostics.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    from tqdm.auto import tqdm
    from tunix.generate import sampler as sampler_lib

    from chat import restore_lora
    from config import (
        DATA_SOURCE,
        GENERATION_CONFIGS,
        MAX_PROMPT_LENGTH,
        NUM_BATCHES,
        NUM_EPOCHS,
        NUM_TEST_BATCHES,
        TEST_DATA_DIR,
        TOTAL_GENERATION_STEPS,
        TRAIN_DATA_DIR,
        TRAIN_FRACTION,
        TRAIN_MICRO_BATCH_SIZE,
    )
    from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
    from model import build_mesh, download_weights, get_lora_model, load_base_model, load_tokenizer
    from rewards import match_format, match_numbers, reward_diagnostics_for_observability

    source = args.source or DATA_SOURCE

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    restored_step = None
    if args.no_restore:
        print("Skipping checkpoint restore - evaluating the base model.")
    else:
        step = None if args.step == 0 else args.step
        restored_step = restore_lora(lora, str(Path(args.ckpt_dir).expanduser()), step)

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES,
        NUM_TEST_BATCHES,
        TRAIN_MICRO_BATCH_SIZE,
        TRAIN_FRACTION,
        NUM_EPOCHS,
        TRAIN_DATA_DIR,
        TEST_DATA_DIR,
        source=source,
    )

    sampler = sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    n, t, acc, pacc, facc, example_rows = evaluate(
        test_ds,
        sampler,
        eos_tokens,
        TEMPLATE,
        SYSTEM_PROMPT,
        match_format,
        match_numbers,
        tqdm,
        TOTAL_GENERATION_STEPS,
        num_passes=args.num_passes,
        diagnostics_fn=reward_diagnostics_for_observability,
        **GENERATION_CONFIGS[args.preset],
    )
    print(f"\nFINAL: correct={n}/{t}  acc={acc:.2f}%  partial={pacc:.2f}%  format={facc:.2f}%")
    example_summary = summarize_examples(example_rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": {
            "policy": "base" if args.no_restore else "lora",
            "checkpoint_restored": not args.no_restore,
            "checkpoint_dir": None if args.no_restore else str(Path(args.ckpt_dir).expanduser()),
            "requested_step": None if args.no_restore else args.step,
            "restored_step": restored_step,
        },
        "data": {
            "source": source,
            "num_test_batches": NUM_TEST_BATCHES,
            "train_micro_batch_size": TRAIN_MICRO_BATCH_SIZE,
        },
        "generation": {
            "preset": args.preset,
            "config": GENERATION_CONFIGS[args.preset],
            "max_generation_steps": TOTAL_GENERATION_STEPS,
            "num_passes": args.num_passes,
        },
        "metrics": {
            "correct": n,
            "total": t,
            "accuracy": acc,
            "partial_accuracy": pacc,
            "format_accuracy": facc,
            **example_summary,
        },
        "outputs": {
            "examples_jsonl": str(Path(args.output_examples_jsonl).expanduser())
            if args.output_examples_jsonl
            else None,
        },
    }
    if args.output_examples_jsonl:
        write_jsonl(args.output_examples_jsonl, example_rows)
    if args.output_json:
        write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
