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
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PRESET_CHOICES = ("greedy", "standard", "liberal")
SOURCE_CHOICES = ("tfds", "kaggle")
DEFAULT_CKPT_ROOT = "~/tpu-2026/ckpts_backup/actor"


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
):
    corr = partially_corr = corr_format = total = 0

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

    return corr, total, corr / total * 100, partially_corr / total * 100, corr_format / total * 100


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
    from rewards import match_format, match_numbers

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
    n, t, acc, pacc, facc = evaluate(
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
        **GENERATION_CONFIGS[args.preset],
    )
    print(f"\nFINAL: correct={n}/{t}  acc={acc:.2f}%  partial={pacc:.2f}%  format={facc:.2f}%")

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
        },
    }
    if args.output_json:
        write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
