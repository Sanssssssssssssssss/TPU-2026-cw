"""GRPO training entry point.

Run under tmux:
    tmux new -s tunix
    source ~/venvs/tunix/bin/activate
    cd ~/tpu-2026/scripts
    python train.py
    # detach: Ctrl-b d   # reattach: tmux attach -t tunix

Resuming a wandb run: set WANDB_RUN_ID in env (or pass --wandb-run-id).
Resuming from checkpoint: just point CKPT_DIR at the existing directory.
Tunix's RLCluster uses Orbax and will pick up the latest step in CKPT_DIR.
"""
import argparse
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio
import optax
import wandb
from huggingface_hub import login as hf_login
from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

import config as config_module
from config import (
    B1, B2,
    BETA,
    CKPT_DIR,
    DATA_SOURCE,
    EPSILON,
    EVAL_EVERY_N_STEPS,
    LEARNING_RATE,
    LR_SCHEDULE_STEPS,
    MAX_GRAD_NORM,
    MAX_PROMPT_LENGTH,
    MAX_STEPS,
    MAX_TO_KEEP,
    NUM_BATCHES,
    NUM_EPOCHS,
    NUM_GENERATIONS,
    NUM_ITERATIONS,
    NUM_TEST_BATCHES,
    SAVE_INTERVAL_STEPS,
    TEMPERATURE,
    TENSORBOARD_DIR,
    TEST_DATA_DIR,
    TOP_K, TOP_P,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_ID,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from data import build_train_val_test
from grpo_observability import GRPOObservability, collect_config_snapshot
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from rewards import REWARD_FNS


def login_services():
    nest_asyncio.apply()  # tunix uses async; jupyter-style nesting helps in tmux too
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    if os.environ.get("HF_TOKEN"):
        hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)


def maybe_init_wandb(run_id: str | None):
    """Init wandb. If run_id is given we resume; otherwise a fresh run is created."""
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — skipping wandb.")
        return None
    kwargs = {"project": WANDB_PROJECT, "entity": WANDB_ENTITY}
    if run_id:
        # "allow" => resume if the run exists on the server, otherwise create
        # a new run with this id. "must" errors out if the run was never synced
        # (which is what happens if the previous training crashed before
        # wandb.init was reached).
        kwargs.update({"id": run_id, "resume": "allow"})
    return wandb.init(**kwargs)


def build_optimizer():
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=LR_SCHEDULE_STEPS,
        end_value=0.0,
    )
    opt = optax.adamw(learning_rate=schedule, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY)
    if MAX_GRAD_NORM is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM), opt)
    return opt


def build_metrics_logging_options():
    run_name = os.environ.get("WANDB_RUN_ID") or WANDB_RUN_ID or ""
    kwargs = {
        "log_dir": TENSORBOARD_DIR,
        "flush_every_n_steps": 20,
    }
    if os.environ.get("WANDB_API_KEY"):
        kwargs.update({
            "project_name": WANDB_PROJECT,
            "run_name": run_name,
        })
        return metrics_logger.MetricsLoggerOptions(**kwargs)

    print("WANDB_API_KEY not set - Tunix metrics logger will use TensorBoard only.")
    return metrics_logger.MetricsLoggerOptions(
        **kwargs,
        backend_kwargs={
            "custom_backend": [
                lambda: metrics_logger.TensorboardBackend(
                    log_dir=TENSORBOARD_DIR,
                    flush_every_n_steps=20,
                )
            ]
        },
    )


def build_cluster_config(mesh, optimizer, eos_tokens):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=MAX_STEPS,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=build_metrics_logging_options(),
            checkpoint_root_directory=CKPT_DIR,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP,
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            eos_tokens=eos_tokens,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--wandb-run-id", default=WANDB_RUN_ID or os.environ.get("RUN_ID"),
                    help="Pass an existing W&B run id to resume.")
    args = ap.parse_args()

    login_services()
    if args.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id

    observability = GRPOObservability.from_env(
        run_id=args.wandb_run_id,
        tensorboard_dir=TENSORBOARD_DIR,
        num_generations=NUM_GENERATIONS,
        config_snapshot=collect_config_snapshot(config_module),
    )
    # Init W&B before RLCluster construction; Tunix can hang if W&B is first
    # initialized inside the metrics backend during cluster setup.
    observability.start_wandb(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    observability.start_tensorboard()
    observability.write_manifest(status="initializing")

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, _ = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    train_ds, val_ds, _ = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )
    print(f"Datasets: train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    optimizer = build_optimizer()
    cluster_cfg = build_cluster_config(mesh, optimizer, eos_tokens)
    grpo_cfg = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora, reference=base, tokenizer=tokenizer, cluster_config=cluster_cfg,
    )
    trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=REWARD_FNS,
        metric_fns=[observability.metric_fn],
        algo_config=grpo_cfg,
    )

    print(
        "Starting GRPO training. "
        f"CKPT_DIR={CKPT_DIR}  MAX_STEPS={MAX_STEPS}  "
        f"LR_SCHEDULE_STEPS={LR_SCHEDULE_STEPS}  WARMUP_STEPS={WARMUP_STEPS}"
    )
    status = "failed"
    try:
        observability.write_manifest(status="running")
        trainer.train(train_ds, val_ds)
        status = "completed"
        print("Training finished.")
    finally:
        observability.finish(
            status=status,
            extra={
                "ckpt_dir": CKPT_DIR,
                "tensorboard_dir": TENSORBOARD_DIR,
                "max_steps": MAX_STEPS,
                "lr_schedule_steps": LR_SCHEDULE_STEPS,
                "eval_every_n_steps": EVAL_EVERY_N_STEPS,
                "save_interval_steps": SAVE_INTERVAL_STEPS,
            },
        )


if __name__ == "__main__":
    main()
