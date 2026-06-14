"""Single source of truth for hyperparameters and paths.

Everything in this file is a knob you might tune. Environment variables can
override the defaults, which lets remote TPU jobs and smoke tests change paths
or shrink runs without editing code.
"""

import os

import jax


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ====== Model ======
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-1b-it")
GEMMA_TOKENIZER_PATH = os.environ.get(
    "GEMMA_TOKENIZER_PATH", "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
)

# ====== Data ======
TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "./data/train")
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "./data/test")
TRAIN_FRACTION = _env_float("TRAIN_FRACTION", 0.9)
DATA_SOURCE = os.environ.get("DATA_SOURCE", "tfds")  # "tfds" or "kaggle"

# ====== LoRA (parameter-efficient finetuning) ======
# Only the LoRA adapters are trained; the base model is frozen and shared with
# the reference model. Smaller rank => fewer trainable params, smaller KL drift.
RANK = _env_int("RANK", 64)
ALPHA = _env_float("ALPHA", 64.0)

# ====== Sharding (TPU mesh) ======
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 4:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [MESH_COUNTS, ("fsdp", "tp")]

# ====== Generation during GRPO rollouts ======
MAX_PROMPT_LENGTH = _env_int("MAX_PROMPT_LENGTH", 256)
TOTAL_GENERATION_STEPS = _env_int("TOTAL_GENERATION_STEPS", 768)
TEMPERATURE = _env_float("TEMPERATURE", 0.9)
TOP_P = _env_float("TOP_P", 1.0)
TOP_K = _env_int("TOP_K", 50)
NUM_GENERATIONS = _env_int("NUM_GENERATIONS", 2)

# ====== GRPO loss ======
NUM_ITERATIONS = _env_int("NUM_ITERATIONS", 1)
BETA = _env_float("BETA", 0.08)
EPSILON = _env_float("EPSILON", 0.2)
_ADVANTAGE_RAW = (
    os.environ.get("GRPO_ADVANTAGE_ESTIMATOR")
    or os.environ.get("GRPO_ADVANTAGE_MODE")
    or "grpo"
).strip().lower()
GRPO_ADVANTAGE_ESTIMATOR = {
    "leave_one_out": "rloo",
    "leave-one-out": "rloo",
    "loo": "rloo",
    "rloo": "rloo",
    "group_mean": "grpo",
    "group-mean": "grpo",
    "grpo": "grpo",
}.get(_ADVANTAGE_RAW, _ADVANTAGE_RAW)

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = _env_int("TRAIN_MICRO_BATCH_SIZE", 1)
NUM_BATCHES = _env_int("NUM_BATCHES", 3738)
NUM_TEST_BATCHES = _env_int("NUM_TEST_BATCHES", 64)
EVAL_EVERY_N_STEPS = _env_int("EVAL_EVERY_N_STEPS", 64)
NUM_EPOCHS = _env_int("NUM_EPOCHS", 1)
_DEFAULT_MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
MAX_STEPS = _env_int("MAX_STEPS", _DEFAULT_MAX_STEPS)

# ====== Optimiser ======
LEARNING_RATE = _env_float("LEARNING_RATE", 3e-6)
B1 = _env_float("B1", 0.9)
B2 = _env_float("B2", 0.99)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.1)
LR_SCHEDULE_STEPS = _env_int("LR_SCHEDULE_STEPS", MAX_STEPS)
WARMUP_STEPS = _env_float("WARMUP_STEPS", 0.1 * LR_SCHEDULE_STEPS)
MAX_GRAD_NORM = _env_float("MAX_GRAD_NORM", 0.1)

# ====== Checkpointing ======
# Defaults preserve the original baseline. Cloud jobs override these to a
# persistent per-run directory under ~/tpu-runs/<RUN_ID>/.
INTERMEDIATE_CKPT_DIR = os.environ.get("INTERMEDIATE_CKPT_DIR", "/tmp/content/intermediate_ckpt/")
CKPT_DIR = os.environ.get("CKPT_DIR", "/tmp/content/ckpts/")
TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR", "/tmp/content/tmp/tensorboard/grpo")
SAVE_INTERVAL_STEPS = _env_int("SAVE_INTERVAL_STEPS", 500)
MAX_TO_KEEP = _env_int("MAX_TO_KEEP", 4)

# ====== Inference presets ======
GENERATION_CONFIGS = {
    "greedy": {"temperature": None, "top_k": 1, "top_p": None},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== Observability ======
OBS_OUTPUT_DIR = os.environ.get("OBS_OUTPUT_DIR", "artifacts/observability")
OBS_TRACE_EVERY_N_STEPS = _env_int("OBS_TRACE_EVERY_N_STEPS", 64)
OBS_TRACE_MAX_ROWS = _env_int("OBS_TRACE_MAX_ROWS", 32)

# ====== W&B ======
# Set WANDB_RUN_ID in env to resume an existing run. If WANDB_ENTITY is unset,
# W&B logs to the default entity for the authenticated account.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-tpu-2026")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID", None)
