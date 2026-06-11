"""Verify that a fetched experiment package contains auditable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def has_any(path: Path, pattern: str) -> bool:
    return any(path.glob(pattern))


def check_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")


def check_any(path: Path, pattern: str, label: str, errors: list[str]) -> None:
    if not path.exists() or not has_any(path, pattern):
        errors.append(f"missing {label}: {path}/{pattern}")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def verify_run(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    check_file(root / "pipeline.log", "pipeline log", errors)
    check_any(root / "artifacts", "*manifest*.json", "run manifest", errors)

    manifests = sorted((root / "artifacts").glob("*manifest*.json"))
    for manifest in manifests:
        data = read_json(manifest)
        if "rank" in data and data.get("rank") in (None, ""):
            warnings.append(f"manifest rank is empty in {manifest}")

    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        errors.append(f"missing per-branch runs directory: {runs_dir}")
        return errors, warnings

    branches = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not branches:
        errors.append(f"no branch directories under {runs_dir}")
        return errors, warnings

    for branch in branches:
        check_file(branch / "run_env.txt", f"{branch.name} run env", errors)
        check_file(branch / "train.log", f"{branch.name} train log", errors)
        check_any(branch / "tensorboard", "events.out.tfevents.*", f"{branch.name} TensorBoard events", errors)
        check_any(branch / "artifacts" / "rollout_traces", "*.jsonl", f"{branch.name} rollout traces", errors)
        check_any(branch / "artifacts" / "checkpoint_eval", "*.json", f"{branch.name} checkpoint eval JSON", errors)
        check_any(branch / "artifacts" / "checkpoint_eval", "*.csv", f"{branch.name} checkpoint eval CSV", errors)
        examples_dir = branch / "artifacts" / "checkpoint_eval_examples"
        if examples_dir.exists() and not has_any(examples_dir, "*.jsonl"):
            warnings.append(f"empty eval examples directory: {examples_dir}")
        env_path = branch / "run_env.txt"
        if env_path.is_file():
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
            if "RANK=" not in env_text:
                warnings.append(f"RANK not recorded in {env_path}")
            if "ALPHA=" not in env_text:
                warnings.append(f"ALPHA not recorded in {env_path}")

    archives = root / "checkpoint_archives"
    manifest = root / "checkpoint_archives.txt"
    if not archives.exists() or not manifest.exists():
        warnings.append("checkpoint archives are not present locally yet")
    elif not has_any(archives, "*.tar.gz"):
        warnings.append(f"no checkpoint tarballs found in {archives}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Fetched run directory, e.g. artifacts/cloud/<run_id>")
    args = parser.parse_args()

    root = args.run_dir.resolve()
    errors, warnings = verify_run(root)

    print(f"run_dir: {root}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nerrors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\npackage verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
