#!/usr/bin/env python3
"""Run the audited QCal Agent 1.0 baseline, LoRA, evaluation and packaging pipeline."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="One CUDA device index; the pipeline is single-process")
    parser.add_argument("--eval-limit", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    model, dataset, root = args.model.resolve(), args.dataset.resolve(), args.run_dir.resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"Model checkpoint is missing: {model}")
    if not (dataset / "manifest.json").is_file():
        raise ValueError(f"Dataset is missing: {dataset}")
    if args.eval_limit < 1 or args.batch_size < 1:
        raise ValueError("Evaluation limit and batch size must be positive")
    root.mkdir(parents=True, exist_ok=False)
    source_paths = [project / "training" / "train_lora.py",
        project / "training" / "assistant_loss.py", Path(__file__).resolve(),
        project / "scripts" / "evaluate_policy.py", project / "src" / "qmagent" / "policies.py"]
    state = {"schema": "qcal-training-run-1.0", "pid": os.getpid(), "status": "running",
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model), "dataset": str(dataset), "gpu": args.gpu, "stages": [],
        "scope": "Synthetic single-qubit research; no hardware or coupler access",
        "source_sha256": {str(path.relative_to(project)): sha256(path) for path in source_paths}}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{project / 'src'}:{project / 'training'}")

    def save() -> None:
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def run(name: str, command: list[object], scientific_failure: bool = False) -> None:
        item = {"stage": name, "command": list(map(str, command)), "status": "running"}
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        with (root / f"{name}.log").open("x") as log:
            result = subprocess.run(list(map(str, command)), cwd=project, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {name}.log")

    python = sys.executable
    adapter = root / "training" / "final_adapter"
    audit_file = root / "tokenizer_audit.json"
    try:
        run("dataset_audit", [python, project / "scripts" / "audit_dataset.py", dataset])
        run("tokenizer_audit", [python, project / "scripts" / "audit_tokens.py",
            "--dataset", dataset, "--model", model, "--output", audit_file])
        run("assistant_loss_verification", [python, project / "training" / "verify_assistant_loss.py",
            "--model", model, "--dataset", dataset / "validation.jsonl",
            "--output", root / "assistant_loss_verification.json"])
        for split in ("test", "ood"):
            run(f"baseline_{split}", [python, project / "scripts" / "evaluate_policy.py",
                "--model", model, "--trust-remote-code", "--test-file", dataset / f"{split}.jsonl",
                "--output", root / f"baseline_{split}", "--limit", args.eval_limit,
                "--batch-size", args.batch_size])
            run(f"baseline_controller_{split}", [python, project / "scripts" / "score_policy_predictions.py",
                "--test-file", dataset / f"{split}.jsonl",
                "--predictions", root / f"baseline_{split}" / "predictions.jsonl",
                "--output", root / f"baseline_{split}" / "controller_score.json"])
        run("training", [python, project / "training" / "train_lora.py", "--model", model,
            "--train-file", dataset / "train.jsonl", "--validation-file", dataset / "validation.jsonl",
            "--output-dir", root / "training", "--baseline-metrics", root / "baseline_test" / "metrics.json",
            "--dataset-audit", audit_file, "--max-length", "20480", "--epochs", "1",
            "--micro-batch", "2", "--gradient-accumulation", "5", "--logging-steps", "5",
            "--eval-steps", "150", "--save-steps", "100", "--assistant-only-projection",
            "--no-load-best-model"])
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError("Training did not produce final_adapter")
        for split in ("test", "ood"):
            adapted = root / f"adapted_{split}"
            run(f"adapted_{split}", [python, project / "scripts" / "evaluate_policy.py",
                "--model", model, "--adapter", adapter, "--trust-remote-code",
                "--test-file", dataset / f"{split}.jsonl", "--output", adapted,
                "--limit", args.eval_limit, "--batch-size", args.batch_size])
            run(f"adapted_controller_{split}", [python, project / "scripts" / "score_policy_predictions.py",
                "--test-file", dataset / f"{split}.jsonl", "--predictions", adapted / "predictions.jsonl",
                "--output", adapted / "controller_score.json"])
            run(f"comparison_{split}", [python, project / "scripts" / "compare_policy.py",
                "--baseline", root / f"baseline_{split}", "--adapted", adapted,
                "--output", root / f"comparison_{split}.json"])
        for arm in ("base", "adapted"):
            command = [python, "-m", "qmagent.cli", "run", "--policy", "hf",
                "--decode-backend", "hf", "--model", model, "--trust-remote-code",
                "--backend", "physical", "--seed", "2026091370", "--episodes", "3",
                "--max-steps", "45", "--max-tool-calls", "10", "--request-timeout", "300",
                "--output-dir", root / f"closed_loop_{arm}"]
            if arm == "adapted":
                command += ["--adapter", adapter]
            run(f"closed_loop_{arm}", command, scientific_failure=True)
        release = root / "release"
        release.mkdir()
        artifact = release / "qcal-agent-1.0.0-lora.safetensors"
        artifact.write_bytes((adapter / "adapter_model.safetensors").read_bytes())
        evidence = {"version": "1.0.0", "model": str(model), "dataset": str(dataset),
            "adapter_sha256": sha256(artifact), "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
            "comparison_test_sha256": sha256(root / "comparison_test.json"),
            "comparison_ood_sha256": sha256(root / "comparison_ood.json")}
        (release / "release_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
        state["status"] = "completed_with_scientific_failures" if any(
            item["status"] == "failed" for item in state["stages"]) else "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
