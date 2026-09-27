"""One isolated, seeded comparison of the best early and later Laya epochs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from train_laya import load_dataset

ROOT = Path(__file__).resolve().parents[1]
GPU = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dataset_summary(samples):
    targets_by_state = {}
    conflicting = set()
    for item in samples:
        targets = json.dumps(item["gold"], sort_keys=True)
        if item["state"] in targets_by_state and targets_by_state[item["state"]] != targets:
            conflicting.add(item["state"])
        targets_by_state[item["state"]] = targets
    return {"samples": len(samples), "unique_states": len(targets_by_state),
            "conflicting_label_states": len(conflicting), "labels": {
        qid: dict(Counter(max(item["gold"][qid]["probabilities"], key=item["gold"][qid]["probabilities"].get)
                          for item in samples if qid in item["gold"]))
        for qid in sorted({qid for item in samples for qid in item["gold"]})}}


def run_experiment(args):
    output = args.output_dir.resolve()
    source = args.run_dir.resolve()
    training_root = (ROOT / "runtime/training").resolve()
    if not output.is_relative_to(training_root) or output == training_root or output == source or source.is_relative_to(output):
        raise ValueError("Choose a separate experiment directory inside runtime/training.")
    if not (source / "evaluation.json").is_file():
        raise ValueError("The source training run has not completed its evaluation.")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "experiment-state.json"
    state = {"status": "starting", "pid": os.getpid(), "seed": args.seed,
             "source_run": source.name, "started_utc": datetime.now(timezone.utc).isoformat()}
    # Exclusive creation survives process exits and prevents a later heartbeat restarting this study.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream)
    started = time.perf_counter()
    lock = None
    try:
        # An open handle prevents the desktop launcher's FileShare.None lock from opening.
        # A byte lock also excludes a second Python experiment using this runner.
        import msvcrt
        lock = (training_root / ".training.lock").open("a+b")
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        train_data, val_data = output / "laya-mail-train.jsonl", output / "laya-mail-val.jsonl"
        for destination in (train_data, val_data):
            shutil.copyfile(source / destination.name, destination)
            if sha256(destination) != sha256(source / destination.name):
                raise ValueError("Dataset snapshot did not match its source.")
        train_samples, val_samples = load_dataset(train_data, val_file_path=val_data)
        model_hash = sha256(args.base_model / "model.safetensors")
        active_weights = ROOT / "runtime/models/laya-mail-custom/model.safetensors"
        active_before = sha256(active_weights) if active_weights.is_file() else None
        state.update(data={"train": dataset_summary(train_samples), "val": dataset_summary(val_samples)},
                     dataset_sha256={"train": sha256(train_data), "val": sha256(val_data)},
                     base_model_sha256=model_hash, active_model_sha256_before=active_before,
                     max_epochs=20, compare_epoch=8, min_epochs=9, patience=3,
                     learning_rate=3e-4, batch_size=8)
        history_path = output / "history.json"
        model_dir = output / "model"
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": str(args.seed),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TORCH_DISABLE_NATIVE_JIT": "1"}

        def execute(stage, arguments):
            state.update(status=stage, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
            process = subprocess.Popen([sys.executable, "-u", *map(str, arguments)], env=environment)
            state["child_pid"] = process.pid
            try:
                write_json(state_path, state)
                result = process.wait()
            except BaseException:
                # Windows venv launchers can have a real Python child; reap the whole owned tree.
                if process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                process.wait()
                state.pop("child_pid", None)
                raise
            state.pop("child_pid", None)
            if result:
                raise RuntimeError(f"{stage} process failed with exit code {result}")

        execute("training", [GPU / "train_laya.py", "--data", train_data, "--val-data", val_data,
                             "--base-model", args.base_model, "--output-dir", model_dir,
                             "--epochs", 20, "--lr", "3e-4", "--batch-size", 8, "--device", "cuda",
                             "--seed", args.seed, "--history", history_path,
                             "--compare-epoch", 8, "--min-epochs", 9, "--patience", 3])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        early, later = model_dir / "best-through-8", model_dir / "best-after-8"
        for checkpoint in (early, later):
            if not (checkpoint / "model.safetensors").is_file():
                raise ValueError("An epoch comparison checkpoint is missing.")
        comparison_path = output / "comparison.json"
        execute("evaluating", [GPU / "evaluate_model.py", "--data", val_data,
                               "--model", later, "--baseline", early, "--device", "cuda",
                               "--report", comparison_path])
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        active_after = sha256(active_weights) if active_weights.is_file() else None
        source_comparison = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
        summary = {"source_run": source.name, "seed": args.seed, "data": state["data"],
                   "history": history, "comparison": comparison, "source_comparison": source_comparison,
                   "elapsed_seconds": time.perf_counter() - started,
                   "active_model_unchanged": active_before == active_after,
                   "validation_only": True}
        write_json(output / "summary.json", summary)
        state.update(status="complete", completed_epochs=history["completed_epochs"],
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     active_model_unchanged=active_before == active_after)
        write_json(state_path, state)
        print("Epoch comparison complete; aggregate results saved. No model was activated.", flush=True)
        return 0
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        return run_experiment(args)
    except Exception as error:
        print(f"Epoch experiment failed ({type(error).__name__}). Check its private log and state.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
