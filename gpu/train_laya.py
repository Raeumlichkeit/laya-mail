"""Train and fine-tune Laya decision heads on exported email review data with PyTorch."""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

# Resolve project roots
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CANDIDATES = [
    REPO_ROOT / "runtime" / "models" / "multilingual",
    REPO_ROOT.parent / "work" / "laya-gpu" / "multilingual",
    REPO_ROOT.parent / "outputs" / "laya-gpu" / "multilingual",
]


def find_default_base_model():
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_dir() and (candidate / "model.safetensors").exists():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL training data file")
    parser.add_argument("--val-data", type=Path, default=None,
                        help="Optional separate path to JSONL validation data file")
    parser.add_argument("--base-model", type=Path, default=find_default_base_model(),
                        help="Path to base multilingual Laya model directory")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runtime" / "models" / "laya-mail-custom",
                        help="Directory to save fine-tuned model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size in samples (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed random seed")
    parser.add_argument("--history", type=Path, default=None, help="Atomically updated aggregate epoch history JSON")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without lower validation loss; 0 disables")
    parser.add_argument("--min-epochs", type=int, default=1, help="Earliest epoch at which early stopping is allowed")
    parser.add_argument("--compare-epoch", type=int, default=0,
                        help="Also save the best checkpoints through/after this epoch; 0 disables")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze the transformer encoder and only train decision heads (default: True)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Unfreeze encoder backbone for full fine-tuning")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda or cpu; default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify dataset formatting and pipeline without running full training")
    return parser.parse_args()


def validate_sample(sample, location="sample", *, allow_gmail_spam=False):
    """Validate an exported target distribution without echoing email contents."""
    def invalid(reason):
        raise ValueError(f"{location}: {reason}") from None

    if not isinstance(sample, dict):
        invalid("expected a JSON object")
    state = sample.get("state")
    if not isinstance(state, str) or not state.strip():
        invalid("state must be a nonempty string")
    try:
        state.encode("utf-8")
    except UnicodeError:
        invalid("state contains invalid Unicode")
    questions, gold = sample.get("questions"), sample.get("gold")
    if not isinstance(questions, dict) or not questions or not isinstance(gold, dict) or set(questions) != set(gold):
        invalid("questions and gold must have the same nonempty set of question IDs")
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("split") not in (None, "train", "val"):
        invalid("metadata.split must be train or val when supplied")
    gmail_spam = metadata.get("label_source") == "gmail_spam"
    if gmail_spam:
        if not allow_gmail_spam:
            invalid("Gmail spam labels are training-only and cannot be used for evaluation")
        if metadata.get("split") != "train":
            invalid("Gmail spam labels require metadata.split=train")
        if set(questions) != {"spamverdacht"}:
            invalid("Gmail spam labels must contain only the spamverdacht question")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid.strip() or not isinstance(question, dict):
            invalid("question IDs must be nonempty strings with object definitions")
        kind = question.get("type")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            invalid("question instructions must be a nonempty string")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria or any(not isinstance(k, str) or not k.strip() for k in criteria):
                invalid("choice criteria must be a nonempty object with nonempty string keys")
            expected = set(criteria)
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                invalid("score criteria must be a nonempty list")
            expected = {str(i) for i in range(len(criteria))}
        elif kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or not set(criteria).issubset({"false", "true"})):
                invalid("noul criteria may only describe false and true")
            expected = {"false", "true"}
        else:
            invalid("unsupported question type")
        if "labels" in question:
            labels = question["labels"]
            if (kind != "noul" or not isinstance(labels, dict) or set(labels) != {"false", "true"}
                    or any(not isinstance(v, str) or not v.strip() for v in labels.values())
                    or labels["false"].strip() == labels["true"].strip()):
                invalid("custom labels require distinct nonempty noul labels for false and true")
        answer = gold[qid]
        if not isinstance(answer, dict) or answer.get("type", kind) != kind:
            invalid("gold type must match its question")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            invalid("gold probabilities must match all question options exactly")
        values = list(probabilities.values())
        if (any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values)
                or not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)):
            invalid("gold probabilities must be finite, nonnegative numbers summing to one")
    if gmail_spam and (questions["spamverdacht"]["type"] != "choice"
                      or gold["spamverdacht"]["probabilities"] != {"ja": 1.0, "nein": 0.0}):
        invalid("Gmail spam labels require a positive ja/nein choice target")


def load_samples(file_path, *, allow_gmail_spam=False):
    """Read nonempty UTF-8 JSONL; invalid rows raise a content-free line error."""
    file_path = Path(file_path)
    samples = []
    for number, raw in enumerate(file_path.read_bytes().splitlines(), 1):
        location = f"{file_path}: line {number}"
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeError:
            raise ValueError(f"{location}: invalid UTF-8") from None
        if not line:
            continue
        try:
            sample = json.loads(line)
        except (ValueError, RecursionError):
            raise ValueError(f"{location}: invalid JSON") from None
        validate_sample(sample, location, allow_gmail_spam=allow_gmail_spam)
        samples.append(sample)
    if not samples:
        raise ValueError(f"{file_path}: dataset is empty; collect and export reviews first")
    return samples


def load_dataset(file_path, val_ratio=0.20, val_file_path=None):
    samples = load_samples(file_path, allow_gmail_spam=True)
    if val_file_path:
        train_samples, val_samples = samples, load_samples(val_file_path)
        if any(item.get("metadata", {}).get("split") == "val" for item in train_samples):
            raise ValueError("Training file contains examples marked for validation")
        if any(item.get("metadata", {}).get("split") == "train" for item in val_samples):
            raise ValueError("Validation file contains examples marked for training")
    else:
        train_samples, val_samples = [], []
        for item in samples:
            split = item.get("metadata", {}).get("split")
            if split is None:
                h = int(hashlib.sha256(item["state"].encode("utf-8")).hexdigest()[:8], 16)
                split = "val" if (h % 100) < int(val_ratio * 100) else "train"
            (val_samples if split == "val" else train_samples).append(item)
    if not train_samples:
        raise ValueError("Training split is empty; at least one training example is required")
    train_states = {sample["state"] for sample in train_samples}
    if any(sample["state"] in train_states for sample in val_samples):
        raise ValueError("Training and validation sets overlap; use disjoint state content")
    return train_samples, val_samples


def build_sample_items(tok, sample, max_len=1024, head_max_len=256):
    from laya.common import QTYPES, build_sequence

    validate_sample(sample, allow_gmail_spam=True)

    state = sample["state"]
    questions = sample["questions"]
    gold = sample["gold"]

    truncate_left = isinstance(state, list)
    state_clean = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    state_ids = tok(
        state_clean.replace(tok.mask_token, " "),
        add_special_tokens=False,
    )["input_ids"]

    items = []
    for qid, qdef in questions.items():
        t = qdef["type"]
        q_internal = {"t": t, "ins": qdef["instructions"], "crit": qdef.get("criteria", {})}
        if "labels" in qdef:
            q_internal["labels"] = qdef["labels"]
        seq, markers = build_sequence(tok, state, q_internal, max_len, head_max_len,
                                      truncate_left=truncate_left, state_ids=state_ids)
        gold_q = gold[qid]
        target_probs = gold_q["probabilities"]

        # Align target array with criteria order
        if t == "choice":
            criteria_keys = list(q_internal["crit"].keys())
            target = [float(target_probs[k]) for k in criteria_keys]
        elif t == "score":
            target = [float(target_probs[str(idx)]) for idx in range(len(q_internal["crit"]))]
        else:
            target = [float(target_probs["false"]), float(target_probs["true"])]

        if len(target) != len(markers):
            raise ValueError("Tokenized question lost option markers; increase max_len or head_max_len")

        items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[t],
            "target": target,
            "qid": qid,
        })
    return items


def batch_samples(tok, samples, batch_size=4, max_len=1024, head_max_len=256, shuffle=False):
    from laya.common import collate_items

    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)

    all_items = []
    for idx in indices:
        sample_items = build_sample_items(tok, samples[idx], max_len=max_len, head_max_len=head_max_len)
        if sample_items:
            all_items.append(sample_items)

    batches = []
    for i in range(0, len(all_items), batch_size):
        chunk = all_items[i:i + batch_size]
        batch = collate_items(chunk, tok.pad_token_id)
        if batch is not None:
            # Flatten qids to match question rows in batch
            batch["qids"] = [item["qid"] for sublist in chunk for item in sublist]
            batches.append(batch)
    return batches


def evaluate(model, batches, device):
    import torch

    if not batches:
        raise ValueError("Validation produced no usable batches")
    model.eval()
    total_loss = 0.0
    total_rows = 0
    correct = defaultdict(int)
    totals = defaultdict(int)

    with torch.no_grad():
        for b in batches:
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            marker_pos = b["marker_pos"].to(device)
            marker_mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)

            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            log_p = torch.log_softmax(logits, dim=-1)
            loss = -(target * log_p).sum(dim=-1).mean()
            if not torch.isfinite(loss).item():
                raise ValueError("Validation loss is not finite; no checkpoint was saved")
            total_loss += loss.item() * target.shape[0]
            total_rows += target.shape[0]

            preds = logits.argmax(dim=-1)
            gold_preds = target.argmax(dim=-1)

            for pred, true, qid in zip(preds.cpu().tolist(), gold_preds.cpu().tolist(), b["qids"]):
                if pred == true:
                    correct[qid] += 1
                totals[qid] += 1

    avg_loss = total_loss / total_rows
    accuracies = {qid: correct[qid] / max(1, totals[qid]) for qid in totals}
    overall_acc = sum(correct.values()) / max(1, sum(totals.values()))
    return avg_loss, accuracies, overall_acc


def set_seed(seed):
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 64))
    torch.cuda.manual_seed_all(seed % (2 ** 64))


def write_history(path, history):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(history, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def record_epoch(history, row):
    """Return checkpoint decisions; selection and stopping use validation loss only."""
    history["epochs"].append(row)
    epoch = row["epoch"]
    validation = [item for item in history["epochs"] if item["val_loss"] is not None]
    best = min(validation, key=lambda item: item["val_loss"]) if validation else row
    history["best_epoch"] = best["epoch"]
    history["completed_epochs"] = epoch
    boundary = history["compare_epoch"]
    through = [item for item in validation if item["epoch"] <= boundary] if boundary else []
    after = [item for item in validation if item["epoch"] > boundary] if boundary else []
    history["best_through_epoch"] = min(through, key=lambda item: item["val_loss"])["epoch"] if through else None
    history["best_after_epoch"] = min(after, key=lambda item: item["val_loss"])["epoch"] if after else None
    history["stopped_early"] = bool(validation and history["patience"]
                                    and history["min_epochs"] <= epoch < history["requested_epochs"]
                                    and epoch - best["epoch"] >= history["patience"])
    return (bool(validation and best["epoch"] == epoch),
            bool(boundary and epoch == boundary),
            bool(boundary and history["best_after_epoch"] == epoch))


def save_checkpoint(path, weights, config, args, completed_epochs, best_epoch):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path / "model.safetensors"))
    for folder in ("encoder", "tokenizer"):
        src, dst = args.base_model / folder, path / folder
        if src.is_dir() and not dst.is_dir():
            shutil.copytree(src, dst)
    cfg = copy.deepcopy(config)
    cfg["training"] = {
        "fine_tuned_from_checkpoint": True,
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "seed": getattr(args, "seed", None),
        "base_model": str(args.base_model.name),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def train(args):
    print("=" * 60)
    print(" Laya Mail · Fine-Tuning Pipeline (RTX 4070 Ti / CUDA)")
    print("=" * 60)

    if type(getattr(args, "epochs", 5)) is not int or getattr(args, "epochs", 5) < 1:
        raise ValueError("epochs must be a positive integer")
    if type(getattr(args, "batch_size", 4)) is not int or getattr(args, "batch_size", 4) < 1:
        raise ValueError("batch-size must be a positive integer")
    learning_rate = getattr(args, "lr", 2e-4)
    if type(learning_rate) not in (int, float) or not 0 < learning_rate < float("inf"):
        raise ValueError("lr must be a finite positive number")
    seed = getattr(args, "seed", None)
    history_path = getattr(args, "history", None)
    patience = getattr(args, "patience", 0)
    min_epochs = getattr(args, "min_epochs", 1)
    compare_epoch = getattr(args, "compare_epoch", 0)
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(patience) is not int or patience < 0:
        raise ValueError("patience must be a nonnegative integer")
    if type(min_epochs) is not int or not 1 <= min_epochs <= getattr(args, "epochs", 5):
        raise ValueError("min-epochs must be between 1 and epochs")
    if type(compare_epoch) is not int or not 0 <= compare_epoch < getattr(args, "epochs", 5):
        raise ValueError("compare-epoch must be nonnegative and below epochs")
    if compare_epoch and patience and min_epochs <= compare_epoch:
        raise ValueError("min-epochs must exceed compare-epoch when early stopping is enabled")
    val_data = getattr(args, "val_data", None)
    train_samples, val_samples = load_dataset(args.data, val_ratio=getattr(args, "val_ratio", 0.20), val_file_path=val_data)
    print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    if (patience or compare_epoch) and not val_samples:
        raise ValueError("Early stopping and checkpoint comparison require validation samples")

    if args.dry_run:
        print("[Dry-Run] Dataset validation successful. Exiting without training.")
        return 0

    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {args.base_model}")

    import torch
    import laya

    set_seed(seed)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"Loading base model from: {args.base_model}")
    agent = laya.load(str(args.base_model), device="cpu")
    tok = agent.tok
    model = agent.model

    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    if args.freeze_backbone:
        print("Freezing encoder backbone. Only decision heads and type embeddings will be trained.")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Full fine-tuning: training encoder and decision heads.")
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    max_len = agent.cfg.get("max_len", 1024)
    head_max_len = agent.cfg.get("head_max_len", 256)

    val_batches = batch_samples(tok, val_samples, batch_size=args.batch_size,
                                max_len=max_len, head_max_len=head_max_len, shuffle=False)
    if val_samples and not val_batches:
        raise ValueError("Validation produced no usable batches")

    best_weights = None
    history = {"schema_version": 1, "seed": seed, "compare_epoch": compare_epoch,
               "patience": patience, "min_epochs": min_epochs, "requested_epochs": args.epochs,
               "epochs": [], "best_epoch": None, "completed_epochs": 0, "stopped_early": False,
               "best_through_epoch": None, "best_after_epoch": None}
    write_history(history_path, history)

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_batches = batch_samples(tok, train_samples, batch_size=args.batch_size,
                                      max_len=max_len, head_max_len=head_max_len, shuffle=True)
        if not train_batches:
            raise ValueError("Training produced no usable batches; no checkpoint was saved")
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0

        for b in train_batches:
            optimizer.zero_grad()
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            marker_pos = b["marker_pos"].to(device)
            marker_mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)

            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype,
                              detach_encoder=args.freeze_backbone)
            log_p = torch.log_softmax(logits, dim=-1)
            loss = -(target * log_p).sum(dim=-1).mean()
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite; no checkpoint was saved")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * target.shape[0]
            total_train_rows += target.shape[0]

        train_loss = total_train_loss / total_train_rows
        train_seconds = time.perf_counter() - t0
        validation_start = time.perf_counter()
        val_loss, val_accs, val_overall = None, {}, None

        if val_batches:
            val_loss, val_accs, val_overall = evaluate(model, val_batches, device)
        validation_seconds = time.perf_counter() - validation_start
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_accuracies": val_accs, "val_overall": val_overall,
               "train_seconds": train_seconds, "validation_seconds": validation_seconds,
               "checkpoint_seconds": 0.0, "seconds": 0.0}
        improved, save_through, save_after = record_epoch(history, row)
        checkpoint_start = time.perf_counter()
        if improved:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if save_through:
            save_checkpoint(args.output_dir / f"best-through-{compare_epoch}", best_weights,
                            agent.cfg, args, epoch, history["best_through_epoch"])
        if save_after:
            later_weights = best_weights if improved else {k: v.cpu() for k, v in model.state_dict().items()}
            save_checkpoint(args.output_dir / f"best-after-{compare_epoch}", later_weights,
                            agent.cfg, args, epoch, history["best_after_epoch"])
            del later_weights
        row["checkpoint_seconds"] = time.perf_counter() - checkpoint_start
        row["seconds"] = time.perf_counter() - t0
        write_history(history_path, history)
        suffix = "(No validation set)"
        if val_loss is not None:
            acc_str = " · ".join(f"{k}: {v * 100:.1f}%" for k, v in val_accs.items())
            suffix = f"Val Loss: {val_loss:.4f} | Overall: {val_overall * 100:.1f}% [{acc_str}]"
        print(f"Epoch {epoch:2d}/{args.epochs:2d} ({row['seconds']:.1f}s) | Train Loss: {train_loss:.4f} | {suffix}")
        if history["stopped_early"]:
            print(f"Early stopping after epoch {epoch}; best validation epoch was {history['best_epoch']}.")
            break

    print(f"\nTraining completed. Saving checkpoint to: {args.output_dir}")
    checkpoint_start = time.perf_counter()
    weights_to_save = best_weights if best_weights is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    save_checkpoint(args.output_dir, weights_to_save, agent.cfg, args,
                    history["completed_epochs"], history["best_epoch"])
    final_checkpoint_seconds = time.perf_counter() - checkpoint_start
    history["epochs"][-1]["checkpoint_seconds"] += final_checkpoint_seconds
    history["epochs"][-1]["seconds"] += final_checkpoint_seconds
    write_history(history_path, history)
    print(f"Checkpoint successfully saved. To serve it with the GPU service:")
    print(f'Set "model_path": "{args.output_dir}" in gpu/config.local.json and restart.')
    return 0


def main():
    args = parse_args()
    try:
        sys.exit(train(args))
    except Exception as err:
        print(f"\nTraining error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
