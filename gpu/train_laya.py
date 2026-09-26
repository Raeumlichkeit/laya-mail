"""Train and fine-tune Laya decision heads on exported email review data with PyTorch."""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
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
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze the transformer encoder and only train decision heads (default: True)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Unfreeze encoder backbone for full fine-tuning")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda or cpu; default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify dataset formatting and pipeline without running full training")
    return parser.parse_args()


def load_dataset(file_path, val_ratio=0.20, val_file_path=None):
    if not file_path.is_file():
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    raw_lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError("Dataset is empty. Collect and export reviews from the dashboard first.")

    train_samples = []
    val_samples = []

    if val_file_path:
        if not val_file_path.is_file():
            raise FileNotFoundError(f"Validation data file not found: {val_file_path}")
        val_lines = [line.strip() for line in val_file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in raw_lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item.get("state"), str) or not isinstance(item.get("questions"), dict) or not isinstance(item.get("gold"), dict):
                continue
            train_samples.append(item)
        for line in val_lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item.get("state"), str) or not isinstance(item.get("questions"), dict) or not isinstance(item.get("gold"), dict):
                continue
            val_samples.append(item)
        if not val_samples:
            raise ValueError(f"Validation dataset is empty or contains no valid samples: {val_file_path}")
        return train_samples, val_samples

    for i, line in enumerate(raw_lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item.get("state"), str) or not isinstance(item.get("questions"), dict) or not isinstance(item.get("gold"), dict):
            continue
        meta = item.get("metadata", {})
        split = meta.get("split")
        if split == "val":
            val_samples.append(item)
        elif split == "train":
            train_samples.append(item)
        else:
            # Deterministic split based on state content hash
            h = int(hashlib.sha256(item["state"].encode("utf-8")).hexdigest()[:8], 16)
            if (h % 100) < int(val_ratio * 100):
                val_samples.append(item)
            else:
                train_samples.append(item)

    return train_samples, val_samples


def build_sample_items(tok, sample, max_len=1024, head_max_len=256):
    from laya.common import QTYPES, build_sequence, render_options

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
        if qid not in gold:
            continue
        t = qdef.get("type", "choice")
        q_internal = {"t": t, "ins": qdef.get("instructions", qid), "crit": qdef.get("criteria", {})}
        seq, markers = build_sequence(tok, state, q_internal, max_len, head_max_len,
                                      truncate_left=truncate_left, state_ids=state_ids)
        gold_q = gold[qid]
        target_probs = gold_q.get("probabilities", {})

        # Align target array with criteria order
        if t == "choice":
            criteria_keys = list(q_internal["crit"].keys())
            target = [float(target_probs.get(k, 0.0)) for k in criteria_keys]
        elif t == "score":
            target = [float(target_probs.get(str(idx), target_probs.get(idx, 0.0))) for idx in range(len(markers))]
        else:
            target = [float(target_probs.get("false", 0.0)), float(target_probs.get("true", 0.0))]

        if len(target) != len(markers):
            continue

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

    model.eval()
    total_loss = 0.0
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
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            gold_preds = target.argmax(dim=-1)

            for pred, true, qid in zip(preds.cpu().tolist(), gold_preds.cpu().tolist(), b["qids"]):
                if pred == true:
                    correct[qid] += 1
                totals[qid] += 1

    avg_loss = total_loss / max(1, len(batches))
    accuracies = {qid: correct[qid] / max(1, totals[qid]) for qid in totals}
    overall_acc = sum(correct.values()) / max(1, sum(totals.values()))
    return avg_loss, accuracies, overall_acc


def train(args):
    print("=" * 60)
    print(" Laya Mail · Fine-Tuning Pipeline (RTX 4070 Ti / CUDA)")
    print("=" * 60)

    val_data = getattr(args, "val_data", None)
    train_samples, val_samples = load_dataset(args.data, val_ratio=getattr(args, "val_ratio", 0.20), val_file_path=val_data)
    print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    if args.dry_run:
        print("[Dry-Run] Dataset validation successful. Exiting without training.")
        return 0

    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {args.base_model}")

    import torch
    from safetensors.torch import save_file
    import laya

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

    best_val_loss = float("inf")
    best_weights = None

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_batches = batch_samples(tok, train_samples, batch_size=args.batch_size,
                                      max_len=max_len, head_max_len=head_max_len, shuffle=True)
        model.train()
        total_train_loss = 0.0

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
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / max(1, len(train_batches))
        elapsed = time.perf_counter() - t0

        if val_batches:
            val_loss, val_accs, val_overall = evaluate(model, val_batches, device)
            acc_str = " · ".join(f"{k}: {v * 100:.1f}%" for k, v in val_accs.items())
            print(f"Epoch {epoch:2d}/{args.epochs:2d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Overall: {val_overall * 100:.1f}% [{acc_str}]")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | (No validation set)")

    print("\nTraining completed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights and configuration
    print(f"Saving checkpoint to: {args.output_dir}")
    weights_to_save = best_weights if best_weights is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(weights_to_save, str(args.output_dir / "model.safetensors"))

    # Copy encoder and tokenizer directories and base config
    for folder in ("encoder", "tokenizer"):
        src = args.base_model / folder
        dst = args.output_dir / folder
        if src.is_dir() and not dst.is_dir():
            shutil.copytree(src, dst)

    cfg = copy.deepcopy(agent.cfg)
    cfg["training"] = {
        "fine_tuned_from_checkpoint": True,
        "epochs_completed": args.epochs,
        "base_model": str(args.base_model.name),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
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
