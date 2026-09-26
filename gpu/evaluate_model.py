"""Evaluate a Laya checkpoint on an email evaluation dataset, with optional baseline comparison."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset file")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional baseline model to compare against (e.g. base multilingual model)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run evaluation on (cuda or cpu; default: cuda if available)")
    return parser.parse_args()


def load_data(file_path):
    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for line in lines:
        try:
            item = json.loads(line)
            if "state" in item and "questions" in item and "gold" in item:
                samples.append(item)
        except json.JSONDecodeError:
            continue
    if not samples:
        raise ValueError(f"No valid JSONL samples found in {file_path}")
    return samples


def run_eval(agent, samples):
    correct = defaultdict(int)
    totals = defaultdict(int)
    confusion_inhalt = defaultdict(lambda: defaultdict(int))
    latencies = []

    for item in samples:
        state = item["state"]
        questions = item["questions"]
        gold = item["gold"]

        t0 = time.perf_counter()
        prediction = agent.predict(state, questions)
        latencies.append(time.perf_counter() - t0)

        for qid, qdef in questions.items():
            if qid not in gold or qid not in prediction.get("answers", {}):
                continue
            pred_choice = prediction["answers"][qid].get("choice")
            gold_probs = gold[qid].get("probabilities", {})
            gold_choice = max(gold_probs.items(), key=lambda kv: kv[1])[0] if gold_probs else None

            if pred_choice == gold_choice:
                correct[qid] += 1
            totals[qid] += 1

            if qid == "inhalt":
                confusion_inhalt[gold_choice][pred_choice] += 1

    accuracies = {qid: correct[qid] / max(1, totals[qid]) for qid in totals}
    overall_acc = sum(correct.values()) / max(1, sum(totals.values()))
    avg_latency = (sum(latencies) / len(latencies)) * 1000.0 if latencies else 0.0

    return {
        "accuracies": accuracies,
        "overall_accuracy": overall_acc,
        "confusion_inhalt": confusion_inhalt,
        "avg_latency_ms": avg_latency,
        "sample_count": len(samples),
    }


def print_report(name, results):
    print("\n" + "=" * 60)
    print(f" Evaluation: {name}")
    print("=" * 60)
    print(f"Samples evaluated: {results['sample_count']}")
    print(f"Average latency:   {results['avg_latency_ms']:.1f} ms per email")
    print(f"Overall Accuracy:  {results['overall_accuracy'] * 100:.1f}%\n")

    print("Accuracies by Question:")
    for qid, acc in sorted(results["accuracies"].items()):
        print(f"  - {qid:14s}: {acc * 100:.1f}%")

    if results["confusion_inhalt"]:
        print("\nConfusion Matrix (Inhalt: Actual rows -> Predicted columns):")
        categories = ["rechnung", "info", "werbung", "sonstiges"]
        header = f"{'Actual':12s} | " + " | ".join(f"{c:>9s}" for c in categories)
        print(header)
        print("-" * len(header))
        for actual in categories:
            row_vals = [results["confusion_inhalt"][actual].get(pred, 0) for pred in categories]
            print(f"{actual:12s} | " + " | ".join(f"{v:9d}" for v in row_vals))


def main():
    args = parse_args()
    samples = load_data(args.data)
    print(f"Loaded {len(samples)} test samples from {args.data.name}")

    import torch
    import laya

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading candidate model from {args.model} on {device}...")
    agent = laya.load(str(args.model), device=device)
    results = run_eval(agent, samples)
    print_report(args.model.name, results)

    if args.baseline:
        print(f"\nLoading baseline model from {args.baseline} on {device}...")
        base_agent = laya.load(str(args.baseline), device=device)
        base_results = run_eval(base_agent, samples)
        print_report(f"Baseline ({args.baseline.name})", base_results)

        print("\n" + "=" * 60)
        print(" Comparison: Baseline vs Fine-Tuned")
        print("=" * 60)
        diff_overall = (results["overall_accuracy"] - base_results["overall_accuracy"]) * 100
        sign = "+" if diff_overall >= 0 else ""
        print(f"Overall Accuracy:  {base_results['overall_accuracy'] * 100:.1f}% -> {results['overall_accuracy'] * 100:.1f}% ({sign}{diff_overall:.1f}%)")
        for qid in sorted(results["accuracies"].keys()):
            b_acc = base_results["accuracies"].get(qid, 0.0) * 100
            c_acc = results["accuracies"].get(qid, 0.0) * 100
            d = c_acc - b_acc
            s = "+" if d >= 0 else ""
            print(f"  - {qid:14s}: {b_acc:.1f}% -> {c_acc:.1f}% ({s}{d:.1f}%)")


if __name__ == "__main__":
    main()
