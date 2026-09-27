"""Evaluate a Laya checkpoint on an email evaluation dataset, with optional baseline comparison."""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from train_laya import load_samples, validate_sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset file")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional baseline model to compare against (e.g. base multilingual model)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run evaluation on (cuda or cpu; default: cuda if available)")
    parser.add_argument("--report", type=Path, help="Optional JSON report containing aggregate metrics only")
    return parser.parse_args(argv)


def load_data(file_path):
    samples = load_samples(file_path)
    validate_eval_samples(samples)
    return samples


def validate_eval_samples(samples):
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("Evaluation requires a nonempty dataset.")
    for index, item in enumerate(samples, 1):
        validate_sample(item, location=f"Evaluation sample {index}")
        if any(question.get("type", "choice") != "choice" for question in item["questions"].values()):
            raise ValueError(f"Evaluation sample {index}: accuracy evaluation supports choice questions only.")


def run_eval(agent, samples):
    validate_eval_samples(samples)
    correct = defaultdict(int)
    totals = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    latencies = []

    for index, item in enumerate(samples, 1):
        state = item["state"]
        questions = item["questions"]
        gold = item["gold"]

        t0 = time.perf_counter()
        prediction = agent.predict(state, questions)
        latencies.append(time.perf_counter() - t0)
        answers = prediction.get("answers") if isinstance(prediction, dict) else None
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError(f"Evaluation sample {index}: model returned incomplete or invalid answers.")

        for qid, qdef in questions.items():
            answer = answers[qid]
            pred_choice = answer.get("choice") if isinstance(answer, dict) else None
            if (not isinstance(pred_choice, str) or pred_choice not in qdef["criteria"]
                    or answer.get("type", "choice") != "choice"):
                raise ValueError(f"Evaluation sample {index}: model returned an invalid choice.")
            gold_probs = gold[qid]["probabilities"]
            gold_choice = max(gold_probs, key=gold_probs.get)

            if pred_choice == gold_choice:
                correct[qid] += 1
            totals[qid] += 1

            confusion[qid][gold_choice][pred_choice] += 1

    accuracies = {qid: correct[qid] / totals[qid] for qid in totals}
    overall_acc = sum(correct.values()) / sum(totals.values())
    avg_latency = (sum(latencies) / len(latencies)) * 1000.0
    confusion_by_question = {qid: {actual: dict(predicted) for actual, predicted in rows.items()}
                             for qid, rows in confusion.items()}

    return {
        "accuracies": accuracies,
        "overall_accuracy": overall_acc,
        "confusion_inhalt": confusion_by_question.get("inhalt", {}),
        "confusion_by_question": confusion_by_question,
        "avg_latency_ms": avg_latency,
        "sample_count": len(samples),
        "question_totals": dict(totals),
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
            row_vals = [results["confusion_inhalt"].get(actual, {}).get(pred, 0) for pred in categories]
            print(f"{actual:12s} | " + " | ".join(f"{v:9d}" for v in row_vals))


def evaluate_checkpoint(path, samples, device):
    import torch
    import laya

    agent = None
    try:
        agent = laya.load(str(path), device=device)
        return run_eval(agent, samples)
    finally:
        del agent
        gc.collect()
        if str(device).split(":", 1)[0] == "cuda":
            torch.cuda.empty_cache()


def evaluate(args):
    samples = load_data(args.data)
    print(f"Loaded {len(samples)} test samples.")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating candidate model on {device}...")
    results = evaluate_checkpoint(args.model, samples, device)
    print_report("Candidate", results)
    report = {"candidate": results, "baseline": None, "improved": None, "nonregressed": None}

    if args.baseline:
        print(f"\nEvaluating baseline model on {device}...")
        base_results = evaluate_checkpoint(args.baseline, samples, device)
        print_report("Baseline", base_results)
        report.update(baseline=base_results,
                      improved=results["overall_accuracy"] > base_results["overall_accuracy"],
                      nonregressed=results["overall_accuracy"] >= base_results["overall_accuracy"]
                      and all(results["accuracies"][qid] >= value for qid, value in base_results["accuracies"].items()))

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

    if args.report:
        descriptor, temporary = tempfile.mkstemp(dir=args.report.parent, prefix=".evaluation-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, args.report)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        evaluate(args)
        return 0
    except Exception:
        # Dataset text and model exceptions can contain private email content.
        print("Evaluation failed; check the dataset, checkpoints, device and report destination.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
