"""Offline tests for training and dataset processing helpers."""

import importlib.util
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
from train_laya import (load_dataset, load_samples, validate_sample, build_sample_items, evaluate,
                       parse_args, record_epoch, save_checkpoint, set_seed, write_history)


HAS_LAYA_COMMON = importlib.util.find_spec("laya") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def sample(state="Synthetic training example", split="train"):
    return {
        "state": state,
        "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic choice",
                                  "criteria": {"a": "First", "b": "Second"}}},
        "gold": {"inhalt": {"type": "choice", "probabilities": {"a": 1.0, "b": 0.0}}},
        "metadata": {"split": split},
    }


def gmail_spam_sample():
    return {
        "state": "Synthetic Gmail spam example",
        "questions": {"spamverdacht": {"type": "choice", "instructions": "Spam?",
                                        "criteria": {"ja": "Spam", "nein": "Kein Spam"}}},
        "gold": {"spamverdacht": {"type": "choice", "probabilities": {"ja": 1.0, "nein": 0.0}}},
        "metadata": {"split": "train", "label_source": "gmail_spam"},
    }


class TrainingPipelineCheck(unittest.TestCase):
    def test_gmail_spam_labels_are_partial_positive_training_targets_only(self):
        weak = gmail_spam_sample()
        validate_sample(weak, allow_gmail_spam=True)
        with self.assertRaisesRegex(ValueError, "training-only"):
            validate_sample(weak)
        bad_samples = []
        for split in (None, "val"):
            row = copy.deepcopy(weak)
            row["metadata"]["split"] = split
            bad_samples.append(row)
        for probabilities in ({"ja": 0.0, "nein": 1.0}, {"ja": .9, "nein": .1},
                              {"false": 0.0, "true": 1.0}):
            row = copy.deepcopy(weak)
            row["gold"]["spamverdacht"]["probabilities"] = probabilities
            bad_samples.append(row)
        row = copy.deepcopy(weak)
        row["questions"].update(sample()["questions"])
        row["gold"].update(sample()["gold"])
        bad_samples.append(row)
        for row in bad_samples:
            with self.subTest(metadata=row["metadata"]), self.assertRaises(ValueError):
                validate_sample(row, allow_gmail_spam=True)

    def test_gmail_spam_stays_in_training_and_is_rejected_by_both_evaluation_paths(self):
        from evaluate_model import load_data, run_eval

        weak = gmail_spam_sample()
        with tempfile.TemporaryDirectory() as folder:
            path, val_path = Path(folder) / "train.jsonl", Path(folder) / "val.jsonl"
            human_train = sample()
            human_val = sample("Synthetic validation example", "val")
            path.write_text("\n".join(json.dumps(row) for row in (human_train, human_val, weak)), encoding="utf-8")
            train, val = load_dataset(path, val_ratio=1.0)
            self.assertEqual(train, [human_train, weak])
            self.assertEqual(val, [human_val])
            val_path.write_text(json.dumps(weak), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-only"):
                load_data(val_path)
            with self.assertRaisesRegex(ValueError, "training-only"):
                run_eval(Mock(), [weak])
            path.write_text(json.dumps(human_train), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-only"):
                load_dataset(path, val_file_path=val_path)

    def test_load_dataset_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            data_file = Path(folder) / "test_data.jsonl"
            samples = [
                {
                    "state": f"Sample message {i}",
                    "questions": {
                        "inhalt": {"type": "choice", "instructions": "Synthetic content question", "criteria": {"rechnung": "", "werbung": "", "info": "", "sonstiges": ""}},
                        "wichtig": {"type": "choice", "instructions": "Synthetic priority question", "criteria": {"ja": "", "nein": ""}},
                        "spamverdacht": {"type": "choice", "instructions": "Synthetic spam question", "criteria": {"ja": "", "nein": ""}},
                    },
                    "gold": {
                        "inhalt": {"probabilities": {"rechnung": 1.0, "werbung": 0.0, "info": 0.0, "sonstiges": 0.0}},
                        "wichtig": {"probabilities": {"ja": 1.0, "nein": 0.0}},
                        "spamverdacht": {"probabilities": {"ja": 0.0, "nein": 1.0}},
                    },
                    "metadata": {"split": "val" if i % 4 == 0 else "train"}
                }
                for i in range(12)
            ]
            data_file.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")

            train, val = load_dataset(data_file, val_ratio=0.25)
            self.assertEqual(len(train), 9)
            self.assertEqual(len(val), 3)

    def test_empty_dataset_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            empty_file = Path(folder) / "empty.jsonl"
            empty_file.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(empty_file)

    @unittest.skipUnless(HAS_LAYA_COMMON, "laya.common is required for build_sample_items test")
    def test_build_sample_items_matching(self):
        class DummyTok:
            mask_token = "[MASK]"
            cls_token_id = 101
            sep_token_id = 102
            mask_token_id = 103
            pad_token_id = 0

            def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
                return {"input_ids": [1] * min(len(text), max_length or 50)}

        sample = {
            "state": "Rechnung über 50 EUR",
            "questions": {
                "inhalt": {
                    "type": "choice",
                    "instructions": "Inhaltskategorie",
                    "criteria": {"rechnung": "Zahlung", "werbung": "Marketing", "info": "Info", "sonstiges": "Rest"}
                }
            },
            "gold": {
                "inhalt": {
                    "type": "choice",
                    "probabilities": {"rechnung": 1.0, "werbung": 0.0, "info": 0.0, "sonstiges": 0.0}
                }
            }
        }
        items = build_sample_items(DummyTok(), sample, max_len=128, head_max_len=64)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["qid"], "inhalt")
        self.assertEqual(items[0]["target"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(items[0]["markers"]), 4)
        spam_items = build_sample_items(DummyTok(), gmail_spam_sample(), max_len=128, head_max_len=64)
        self.assertEqual(len(spam_items), 1)
        self.assertEqual(spam_items[0]["qid"], "spamverdacht")
        self.assertEqual(spam_items[0]["target"], [1.0, 0.0])
        self.assertEqual(len(spam_items[0]["markers"]), 2)
        with self.assertRaisesRegex(ValueError, "lost option markers"):
            build_sample_items(DummyTok(), sample, max_len=8, head_max_len=64)

    def test_train_dry_run(self):
        with tempfile.TemporaryDirectory() as folder:
            data_file = Path(folder) / "test.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
                "metadata": {"split": "train"}
            }
            data_file.write_text(json.dumps(sample) + "\n" + json.dumps(sample), encoding="utf-8")
            from train_laya import train
            import argparse
            args = argparse.Namespace(data=data_file, val_data=None, dry_run=True, val_ratio=0.5, base_model=Path(folder))
            self.assertEqual(train(args), 0)

    def test_separate_val_data(self):
        with tempfile.TemporaryDirectory() as folder:
            train_file = Path(folder) / "train.jsonl"
            val_file = Path(folder) / "val.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
            }
            train_file.write_text("\n".join(json.dumps(sample) for _ in range(5)), encoding="utf-8")
            val_file.write_text("\n".join(json.dumps({**sample, "state": "Separate validation message"})
                                          for _ in range(2)), encoding="utf-8")

            train_samples, val_samples = load_dataset(train_file, val_file_path=val_file)
            self.assertEqual(len(train_samples), 5)
            self.assertEqual(len(val_samples), 2)

    def test_no_val_fallback_does_not_pop(self):
        with tempfile.TemporaryDirectory() as folder:
            train_file = Path(folder) / "train_only.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
                "metadata": {"split": "train"}
            }
            train_file.write_text("\n".join(json.dumps(sample) for _ in range(6)), encoding="utf-8")

            train_samples, val_samples = load_dataset(train_file)
            # All 6 should remain in train_samples, nothing popped into val_samples
            self.assertEqual(len(train_samples), 6)
            self.assertEqual(len(val_samples), 0)

    def test_empty_separate_val_data_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            train_file = Path(folder) / "train.jsonl"
            empty_val_file = Path(folder) / "empty_val.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
            }
            train_file.write_text(json.dumps(sample), encoding="utf-8")
            empty_val_file.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_dataset(train_file, val_file_path=empty_val_file)

    def test_invalid_rows_fail_with_line_number_without_contents(self):
        bad_samples = [None, [], {**sample(), "state": " "},
                       {**sample(), "questions": {}, "gold": {}},
                       {**sample(), "gold": {}}, {**sample(), "metadata": []}]
        for probabilities in ({"a": 1.0}, {"a": 1.0, "b": 0.0, "c": 0.0},
                              {"a": 0.0, "b": 0.0}, {"a": -0.1, "b": 1.1},
                              {"a": float("nan"), "b": 0.0}, {"a": float("inf"), "b": 0.0},
                              {"a": "1", "b": 0}, {"a": True, "b": False}):
            bad = sample()
            bad["gold"]["inhalt"]["probabilities"] = probabilities
            bad_samples.append(bad)
        for change in ({"type": "unknown"}, {"type": None}, {"criteria": {}}, {"criteria": []},
                       {"instructions": None}, {"instructions": " "}):
            bad = sample()
            bad["questions"]["inhalt"].update(change)
            bad_samples.append(bad)
        bad = sample()
        bad["gold"]["inhalt"]["type"] = "score"
        bad_samples.append(bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            for row in [json.dumps(item) for item in bad_samples] + ['PRIVATE-CONTENT invalid JSON']:
                with self.subTest(row_type=row[:1]):
                    path.write_text(json.dumps(sample()) + "\n\n" + row, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "line 3") as error:
                        load_samples(path)
                    self.assertNotIn("PRIVATE-CONTENT", str(error.exception))
                    self.assertNotIn("Synthetic training example", str(error.exception))
            path.write_bytes(b"\n\xffPRIVATE-CONTENT")
            with self.assertRaisesRegex(ValueError, "line 2: invalid UTF-8"):
                load_samples(path)

    def test_choice_score_and_noul_targets(self):
        for kind, criteria, probabilities in (
                ("choice", {"a": "First", "b": "Second"}, {"a": 0.25, "b": 0.75}),
                ("score", ["Low", "High"], {"0": 0.25, "1": 0.75}),
                ("noul", {"true": "Present"}, {"false": 0.25, "true": 0.75})):
            item = {"state": "Synthetic", "questions": {"q": {"type": kind, "instructions": "Synthetic", "criteria": criteria}},
                    "gold": {"q": {"type": kind, "probabilities": probabilities}}}
            validate_sample(item)
            item["gold"]["q"]["probabilities"] = {"wrong": 1.0}
            with self.assertRaisesRegex(ValueError, "match all question options"):
                validate_sample(item)

    def test_validation_only_and_overlaps_fail_including_dry_run(self):
        from argparse import Namespace
        from train_laya import train

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            val_path = Path(folder) / "val.jsonl"
            path.write_text(json.dumps(sample(split="val")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Training split is empty"):
                train(Namespace(data=path, val_data=None, dry_run=True))
            path.write_text("\n".join(json.dumps(sample(split=split)) for split in ("train", "val")),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_dataset(path)
            path.write_text(json.dumps(sample()), encoding="utf-8")
            val_path.write_text(json.dumps(sample(split="val")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_dataset(path, val_file_path=val_path)
            path.write_text(json.dumps(sample("Separate state", split="val")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marked for validation"):
                load_dataset(path, val_file_path=val_path)

    def test_imports_do_not_load_torch_during_discovery(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import train_laya, test_train; import sys; "
             "assert 'torch' not in sys.modules"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(HAS_TORCH, "CPU tensor checks require torch")
    def test_validation_loss_weights_question_rows_and_rejects_nonfinite(self):
        import torch

        class FixedLogits:
            def eval(self):
                pass

            def __call__(self, input_ids, *args):
                return input_ids, None

        def batch(probability, count):
            logits = torch.tensor([[math.log(probability), math.log(1 - probability)]] * count)
            return {"input_ids": logits, "attention_mask": torch.ones(count, 2),
                    "marker_pos": torch.zeros(count, 2), "marker_mask": torch.ones(count, 2, dtype=torch.bool),
                    "qtype": torch.zeros(count), "target": torch.tensor([[1., 0.]] * count),
                    "qids": ["inhalt"] * count}

        batches = [batch(0.9, 4), batch(0.1, 1)]
        loss, accuracies, overall = evaluate(FixedLogits(), batches, "cpu")
        self.assertAlmostEqual(loss, -(4 * math.log(.9) + math.log(.1)) / 5, places=6)
        self.assertEqual(accuracies, {"inhalt": .8})
        self.assertEqual(overall, .8)
        batches[0]["input_ids"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "loss is not finite"):
            evaluate(FixedLogits(), batches, "cpu")
        with self.assertRaisesRegex(ValueError, "no usable batches"):
            evaluate(FixedLogits(), [], "cpu")

    def test_epoch_comparison_keeps_later_best_even_when_earlier_is_better(self):
        history = {"epochs": [], "compare_epoch": 8, "patience": 3,
                   "min_epochs": 9, "requested_epochs": 20}
        decisions = []
        for epoch, loss in enumerate([1., .8, .7, .6, .5, .4, .3, .2, .4, .35, .36], 1):
            decisions.append(record_epoch(history, {"epoch": epoch, "val_loss": loss}))
            if epoch < 11:
                self.assertFalse(history["stopped_early"])
        self.assertEqual(history["completed_epochs"], 11)
        self.assertTrue(history["stopped_early"])
        self.assertEqual(history["best_epoch"], 8)
        self.assertEqual(history["best_through_epoch"], 8)
        self.assertEqual(history["best_after_epoch"], 10)
        self.assertEqual([i + 1 for i, decision in enumerate(decisions) if decision[1]], [8])
        self.assertEqual([i + 1 for i, decision in enumerate(decisions) if decision[2]], [9, 10])

    def test_early_stop_respects_minimum_ties_and_disabled_default(self):
        history = {"epochs": [], "compare_epoch": 8, "patience": 3,
                   "min_epochs": 9, "requested_epochs": 20}
        for epoch in range(1, 10):
            decision = record_epoch(history, {"epoch": epoch, "val_loss": 1.0})
            self.assertEqual(history["stopped_early"], epoch == 9)
        self.assertEqual(history["best_epoch"], 1)
        self.assertEqual(history["best_through_epoch"], 1)
        self.assertEqual(history["best_after_epoch"], 9)
        self.assertEqual(decision, (False, False, True))

        for validation_loss in (None, 1.0):
            history = {"epochs": [], "compare_epoch": 0, "patience": 0,
                       "min_epochs": 1, "requested_epochs": 5}
            for epoch in range(1, 6):
                record_epoch(history, {"epoch": epoch, "val_loss": validation_loss})
                self.assertFalse(history["stopped_early"])
            self.assertEqual(history["best_epoch"], 5 if validation_loss is None else 1)
            self.assertIsNone(history["best_through_epoch"])
            self.assertIsNone(history["best_after_epoch"])

    def test_history_replacement_is_atomic_and_keeps_previous_file_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.json"
            write_history(path, {"completed_epochs": 1, "epochs": [{"epoch": 1, "val_loss": .5}]})
            previous = path.read_bytes()
            with patch("train_laya.os.replace", side_effect=OSError("synthetic replacement failure")):
                with self.assertRaises(OSError):
                    write_history(path, {"completed_epochs": 2, "epochs": []})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ["history.json"])
            with self.assertRaises(ValueError):
                write_history(path, {"val_loss": float("nan")})
            self.assertEqual(path.read_bytes(), previous)
            write_history(path, {"completed_epochs": 2, "epochs": [{"epoch": 2, "val_loss": .4}]})
            self.assertEqual(json.loads(path.read_text())["completed_epochs"], 2)

    def test_checkpoint_is_complete_and_records_selected_and_completed_epochs(self):
        fake_safetensors = ModuleType("safetensors")
        fake_torch = ModuleType("safetensors.torch")
        fake_torch.save_file = lambda weights, path: Path(path).write_text(str(weights["synthetic_epoch"]))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base"
            for name in ("encoder", "tokenizer"):
                (base / name).mkdir(parents=True)
                (base / name / "config.json").write_text('{"synthetic": true}')
            args = SimpleNamespace(base_model=base, epochs=20, seed=42)
            config = {"head_layers": 2, "training": {"old": True}}
            with patch.dict(sys.modules, {"safetensors": fake_safetensors, "safetensors.torch": fake_torch}):
                for name, selected, completed in (("best-through-8", 3, 8), ("best-after-8", 10, 10), ("overall", 3, 11)):
                    destination = Path(folder) / name
                    save_checkpoint(destination, {"synthetic_epoch": selected}, config, args, completed, selected)
                    saved = json.loads((destination / "rl_agent_config.json").read_text())
                    self.assertEqual(saved["training"]["best_epoch"], selected)
                    self.assertEqual(saved["training"]["epochs_completed"], completed)
                    self.assertEqual(saved["training"]["epochs_requested"], 20)
                    self.assertEqual(saved["training"]["seed"], 42)
                    self.assertEqual((destination / "model.safetensors").read_text(), str(selected))
                    for name in ("encoder", "tokenizer"):
                        self.assertEqual((destination / name / "config.json").read_bytes(), (base / name / "config.json").read_bytes())
            self.assertEqual(config["training"], {"old": True})

    def test_optional_arguments_preserve_defaults_and_seed_all_rngs(self):
        with patch.object(sys, "argv", ["train_laya.py", "--data", "synthetic.jsonl"]):
            args = parse_args()
        self.assertIsNone(args.seed)
        self.assertIsNone(args.history)
        self.assertEqual((args.patience, args.min_epochs, args.compare_epoch), (0, 1, 0))
        fake_numpy, fake_torch = ModuleType("numpy"), ModuleType("torch")
        fake_numpy.random = SimpleNamespace(seed=Mock())
        fake_torch.manual_seed = Mock()
        fake_torch.cuda = SimpleNamespace(manual_seed_all=Mock())
        with patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch}), patch("train_laya.random.seed") as python_seed:
            set_seed(42)
            python_seed.assert_called_once_with(42)
            fake_numpy.random.seed.assert_called_once_with(42)
            fake_torch.manual_seed.assert_called_once_with(42)
            fake_torch.cuda.manual_seed_all.assert_called_once_with(42)


if __name__ == "__main__":
    unittest.main()
