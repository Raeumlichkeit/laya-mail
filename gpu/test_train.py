"""Offline tests for training and dataset processing helpers."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from train_laya import load_dataset, build_sample_items


def _can_import_laya_common():
    try:
        import importlib.util
        return bool(importlib.util.find_spec("laya") and importlib.util.find_spec("laya.common"))
    except Exception:
        return False


HAS_LAYA_COMMON = _can_import_laya_common()


class TrainingPipelineCheck(unittest.TestCase):
    def test_load_dataset_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            data_file = Path(folder) / "test_data.jsonl"
            samples = [
                {
                    "state": f"Sample message {i}",
                    "questions": {
                        "inhalt": {"type": "choice", "criteria": {"rechnung": "", "werbung": "", "info": "", "sonstiges": ""}},
                        "wichtig": {"type": "choice", "criteria": {"ja": "", "nein": ""}},
                        "spamverdacht": {"type": "choice", "criteria": {"ja": "", "nein": ""}},
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

    def test_train_dry_run(self):
        with tempfile.TemporaryDirectory() as folder:
            data_file = Path(folder) / "test.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "criteria": {"rechnung": ""}}},
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
                "questions": {"inhalt": {"type": "choice", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
            }
            train_file.write_text("\n".join(json.dumps(sample) for _ in range(5)), encoding="utf-8")
            val_file.write_text("\n".join(json.dumps(sample) for _ in range(2)), encoding="utf-8")

            train_samples, val_samples = load_dataset(train_file, val_file_path=val_file)
            self.assertEqual(len(train_samples), 5)
            self.assertEqual(len(val_samples), 2)

    def test_no_val_fallback_does_not_pop(self):
        with tempfile.TemporaryDirectory() as folder:
            train_file = Path(folder) / "train_only.jsonl"
            sample = {
                "state": "Test message",
                "questions": {"inhalt": {"type": "choice", "criteria": {"rechnung": ""}}},
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
                "questions": {"inhalt": {"type": "choice", "criteria": {"rechnung": ""}}},
                "gold": {"inhalt": {"probabilities": {"rechnung": 1.0}}},
            }
            train_file.write_text(json.dumps(sample), encoding="utf-8")
            empty_val_file.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_dataset(train_file, val_file_path=empty_val_file)


if __name__ == "__main__":
    unittest.main()
