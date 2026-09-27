"""Offline export checks; use synthetic messages and no dashboard connection."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from export_training_data import main, save_export


def sample(state, split):
    return {"state": state,
            "questions": {"inhalt": {"type": "choice", "instructions": "Kategorie", "criteria": {"info": "Information", "werbung": "Werbung"}}},
            "gold": {"inhalt": {"probabilities": {"info": 1.0, "werbung": 0.0}}},
            "metadata": {"split": split}}


def payload(*samples):
    return ("\n".join(json.dumps(item) for item in samples) + "\n").encode()


class ExportCheck(unittest.TestCase):
    def test_one_snapshot_is_split_without_changing_review_data(self):
        train = sample("Synthetic train message", "train")
        val = sample("Synthetic validation message", "val")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            self.assertEqual(save_export(payload(train, val), output), (1, 1))
            self.assertEqual(json.loads((output / "laya-mail-train.jsonl").read_text()), train)
            self.assertEqual(json.loads((output / "laya-mail-val.jsonl").read_text()), val)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["new-run"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_export(payload(val), output)
            self.assertEqual(json.loads((output / "laya-mail-train.jsonl").read_text()), train)

    def test_bad_export_leaves_no_reusable_partial_files(self):
        train = sample("Synthetic train message", "train")
        invalid = (b"", b"<html>login</html>", payload(train),
                   payload(train, sample(train["state"], "val")))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            for content in invalid:
                with self.subTest(content=content[:10]), self.assertRaises(ValueError):
                    save_export(content, output)
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_network_failure_is_nonzero_and_does_not_reuse_local_files(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing-run"
            output.mkdir()
            old_data = output / "laya-mail-train.jsonl"
            old_data.write_text("old synthetic export", encoding="utf-8")
            args = ["export_training_data.py", "--url", "http://127.0.0.1:8001",
                    "--key-file", "unused", "--output-dir", str(output)]
            with patch("sys.argv", args), patch("export_training_data.fetch_export", side_effect=URLError("offline")), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual(old_data.read_text(), "old synthetic export")


if __name__ == "__main__":
    unittest.main()
