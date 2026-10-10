"""Offline export checks; use synthetic messages and no dashboard connection."""
import contextlib
from http.client import IncompleteRead
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
    def test_incomplete_download_never_saves_partial_or_reuses_old_data(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder) / "existing-run"
            existing.mkdir()
            old_data = existing / "laya-mail-train.jsonl"
            old_data.write_text("old synthetic export", encoding="utf-8")
            for output in (Path(folder) / "new-run", existing):
                errors = io.StringIO()
                args = ["export_training_data.py", "--url", "https://dashboard.example.invalid",
                        "--key-file", "unused", "--output-dir", str(output)]
                with self.subTest(output=output.name), patch("sys.argv", args), \
                        patch("export_training_data.fetch_export", side_effect=IncompleteRead(b"PRIVATE_EMAIL_BODY", 100)), \
                        patch("export_training_data.save_export") as save, contextlib.redirect_stderr(errors):
                    self.assertEqual(main(), 1)
                save.assert_not_called()
                self.assertEqual(errors.getvalue(),
                                 "Export failed: download incomplete. No partial or old data will be used. Please retry.\n")
                self.assertEqual(old_data.read_text(), "old synthetic export")
                self.assertFalse((Path(folder) / "new-run").exists())

    def test_gmail_spam_provenance_is_preserved_in_training_and_never_validation(self):
        weak = {"state": "Synthetic Gmail spam",
                "questions": {"spamverdacht": {"type": "choice", "instructions": "Spam?",
                                                "criteria": {"ja": "Spam", "nein": "Kein Spam"}}},
                "gold": {"spamverdacht": {"type": "choice", "probabilities": {"ja": 1.0, "nein": 0.0}}},
                "metadata": {"label_source": "gmail_spam", "split": "train"}}
        human_val = sample("Synthetic human-reviewed validation", "val")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            self.assertEqual(save_export(payload(weak, human_val), output), (1, 1))
            self.assertEqual(json.loads((output / "laya-mail-train.jsonl").read_text()), weak)
            self.assertEqual(json.loads((output / "laya-mail-val.jsonl").read_text()), human_val)
            weak["metadata"]["split"] = "val"
            invalid_output = Path(folder) / "invalid-run"
            with self.assertRaisesRegex(ValueError, "metadata.split=train"):
                save_export(payload(sample("Synthetic human training", "train"), weak), invalid_output)
            self.assertFalse(invalid_output.exists())

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
