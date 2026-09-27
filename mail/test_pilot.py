"""Offline regression checks for optional Gmail label application."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import labeler
import pilot_gmail as pilot
from test_pilot_gmail import Gmail, Laya


URL = "http://127.0.0.1:8000/v1/systemone"
CACHE = {key: f"Label_{index}" for index, key in enumerate(labeler.LABEL_MAP)}
PRIVATE_ERROR = "private-response-body-and-access-token"


class PilotLabelTests(unittest.TestCase):
    def test_label_initialization_failure_aborts_before_listing_and_is_redacted(self):
        gmail, laya = Gmail(count=1), Laya()
        output = io.StringIO()
        with patch.object(labeler, "get_or_create_labels", side_effect=RuntimeError(PRIVATE_ERROR)), \
                patch.object(labeler, "apply_labels") as apply, redirect_stdout(output), redirect_stderr(output):
            with self.assertRaises(pilot.PilotError) as caught:
                pilot.run_pilot(gmail, laya, "test@gmail.com", URL, "FAKE_KEY", apply_labels_mode=True)
        self.assertEqual(str(caught.exception), "label_initialization_failed")
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception) + output.getvalue())
        self.assertEqual([url for url, _ in gmail.calls], [pilot.GMAIL + "/profile"])
        self.assertEqual(laya.calls, [])
        apply.assert_not_called()

    def test_label_write_error_marks_report_failed_and_keeps_classification(self):
        gmail, laya = Gmail(count=1), Laya()
        output = io.StringIO()
        with patch.object(labeler, "get_or_create_labels", return_value=CACHE), \
                patch.object(labeler, "apply_labels", side_effect=RuntimeError(PRIVATE_ERROR)) as apply, \
                redirect_stdout(output), redirect_stderr(output):
            report = pilot.run_pilot(gmail, laya, "test@gmail.com", URL, "FAKE_KEY", apply_labels_mode=True)
        row = report["messages"][0]
        self.assertEqual(report["status"], "error")
        self.assertTrue(row["label_error"])
        self.assertEqual(row["status"], "suggestion")
        self.assertEqual(row["model_input"], laya.calls[0]["json"])
        self.assertNotIn("applied_labels", row)
        self.assertNotIn(PRIVATE_ERROR, json.dumps(report) + output.getvalue())
        apply.assert_called_once_with(gmail, "0", row["suggestions"], CACHE, archive=False)

    def test_cli_label_error_exits_unsuccessfully_and_saves_redacted_report(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            private.chmod(0o700)
            key_path = private / "laya-key"
            key_path.write_text("FAKE_KEY", encoding="utf-8")
            report_path = private / "report.json"
            credentials = Mock(scopes=[pilot.MODIFY_SCOPE])
            output = io.StringIO()
            with patch.object(pilot.Credentials, "from_authorized_user_file", return_value=credentials), \
                    patch.object(pilot, "AuthorizedSession") as gmail_context, \
                    patch.object(pilot.requests, "Session") as laya_context, \
                    patch.object(labeler, "get_or_create_labels", return_value=CACHE), \
                    patch.object(labeler, "apply_labels", side_effect=RuntimeError(PRIVATE_ERROR)), \
                    redirect_stdout(output), redirect_stderr(output):
                gmail_context.return_value.__enter__.return_value = Gmail(count=1)
                laya_context.return_value.__enter__.return_value = Laya()
                result = pilot.main(["--token", str(private / "unused-token.json"),
                                     "--laya-key-file", str(key_path), "--output", str(report_path),
                                     "--expected-account", "test@gmail.com", "--apply-labels"])
            self.assertEqual(result, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertIn("1 Fehler", output.getvalue())
            self.assertNotIn(PRIVATE_ERROR, json.dumps(report) + output.getvalue())
            self.assertNotIn("FAKE_KEY", output.getvalue())


if __name__ == "__main__":
    unittest.main()
