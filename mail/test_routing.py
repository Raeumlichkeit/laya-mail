"""Regression checks for mailbox isolation and the shared manual/live worker."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dashboard import Dashboard, FeedbackError
from pilot_gmail import INPUT_VERSION, QUESTIONS, save_report


PRIMARY = "primary@example.invalid"
SECONDARY = "second@example.invalid"
LABELS = {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"}


def message(account, message_id="abc123"):
    return {"message_id": message_id, "account_email": account, "status": "suggestion",
            "suggestions": dict(LABELS), "model_input": {
                "model": "multilingual", "state": "Identischer synthetischer Mailtext", "questions": QUESTIONS}}


class RoutingCheck(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        (self.directory / "dashboard-key").write_text("synthetic-dashboard-key-for-test")
        (self.directory / "token.json").write_text("{}")
        self.watcher = patch("dashboard.LiveWatcher.start")
        self.watcher.start()
        self.addCleanup(self.watcher.stop)
        self.app = Dashboard(self.directory, PRIMARY, "http://127.0.0.1:8000/v1/systemone")
        self.app.accounts_mgr.add_account(SECONDARY, "token_second.json")
        (self.directory / "token_second.json").write_text("{}")

    def set_report(self, rows):
        path = self.directory / "pilot-synthetic.json"
        save_report(path, {"input_version": INPUT_VERSION, "messages": rows, "status": "complete"})
        self.app.report_path = path

    def test_single_secondary_batch_keeps_owner_token_and_removal_success(self):
        self.app.accounts_mgr.select(SECONDARY)

        def pilot(command, **kwargs):
            self.assertEqual(command[command.index("--expected-account") + 1], SECONDARY)
            self.assertEqual(Path(command[command.index("--token") + 1]).name, "token_second.json")
            row = message(SECONDARY)
            del row["account_email"]  # Ownership must also survive older pilot output.
            save_report(Path(command[command.index("--output") + 1]), {"status": "complete", "messages": [row]})
            return SimpleNamespace(returncode=0)

        with patch("dashboard.subprocess.run", side_effect=pilot):
            self.app.run(limit=1)
        self.assertEqual(self.app.report()["messages"][0]["account_email"], SECONDARY)
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as credentials, \
                patch("google.auth.transport.requests.AuthorizedSession"), \
                patch("dashboard.gmail_get", return_value={"emailAddress": SECONDARY}), \
                patch("labeler.get_or_create_labels"), patch("labeler.apply_labels", return_value=[]) as apply:
            credentials.return_value.scopes = ["https://www.googleapis.com/auth/gmail.modify"]
            result = self.app.apply_current_labels()
            self.assertEqual(result["applied_count"], 1)
            credentials.assert_called_once_with(str(self.directory / "token_second.json"))
            apply.assert_called_once()
            self.assertEqual(self.app.report()["messages"][0]["applied_labels"], [])

    def test_missing_or_wrong_account_never_uses_primary_credentials(self):
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as credentials, \
                patch("google.auth.transport.requests.AuthorizedSession"), \
                patch("dashboard.gmail_get", return_value={"emailAddress": PRIMARY}), \
                patch("labeler.get_or_create_labels") as create, patch("labeler.apply_labels") as apply:
            credentials.return_value.scopes = ["https://www.googleapis.com/auth/gmail.modify"]
            self.set_report([message(SECONDARY)])
            self.assertTrue(self.app.apply_current_labels()["errors"])
            create.assert_not_called()
            apply.assert_not_called()
            credentials.reset_mock()
            (self.directory / "token_second.json").unlink()
            self.assertTrue(self.app.apply_current_labels()["errors"])
            legacy = message(PRIMARY)
            del legacy["account_email"]
            self.set_report([legacy])
            self.assertTrue(self.app.apply_current_labels()["errors"])
            credentials.assert_not_called()
            apply.assert_not_called()

    def test_same_id_reviews_and_seen_history_are_account_scoped(self):
        self.set_report([message(PRIMARY), message(SECONDARY), message(PRIMARY, "def456")])
        report = self.app.review_report()
        data = {"message_id": "abc123", "report_id": report["report_id"],
                "snapshot_id": report["messages"][0]["snapshot_id"], "action": "accepted"}
        with self.assertRaises(FeedbackError):
            self.app.save_feedback(data)
        result = self.app.save_feedback({**data, "account_email": SECONDARY})
        self.assertIsNone(result["messages"][0]["review"])
        self.assertIsNotNone(result["messages"][1]["review"])
        self.app.save_feedback({**data, "account_email": PRIMARY})
        self.assertEqual(len(self.app.feedback["items"]), 2)
        # Identical texts from distinct accounts cannot leak across train/validation.
        exported = [json.loads(line) for line in self.app.training_export().splitlines()]
        self.assertEqual(len({row["metadata"]["split"] for row in exported}), 1)
        self.assertEqual({row["metadata"]["account_email"] for row in exported}, {PRIMARY, SECONDARY})
        with patch("dashboard.threading.Thread"):
            self.assertTrue(self.app.start("next", 3))
        self.assertEqual(json.loads(self.app.seen_path(PRIMARY).read_text()), ["abc123", "def456"])
        self.assertEqual(json.loads(self.app.seen_path(SECONDARY).read_text()), ["abc123"])

    def test_batch_total_is_bounded_and_empty_live_run_preserves_report(self):
        limits = []

        def empty_pilot(command, **kwargs):
            limits.append(int(command[command.index("--limit") + 1]))
            save_report(Path(command[command.index("--output") + 1]), {"status": "complete", "messages": []})
            return SimpleNamespace(returncode=0)

        self.set_report([message(PRIMARY)])
        previous = self.app.report_path
        self.app._run_live = True
        with patch("dashboard.subprocess.run", side_effect=empty_pilot):
            self.app.run("next", 3)
        self.assertEqual(limits, [2, 3])  # Unused capacity passes to later mailboxes.
        self.assertEqual(self.app.report_path, previous)
        self.assertEqual(len(self.app.report()["messages"]), 1)
        limits.clear()
        with patch("dashboard.subprocess.run", side_effect=empty_pilot):
            self.app.run("next", 1)
        self.assertEqual(limits, [1, 1])  # A small batch still checks the second empty mailbox.

        def full_pilot(command, **kwargs):
            quota = int(command[command.index("--limit") + 1])
            save_report(Path(command[command.index("--output") + 1]), {
                "status": "complete", "messages": [message(PRIMARY, str(i)) for i in range(quota)]})
            return SimpleNamespace(returncode=0)

        with patch("dashboard.subprocess.run", side_effect=full_pilot):
            self.app.run("next", 3)
        self.assertEqual(len(self.app.report()["messages"]), 3)

    def test_live_uses_shared_worker_and_disabled_selection_does_not_fallback(self):
        with patch.object(self.app, "start", return_value=True) as start:
            self.app.watcher._check_cycle({"live_mode": True})
            start.assert_called_once_with("next", 10, live=True)
        self.app.accounts_mgr.select(SECONDARY)
        self.app.accounts_mgr.toggle_account(SECONDARY, False)
        with patch("dashboard.threading.Thread") as thread:
            with self.assertRaises(FeedbackError):
                self.app.start()
            thread.assert_not_called()

    def test_partial_label_failure_keeps_rows_and_marks_run_failed(self):
        self.app.accounts_mgr.select(SECONDARY)

        def failed_pilot(command, **kwargs):
            row = {**message(SECONDARY), "label_error": "Labels konnten nicht gesetzt werden."}
            save_report(Path(command[command.index("--output") + 1]), {"status": "error", "messages": [row]})
            return SimpleNamespace(returncode=1)

        with patch("dashboard.subprocess.run", side_effect=failed_pilot):
            self.app.run()
        self.assertEqual(self.app.state["phase"], "error")
        self.assertEqual(len(self.app.report()["messages"]), 1)
        self.assertTrue(self.app.report()["messages"][0]["label_error"])


if __name__ == "__main__":
    unittest.main()
