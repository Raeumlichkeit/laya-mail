"""Offline regression tests for the separate, automatically labelled training source."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests

from accounts import AccountError
from dashboard import Dashboard, FeedbackError, Handler
from pilot_gmail import INPUT_VERSION, QUESTIONS, save_report


ACCOUNT = "primary@example.invalid"
SECOND = "second@example.invalid"


def example(message_id="abc", state="Synthetic training mail", account=ACCOUNT):
    return {"message_id": message_id, "account_email": account, "input_version": INPUT_VERSION,
            "model_input": {"model": "multilingual", "state": state, "questions": QUESTIONS},
            "label_source": "gmail_spam", "imported_at": "2026-01-01T00:00:00Z"}


class SpamDashboardCheck(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        (self.directory / "dashboard-key").write_text("synthetic-dashboard-key-for-test")
        (self.directory / "token.json").write_text("{}")
        for patcher in (patch("dashboard.LiveWatcher.start"),
                        patch.object(Dashboard, "gpu_status", return_value={"online": False})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = Dashboard(self.directory, ACCOUNT, "http://127.0.0.1:8000/v1/systemone")

    def start_without_thread(self, limit=20):
        with patch("dashboard.threading.Thread") as thread:
            self.assertEqual(self.app.start_spam_training(limit), {"status": "started"})
            return thread.call_args.kwargs["args"]

    def test_export_is_train_only_deduplicated_and_human_decisions_win(self):
        self.app.spam_training["items"] = {"abc": example(), "def": example("def"),
                                             "fed": example("fed", "Human-reviewed text")}
        first = [json.loads(line) for line in self.app.training_export().splitlines()]
        self.assertEqual(len(first), 2)
        self.assertEqual(self.app.training_export("val"), b"")
        self.assertEqual(self.app.training_export("train"), self.app.training_export())
        self.assertEqual(first[0]["metadata"]["split"], "train")
        self.assertEqual(first[0]["metadata"]["label_source"], "gmail_spam")
        self.assertEqual(set(first[0]["gold"]), {"spamverdacht"})
        self.assertEqual(set(first[0]["questions"]), {"spamverdacht"})
        self.app.feedback["items"]["a12"] = {**example("a12", "Human-reviewed text"), "action": "corrected",
                                             "labels": {"inhalt": "info", "wichtig": "ja", "spamverdacht": "nein"}}
        # Same-state automatic labels cannot contradict a human label or leak into validation.
        exported = [json.loads(line) for line in self.app.training_export().splitlines()]
        self.assertEqual(len(exported), 2)
        self.assertEqual(sum(row["metadata"].get("label_source") == "gmail_spam" for row in exported), 1)
        # A skipped human review also excludes the automatic source by identity, even after text changes.
        self.app.feedback["items"]["abc"] = {**example(state="Changed text"), "action": "skipped", "labels": None}
        self.app.feedback["items"]["def"] = {**example("def"), "action": "skipped", "labels": None}
        self.assertEqual(self.app.spam_training_examples(), [])
        self.assertEqual(self.app.review_report()["summary"]["training_examples"], 1)
        self.assertEqual(self.app.review_report()["messages"], [])

    def test_worker_routes_selected_accounts_preserves_report_and_persists(self):
        self.app.accounts_mgr.add_account(SECOND, "token_second.json")
        (self.directory / "token_second.json").write_text("{}")
        self.app.accounts_mgr.select(SECOND)
        args = self.start_without_thread(4)
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as credentials, \
                patch("google.auth.transport.requests.AuthorizedSession") as session, \
                patch("dashboard.collect_spam", return_value={"items": [example(account=SECOND)],
                                                             "processed": 1, "total": 1, "errors": []}) as collect:
            self.app.run_spam_training(*args)
        credentials.assert_called_once_with(str(self.directory / "token_second.json"))
        collect.assert_called_once_with(session.return_value.__enter__.return_value, SECOND, set(), 4)
        self.assertIsNone(self.app.report_path)
        self.assertEqual(self.app.feedback["items"], {})
        self.assertEqual(self.app.spam_training_status()["available"], 1)
        self.assertEqual(self.app.spam_training_status()["phase"], "complete")
        status = self.app.status()
        self.assertNotIn("items", status["spam_training"])
        self.assertNotIn("Synthetic training mail", json.dumps(status))
        restarted = Dashboard(self.directory, ACCOUNT, self.app.laya_url)
        self.assertEqual(restarted.training_export(), self.app.training_export())
        self.assertEqual(restarted.review_report()["summary"]["spam_training_examples"], 1)

    def test_global_limit_exclusions_errors_and_recovery(self):
        self.app.accounts_mgr.add_account(SECOND, "token_second.json")
        (self.directory / "token_second.json").write_text("{}")
        self.app.accounts_mgr.select("all")
        self.app.spam_training["items"]["abc"] = example()
        self.app.feedback["items"]["ddd"] = {**example("ddd", "Skipped synthetic mail"), "action": "skipped", "labels": None}
        args = self.start_without_thread(3)
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file"), \
                patch("google.auth.transport.requests.AuthorizedSession"), \
                patch("dashboard.collect_spam", side_effect=[
                    {"items": [], "processed": 2, "total": 2, "errors": ["gmail_message_not_found"]},
                    {"items": [example("fed", "Another synthetic mail", SECOND)],
                     "processed": 1, "total": 1, "errors": []}]) as collect:
            self.app.run_spam_training(*args)
        self.assertEqual([call.args[3] for call in collect.call_args_list], [2, 1])
        self.assertEqual(collect.call_args_list[0].args[2], {"abc", "ddd"})
        self.assertEqual(collect.call_args_list[1].args[2], set())
        self.assertEqual(self.app.spam_training["processed"], 3)
        self.assertEqual(self.app.spam_training["phase"], "error")
        self.assertEqual(self.app.spam_training_status()["available"], 2)
        self.assertNotIn("gmail_message_not_found", self.app.spam_training["errors"])
        self.app.spam_training["phase"] = "running"
        save_report(self.app.spam_training_file, self.app.spam_training)
        restarted = Dashboard(self.directory, ACCOUNT, self.app.laya_url)
        self.assertEqual(restarted.spam_training["phase"], "error")
        self.assertEqual(restarted.spam_training_status()["available"], 2)

    def test_import_shares_busy_guard_with_mail_and_account_actions(self):
        for invalid in (0, 101, True, "20", 1.5):
            with self.assertRaises(FeedbackError):
                self.app.start_spam_training(invalid)
        self.app.state["phase"] = "running"
        with self.assertRaises(FeedbackError):
            self.app.start_spam_training()
        self.app.state["phase"] = "idle"
        self.start_without_thread()
        self.assertFalse(self.app.start())
        for action in (self.app.start_spam_training, self.app.apply_current_labels, self.app.training_export):
            with self.assertRaises(FeedbackError):
                action()
        with self.assertRaises(AccountError):
            self.app.change_account("select", "all")

    def test_import_http_requires_login_origin_and_valid_limit(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        origin = "http://127.0.0.1:" + str(server.server_port)
        server.RequestHandlerClass = partial(Handler, app=self.app, origin=origin)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with requests.Session() as client:
                url = origin + "/api/spam-training"
                self.assertEqual(client.post(url, json={}, headers={"Origin": origin}, timeout=5).status_code, 401)
                client.post(origin + "/login", data={"code": self.app.key}, headers={"Origin": origin}, timeout=5)
                self.assertEqual(client.post(url, json={}, headers={"Origin": "https://evil.invalid"}, timeout=5).status_code, 403)
                for payload in ([], {"limit": 0}, {"limit": True}):
                    self.assertEqual(client.post(url, json=payload, headers={"Origin": origin}, timeout=5).status_code, 400)
                with patch.object(self.app, "run_spam_training"):
                    self.assertEqual(client.post(url, json={"limit": 20}, headers={"Origin": origin}, timeout=5).status_code, 202)
                self.assertEqual(client.get(origin + "/api/training-export", timeout=5).status_code, 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
