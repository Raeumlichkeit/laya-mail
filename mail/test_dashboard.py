"""Offline HTTP checks; never accesses Google or Laya."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests
from dashboard import Dashboard, Handler, LABELS


class DashboardCheck(unittest.TestCase):
    def test_private_http_and_review_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            key = "synthetic-test-key-not-a-real-credential"
            (directory / "dashboard-key").write_text(key)
            report = {"input_version": "mail-v2", "created_utc": "2026-09-26T09:00:00Z", "status": "complete", "messages": [{
                "message_id": "abc123",
                "model_input": {"model": "multilingual", "state": "Testrechnung <img src=x onerror=alert(1)>",
                                "questions": {name: {"type": "choice", "instructions": name,
                                                "criteria": {option: option for option in sorted(choices)}}
                                              for name, choices in LABELS.items()}},
                "status": "suggestion", "subject": "<img src=x onerror=alert(1)>", "truncated": True,
                "suggestions": {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"},
                "raw_scores": {"inhalt": {"probabilities": {"rechnung": .49, "info": .48, "sonstiges": .03}}}
            }]}
            (directory / "pilot-test.json").write_text(json.dumps(report))
            app = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            origin = "http://127.0.0.1:" + str(server.server_port)
            server.RequestHandlerClass = partial(Handler, app=app, origin=origin)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            release = threading.Event()
            try:
                with requests.Session() as client:
                    self.assertEqual(client.get(origin + "/api/report", timeout=5).status_code, 401)
                    self.assertEqual(client.get(origin + "/api/training-export", timeout=5).status_code, 401)
                    self.assertEqual(client.post(origin + "/api/feedback", json={}, headers={"Origin": origin}, timeout=5).status_code, 401)
                    login = client.get(origin + "/", timeout=5)
                    self.assertNotIn(report["messages"][0]["subject"], login.text)
                    self.assertEqual(login.headers["Referrer-Policy"], "same-origin")
                    self.assertEqual(client.get(origin + "/health", headers={"Host": "evil.invalid"}, timeout=5).status_code, 400)
                    self.assertEqual(client.post(origin + "/login", data={"code": key}, headers={"Origin": "https://evil.invalid"}, timeout=5).status_code, 403)
                    self.assertEqual(client.post(origin + "/login", data={"code": "ü"}, headers={"Origin": origin}, timeout=5).status_code, 401)
                    response = client.post(origin + "/login", data={"code": key}, headers={"Origin": origin}, allow_redirects=False, timeout=5)
                    self.assertEqual(response.status_code, 303)
                    self.assertIn("HttpOnly", response.headers["Set-Cookie"])
                    self.assertIn("SameSite=Strict", response.headers["Set-Cookie"])
                    private = client.get(origin + "/api/report", timeout=5).json()
                    self.assertTrue(private["messages"][0]["needs_review"])
                    self.assertEqual(len(private["messages"][0]["review_reasons"]), 2)
                    self.assertTrue(private["messages"][0]["training_ready"])
                    self.assertEqual(private["summary"]["training_examples"], 0)
                    self.assertEqual(client.get(origin + "/api/training-export", timeout=5).content, b"")
                    feedback = {"report_id": private["report_id"], "message_id": "abc123",
                                "snapshot_id": private["messages"][0]["snapshot_id"], "action": "accepted"}
                    headers = {"Origin": origin}
                    self.assertEqual(client.post(origin + "/api/feedback", json=feedback,
                                                 headers={"Origin": "https://evil.invalid"}, timeout=5).status_code, 403)
                    for malformed in ([], {"action": []}, {**feedback, "action": "corrected", "labels": {"inhalt": "invented"}}):
                        self.assertEqual(client.post(origin + "/api/feedback", json=malformed, headers=headers, timeout=5).status_code, 400)
                    for stale in ({**feedback, "report_id": "../token.json"}, {**feedback, "snapshot_id": "wrong"}):
                        self.assertEqual(client.post(origin + "/api/feedback", json=stale, headers=headers, timeout=5).status_code, 409)
                    # Accept uses server-side suggestions, never a forged client label/input.
                    accepted = client.post(origin + "/api/feedback", json={**feedback, "labels": {"inhalt": "werbung"},
                                             "model_input": {"state": "FORGED"}}, headers=headers, timeout=5)
                    self.assertEqual(accepted.status_code, 200)
                    self.assertFalse(accepted.json()["messages"][0]["needs_review"])
                    self.assertEqual(accepted.json()["summary"]["reviewed"], 1)
                    self.assertEqual(accepted.json()["summary"]["training_examples"], 1)
                    client.post(origin + "/api/feedback", json=feedback, headers=headers, timeout=5)
                    exported = client.get(origin + "/api/training-export", timeout=5)
                    self.assertIn("attachment", exported.headers["Content-Disposition"])
                    self.assertEqual(len(exported.text.splitlines()), 1)  # No duplicate after repeated review.
                    example = exported.json()
                    self.assertEqual(example["state"], report["messages"][0]["model_input"]["state"])
                    self.assertEqual(accepted.json()["summary"]["distribution"]["inhalt"]["rechnung"], 1)
                    self.assertEqual(example["metadata"]["split"] in {"train", "val"}, True)
                    split_train = client.get(origin + "/api/training-export?split=train", timeout=5)
                    self.assertIn("laya-mail-train.jsonl", split_train.headers["Content-Disposition"])
                    self.assertNotIn(key, exported.text)
                    self.assertEqual(client.post(origin + "/api/feedback", json={**feedback, "action": "skipped"},
                                                 headers=headers, timeout=5).json()["summary"]["skipped"], 1)
                    self.assertEqual(client.get(origin + "/api/training-export", timeout=5).content, b"")
                    corrected = {**feedback, "action": "corrected", "labels": {"inhalt": "werbung", "wichtig": "nein", "spamverdacht": "nein"}}
                    self.assertEqual(client.post(origin + "/api/feedback", json=corrected, headers=headers, timeout=5).status_code, 200)
                    persisted = Dashboard(directory, "test@example.invalid", app.laya_url)
                    self.assertEqual(persisted.review_report()["messages"][0]["review"]["labels"]["inhalt"], "werbung")
                    # A changed input must not inherit a previous label as though newly reviewed.
                    changed = json.loads(json.dumps(report))
                    changed["messages"][0]["model_input"]["state"] += " Another input"
                    (directory / "pilot-test.json").write_text(json.dumps(changed))
                    fresh = client.get(origin + "/api/report", timeout=5).json()
                    self.assertIsNone(fresh["messages"][0]["review"])
                    self.assertEqual(client.post(origin + "/api/feedback", json=corrected, headers=headers, timeout=5).status_code, 409)
                    # Reports predating exact input capture cannot silently become training examples.
                    del changed["messages"][0]["model_input"]
                    (directory / "pilot-test.json").write_text(json.dumps(changed))
                    legacy = client.get(origin + "/api/report", timeout=5).json()
                    self.assertFalse(legacy["messages"][0]["training_ready"])
                    self.assertEqual(client.post(origin + "/api/feedback", json={**corrected, "snapshot_id": None}, headers=headers, timeout=5).status_code, 409)
                    (directory / "pilot-test.json").write_text(json.dumps(report))
                    before = (directory / "feedback.json").read_bytes()
                    with patch("dashboard.save_report", side_effect=OSError("disk full")):
                        self.assertEqual(client.post(origin + "/api/feedback", json=feedback, headers=headers, timeout=5).status_code, 503)
                    self.assertEqual((directory / "feedback.json").read_bytes(), before)
                    self.assertEqual(app.feedback["items"]["abc123"]["action"], "corrected")
                    self.assertFalse(app.authorized("laya_session=invalid.invalid.invalid"))
                    self.assertFalse(app.authorized('laya_session="1234.abc.é"'))
                    with patch("dashboard.save_report", side_effect=OSError("test disk error")):
                        self.assertEqual(client.post(origin + "/api/run", json={}, headers={"Origin": origin}, timeout=5).status_code, 503)
                        self.assertEqual(app.status()["phase"], "complete")
                    historical = json.loads(json.dumps(report))
                    historical["messages"].append({"message_id": "def456", "status": "error"})
                    (directory / "pilot-history.json").write_text(json.dumps(historical))
                    for bad_mode in ([], {"mode": []}, {"mode": "arbitrary"}):
                        self.assertEqual(client.post(origin + "/api/run", json=bad_mode, headers=headers, timeout=5).status_code, 400)
                    for invalid_limit in (0, 101, True, 2.5, "20", None):
                        self.assertEqual(client.post(origin + "/api/run", json={"mode": "next", "limit": invalid_limit}, headers=headers, timeout=5).status_code, 400)
                    with patch.object(app, "run", side_effect=lambda mode, limit: release.wait(5)) as worker:
                        self.assertEqual(client.post(origin + "/api/run", json={"mode": "next", "limit": 7}, headers=headers, timeout=5).status_code, 202)
                        self.assertEqual(client.post(origin + "/api/run", json={}, headers={"Origin": origin}, timeout=5).status_code, 409)
                        self.assertEqual(client.get(origin + "/api/status", timeout=5).json()["phase"], "running")
                        self.assertEqual(app.status()["batch_mode"], "next")
                        self.assertEqual(app.status()["requested_limit"], 7)
                        self.assertEqual(app.status()["total"], 7)
                        self.assertEqual(json.loads((directory / "seen-message-ids.json").read_text()), ["abc123", "def456"])
                        self.assertEqual((directory / "feedback.json").read_bytes(), before)
                        worker.assert_called_once_with("next", 7)
                        release.set()
                    restored = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")
                    self.assertEqual(restored.status()["phase"], "error")
                    client.post(origin + "/api/logout", json={}, headers={"Origin": origin}, timeout=5)
                    self.assertEqual(client.get(origin + "/api/report", timeout=5).status_code, 401)
            finally:
                release.set()
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
