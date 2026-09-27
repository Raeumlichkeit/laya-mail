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
    def setUp(self):
        # Background scheduling and network health probes are tested separately.
        self.start_patch = patch("dashboard.LiveWatcher.start")
        self.start_patch.start()
        self.addCleanup(self.start_patch.stop)
        self.health_patch = patch.object(Dashboard, "gpu_status", return_value={"online": False})
        self.health_patch.start()
        self.addCleanup(self.health_patch.stop)

    def test_failed_fetch_preserves_feedback_without_changed_input_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "dashboard-key").write_text("synthetic-test-key-not-a-real-credential")
            (directory / "token.json").write_text("{}")
            row = {
                "message_id": "abc123", "status": "suggestion",
                "suggestions": {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"},
                "model_input": {
                    "model": "multilingual", "state": "Synthetische Testrechnung",
                    "questions": {name: {"type": "choice", "instructions": name,
                                         "criteria": {option: option for option in sorted(choices)}}
                                  for name, choices in LABELS.items()},
                },
            }
            report = {"input_version": "mail-v2", "status": "complete", "messages": [row]}
            report_path = directory / "pilot-test.json"
            report_path.write_text(json.dumps(report))
            app = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")
            original = app.review_report()["messages"][0]
            app.save_feedback({"report_id": report_path.name, "message_id": row["message_id"],
                               "snapshot_id": original["snapshot_id"], "action": "accepted"})
            saved_feedback = (directory / "feedback.json").read_bytes()
            saved_export = app.training_export()
            report["messages"] = [{"message_id": row["message_id"], "status": "error",
                                   "error": "gmail_request_failed"}]
            report_path.write_text(json.dumps(report))

            failed = app.review_report()
            failed_row = failed["messages"][0]
            self.assertIsNone(failed_row["snapshot_id"])
            self.assertIsNone(failed_row["review"])
            self.assertFalse(failed_row["training_ready"])
            self.assertTrue(failed_row["needs_review"])
            self.assertEqual(failed_row["review_reasons"], ["Technischer Fehler bei der Auswertung"])
            self.assertEqual(failed["summary"]["training_examples"], 1)
            self.assertEqual((directory / "feedback.json").read_bytes(), saved_feedback)
            self.assertEqual(app.training_export(), saved_export)

            # An available, genuinely changed input must still invalidate the displayed review.
            row["model_input"]["state"] += " Anderer Text"
            report["messages"] = [row]
            report_path.write_text(json.dumps(report))
            changed = app.review_report()["messages"][0]
            self.assertIsNone(changed["review"])
            self.assertIn("Die frühere Bewertung gilt für eine andere Texteingabe; bitte erneut prüfen.",
                          changed["review_reasons"])

    def test_private_http_and_review_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            key = "synthetic-test-key-not-a-real-credential"
            (directory / "dashboard-key").write_text(key)
            (directory / "token.json").write_text("{}")
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
                        # Failed historical rows remain eligible for another fetch.
                        self.assertEqual(json.loads(app.seen_path(app.account).read_text()), ["abc123"])
                        self.assertEqual((directory / "feedback.json").read_bytes(), before)
                        worker.assert_called_once_with("next", 7)
                        release.set()
                    restored = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")
                    self.assertEqual(restored.status()["phase"], "error")
                    app.state = dict(restored.state)
                    with patch.object(app, "run"):
                        self.assertEqual(client.post(origin + "/api/run", json={"mode": "all"}, headers=headers, timeout=5).status_code, 202)
                    self.assertIsNone(app.state["requested_limit"])
                    self.assertEqual(client.post(origin + "/api/run/stop", json={}, headers={"Origin": "https://evil.invalid"}, timeout=5).status_code, 403)
                    self.assertFalse((directory / "stop-request.json").exists())
                    stopped = client.post(origin + "/api/run/stop", json={}, headers=headers, timeout=5)
                    self.assertEqual(stopped.status_code, 200)
                    self.assertEqual(stopped.json(), {"status": "stopping"})
                    self.assertTrue(app.state["stop_requested"])
                    self.assertEqual(app.state["phase"], "running")  # The worker, not the request, finishes the run.
                    client.post(origin + "/api/logout", json={}, headers={"Origin": origin}, timeout=5)
                    self.assertEqual(client.get(origin + "/api/report", timeout=5).status_code, 401)
            finally:
                release.set()
                server.shutdown()
                server.server_close()
                thread.join()

    def test_gpu_accounts_and_settings_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            key = "synthetic-test-key-not-a-real-credential"
            (directory / "dashboard-key").write_text(key)
            (directory / "token_second.json").write_text("{}")
            app = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            origin = "http://127.0.0.1:" + str(server.server_port)
            server.RequestHandlerClass = partial(Handler, app=app, origin=origin)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with requests.Session() as client:
                    login = client.post(origin + "/login", data={"code": key}, headers={"Origin": origin}, allow_redirects=False)
                    self.assertEqual(login.status_code, 303)
                    headers = {"Origin": origin}

                    gpu_res = client.get(origin + "/api/gpu-status", headers=headers)
                    self.assertEqual(gpu_res.status_code, 200)
                    self.assertIn("online", gpu_res.json())

                    acc_res = client.get(origin + "/api/accounts", headers=headers)
                    self.assertEqual(acc_res.status_code, 200)
                    self.assertIn("accounts", acc_res.json())
                    self.assertEqual(acc_res.json()["selected"], "all")

                    add_res = client.post(origin + "/api/accounts/add", json={"email": "second@example.invalid"}, headers=headers)
                    self.assertEqual(add_res.status_code, 200)
                    self.assertTrue(any(a["email"] == "second@example.invalid" for a in add_res.json()["accounts"]))
                    token_name = next(a["token_file"] for a in add_res.json()["accounts"] if a["email"] == "second@example.invalid")
                    (directory / token_name).write_text("{}")

                    sel_res = client.post(origin + "/api/accounts/select", json={"selected": "second@example.invalid"}, headers=headers)
                    self.assertEqual(sel_res.status_code, 200)
                    self.assertEqual(sel_res.json()["selected"], "second@example.invalid")

                    sett_res = client.get(origin + "/api/settings", headers=headers)
                    self.assertEqual(sett_res.status_code, 200)
                    self.assertFalse(sett_res.json()["live_mode"])
                    self.assertFalse(sett_res.json()["sort_inbox"])

                    upd_res = client.post(origin + "/api/settings", json={"live_mode": True, "sort_inbox": True}, headers=headers)
                    self.assertEqual(upd_res.status_code, 200)
                    self.assertTrue(upd_res.json()["live_mode"])
                    self.assertTrue(upd_res.json()["sort_inbox"])

                    status_res = client.get(origin + "/api/status", headers=headers).json()
                    self.assertFalse(status_res["read_only"])
                    self.assertTrue(status_res["settings"]["live_mode"])

                    # Test apply-current without report
                    apply_none = client.post(origin + "/api/labels/apply-current", json={}, headers=headers)
                    self.assertEqual(apply_none.status_code, 404)

                    with patch.object(app, "apply_current_labels", return_value={"status": "ok", "applied_count": 2, "total_messages": 2, "errors": []}):
                        apply_res = client.post(origin + "/api/labels/apply-current", json={}, headers=headers)
                        self.assertEqual(apply_res.status_code, 200)
                        self.assertEqual(apply_res.json()["applied_count"], 2)

                    # Test rescan mode in start
                    with patch.object(app, "run"):
                        self.assertTrue(app.start(mode="rescan", limit=15))
                        self.assertEqual(app.status()["batch_mode"], "rescan")
                        self.assertEqual(app.status()["requested_limit"], 15)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


    def test_apply_current_labels_logic(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            key = "synthetic-test-key-not-a-real-credential"
            (directory / "dashboard-key").write_text(key)
            (directory / "token.json").write_text(json.dumps({"token": "fake"}))

            valid_questions = {name: {"type": "choice", "instructions": name,
                                      "criteria": {option: option for option in sorted(choices)}}
                               for name, choices in LABELS.items()}
            msg1 = {
                "message_id": "msg1",
                "status": "suggestion",
                "model_input": {"model": "multilingual", "state": "Invoice 123", "questions": valid_questions},
                "suggestions": {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"},
            }
            msg2 = {
                "message_id": "msg2",
                "status": "suggestion",
                "model_input": {"model": "multilingual", "state": "Newsletter", "questions": valid_questions},
                "suggestions": {"inhalt": "werbung", "wichtig": "nein", "spamverdacht": "nein"},
            }
            msg3 = {
                "message_id": "msg3",
                "status": "error",
                "model_input": {"model": "multilingual", "state": "Broken", "questions": valid_questions},
                "suggestions": {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"},
            }
            msg4 = {
                "message_id": "msg4",
                "status": "suggestion",
                "model_input": {"model": "multilingual", "state": "Invalid choices", "questions": valid_questions},
                "suggestions": {"inhalt": "invalid_choice", "wichtig": "nein", "spamverdacht": "nein"},
            }

            from dashboard import snapshot_id
            snap1 = snapshot_id(msg1, "mail-v2")
            feedback_data = {
                "revision": 1,
                "items": {
                    "msg1": {
                        "action": "corrected",
                        "snapshot_id": snap1,
                        "labels": {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"},
                    },
                    "msg2": {
                        "action": "corrected",
                        "snapshot_id": "mismatched-snap",
                        "labels": {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"},
                    }
                }
            }
            (directory / "feedback.json").write_text(json.dumps(feedback_data))
            report = {
                "input_version": "mail-v2",
                "created_utc": "2026-09-27T08:00:00Z",
                "messages": [msg1, msg2, msg3, msg4]
            }
            (directory / "pilot-test.json").write_text(json.dumps(report))

            app = Dashboard(directory, "test@example.invalid", "http://127.0.0.1:8000/v1/systemone")

            # 1. Test concurrency lock
            app.applying_labels = True
            with self.assertRaises(Exception):
                app.apply_current_labels()
            app.applying_labels = False

            # 2. Test label selection and application
            applied_records = []
            def fake_apply(session, message_id, effective, label_cache, archive=False):
                applied_records.append((message_id, effective))
                return ["FakeLabel"]

            with patch("labeler.get_or_create_labels", return_value={"FakeLabel": "lbl_1"}), \
                 patch("labeler.apply_labels", side_effect=fake_apply), \
                 patch("dashboard.gmail_get", return_value={"emailAddress": "test@example.invalid"}), \
                 patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=object()), \
                 patch("google.auth.transport.requests.AuthorizedSession", return_value=unittest.mock.MagicMock()):
                res = app.apply_current_labels()
                self.assertEqual(res["status"], "ok")
                self.assertEqual(res["applied_count"], 2)
                # msg1 used corrected feedback labels
                self.assertEqual(applied_records[0], ("msg1", {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"}))
                # msg2 fell back to valid plain suggestions because snapshot mismatched
                self.assertEqual(applied_records[1], ("msg2", {"inhalt": "werbung", "wichtig": "nein", "spamverdacht": "nein"}))
                # msg3 (error status) and msg4 (invalid choices) were skipped
                self.assertEqual(len(applied_records), 2)
                self.assertFalse(app.applying_labels)

            # 3. Test report path change during run
            with patch("labeler.get_or_create_labels", return_value={"FakeLabel": "lbl_1"}), \
                 patch("labeler.apply_labels", side_effect=fake_apply), \
                 patch("dashboard.gmail_get", return_value={"emailAddress": "test@example.invalid"}), \
                 patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=object()), \
                 patch("google.auth.transport.requests.AuthorizedSession", return_value=unittest.mock.MagicMock()):
                orig_target = app.report_path
                def switch_report(session, msg_id, eff, cache, archive=False):
                    app.report_path = directory / "other-report.json"
                    return ["FakeLabel"]
                with patch("labeler.apply_labels", side_effect=switch_report):
                    res2 = app.apply_current_labels()
                    self.assertTrue(any("Berichtspfad hat sich" in e for e in res2["errors"]))
                app.report_path = orig_target

            # 4. Test missing modify scope returns clear error
            fake_readonly_creds = unittest.mock.MagicMock()
            fake_readonly_creds.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
            with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=fake_readonly_creds):
                res3 = app.apply_current_labels()
                self.assertEqual(res3["applied_count"], 0)
                self.assertTrue(any("gmail.modify" in e for e in res3["errors"]))


if __name__ == "__main__":
    unittest.main()


