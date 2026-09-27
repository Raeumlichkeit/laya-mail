"""Offline retry tests: no network and no actual waiting."""
from contextlib import redirect_stderr, redirect_stdout
from email.utils import formatdate
import io
import unittest
from unittest.mock import Mock, call, patch

import requests

import gmail_http as http
import labeler
import pilot_gmail as pilot
from test_pilot_gmail import Gmail, Laya


PRIVATE = "PRIVATE-RESPONSE-BODY-AND-AUTHORIZATION"
URL = pilot.GMAIL + "/messages/abc123"


class Response:
    def __init__(self, status=200, reason=None, data=None, retry_after=None):
        self.status_code = status
        self.data = data if data is not None else {"error": {"message": PRIVATE, "errors": [{"reason": reason}]}}
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.text = PRIVATE

    def json(self):
        return self.data


class GmailHTTPCheck(unittest.TestCase):
    def setUp(self):
        sleep = patch("gmail_http.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        jitter = patch("gmail_http.random.random", return_value=0.25)
        jitter.start()
        self.addCleanup(jitter.stop)

    def test_transient_statuses_retry_at_most_seven_times(self):
        for status, reason in ((429, None), (500, None), (502, None), (503, None), (504, None),
                               (403, "rateLimitExceeded"), (403, "userRateLimitExceeded")):
            with self.subTest(status=status, reason=reason):
                failed = Response(status, reason)
                session = Mock()
                session.get.return_value = failed
                self.sleep.reset_mock()
                self.assertIs(http.request_with_retry(session, "GET", URL), failed)
                self.assertEqual(session.get.call_count, 7)
                self.assertEqual(self.sleep.call_args_list, [call(delay + 0.25) for delay in (1, 2, 4, 8, 16, 32)])
                self.assertTrue(all(args.kwargs["allow_redirects"] is False for args in session.get.call_args_list))

    def test_permanent_errors_are_not_retried_and_codes_never_contain_private_data(self):
        for status, reason, code in ((400, None, "gmail_request_failed"), (401, None, "gmail_authentication_failed"),
                                    (403, "insufficientPermissions", "gmail_permission_denied"),
                                    (403, "dailyLimitExceeded", "gmail_quota_exceeded"),
                                    (403, PRIVATE, "gmail_permission_denied"), (404, None, "gmail_message_not_found")):
            with self.subTest(status=status, reason=reason):
                response = Response(status, reason)
                session = Mock()
                session.get.return_value = response
                self.assertIs(http.request_with_retry(session, "GET", URL), response)
                session.get.assert_called_once()
                self.assertEqual(http.error_code(response), code)
                self.sleep.assert_not_called()
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output), self.assertRaises(pilot.PilotError) as caught:
                    pilot.gmail_get(session, "/messages/abc123")
                self.assertEqual(str(caught.exception), code)
                self.assertNotIn(PRIVATE, str(caught.exception) + output.getvalue())

    def test_transport_failures_retry_and_exhaustion_is_redacted(self):
        success = Response(data={"id": "abc123"})
        session = Mock()
        session.get.side_effect = [requests.Timeout(PRIVATE), requests.ConnectionError(PRIVATE), success]
        self.assertIs(http.request_with_retry(session, "GET", URL), success)
        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(1.25), call(2.25)])
        session.get.reset_mock()
        session.get.side_effect = requests.Timeout(PRIVATE)
        with self.assertRaises(pilot.PilotError) as caught:
            pilot.gmail_get(session, "/messages/abc123")
        self.assertEqual(str(caught.exception), "gmail_temporary_error")
        self.assertEqual(session.get.call_count, 7)
        self.assertNotIn(PRIVATE, str(caught.exception))
        session.get.side_effect = requests.RequestException(PRIVATE)
        session.get.reset_mock()
        with self.assertRaises(pilot.PilotError) as caught:
            pilot.gmail_get(session, "/messages/abc123")
        self.assertEqual(str(caught.exception), "gmail_request_failed")
        session.get.assert_called_once()

    def test_retry_after_is_respected_and_long_waits_return_original_response(self):
        for header, delay in (("7", 7), (formatdate(1012, usegmt=True), 12), ("malformed", 1.25)):
            with self.subTest(header=header), patch("gmail_http.time.time", return_value=1000):
                self.sleep.reset_mock()
                success = Response(data={"id": "abc123"})
                session = Mock()
                session.get.side_effect = [Response(429, retry_after=header), success]
                self.assertIs(http.request_with_retry(session, "GET", URL), success)
                self.sleep.assert_called_once_with(delay)
        for header in ("65", "99999999999999999", formatdate(1065, usegmt=True)):
            with self.subTest(header=header), patch("gmail_http.time.time", return_value=1000):
                self.sleep.reset_mock()
                blocked = Response(429, retry_after=header)
                session = Mock()
                session.get.return_value = blocked
                self.assertIs(http.request_with_retry(session, "GET", URL), blocked)
                session.get.assert_called_once()
                self.sleep.assert_not_called()

    def test_modify_retries_identical_payload_and_rate_limit_is_not_permission_error(self):
        session = Mock()
        session.post.side_effect = [Response(403, "rateLimitExceeded"), requests.ConnectionError(PRIVATE), Response(503), Response()]
        cache = {key: "Label_" + key for key in labeler.LABEL_MAP}
        suggestions = {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"}
        self.assertEqual(labeler.apply_labels(session, "abc123", suggestions, cache), ["Info"])
        self.assertEqual(session.post.call_count, 4)
        self.assertTrue(all(item == session.post.call_args_list[0] for item in session.post.call_args_list))
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["addLabelIds"], [cache["info"]])
        self.assertEqual(set(payload["removeLabelIds"]), {value for key, value in cache.items() if key != "info"})
        self.assertNotIn("INBOX", payload["removeLabelIds"])
        session.post.side_effect = None
        session.post.return_value = Response(403, "userRateLimitExceeded")
        with self.assertRaises(labeler.LabelerError) as caught:
            labeler.apply_labels(session, "abc123", suggestions, cache)
        self.assertEqual(str(caught.exception), "gmail_rate_limited")

    def test_label_reads_retry_including_conflict_refetch_but_creation_never_retries(self):
        labels = [{"name": name, "id": "Label_" + key} for key, name in labeler.LABEL_MAP.items()]
        session = Mock()
        session.get.side_effect = [Response(503), Response(data={"labels": labels[1:]}),
                                   Response(403, "rateLimitExceeded"), Response(data={"labels": labels})]
        session.post.return_value = Response(409)
        self.assertEqual(labeler.get_or_create_labels(session), {key: "Label_" + key for key in labeler.LABEL_MAP})
        self.assertEqual(session.get.call_count, 4)
        session.post.assert_called_once()
        for response in (Response(503), Response(403, "rateLimitExceeded")):
            with self.subTest(status=response.status_code):
                self.sleep.reset_mock()
                session = Mock()
                session.get.return_value = Response(data={"labels": []})
                session.post.return_value = response
                with self.assertRaises(labeler.LabelerError) as caught:
                    labeler.get_or_create_labels(session)
                session.post.assert_called_once()
                self.sleep.assert_not_called()
                self.assertNotIn(PRIVATE, str(caught.exception))
                if response.status_code == 403:
                    self.assertEqual(str(caught.exception), "gmail_rate_limited")

    def test_exhausted_get_or_modify_quota_stops_batch_and_saves_partial_results(self):
        for failed_method, reason, attempts in (("GET", "rateLimitExceeded", 7), ("modify", "userRateLimitExceeded", 7),
                                                ("GET", "dailyLimitExceeded", 1), ("modify", "dailyLimitExceeded", 1)):
            with self.subTest(failed_method=failed_method, reason=reason):
                gmail, laya = Gmail(count=3), Laya()
                original_get = gmail.get
                fetched = []

                def get(url, **kwargs):
                    if "/messages/" in url:
                        fetched.append(url.rsplit("/", 1)[1])
                    if failed_method == "GET" and url.endswith("/messages/1"):
                        return Response(403, reason)
                    return original_get(url, **kwargs)

                def modify(url, **kwargs):
                    return Response(403, reason) if failed_method == "modify" and url.endswith("/messages/1/modify") else Response()

                gmail.get = get
                gmail.post = Mock(side_effect=modify)
                cache = {key: "Label_" + key for key in labeler.LABEL_MAP}
                progress = []
                with patch.object(labeler, "get_or_create_labels", return_value=cache):
                    report = pilot.run_pilot(gmail, laya, "test@gmail.com", "http://127.0.0.1:8000/v1/systemone", "FAKE_KEY",
                                             apply_labels_mode=True, progress=lambda *value: progress.append(value))
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["total"], 3)
                self.assertEqual([row["message_id"] for row in report["messages"]], ["0", "1"])
                self.assertIn("applied_labels", report["messages"][0])
                self.assertEqual(progress, [(0, 3), (1, 3), (2, 3)])
                self.assertNotIn("2", fetched)
                self.assertEqual(len(laya.calls), 1 if failed_method == "GET" else 2)
                self.assertEqual(gmail.post.call_count, 1 if failed_method == "GET" else attempts + 1)
                self.assertEqual(fetched.count("1"), attempts if failed_method == "GET" else 1)
                failed = report["messages"][1]
                expected_code = "gmail_quota_exceeded" if reason == "dailyLimitExceeded" else "gmail_rate_limited"
                if failed_method == "GET":
                    self.assertEqual(failed["error"], expected_code)
                else:
                    self.assertEqual(failed["label_error_code"], expected_code)
                    self.assertEqual(failed["label_error"], "Die Gmail-Labels konnten nicht aktualisiert werden.")
                self.assertNotIn(PRIVATE, str(report))


if __name__ == "__main__":
    unittest.main()
