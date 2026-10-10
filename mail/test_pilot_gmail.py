"""Entirely offline: python -m unittest discover -s mail."""

import base64
import copy
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pilot_gmail as pilot


def encoded(text):
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


MESSAGE = {"payload": {
    "mimeType": "multipart/mixed",
    "headers": [{"name": "Subject", "value": "Test Rechnung"}, {"name": "From", "value": "example@example.invalid"}],
    "parts": [
        {"mimeType": "multipart/alternative", "parts": [
            {"mimeType": "text/plain", "body": {"data": encoded("Bitte die Rechnung pruefen. " * 100)}},
            {"mimeType": "text/html", "body": {"data": encoded("<p>ALTERNATIVE_HTML_SECRET</p>")}},
        ]},
        {"mimeType": "text/plain", "filename": "rechnung.txt", "body": {"data": encoded("ATTACHMENT_SECRET")}},
        {"mimeType": "text/plain", "headers": [{"name": "Content-Disposition", "value": "attachment"}], "body": {"data": encoded("UNNAMED_ATTACHMENT_SECRET")}},
    ],
}}


class Response:
    status_code = 200

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class Gmail:
    """Only GET exists, so accidental mutations fail this test."""
    def __init__(self, count=25, message=MESSAGE, limit=20):
        self.calls = []
        self.count = count
        self.message = message
        self.limit = limit

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        if url == pilot.GMAIL + "/profile":
            return Response({"emailAddress": "test@gmail.com"})
        if url == pilot.GMAIL + "/messages":
            assert kwargs["params"] == {"labelIds": "INBOX", "maxResults": self.limit, "includeSpamTrash": "false"}
            return Response({"messages": [{"id": format(index, "x")} for index in range(self.count)]})
        assert url.startswith(pilot.GMAIL + "/messages/")
        assert kwargs["params"] == {"format": "full"}
        return Response(copy.deepcopy(self.message))


class Laya:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        assert url == "http://127.0.0.1:8000/v1/systemone"
        assert kwargs["allow_redirects"] is False
        assert kwargs["json"]["model"] == "multilingual"
        assert len(kwargs["json"]["state"]) <= pilot.STATE_CHARS
        assert "SECRET" not in kwargs["json"]["state"]
        if self.fail:
            return Response({"routing": {"model": "multilingual"}, "answers": {}})
        answers = {}
        for name, question in pilot.QUESTIONS.items():
            options = list(question["criteria"])
            answers[name] = {"type": "choice", "choice": options[0], "confidence": 0.1, "answer_confidence": 0.5,
                             "probabilities": {key: 1 / len(options) for key in options}}
        return Response({"routing": {"model": "multilingual"}, "answers": answers, "usage": {"input_tokens": 400}})


class PaginatedGmail(Gmail):
    def __init__(self, pages):
        super().__init__()
        self.pages = pages
        self.list_calls = 0

    def get(self, url, **kwargs):
        if url != pilot.GMAIL + "/messages":
            return super().get(url, **kwargs)
        self.calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        expected_token, response = self.pages[self.list_calls]
        expected_params = {"labelIds": "INBOX", "maxResults": 100, "includeSpamTrash": "false"}
        if expected_token is not None:
            expected_params["pageToken"] = expected_token
        assert kwargs["params"] == expected_params
        self.list_calls += 1
        return Response(response)


class PilotCheck(unittest.TestCase):
    def test_tailnet_model_urls_pass_validation_and_outside_addresses_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            directory.chmod(0o700)
            output = directory / "unused-report.json"
            for host, accepted in (("100.64.0.10", True), ("100.64.0.20", True),
                                   ("100.63.255.255", False), ("100.128.0.0", False), ("8.8.8.8", False)):
                errors = io.StringIO()
                with self.subTest(host=host), redirect_stderr(errors), \
                        patch.object(pilot.Credentials, "from_authorized_user_file",
                                     side_effect=pilot.PilotError("synthetic_credentials_boundary")) as credentials:
                    result = pilot.main(["--token", "unused-token.json", "--laya-key-file", "unused-key",
                                         "--expected-account", "test@example.invalid", "--output", str(output),
                                         "--laya-url", f"http://{host}:8002/v1/systemone"])
                self.assertEqual(result, 1)
                if accepted:
                    credentials.assert_called_once_with("unused-token.json")
                    self.assertIn("synthetic_credentials_boundary", errors.getvalue())
                else:
                    credentials.assert_not_called()
                    self.assertIn("laya_url_must_be_local", errors.getvalue())
                self.assertFalse(output.exists())

    def test_adjustable_limit_latest_next_and_invalid_values(self):
        url = "http://127.0.0.1:8000/v1/systemone"
        for limit in (3, 7, 100):
            with self.subTest(limit=limit):
                gmail, laya = Gmail(count=105, limit=limit), Laya()
                report = pilot.run_pilot(gmail, laya, "test@gmail.com", url, "FAKE_KEY", limit=limit)
                self.assertEqual(report["limit"], limit)
                self.assertEqual(len(report["messages"]), limit)
                self.assertEqual(len(laya.calls), limit)
                self.assertEqual(len(gmail.calls), limit + 2)
                gmail = PaginatedGmail([
                    (None, {"messages": [{"id": format(index, "x")} for index in range(100)], "nextPageToken": "page2"}),
                    ("page2", {"messages": [{"id": format(index, "x")} for index in range(100, 110)]}),
                ])
                laya = Laya()
                report = pilot.run_pilot(gmail, laya, "test@gmail.com", url, "FAKE_KEY", exclude_ids={"0"}, limit=limit)
                self.assertEqual(report["limit"], limit)
                self.assertEqual([row["message_id"] for row in report["messages"]], [format(index, "x") for index in range(1, limit + 1)])
                self.assertEqual(len(laya.calls), limit)
                self.assertEqual(gmail.list_calls, 2 if limit == 100 else 1)
        for value in (0, 101, True, False, 7.0, "7", None):
            gmail = Gmail()
            with self.subTest(invalid=value), self.assertRaisesRegex(pilot.PilotError, "limit_invalid"):
                pilot.run_pilot(gmail, Laya(), "test@gmail.com", url, "FAKE_KEY", limit=value)
            self.assertEqual(gmail.calls, [])
        for value in (0, 101):
            errors = io.StringIO()
            with patch.object(pilot.Credentials, "from_authorized_user_file") as credentials, redirect_stderr(errors):
                result = pilot.main(["--token", "unused.json", "--laya-key-file", "unused-key", "--expected-account", "test@gmail.com",
                                     "--output", "unused-report.json", "--limit", str(value)])
            self.assertEqual(result, 1)
            credentials.assert_not_called()
            self.assertIn("limit_invalid", errors.getvalue())

    def test_next_batch_pagination_exclusions_exhaustion_and_guards(self):
        url = "http://127.0.0.1:8000/v1/systemone"
        excluded = {format(index, "x") for index in range(15)}
        gmail = PaginatedGmail([
            (None, {"messages": [{"id": value} for value in [*sorted(excluded), "f", "f", "10"]], "nextPageToken": "page2"}),
            ("page2", {"messages": [{"id": value} for value in ["e", "10", *[format(index, "x") for index in range(17, 50)]]], "nextPageToken": "unused"}),
        ])
        laya = Laya()
        report = pilot.run_pilot(gmail, laya, "test@gmail.com", url, "FAKE_KEY", exclude_ids=excluded)
        wanted = [format(index, "x") for index in range(15, 35)]
        self.assertEqual(report["batch_mode"], "next")
        self.assertEqual(report["input_version"], "mail-v2")
        self.assertEqual([row["message_id"] for row in report["messages"]], wanted)
        self.assertEqual([path.rsplit("/", 1)[1] for path, _ in gmail.calls if "/messages/" in path], wanted)
        self.assertEqual(gmail.list_calls, 2)
        self.assertEqual(len(laya.calls), 20)
        self.assertEqual(excluded, {format(index, "x") for index in range(15)})

        exhausted = PaginatedGmail([
            (None, {"messages": [{"id": "abc"}], "nextPageToken": "page2"}),
            ("page2", {"messages": [{"id": "abc"}]}),
        ])
        progress = []
        laya = Laya()
        empty = pilot.run_pilot(exhausted, laya, "test@gmail.com", url, "FAKE_KEY",
                                lambda done, total: progress.append((done, total)), {"ABC"})
        self.assertEqual(empty["messages"], [])
        self.assertEqual(empty["status"], "complete")
        self.assertEqual(progress, [(0, 0)])
        self.assertEqual(len(exhausted.calls), 3)  # Profile and two lists, never a full message.
        self.assertEqual(laya.calls, [])

        wrong = PaginatedGmail([])
        with self.assertRaisesRegex(pilot.PilotError, "account_mismatch"):
            pilot.run_pilot(wrong, laya, "other@gmail.com", url, "FAKE_KEY", exclude_ids=set())
        self.assertEqual(len(wrong.calls), 1)
        for bad in ({"messages": {}}, {"messages": ["bad"]}, {"messages": [{"id": "../bad"}]}, {"nextPageToken": False}):
            with self.subTest(bad=bad), self.assertRaises(pilot.PilotError):
                pilot.run_pilot(PaginatedGmail([(None, bad)]), laya, "test@gmail.com", url, "FAKE_KEY", exclude_ids=set())
        repeating = PaginatedGmail([(None, {"nextPageToken": "same"}), ("same", {"nextPageToken": "same"})])
        with self.assertRaisesRegex(pilot.PilotError, "gmail_pagination_repeated_token"):
            pilot.run_pilot(repeating, laya, "test@gmail.com", url, "FAKE_KEY", exclude_ids=set())
        endless = PaginatedGmail([(None if index == 0 else f"page{index}", {"nextPageToken": f"page{index + 1}"}) for index in range(100)])
        with self.assertRaisesRegex(pilot.PilotError, "gmail_pagination_limit"):
            pilot.run_pilot(endless, laya, "test@gmail.com", url, "FAKE_KEY", exclude_ids=set())
        self.assertEqual(endless.list_calls, 100)
        self.assertEqual(laya.calls, [])

    def test_exclusion_file_validation_precedes_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "excluded.json"
            pilot.save_report(path, ["ABC", "abc", "12"])
            self.assertEqual(pilot.load_excluded_ids(path), {"abc", "12"})
            for malformed in ({"SECRET": []}, ["../SECRET"], [42]):
                pilot.save_report(path, malformed)
                errors = io.StringIO()
                with patch.object(pilot.Credentials, "from_authorized_user_file") as credentials, redirect_stderr(errors):
                    result = pilot.main(["--token", "unused.json", "--laya-key-file", "unused-key", "--expected-account", "test@gmail.com",
                                         "--output", str(path.with_name("report.json")), "--exclude-ids-file", str(path)])
                self.assertEqual(result, 1)
                credentials.assert_not_called()
                self.assertNotIn("SECRET", errors.getvalue())
                self.assertIn("exclude_ids_file_invalid", errors.getvalue())

    def test_standalone_joiner_padding_preserves_words(self):
        padding = "\u034f\u200c " * 100 + "\u200d\u200c \u200d"
        persian = "\u0645\u06cc\u200c\u0631\u0648\u0645"
        emoji = "\U0001f469\u200d\U0001f4bb"
        indic = "\u0915\u094d\u200d\u0937"
        self.assertEqual(pilot.clean(f"Anfang {padding} {persian} {emoji} {indic} Ende"),
                         f"Anfang {persian} {emoji} {indic} Ende")

    def test_mime_readonly_limit_account_and_incomplete_laya(self):
        gmail, laya = Gmail(), Laya()
        url = "http://127.0.0.1:8000/v1/systemone"
        progress = []
        report = pilot.run_pilot(gmail, laya, "test@gmail.com", url, "FAKE_KEY", lambda done, total: progress.append((done, total)))
        self.assertEqual(progress, [(done, 20) for done in range(21)])
        self.assertEqual(len(report["messages"]), 20)
        self.assertEqual(len(gmail.calls), 22)
        self.assertEqual(len(laya.calls), 20)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["batch_mode"], "latest")
        self.assertEqual(report["input_version"], "mail-v2")
        self.assertIn("werbung", report["category_definitions"]["inhalt"]["criteria"])
        for row, call in zip(report["messages"], laya.calls):
            self.assertEqual(row["model_input"], call["json"])
            self.assertIsNot(row["model_input"]["questions"], pilot.QUESTIONS)
        self.assertTrue(report["messages"][0]["truncated"])
        self.assertLessEqual(len(report["messages"][0]["excerpt"]), 240)
        self.assertEqual(report["messages"][0]["attachment_names"], ["rechnung.txt", "(Anhang ohne Dateinamen)"])
        self.assertNotIn("SECRET", json.dumps(report))
        self.assertNotIn("FAKE_KEY", json.dumps(report))

        html = {"payload": {"mimeType": "text/html", "body": {"data": encoded('<head><title>IGNORE</title></head><style>IGNORE</style><p>Hallo <a href="https://tracking.invalid">Welt</a></p><script>IGNORE</script><img src="https://tracking.invalid/pixel">')}}}
        text, _, source, _, _ = pilot.mail_text(html)
        self.assertEqual(text, "Hallo Welt")
        self.assertEqual(source, "html")
        self.assertEqual(len(gmail.calls), 22)  # Parsing causes no network traffic.

        wrong = Gmail()
        wrong_progress = []
        with self.assertRaisesRegex(pilot.PilotError, "account_mismatch"):
            pilot.run_pilot(wrong, laya, "other@gmail.com", url, "FAKE_KEY", lambda *counts: wrong_progress.append(counts))
        self.assertEqual(len(wrong.calls), 1)
        self.assertEqual(wrong_progress, [])
        failed = pilot.run_pilot(Gmail(), Laya(fail=True), "test@gmail.com", url, "FAKE_KEY")
        self.assertEqual(failed["status"], "error")
        self.assertTrue(all(row["status"] == "error" and row["error"] == "laya_response_invalid" and "suggestions" not in row for row in failed["messages"]))

        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "progress.json"
            progress_path.write_text("old progress")
            snapshots = []

            def update_progress(processed, total):
                pilot.save_report(progress_path, {"processed": processed, "total": total})
                snapshots.append(json.loads(progress_path.read_text()))

            pilot.run_pilot(Gmail(count=3), Laya(fail=True), "test@gmail.com", url, "FAKE_KEY", update_progress)
            self.assertEqual(snapshots, [{"processed": done, "total": 3} for done in range(4)])
            self.assertEqual(list(Path(directory).iterdir()), [progress_path])
            if os.name != "nt":
                self.assertEqual(progress_path.stat().st_mode & 0o777, 0o600)
            snapshots.clear()
            pilot.run_pilot(Gmail(count=0), laya, "test@gmail.com", url, "FAKE_KEY", update_progress)
            self.assertEqual(snapshots, [{"processed": 0, "total": 0}])

    def test_formatting_noise_long_context_and_failed_input_capture(self):
        accents = "Caf\u00e9 Cafe\u0301 Fran\u00e7ais \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac \u4e2d\u6587 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0928\u092e\u0938\u094d\u0924\u0947"
        noisy = f"\ufeff<!-- invisible\npreheader -->\u034f\u200b{accents} Rech\u00adnung"
        self.assertEqual(pilot.clean(noisy), accents + " Rechnung")
        self.assertEqual(pilot.clean("Text <!-- ungeschlossen"), "Text <!-- ungeschlossen")

        prefix = "Sicherheitsmeldung nach deiner Bestellung. "
        suffix = "Bitte bis Freitag antworten. Keine Zahlung erforderlich."
        body = noisy + " " + prefix + " F\u00fcllinformation" * 500 + " " + suffix
        message = {"payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "Subject", "value": "Hinweis " + "x" * 290},
                        {"name": "From", "value": "Absender " + "x" * 190}],
            "body": {"data": encoded(body)},
        }}
        laya = Laya(fail=True)
        report = pilot.run_pilot(Gmail(count=1, message=message), laya, "test@gmail.com",
                                 "http://127.0.0.1:8000/v1/systemone", "FAKE_KEY")
        row = report["messages"][0]
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error"], "laya_response_invalid")
        self.assertTrue(row["truncated"])
        self.assertEqual(row["model_input"], laya.calls[0]["json"])
        state = row["model_input"]["state"]
        self.assertLessEqual(len(state), pilot.STATE_CHARS)
        self.assertIn(accents, state)
        self.assertIn(prefix.strip(), state)
        self.assertIn(suffix, state)
        self.assertIn("[Mittelteil gekuerzt]", state)
        self.assertNotIn("preheader", state)
        self.assertFalse(any(char in state for char in "\u034f\u200b\ufeff\u00ad"))
        self.assertNotIn("suggestions", row)

    def test_strip_boilerplate_preservation_and_truncation_reporting(self):
        # 1. 'keine weiteren' alone must NOT be stripped
        t1 = "Bitte pruefen Sie die Unterlagen. Wenn Sie keine weiteren Fragen haben, ist alles in Ordnung."
        self.assertEqual(pilot.strip_boilerplate(t1), t1)

        # 2. Reply requests or deadlines in tail after 40% threshold must be preserved
        t2 = "Projektupdate fuer Oktober. " * 5 + "Wenn Sie diesen Newsletter nicht mehr erhalten moechten, bitte antworten Sie bis Freitag."
        self.assertEqual(pilot.strip_boilerplate(t2), t2)

        # 3. Legitimate boilerplate should be stripped
        prefix = "Unsere Herbstangebote sind da und gelten bis Ende des Monats. " * 3
        footer = "Wenn Sie diesen Newsletter nicht mehr erhalten moechten, klicken Sie hier zum Abmelden."
        t3 = prefix + footer
        stripped = pilot.strip_boilerplate(t3)
        self.assertLess(len(stripped), len(t3))
        self.assertEqual(stripped, prefix.rstrip())

        # 4. Truncation reporting: when clean_text != text, pilot marks truncated=True even under BODY_CHARS
        message = {"payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "Subject", "value": "Newsletter"}, {"name": "From", "value": "shop@example.invalid"}],
            "body": {"data": encoded(t3)},
        }}
        laya = Laya(fail=False)
        report = pilot.run_pilot(Gmail(count=1, message=message), laya, "test@gmail.com",
                                 "http://127.0.0.1:8000/v1/systemone", "FAKE_KEY")
        row = report["messages"][0]
        self.assertTrue(row["truncated"])
        self.assertIn("[Text oder Metadaten gekuerzt]", row["model_input"]["state"])


if __name__ == "__main__":
    unittest.main()
