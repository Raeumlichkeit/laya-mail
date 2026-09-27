"""Offline checks for the explicit complete-inbox labeling action."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from dashboard import snapshot_id
import labeler
import pilot_gmail as pilot
from test_pilot_gmail import Gmail, Laya, PaginatedGmail


URL = "http://127.0.0.1:8000/v1/systemone"
CACHE = {key: f"Label_{index}" for index, key in enumerate(labeler.LABEL_MAP)}


def mailbox(count=3):
    return PaginatedGmail([(None, {"messages": [{"id": format(index, "x")} for index in range(count)]})])


class AllInboxCheck(unittest.TestCase):
    def run_all(self, gmail, laya=None, **kwargs):
        return pilot.run_pilot(gmail, laya or Laya(), "test@gmail.com", URL, "FAKE_KEY",
                               apply_labels_mode=True, all_inbox=True, **kwargs)

    def test_all_pages_deduplicate_and_exceed_normal_limits(self):
        gmail = PaginatedGmail([
            (None, {"messages": [{"id": format(index, "x")} for index in range(100)], "nextPageToken": "second"}),
            ("second", {"messages": [{"id": "A"}, *[{"id": format(index, "x")} for index in range(95, 151)]]}),
        ])
        laya = Laya()
        progress = []
        with patch.object(labeler, "get_or_create_labels", return_value=CACHE), \
                patch.object(labeler, "apply_labels", return_value=[]) as apply:
            report = self.run_all(gmail, laya, limit=None, progress=lambda *counts: progress.append(counts))
        self.assertEqual(report["batch_mode"], "all")
        self.assertIsNone(report["limit"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["total"], 151)
        self.assertEqual([row["message_id"] for row in report["messages"]], [format(index, "x") for index in range(151)])
        self.assertEqual(len(laya.calls), 151)
        self.assertEqual(apply.call_count, 151)
        self.assertEqual(progress, [(processed, 151) for processed in range(152)])

        # The all-inbox action is not capped at the next-batch mode's 100 pages.
        pages = [(None if index == 0 else str(index),
                  {"messages": [{"id": format(index, "x")}], **({"nextPageToken": str(index + 1)} if index < 100 else {})})
                 for index in range(101)]
        self.assertEqual(len(pilot.next_entries(PaginatedGmail(pages), set(), limit=None)), 101)

    def test_repeated_tokens_and_invalid_entries_fail_before_label_writes(self):
        for pages, code in (
            ([(None, {"nextPageToken": "same"}), ("same", {"nextPageToken": "same"})], "gmail_pagination_repeated_token"),
            ([(None, {"messages": [{"id": "../private"}]})], "gmail_message_id_invalid"),
        ):
            with self.subTest(code=code), patch.object(labeler, "get_or_create_labels") as create, \
                    self.assertRaisesRegex(pilot.PilotError, code):
                self.run_all(PaginatedGmail(pages))
            create.assert_not_called()

    def test_stop_during_listing_does_not_fetch_or_modify_messages(self):
        for final_page in (False, True):
            with self.subTest(final_page=final_page):
                stop = threading.Event()
                gmail = PaginatedGmail([(None, {"messages": [{"id": "abc"}],
                                               **({} if final_page else {"nextPageToken": "unused"})})])
                original_get = gmail.get

                def get(url, **kwargs):
                    response = original_get(url, **kwargs)
                    if url.endswith("/messages"):
                        stop.set()
                    return response

                gmail.get = get
                laya = Laya()
                progress = []
                with patch.object(labeler, "get_or_create_labels") as create, patch.object(labeler, "apply_labels") as apply:
                    report = self.run_all(gmail, laya, should_stop=stop.is_set, progress=lambda *counts: progress.append(counts))
                self.assertEqual(report["status"], "stopped")
                self.assertEqual(report["messages"], [])
                self.assertEqual(report["total"], 1)
                self.assertEqual(progress, [(0, 1)])
                self.assertEqual(len(gmail.calls), 2)
                self.assertEqual(laya.calls, [])
                create.assert_not_called()
                apply.assert_not_called()

    def test_stop_keeps_partial_report_and_existing_errors_take_priority(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                stop = threading.Event()
                laya = Laya(fail=fail)

                def progress(processed, total):
                    if processed == 1:
                        stop.set()

                with patch.object(labeler, "get_or_create_labels", return_value=CACHE), \
                        patch.object(labeler, "apply_labels", return_value=[]) as apply:
                    report = self.run_all(mailbox(), laya, should_stop=stop.is_set, progress=progress)
                self.assertEqual(report["status"], "error" if fail else "stopped")
                self.assertEqual(report["total"], 3)
                self.assertEqual(len(report["messages"]), 1)
                self.assertEqual(len(laya.calls), 1)
                self.assertEqual(apply.call_count, 0 if fail else 1)

        stop = threading.Event()
        laya = Laya()
        original_post = laya.post

        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            stop.set()
            return response

        laya.post = post
        with patch.object(labeler, "get_or_create_labels", return_value=CACHE), patch.object(labeler, "apply_labels", return_value=[]) as apply:
            report = self.run_all(mailbox(), laya, should_stop=stop.is_set)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(len(report["messages"]), 1)
        self.assertEqual(len(laya.calls), 1)
        self.assertEqual(report["messages"][0]["applied_labels"], [])
        apply.assert_called_once()

        # A stop requested during the final mail still reports stopped after
        # completing that mail's label write.
        stop.clear()
        with patch.object(labeler, "get_or_create_labels", return_value=CACHE), patch.object(labeler, "apply_labels", return_value=[]) as apply:
            report = self.run_all(mailbox(1), laya, should_stop=stop.is_set)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["messages"][0]["applied_labels"], [])
        apply.assert_called_once()

    def test_exact_review_overrides_written_labels_but_keeps_model_suggestion(self):
        original = pilot.run_pilot(Gmail(count=1), Laya(), "test@gmail.com", URL, "FAKE_KEY")["messages"][0]
        correction = {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"}
        self.assertNotEqual(original["suggestions"], correction)
        fingerprint = snapshot_id(original, pilot.INPUT_VERSION)
        valid = {"action": "corrected", "labels": correction, "snapshot_id": fingerprint}
        cases = [(valid, True), ({**valid, "action": "accepted"}, True),
                 ({**valid, "action": "skipped"}, False), ({**valid, "action": []}, False),
                 ({**valid, "snapshot_id": "old-input"}, False),
                 ({**valid, "labels": {"inhalt": "info"}}, False),
                 ({**valid, "labels": {**correction, "wichtig": "maybe"}}, False)]
        for item, expected_review in cases:
            reviews = {"0": copy.deepcopy(item)}
            unchanged = copy.deepcopy(reviews)
            with self.subTest(item=item), patch.object(labeler, "get_or_create_labels", return_value=CACHE), \
                    patch.object(labeler, "apply_labels", return_value=["Info"]) as apply:
                report = self.run_all(mailbox(1), reviews=reviews)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["messages"][0]["suggestions"], original["suggestions"])
            self.assertEqual(apply.call_args.args[2], correction if expected_review else original["suggestions"])
            self.assertEqual(reviews, unchanged)

    def test_cli_all_requires_explicit_modify_scope_and_reads_private_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            private.chmod(0o700)
            output = private / "report.json"
            key = private / "laya-key"
            key.write_text("FAKE_KEY", encoding="utf-8")
            args = ["--token", str(private / "token.json"), "--laya-key-file", str(key),
                    "--output", str(output), "--expected-account", "test@gmail.com", "--all-inbox"]
            for flags, code in (([], "all_inbox_requires_apply_labels"),
                                (["--apply-labels", "--exclude-ids-file", "unused.json"], "all_inbox_excludes_not_allowed")):
                stderr = io.StringIO()
                with patch.object(pilot.Credentials, "from_authorized_user_file") as credentials, redirect_stderr(stderr):
                    self.assertEqual(pilot.main(args + flags), 1)
                credentials.assert_not_called()
                self.assertIn(code, stderr.getvalue())
            stderr = io.StringIO()
            with patch.object(pilot.Credentials, "from_authorized_user_file", return_value=Mock(scopes=[pilot.READONLY_SCOPE])), \
                    patch.object(pilot, "AuthorizedSession") as session, redirect_stderr(stderr):
                self.assertEqual(pilot.main(args + ["--apply-labels"]), 1)
            session.assert_not_called()
            self.assertIn("modify_token_required", stderr.getvalue())

            stop = private / "stop.json"
            reviews = private / "reviews.json"
            pilot.save_report(stop, {"stop": True})
            pilot.save_report(reviews, {})
            stop_bytes, review_bytes = stop.read_bytes(), reviews.read_bytes()
            with patch.object(pilot.Credentials, "from_authorized_user_file", return_value=Mock(scopes=[pilot.MODIFY_SCOPE])), \
                    patch.object(pilot, "AuthorizedSession") as gmail_context, patch.object(pilot.requests, "Session") as laya_context, \
                    patch.object(labeler, "get_or_create_labels") as create, redirect_stdout(io.StringIO()):
                gmail_context.return_value.__enter__.return_value = Gmail()
                laya_context.return_value.__enter__.return_value = Laya()
                self.assertEqual(pilot.main(args + ["--apply-labels", "--stop-file", str(stop), "--reviews-file", str(reviews)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "stopped")
            self.assertEqual(report["messages"], [])
            self.assertEqual(report["total"], 0)
            self.assertEqual(stop.read_bytes(), stop_bytes)
            self.assertEqual(reviews.read_bytes(), review_bytes)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
