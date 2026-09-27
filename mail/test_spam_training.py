"""Synthetic, offline checks for the training-only Gmail spam reader."""

import copy
import json
import unittest
from unittest.mock import patch

import pilot_gmail as pilot
import spam_training
from test_pilot_gmail import MESSAGE, Laya, Gmail


class Response:
    def __init__(self, value, status=200):
        self.value = value
        self.status_code = status
        self.headers = {}

    def json(self):
        if isinstance(self.value, Exception):
            raise self.value
        return copy.deepcopy(self.value)


def message(message_id, labels=None):
    value = copy.deepcopy(MESSAGE)
    value.update({"id": message_id, "labelIds": ["SPAM"] if labels is None else labels})
    return value


class SpamGmail:
    """Only GET exists; attachments, label writes, and inference cannot succeed."""
    def __init__(self, pages=None, messages=None, account="test@gmail.com"):
        self.pages = pages if pages is not None else {None: {"messages": [{"id": "abc"}]}}
        self.messages = messages if messages is not None else {"abc": message("abc")}
        self.account = account
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, copy.deepcopy(kwargs)))
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == 30
        if url == pilot.GMAIL + "/profile":
            return Response({"emailAddress": self.account})
        if url == pilot.GMAIL + "/messages":
            params = kwargs["params"]
            assert params["labelIds"] == "SPAM"
            assert params["includeSpamTrash"] == "true"
            assert params["maxResults"] == 100
            value = self.pages[params.get("pageToken")]
        else:
            assert url.startswith(pilot.GMAIL + "/messages/")
            assert kwargs["params"] == {"format": "full"}
            value = self.messages[url.rsplit("/", 1)[1]]
        return value if isinstance(value, Response) else Response(value)


class SpamTrainingCheck(unittest.TestCase):
    def test_read_only_input_matches_pilot_without_inference_or_attachments(self):
        laya = Laya()
        ordinary = pilot.run_pilot(Gmail(count=1), laya, "test@gmail.com",
                                   "http://127.0.0.1:8000/v1/systemone", "FAKE")
        gmail = SpamGmail()
        with patch.object(pilot, "classify", side_effect=AssertionError("Must not infer")):
            result = spam_training.collect_spam(gmail, "TEST@gmail.com", set())
        self.assertEqual(result["errors"], [])
        self.assertEqual((result["processed"], result["total"]), (1, 1))
        self.assertEqual(result["items"], [{
            "message_id": "abc", "account_email": "test@gmail.com",
            "input_version": pilot.INPUT_VERSION,
            "model_input": ordinary["messages"][0]["model_input"],
        }])
        self.assertEqual(len(gmail.calls), 3)
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(set(result["items"][0]["model_input"]["questions"]), set(pilot.QUESTIONS))

    def test_exclusions_pagination_and_limit_do_not_change_inputs(self):
        gmail = SpamGmail(
            pages={None: {"messages": [{"id": "ABC"}, {"id": "def"}, {"id": "def"}], "nextPageToken": "p2"},
                   "p2": {"messages": [{"id": "ab1"}, {"id": "ab2"}]},},
            messages={"def": message("def"), "ab1": message("ab1")},
        )
        excluded = {"abc"}
        result = spam_training.collect_spam(gmail, "test@gmail.com", excluded, limit=2)
        self.assertEqual([item["message_id"] for item in result["items"]], ["def", "ab1"])
        self.assertEqual(excluded, {"abc"})
        self.assertEqual((result["processed"], result["total"]), (2, 2))
        empty = spam_training.collect_spam(SpamGmail(), "test@gmail.com", {"ABC"})
        self.assertEqual(empty, {"items": [], "errors": [], "processed": 0, "total": 0})

    def test_invalid_arguments_and_wrong_identity_prevent_message_reads(self):
        for limit in (0, 101, True, "20", 20.0, None):
            gmail = SpamGmail()
            with self.subTest(limit=limit), self.assertRaisesRegex(pilot.PilotError, "limit_invalid"):
                spam_training.collect_spam(gmail, "test@gmail.com", set(), limit)
            self.assertEqual(gmail.calls, [])
        for excluded in ([], {"../bad"}, {None}):
            gmail = SpamGmail()
            with self.assertRaisesRegex(pilot.PilotError, "exclude_ids_invalid"):
                spam_training.collect_spam(gmail, "test@gmail.com", excluded)
            self.assertEqual(gmail.calls, [])
        gmail = SpamGmail(account="other@gmail.com")
        with self.assertRaisesRegex(pilot.PilotError, "account_mismatch"):
            spam_training.collect_spam(gmail, "test@gmail.com", set())
        self.assertEqual(len(gmail.calls), 1)

    def test_full_message_identity_and_spam_membership_are_checked(self):
        values = [message("abc", ["INBOX"]), message("abc", ["SPAM", "TRASH"]),
                  message("def"), message("abc", "SPAM"), {"id": "abc", "labelIds": ["SPAM"]}]
        for value in values:
            with self.subTest(value=value):
                result = spam_training.collect_spam(SpamGmail(messages={"abc": value}), "test@gmail.com", set())
                self.assertEqual(result["items"], [])
                self.assertEqual(len(result["errors"]), 1)
                self.assertEqual(result["processed"], 1)
                self.assertNotIn("SECRET", json.dumps(result))

    def test_invalid_list_id_or_response_stops_before_full_reads(self):
        for listing in ({"messages": [{"id": "../SECRET"}]}, {"messages": "SECRET"},
                        ValueError("PRIVATE_RESPONSE_SECRET")):
            gmail = SpamGmail(pages={None: Response(listing)})
            with self.subTest(listing=listing), self.assertRaises(pilot.PilotError) as raised:
                spam_training.collect_spam(gmail, "test@gmail.com", set())
            self.assertEqual(len(gmail.calls), 2)
            self.assertNotIn("SECRET", str(raised.exception))

    def test_message_failures_are_explicit_and_rate_limits_stop_batch(self):
        listing = {None: {"messages": [{"id": "abc"}, {"id": "def"}]}}
        not_found = SpamGmail(pages=listing, messages={
            "abc": Response({"error": {"message": "PRIVATE_SECRET"}}, status=404), "def": message("def"),
        })
        result = spam_training.collect_spam(not_found, "test@gmail.com", set())
        self.assertEqual(result["errors"], ["gmail_message_not_found"])
        self.assertEqual([item["message_id"] for item in result["items"]], ["def"])
        self.assertEqual(result["processed"], 2)
        limited = SpamGmail(pages=listing, messages={
            "abc": Response({"error": {"message": "PRIVATE_SECRET"}}, status=429),
        })
        with patch("gmail_http.time.sleep"):
            result = spam_training.collect_spam(limited, "test@gmail.com", set())
        self.assertEqual(result, {"items": [], "errors": ["gmail_rate_limited"], "processed": 1, "total": 2})
        self.assertEqual(len(limited.calls), 9)  # Profile, listing, seven bounded GET attempts.


if __name__ == "__main__":
    unittest.main()
