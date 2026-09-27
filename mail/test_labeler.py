"""Unit tests for mail/labeler.py."""
import unittest
from unittest.mock import MagicMock, patch
import requests

from labeler import (
    LABEL_MAP,
    LabelerError,
    get_or_create_labels,
    labels_for_suggestions,
    apply_labels,
)


class LabelerTests(unittest.TestCase):
    def setUp(self):
        self.cache = {key: f"Label_{key}" for key in LABEL_MAP}

    def test_labels_for_suggestions(self):
        self.assertEqual(
            labels_for_suggestions({"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"}),
            ["rechnung", "wichtig"],
        )
        self.assertEqual(
            labels_for_suggestions({"inhalt": "werbung", "wichtig": "nein", "spamverdacht": "nein"}),
            ["werbung"],
        )
        self.assertEqual(
            labels_for_suggestions({"inhalt": "sonstiges", "wichtig": "nein", "spamverdacht": "ja"}),
            ["spamverdacht"],
        )
        for invalid in (None, {}, {"inhalt": "rechnung"}, {"inhalt": "info", "wichtig": "nein", "spamverdacht": []}):
            with self.subTest(invalid=invalid), self.assertRaises(LabelerError):
                labels_for_suggestions(invalid)

    def test_get_or_create_labels(self):
        session = MagicMock()
        # Initial call returns empty label list
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": []}

        # Creation succeeds with mocked IDs
        def mock_post(url, json=None, timeout=None, allow_redirects=None):
            self.assertFalse(allow_redirects)
            resp = MagicMock()
            resp.status_code = 200
            name = (json or {}).get("name", "unknown")
            resp.json.return_value = {"id": f"id_{name}", "name": name}
            return resp

        session.post.side_effect = mock_post

        cache = get_or_create_labels(session)
        self.assertEqual(len(cache), len(LABEL_MAP))
        for key, name in LABEL_MAP.items():
            self.assertEqual(cache[key], f"id_{name}")

    def test_apply_labels(self):
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {"id": "msg_123"}

        suggestions = {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "nein"}

        applied = apply_labels(session, "msg_123", suggestions, self.cache, archive=False)
        self.assertEqual(applied, ["Rechnung", "Laya/Wichtig"])

        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertIn("/messages/msg_123/modify", args[0])
        payload = kwargs.get("json", {})
        self.assertEqual(set(payload["addLabelIds"]), {"Label_rechnung", "Label_wichtig"})
        self.assertEqual(set(payload["removeLabelIds"]), {"Label_werbung", "Label_info", "Label_spamverdacht"})
        self.assertFalse(kwargs["allow_redirects"])

    def test_corrections_replace_managed_labels_and_preserve_unrelated(self):
        session = MagicMock()
        present = {"INBOX", "Unrelated_Label"}
        cache = {**self.cache, "unrelated": "Unrelated_Label"}

        def modify(url, json, **kwargs):
            self.assertTrue(set(json["addLabelIds"] + json["removeLabelIds"]) <= set(self.cache.values()))
            present.difference_update(json["removeLabelIds"])
            present.update(json["addLabelIds"])
            response = MagicMock()
            response.status_code = 200
            return response

        session.post.side_effect = modify
        apply_labels(session, "msg_123", {"inhalt": "rechnung", "wichtig": "nein", "spamverdacht": "ja"}, cache)
        self.assertEqual(present, {"INBOX", "Unrelated_Label", "Label_rechnung", "Label_spamverdacht"})
        apply_labels(session, "msg_123", {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"}, cache)
        self.assertEqual(present, {"INBOX", "Unrelated_Label", "Label_info"})
        applied = apply_labels(session, "msg_123", {"inhalt": "sonstiges", "wichtig": "nein", "spamverdacht": "nein"}, cache)
        self.assertEqual(applied, [])
        self.assertEqual(present, {"INBOX", "Unrelated_Label"})
        self.assertEqual(session.post.call_count, 3)

    def test_get_or_create_labels_permission_denied(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": []}
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "Forbidden"
        session.post.return_value = resp

        with self.assertRaises(LabelerError) as ctx:
            get_or_create_labels(session)
        self.assertIn("403", str(ctx.exception))
        self.assertIn("gmail.modify", str(ctx.exception))

    def test_apply_labels_permission_denied(self):
        session = MagicMock()
        session.post.return_value.status_code = 403
        with self.assertRaises(LabelerError) as ctx:
            apply_labels(session, "msg_123", {"inhalt": "rechnung", "wichtig": "nein", "spamverdacht": "nein"}, self.cache)
        self.assertIn("403", str(ctx.exception))
        self.assertIn("gmail.modify", str(ctx.exception))

    def test_apply_labels_missing_cache_key(self):
        session = MagicMock()
        suggestions = {"inhalt": "sonstiges", "wichtig": "nein", "spamverdacht": "nein"}
        for cache in ({}, {"rechnung": "id_1"}, {**self.cache, "info": "INBOX"}, {key: "same" for key in LABEL_MAP}):
            with self.subTest(cache=cache), self.assertRaises(LabelerError):
                apply_labels(session, "msg_123", suggestions, cache)
        with self.assertRaises(LabelerError):
            apply_labels(session, "msg_123", {"inhalt": "sonstiges"}, self.cache)
        with self.assertRaises(LabelerError):
            apply_labels(session, "msg_123", suggestions, self.cache, archive=True)
        session.post.assert_not_called()

    def test_missing_conflict_label_and_sensitive_errors_fail_closed(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": []}
        session.post.return_value.status_code = 409
        with self.assertRaises(LabelerError):
            get_or_create_labels(session)

        session.post.return_value.status_code = 500
        session.post.return_value.text = "PRIVATE_MAIL_BODY"
        with self.assertRaises(LabelerError) as ctx:
            get_or_create_labels(session)
        self.assertNotIn("PRIVATE", str(ctx.exception))
        session.post.side_effect = requests.ConnectionError("PRIVATE_AUTHORIZATION_HEADER")
        with patch("gmail_http.time.sleep"), self.assertRaises(LabelerError) as ctx:
            apply_labels(session, "msg_123", {"inhalt": "info", "wichtig": "nein", "spamverdacht": "nein"}, self.cache)
        self.assertEqual(str(ctx.exception), "modify_request_failed")

    def test_only_exact_managed_names_are_reused(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": [
            {"name": " Info ", "id": "unrelated-id"},
            *[{"name": name, "id": self.cache[key]} for key, name in LABEL_MAP.items() if key != "info"],
        ]}
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {"id": "new-info-id"}
        labels = get_or_create_labels(session)
        self.assertEqual(labels["info"], "new-info-id")
        self.assertNotIn("unrelated-id", labels.values())
        session.post.assert_called_once()

    def test_importance_uses_custom_name_instead_of_reserved_system_label(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": [
            {"name": "Wichtig", "id": "IMPORTANT", "type": "system"},
            *[{"name": name, "id": self.cache[key]} for key, name in LABEL_MAP.items() if key != "wichtig"],
        ]}

        def create(url, json, **kwargs):
            response = MagicMock()
            response.status_code = 400 if json["name"] == "Wichtig" else 200
            response.json.return_value = {"id": "custom-importance-id"}
            return response

        session.post.side_effect = create
        labels = get_or_create_labels(session)
        self.assertEqual(labels["wichtig"], "custom-importance-id")
        self.assertNotIn("IMPORTANT", labels.values())
        session.post.assert_called_once()
        self.assertEqual(session.post.call_args.kwargs["json"]["name"], "Laya/Wichtig")


if __name__ == "__main__":
    unittest.main()

