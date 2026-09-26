"""Unit tests for mail/labeler.py."""
import unittest
from unittest.mock import MagicMock

from labeler import (
    LABEL_MAP,
    LabelerError,
    get_or_create_labels,
    labels_for_suggestions,
    apply_labels,
)


class LabelerTests(unittest.TestCase):
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
        self.assertEqual(labels_for_suggestions(None), [])
        self.assertEqual(labels_for_suggestions({}), [])

    def test_get_or_create_labels(self):
        session = MagicMock()
        # Initial call returns empty label list
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"labels": []}

        # Creation succeeds with mocked IDs
        def mock_post(url, json=None, timeout=None):
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

        label_cache = {
            "rechnung": "Label_Rechnung_ID",
            "wichtig": "Label_Wichtig_ID",
        }
        suggestions = {"inhalt": "rechnung", "wichtig": "ja"}

        applied = apply_labels(session, "msg_123", suggestions, label_cache, archive=False)
        self.assertEqual(applied, ["Rechnung", "Wichtig"])

        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertIn("/messages/msg_123/modify", args[0])
        payload = kwargs.get("json", {})
        self.assertIn("Label_Rechnung_ID", payload["addLabelIds"])
        self.assertIn("Label_Wichtig_ID", payload["addLabelIds"])
        self.assertEqual(payload["removeLabelIds"], [])

    def test_apply_labels_no_op(self):
        session = MagicMock()
        label_cache = {"rechnung": "id_1"}
        applied = apply_labels(session, "msg_123", {"inhalt": "sonstiges"}, label_cache, archive=False)
        self.assertEqual(applied, [])
        session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
