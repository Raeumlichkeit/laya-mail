"""Unit tests for AccountManager."""
import json
from pathlib import Path
import tempfile
import unittest

from accounts import AccountManager, AccountError


class AccountManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "token.json").write_text('{"token": "dummy"}', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_default_account(self):
        mgr = AccountManager(self.dir, "test@example.com")
        state = mgr.get_state()
        self.assertEqual(state["selected"], "all")
        self.assertEqual(len(state["accounts"]), 1)
        self.assertEqual(state["accounts"][0]["email"], "test@example.com")
        self.assertTrue(state["accounts"][0]["token_exists"])

    def test_add_and_select_account(self):
        mgr = AccountManager(self.dir, "first@example.com")
        (self.dir / "token_second.json").write_text('{"token": "dummy2"}', encoding="utf-8")
        mgr.add_account("second@example.com", "token_second.json")

        state = mgr.get_state()
        self.assertEqual(len(state["accounts"]), 2)

        mgr.select("second@example.com")
        self.assertEqual(mgr.get_state()["selected"], "second@example.com")

        active = mgr.get_active_targets()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["email"], "second@example.com")

        # Select all
        mgr.select("all")
        active_all = mgr.get_active_targets()
        self.assertEqual(len(active_all), 2)

    def test_toggle_and_remove(self):
        mgr = AccountManager(self.dir, "first@example.com")
        (self.dir / "token_second.json").write_text('{"token": "dummy2"}', encoding="utf-8")
        mgr.add_account("second@example.com", "token_second.json")

        mgr.toggle_account("second@example.com", False)
        active = mgr.get_active_targets()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["email"], "first@example.com")

        mgr.select("second@example.com")
        mgr.remove_account("second@example.com")
        # Selection resets to all if selected account is deleted
        self.assertEqual(mgr.get_state()["selected"], "all")
        self.assertEqual(len(mgr.list_accounts()), 1)

    def test_invalid_email_handling(self):
        mgr = AccountManager(self.dir, "valid@example.com")
        with self.assertRaises(AccountError):
            mgr.add_account("invalid-email")

    def test_get_token_path(self):
        mgr = AccountManager(self.dir, "first@example.com")
        mgr.add_account("second@example.com", "token_custom.json")
        self.assertEqual(mgr.get_token_path("first@example.com"), self.dir / "token.json")
        self.assertEqual(mgr.get_token_path("second@example.com"), self.dir / "token_custom.json")
        self.assertIsNone(mgr.get_token_path("unknown@example.com"))


if __name__ == "__main__":
    unittest.main()
