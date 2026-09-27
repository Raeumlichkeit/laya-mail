"""Unit tests for AccountManager."""
import json
import os
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

    def test_same_local_part_gets_distinct_stable_token_paths(self):
        mgr = AccountManager(self.dir)
        emails = ("same@first.invalid", "same@second.invalid")
        for email in emails:
            mgr.add_account(email)
        paths = [mgr.get_token_path(email) for email in emails]
        self.assertNotEqual(*paths)
        for path in paths:
            self.assertEqual(path.parent, self.dir)
            path.write_text('{"token": "synthetic"}', encoding="utf-8")
        mgr.add_account(" SAME@FIRST.INVALID ")
        reloaded = AccountManager(self.dir)
        self.assertEqual([reloaded.get_token_path(email) for email in emails], paths)
        self.assertEqual({target["email"]: target["token_path"] for target in reloaded.get_active_targets()},
                         dict(zip(emails, map(str, paths))))

    def test_rejects_token_paths_and_duplicate_token_without_changing_config(self):
        mgr = AccountManager(self.dir, "first@example.invalid")
        original = mgr.get_state()
        saved = mgr.config_path.read_bytes()
        for token in ("../token_other.json", "..\\token_other.json", "sub/token_other.json",
                      "sub\\token_other.json", str(self.dir / "token_other.json"), "token.json", 42):
            with self.subTest(token=token), self.assertRaises(AccountError):
                mgr.add_account("second@example.invalid", token)
            self.assertEqual(mgr.get_state(), original)
            self.assertEqual(mgr.config_path.read_bytes(), saved)

    @unittest.skipUnless(os.name == "nt", "Windows token paths are case insensitive")
    def test_rejects_case_alias_of_another_accounts_token_on_windows(self):
        mgr = AccountManager(self.dir)
        mgr.add_account("first@example.invalid", "token_shared.json")
        (self.dir / "token_shared.json").write_text("synthetic", encoding="utf-8")
        with self.assertRaises(AccountError):
            mgr.add_account("second@example.invalid", "token_SHARED.json")


if __name__ == "__main__":
    unittest.main()
