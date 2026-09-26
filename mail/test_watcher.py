"""Unit tests for mail/watcher.py."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from watcher import LiveWatcher


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.key_file = self.dir / "gpu-api-key"
        self.key_file.write_text("test_key")
        self.accounts_mgr = MagicMock()

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_and_status(self):
        watcher = LiveWatcher(
            directory=self.dir,
            accounts_mgr=self.accounts_mgr,
            laya_url="http://127.0.0.1:8002/v1/systemone",
            key_path=self.key_file,
            interval=60,
        )
        status = watcher.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["interval_seconds"], 60)
        self.assertEqual(status["last_status"], "idle")
        self.assertEqual(status["recent_activities"], [])

    def test_log_activity_cap(self):
        watcher = LiveWatcher(
            directory=self.dir,
            accounts_mgr=self.accounts_mgr,
            laya_url="http://127.0.0.1:8002/v1/systemone",
            key_path=self.key_file,
        )
        for i in range(25):
            watcher.log_activity(f"Event {i}")
        status = watcher.get_status()
        self.assertEqual(len(status["recent_activities"]), 20)
        self.assertIn("Event 24", status["recent_activities"][0]["message"])

    def test_start_and_stop_lifecycle(self):
        watcher = LiveWatcher(
            directory=self.dir,
            accounts_mgr=self.accounts_mgr,
            laya_url="http://127.0.0.1:8002/v1/systemone",
            key_path=self.key_file,
            interval=10,
        )
        watcher.start()
        self.assertTrue(watcher.is_running())
        watcher.stop()
        self.assertFalse(watcher.is_running())

    def test_check_cycle_paused_when_not_live(self):
        settings = {"live_mode": False, "sort_inbox": False}
        watcher = LiveWatcher(
            directory=self.dir,
            accounts_mgr=self.accounts_mgr,
            laya_url="http://127.0.0.1:8002/v1/systemone",
            key_path=self.key_file,
            get_settings_fn=lambda: settings,
        )
        watcher._process_account = MagicMock()
        watcher._check_cycle(settings)
        watcher._process_account.assert_not_called()

    def test_check_cycle_processes_active_accounts(self):
        settings = {"live_mode": True, "sort_inbox": True}
        self.accounts_mgr.get_active_targets.return_value = [
            {"email": "user@example.com", "token_path": str(self.dir / "token.json")}
        ]
        watcher = LiveWatcher(
            directory=self.dir,
            accounts_mgr=self.accounts_mgr,
            laya_url="http://127.0.0.1:8002/v1/systemone",
            key_path=self.key_file,
            get_settings_fn=lambda: settings,
        )
        watcher._process_account = MagicMock(return_value=3)
        watcher._check_cycle(settings)
        watcher._process_account.assert_called_once_with(
            "user@example.com", str(self.dir / "token.json"), "test_key", True
        )
        self.assertEqual(watcher.last_status, "complete")
        self.assertTrue(any("3 neue E-Mails" in a["message"] for a in watcher.recent_activities))


if __name__ == "__main__":
    unittest.main()
