"""Offline tests for the dashboard batch scheduler."""
import threading
import unittest
from unittest.mock import Mock, patch

from watcher import CHECK_ERROR, LiveWatcher


class WatcherTests(unittest.TestCase):
    def test_default_and_disabled_cycles_do_not_start_batches(self):
        start_run = Mock()
        watcher = LiveWatcher(start_run, interval=60)
        self.assertEqual(watcher.get_settings_fn(), {"live_mode": False})
        watcher._check_cycle({})
        watcher._check_cycle({"live_mode": False, "sort_inbox": True})
        start_run.assert_not_called()
        status = watcher.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["interval_seconds"], 60)
        self.assertEqual(status["last_status"], "paused")
        self.assertIsNone(status["last_run_utc"])
        self.assertEqual(status["recent_activities"], [])

    def test_live_cycle_requests_exactly_one_shared_batch(self):
        start_run = Mock(side_effect=[True, False])
        watcher = LiveWatcher(start_run)
        watcher._check_cycle({"live_mode": True, "sort_inbox": True})
        start_run.assert_called_once_with()
        self.assertEqual(watcher.get_status()["last_status"], "started")
        watcher._check_cycle({"live_mode": True})
        self.assertEqual(start_run.call_count, 2)
        status = watcher.get_status()
        self.assertEqual(status["last_status"], "busy")
        self.assertIsNotNone(status["last_run_utc"])
        self.assertIsNone(status["last_error"])
        self.assertEqual(len(status["recent_activities"]), 1)

    def test_callback_error_is_redacted_and_next_cycle_can_recover(self):
        start_run = Mock(side_effect=[RuntimeError("token=private-mail-secret"), True])
        watcher = LiveWatcher(start_run)
        watcher._check_cycle({"live_mode": True})
        status = watcher.get_status()
        self.assertEqual(status["last_status"], "error")
        self.assertEqual(status["last_error"], CHECK_ERROR)
        self.assertNotIn("private-mail-secret", str(status))
        watcher._check_cycle({"live_mode": True})
        self.assertEqual(watcher.get_status()["last_status"], "started")
        self.assertIsNone(watcher.get_status()["last_error"])

    def test_settings_error_is_redacted_without_killing_loop(self):
        start_run = Mock()
        watcher = LiveWatcher(start_run, Mock(side_effect=ValueError("private-settings")))
        with patch.object(watcher._wake_event, "wait", side_effect=lambda **kwargs: watcher._stop_event.set()):
            watcher._loop()
        start_run.assert_not_called()
        self.assertEqual(watcher.last_status, "error")
        self.assertNotIn("private-settings", str(watcher.get_status()))

    def test_log_activity_cap_and_status_snapshot(self):
        watcher = LiveWatcher(Mock())
        for index in range(25):
            watcher.log_activity(f"Event {index}")
        status = watcher.get_status()
        self.assertEqual(len(status["recent_activities"]), 20)
        self.assertEqual(status["recent_activities"][0]["message"], "Event 24")
        status["recent_activities"][0]["message"] = "changed"
        self.assertEqual(watcher.get_status()["recent_activities"][0]["message"], "Event 24")

    def test_start_stop_and_wakeup_use_shared_callback(self):
        initial_cycle = threading.Event()
        started = threading.Event()
        live = threading.Event()

        def settings():
            enabled = live.is_set()
            initial_cycle.set()
            return {"live_mode": enabled}

        start_run = Mock(side_effect=lambda: started.set() or True)
        watcher = LiveWatcher(start_run, settings, interval=3600)
        try:
            watcher.start()
            self.assertTrue(initial_cycle.wait(2))
            thread = watcher._thread
            watcher.start()
            self.assertIs(watcher._thread, thread)
            start_run.assert_not_called()
            live.set()
            watcher.trigger_now()
            self.assertTrue(started.wait(2))
        finally:
            watcher.stop()
        self.assertFalse(watcher.is_running())
        start_run.assert_called_once_with()
        watcher.stop()


if __name__ == "__main__":
    unittest.main()
