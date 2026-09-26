"""Autopilot live watcher for Laya Mail.

Periodically checks active Gmail accounts for incoming messages, runs classification,
and applies mailbox labels if configured.
"""
from datetime import datetime, timezone
import logging
import threading
import time

try:
    from pilot_gmail import run_pilot, save_report, load_excluded_ids, PilotError
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials
    import requests
except ImportError:
    from mail.pilot_gmail import run_pilot, save_report, load_excluded_ids, PilotError
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials
    import requests


DEFAULT_INTERVAL = 180  # 3 minutes


def now_utc():
    return datetime.now(timezone.utc).isoformat()


class LiveWatcher:
    def __init__(self, directory, accounts_mgr, laya_url, key_path, get_settings_fn=None, interval=DEFAULT_INTERVAL):
        self.directory = directory
        self.accounts_mgr = accounts_mgr
        self.laya_url = laya_url
        self.key_path = key_path
        self.get_settings_fn = get_settings_fn or (lambda: {"live_mode": False, "sort_inbox": False})
        self.interval = interval

        self._thread = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self.last_run_utc = None
        self.last_status = "idle"
        self.last_error = None
        self.recent_activities = []
        self._lock = threading.Lock()

    def log_activity(self, message):
        with self._lock:
            entry = {"timestamp": now_utc(), "message": message}
            self.recent_activities.insert(0, entry)
            self.recent_activities = self.recent_activities[:20]

    def get_status(self):
        with self._lock:
            return {
                "active": self.is_running(),
                "interval_seconds": self.interval,
                "last_run_utc": self.last_run_utc,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "recent_activities": list(self.recent_activities),
            }

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(target=self._loop, name="LayaLiveWatcher", daemon=True)
        self._thread.start()
        self.log_activity("Live-Watcher Hintergrunddienst gestartet.")

    def stop(self):
        if not self.is_running():
            return
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=5)
        self.log_activity("Live-Watcher Hintergrunddienst angehalten.")

    def trigger_now(self):
        """Immediately wake up the loop to run a check cycle."""
        self._wake_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            settings = self.get_settings_fn()
            if settings.get("live_mode", False):
                try:
                    self._check_cycle(settings)
                except Exception as err:
                    logging.exception("LiveWatcher check cycle exception")
                    self.last_error = str(err)
                    self.last_status = "error"
            else:
                self.last_status = "paused"

            # Wait for next interval or early trigger
            self._wake_event.wait(timeout=self.interval)
            self._wake_event.clear()

    def _check_cycle(self, settings):
        sort_inbox = bool(settings.get("sort_inbox", False))
        targets = self.accounts_mgr.get_active_targets()
        if not targets:
            self.last_status = "no_active_accounts"
            return

        self.last_status = "checking"
        self.last_run_utc = now_utc()
        key = self.key_path.read_text().strip() if self.key_path.is_file() else ""
        if not key:
            self.last_status = "missing_laya_key"
            self.last_error = "GPU-API-Schluessel fehlt."
            return

        total_processed = 0
        for target in targets:
            if self._stop_event.is_set():
                break
            email = target["email"]
            token_path = target["token_path"]
            try:
                processed = self._process_account(email, token_path, key, sort_inbox)
                total_processed += processed
            except Exception as err:
                logging.warning("LiveWatcher error for %s: %s", email, err)
                self.log_activity(f"Fehler bei {email}: {err}")

        self.last_status = "complete"
        self.last_error = None
        if total_processed > 0:
            self.log_activity(f"{total_processed} neue E-Mails automatisch klassifiziert.")

    def _process_account(self, email, token_path, key, sort_inbox):
        creds = Credentials.from_authorized_user_file(token_path)
        report_path = self.directory / "report.json"
        exclude_ids = set()
        if report_path.is_file():
            try:
                import json
                data = json.loads(report_path.read_text(encoding="utf-8"))
                for msg in data.get("messages", []):
                    if msg.get("message_id"):
                        exclude_ids.add(msg["message_id"])
            except Exception:
                pass

        with AuthorizedSession(creds) as gmail_session, requests.Session() as laya_session:
            laya_session.trust_env = False
            # Fetch up to 10 incoming unreviewed messages
            report = run_pilot(
                gmail_session,
                laya_session,
                email,
                self.laya_url,
                key,
                progress=None,
                exclude_ids=exclude_ids,
                limit=10,
                apply_labels_mode=sort_inbox,
            )

        messages = report.get("messages", [])
        valid_suggestions = [m for m in messages if m.get("status") == "suggestion"]
        if valid_suggestions:
            # Merge with existing report if present
            merged = report
            if report_path.is_file():
                try:
                    import json
                    existing_data = json.loads(report_path.read_text(encoding="utf-8"))
                    existing_messages = existing_data.get("messages", [])
                    merged_messages = valid_suggestions + existing_messages
                    merged["messages"] = merged_messages
                except Exception:
                    pass
            save_report(report_path, merged)
            label_note = " (mit Labels)" if sort_inbox else ""
            self.log_activity(f"{len(valid_suggestions)} E-Mails verarbeitet fuer {email}{label_note}")
        return len(valid_suggestions)
