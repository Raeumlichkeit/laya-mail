"""Periodically request a batch from the dashboard's shared worker."""
from datetime import datetime, timezone
import threading


DEFAULT_INTERVAL = 180
CHECK_ERROR = "Die automatische Prüfung konnte nicht gestartet werden."


def now_utc():
    return datetime.now(timezone.utc).isoformat()


class LiveWatcher:
    def __init__(self, start_run, get_settings_fn=None, interval=DEFAULT_INTERVAL):
        self.start_run = start_run
        self.get_settings_fn = get_settings_fn or (lambda: {"live_mode": False})
        self.interval = interval
        self._thread = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self.last_run_utc = None
        self.last_status = "paused"
        self.last_error = None
        self.recent_activities = []
        self._lock = threading.Lock()

    def log_activity(self, message):
        with self._lock:
            self.recent_activities.insert(0, {"timestamp": now_utc(), "message": message})
            del self.recent_activities[20:]

    def get_status(self):
        with self._lock:
            return {
                "active": self.is_running(),
                "interval_seconds": self.interval,
                "last_run_utc": self.last_run_utc,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "recent_activities": [dict(entry) for entry in self.recent_activities],
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
        self._wake_event.set()

    def _record_error(self):
        with self._lock:
            self.last_status = "error"
            self.last_error = CHECK_ERROR
        self.log_activity(CHECK_ERROR)

    def _loop(self):
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                self._check_cycle(self.get_settings_fn())
            except Exception:
                self._record_error()
            self._wake_event.wait(timeout=self.interval)

    def _check_cycle(self, settings):
        if not settings.get("live_mode", False):
            with self._lock:
                self.last_status = "paused"
                self.last_error = None
            return
        with self._lock:
            self.last_run_utc = now_utc()
        try:
            started = self.start_run()
        except Exception:
            self._record_error()
            return
        with self._lock:
            self.last_status = "started" if started else "busy"
            self.last_error = None
        if started:
            self.log_activity("Automatische Prüfung gestartet.")
