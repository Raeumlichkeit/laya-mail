"""Private LAN dashboard for read-only Laya/Gmail pilot runs."""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import hmac
import ipaddress
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import requests

from pilot_gmail import save_report, gmail_get, INPUT_VERSION, QUESTIONS
import account_setup

try:
    from accounts import AccountManager, AccountError
    from watcher import LiveWatcher
except ImportError:
    from mail.accounts import AccountManager, AccountError
    from mail.watcher import LiveWatcher

LABELS = {"inhalt": {"rechnung", "info", "werbung", "sonstiges"},
          "wichtig": {"ja", "nein"}, "spamverdacht": {"ja", "nein"}}
PILOT_FAILURE_MESSAGES = {
    "modify_token_required": "Die Gmail-Freigabe erlaubt nur Lesen. Für Sortieren die Anmeldung mit gmail.modify erneuern oder Sortieren deaktivieren.",
    "readonly_token_required": "Der Gmail-Anmeldung fehlt eine gültige Lesefreigabe. Bitte die Anmeldung erneuern.",
    "account_mismatch": "Die Gmail-Anmeldung gehört nicht zum ausgewählten Postfach. Bitte die Kontozuordnung prüfen.",
    "label_initialization_failed": "Die Gmail-Labels konnten nicht vorbereitet werden. Bitte Schreibfreigabe und Verbindung prüfen.",
    "gmail_request_failed": "Gmail konnte nicht erreicht werden. Bitte Anmeldung und Verbindung prüfen.",
    "gmail_rate_limited": "Gmail begrenzt gerade die Anfragen. Bitte später erneut versuchen.",
    "gmail_quota_exceeded": "Das Gmail-Tageskontingent ist ausgeschöpft. Bitte nach dessen Zurücksetzung erneut versuchen.",
    "gmail_temporary_error": "Gmail meldet einen vorübergehenden Fehler. Bitte später erneut versuchen.",
    "gmail_message_not_found": "Die Nachricht ist bei Gmail nicht mehr verfügbar.",
    "gmail_permission_denied": "Gmail hat den Zugriff abgelehnt. Bitte die Google-Freigabe prüfen.",
    "gmail_authentication_failed": "Die Gmail-Anmeldung ist nicht mehr gültig. Bitte erneut bei Google anmelden.",
    "laya_key_missing": "Der API-Schlüssel für das ausgewählte Laya-Modell fehlt.",
    "output_directory_not_private": "Das Verzeichnis für die Auswertung ist nicht ausreichend geschützt. Bitte die Dateirechte prüfen.",
    "output_destination_unavailable": "Der neue Auswertungsbericht kann am vorgesehenen Ort nicht gespeichert werden.",
}


def pilot_failure_message(stderr):
    prefix = "Pilot fehlgeschlagen: "
    if isinstance(stderr, str) and stderr.startswith(prefix):
        return PILOT_FAILURE_MESSAGES.get(stderr[len(prefix):].strip())
    return None


class FeedbackError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def valid_input(value):
    if not isinstance(value, dict) or value.get("model") != "multilingual":
        return False
    if not isinstance(value.get("state"), str) or not value["state"].strip():
        return False
    questions = value.get("questions")
    return (isinstance(questions, dict) and set(questions) == set(LABELS)
            and all(isinstance(questions[name], dict)
                    and questions[name].get("type") == "choice"
                    and isinstance(questions[name].get("criteria"), dict)
                    and set(questions[name]["criteria"]) == choices
                    for name, choices in LABELS.items()))


def snapshot_id(row, version):
    if not valid_input(row.get("model_input")):
        return None
    encoded = json.dumps([row.get("message_id"), version, row["model_input"]],
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_labels(value):
    return (isinstance(value, dict) and set(value) == set(LABELS)
            and all(isinstance(value[name], str) and value[name] in choices
                    for name, choices in LABELS.items()))


def training_eligible(item):
    return (item.get("action") in {"accepted", "corrected"}
            and valid_labels(item.get("labels")) and valid_input(item.get("model_input")))


def now():
    return datetime.now(timezone.utc).isoformat()


def for_review(report):
    result = dict(report)
    result["messages"] = []
    names = {"inhalt": "Inhalt", "wichtig": "Wichtigkeit", "spamverdacht": "Spamverdacht"}
    for original in report.get("messages", []):
        row = dict(original)
        reasons = []
        if row.get("status") != "suggestion":
            reasons.append("Technischer Fehler bei der Auswertung")
        if row.get("label_error"):
            reasons.append("Gmail-Labels konnten nicht aktualisiert werden")
        if row.get("truncated"):
            reasons.append("Text oder Betreff wurde gekürzt")
        if row.get("content_omitted"):
            reasons.append("Nicht alle Nachrichteninhalte konnten ausgewertet werden")
        if row.get("attachment_names"):
            reasons.append("Anhanginhalt wurde nicht ausgewertet")
        for question, answer in row.get("raw_scores", {}).items():
            values = sorted(answer.get("probabilities", {}).values(), reverse=True)
            if len(values) > 1 and values[0] - values[1] < 0.20:
                reasons.append("Knappes Modellergebnis: " + names.get(question, question))
        if row.get("suggestions", {}).get("spamverdacht") == "ja":
            reasons.append("Spamverdacht bitte persönlich überprüfen")
        row.update(needs_review=bool(reasons), review_reasons=reasons)
        result["messages"].append(row)
    result["review_policy"] = {"margin_below": 0.20, "calibrated": False}
    return result


class Dashboard:
    def __init__(self, directory, account, laya_url):
        self.directory = directory
        self.account = account
        self.laya_url = laya_url
        self.key = (directory / "dashboard-key").read_text().strip()
        if len(self.key) < 24:
            raise ValueError("Dashboard key must be random and at least 24 characters")
        self.lock = threading.Lock()
        self.applying_labels = False
        self.connecting_account = False
        feedback_file = directory / "feedback.json"
        self.feedback = (json.loads(feedback_file.read_text(encoding="utf-8"))
                         if feedback_file.exists() else {"revision": None, "items": {}})
        self.state = {"phase": "idle", "processed": 0, "total": 0, "batch_mode": "latest", "requested_limit": 20,
                      "started_at": None, "finished_at": None, "error": None,
                      "model": {"name": "Laya multilingual", "device": "cpu", "host": "Mac mini"},
                      "processing_note": None}
        reports = sorted(directory.glob("pilot-*.json"), key=lambda p: p.stat().st_mtime)
        self.report_path = reports[-1] if reports else None
        if self.report_path:
            report = self.report()
            self.state.update(phase="complete" if report.get("status") == "complete" else "error",
                              processed=len(report["messages"]), total=len(report["messages"]),
                              batch_mode=report.get("batch_mode", "latest"),
                              requested_limit=report.get("limit", 20),
                              model=report.get("processing_backend", self.state["model"]),
                              finished_at=report.get("created_utc"))
        state_file = directory / "status.json"
        if state_file.exists():
            self.state.update(json.loads(state_file.read_text(encoding="utf-8")))
            if self.state["phase"] == "running":
                progress_file = directory / "progress.json"
                if progress_file.exists():
                    self.state.update(json.loads(progress_file.read_text(encoding="utf-8")))
                self.state.update(phase="error", finished_at=now(),
                                  error="Der letzte Lesetest wurde durch einen Neustart unterbrochen.")
        save_report(state_file, self.state)

        # Multi-Account Manager
        self.accounts_mgr = AccountManager(self.directory, default_email=self.account)

        # Persistent Settings
        self.settings_file = self.directory / "settings.json"
        self.settings = self._load_settings()

        # Live and manual checks share account routing, history and one worker.
        self.watcher = LiveWatcher(lambda: self.start("next", 10, live=True), self.get_settings)
        self.watcher.start()

    def _load_settings(self):
        defaults = {"live_mode": False, "sort_inbox": False}
        if self.settings_file.is_file():
            try:
                data = json.loads(self.settings_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {**defaults, **data}
            except Exception:
                pass
        return defaults

    def account_setup(self, value):
        email, permission = account_setup.selection(value)
        with self.lock:
            self.require_accounts_idle()
            account_setup.require_new_account(self.accounts_mgr, email)
            return account_setup.prepare(self.directory, email, permission)

    def require_accounts_idle(self):
        if self.state["phase"] == "running" or self.applying_labels or self.connecting_account:
            raise AccountError("Bitte warten, bis der laufende Vorgang beendet ist.", 409)

    def change_account(self, action, *args):
        with self.lock:
            self.require_accounts_idle()
            return getattr(self.accounts_mgr, action)(*args)

    def connect_account(self, value):
        email, permission = account_setup.selection(value)
        with self.lock:
            self.require_accounts_idle()
            if self.settings.get("live_mode"):
                raise AccountError("Bitte schalte den Live-Modus in der Übersicht aus, bevor du ein Postfach verbindest.", 409)
            account_setup.require_new_account(self.accounts_mgr, email)
            config = account_setup.read_config(self.directory) or {}
            self.connecting_account = True
        try:
            token = account_setup.verify_grant(value.get("token"), email, permission, config.get("client_id"))
            with self.lock:
                accounts = account_setup.store_new_account(self.accounts_mgr, email, token)
            return {"status": "connected", "email": email, "permission": permission, "accounts": accounts}
        finally:
            with self.lock:
                self.connecting_account = False

    def get_settings(self):
        with self.lock:
            s = dict(self.settings)
        s["watcher"] = self.watcher.get_status()
        return s

    def update_settings(self, updates):
        if not isinstance(updates, dict):
            raise FeedbackError("Ungültige Einstellungen.")
        with self.lock:
            if updates.get("live_mode") and self.connecting_account:
                raise FeedbackError("Bitte warte, bis die Verbindung des Postfachs abgeschlossen ist, bevor du den Live-Modus einschaltest.", 409)
            if "live_mode" in updates:
                self.settings["live_mode"] = bool(updates["live_mode"])
            if "sort_inbox" in updates:
                self.settings["sort_inbox"] = bool(updates["sort_inbox"])
            try:
                save_report(self.settings_file, self.settings)
            except OSError as err:
                raise FeedbackError("Einstellungen konnten nicht gespeichert werden.", 503) from err
        if self.settings.get("live_mode"):
            self.watcher.trigger_now()
        return self.get_settings()

    def gpu_status(self):
        config_path = self.directory / "gpu-backend.json"
        url = self.laya_url
        if config_path.is_file():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(cfg, dict) and cfg.get("url"):
                    url = cfg["url"]
            except Exception:
                pass

        endpoint = urlsplit(url)
        t0 = time.time()
        try:
            with requests.Session() as probe:
                probe.trust_env = False
                health_url = endpoint._replace(path="/health").geturl()
                resp = probe.get(health_url, timeout=2, allow_redirects=False)
                latency_ms = round((time.time() - t0) * 1000, 1)
                if resp.status_code == 200:
                    data = resp.json()
                    device = data.get("device", "cpu")
                    is_cuda = device in ("cuda", "cuda:0")
                    return {
                        "online": True,
                        "device": device,
                        "is_gpu": is_cuda,
                        "model": "Laya multilingual (RTX 4070 Ti)" if is_cuda else "Laya multilingual (CPU)",
                        "host": "Hauptrechner (NVIDIA RTX 4070 Ti)" if is_cuda else "Host",
                        "latency_ms": latency_ms,
                        "loaded": data.get("loaded", []),
                    }
        except Exception:
            pass

        return {
            "online": False,
            "device": "cpu",
            "is_gpu": False,
            "model": "Fallback Mac mini CPU",
            "host": "Mac mini",
            "latency_ms": None,
            "loaded": [],
        }

    def report(self):
        if self.report_path is None:
            return {"created_utc": None, "messages": [], "limitations": []}
        return json.loads(self.report_path.read_text(encoding="utf-8"))

    def review_key(self, row):
        account = row.get("account_email", self.account).casefold()
        message_id = row["message_id"]
        # Keep existing primary-account reviews addressable without rewriting data.
        return message_id if account == self.account.casefold() else account + ":" + message_id

    def seen_path(self, account):
        suffix = hashlib.sha256(account.casefold().encode()).hexdigest()[:16]
        return self.directory / ("seen-message-ids-" + suffix + ".json")

    def seen_ids(self, account):
        seen = set()
        for key, item in self.feedback["items"].items():
            owner = item.get("account_email") or (key.rsplit(":", 1)[0] if ":" in key else self.account)
            if owner.casefold() == account.casefold():
                seen.add(item.get("message_id") or key.rsplit(":", 1)[-1])
        # The latest attempt wins: failed reads/labels must remain eligible for retry,
        # even when an older run or a preserved review exists for the same message.
        for path in sorted(self.directory.glob("pilot-*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name)):
            previous = json.loads(path.read_text(encoding="utf-8"))
            for row in previous["messages"]:
                owner = row.get("account_email") or previous.get("account_email") or self.account
                if row.get("message_id") and owner.casefold() == account.casefold():
                    if row.get("status") == "suggestion" and not row.get("label_error"):
                        seen.add(row["message_id"])
                    else:
                        seen.discard(row["message_id"])
        return seen

    def review_report(self):
        # Caller holds the same lock used for report switches and feedback writes.
        report = for_review(self.report())
        report["report_id"] = self.report_path.name if self.report_path else None
        reviewed = skipped = 0
        for row in report["messages"]:
            fingerprint = snapshot_id(row, report.get("input_version"))
            row.update(snapshot_id=fingerprint, training_ready=fingerprint is not None, review=None)
            item = self.feedback["items"].get(self.review_key(row)) if row.get("message_id") else None
            if item and item.get("snapshot_id") == fingerprint:
                row["review"] = {name: item[name] for name in ("action", "labels", "updated_utc")}
                row["review"]["training_eligible"] = training_eligible(item)
                row["needs_review"] = False
                reviewed += item["action"] in {"accepted", "corrected"}
                skipped += item["action"] == "skipped"
            elif item and fingerprint is not None:
                row["review_reasons"].append("Die frühere Bewertung gilt für eine andere Texteingabe; bitte erneut prüfen.")
                row["needs_review"] = True
        distribution = {"inhalt": {k: 0 for k in sorted(LABELS["inhalt"])},
                        "wichtig": {k: 0 for k in sorted(LABELS["wichtig"])},
                        "spamverdacht": {k: 0 for k in sorted(LABELS["spamverdacht"])}}
        for item in self.feedback["items"].values():
            if training_eligible(item):
                labels = item.get("labels") or {}
                for cat in LABELS:
                    val = labels.get(cat)
                    if val in distribution[cat]:
                        distribution[cat][val] += 1
        report["summary"] = {"reviewed": reviewed, "skipped": skipped,
                             "pending": len(report["messages"]) - reviewed - skipped,
                             "training_examples": sum(training_eligible(item) for item in self.feedback["items"].values()),
                             "distribution": distribution}
        return report

    def apply_current_labels(self):
        with self.lock:
            if self.state["phase"] == "running" or self.applying_labels or self.connecting_account:
                raise FeedbackError("Bitte warten, bis der aktuelle Lauf beendet ist.", 409)
            target_path = self.report_path
            if not target_path or not target_path.is_file():
                raise FeedbackError("Keine Mails zum Labeln vorhanden.", 404)
            try:
                report = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception as err:
                raise FeedbackError("Bericht konnte nicht gelesen werden.", 500) from err

            messages = report.get("messages", [])
            if not messages:
                return {"status": "ok", "applied_count": 0, "total_messages": 0, "errors": []}

            input_version = report.get("input_version")
            self.applying_labels = True

        try:
            try:
                try:
                    from labeler import get_or_create_labels, apply_labels
                except ImportError:
                    from mail.labeler import get_or_create_labels, apply_labels
            except ImportError as err:
                raise FeedbackError("Labeler-Modul nicht verfuegbar.", 500) from err

            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import AuthorizedSession
            except ImportError as err:
                raise FeedbackError("Google Auth nicht verfuegbar.", 500) from err

            grouped = {}
            errors = []
            known_accounts = self.accounts_mgr.list_accounts()
            for msg in messages:
                email = msg.get("account_email") or report.get("account_email")
                if not email and len(known_accounts) == 1 and known_accounts[0]["email"].casefold() == self.account.casefold():
                    email = self.account  # Unambiguous reports from before multi-account support.
                if not email:
                    errors.append("Bei einer Nachricht fehlt die eindeutige Kontozuordnung. Bitte das Postfach erneut prüfen.")
                    continue
                grouped.setdefault(email, []).append(msg)

            applied_count = 0

            for email, account_messages in grouped.items():
                token_path = self.accounts_mgr.get_token_path(email)
                if not token_path or not Path(token_path).is_file():
                    errors.append(f"Token fuer {email} nicht gefunden.")
                    continue

                try:
                    creds = Credentials.from_authorized_user_file(str(token_path))
                    scopes = getattr(creds, "scopes", None)
                    if scopes is not None and "https://www.googleapis.com/auth/gmail.modify" not in scopes:
                        errors.append(
                            f"Konto {email}: Token verfügt nur über Lesezugriff (gmail.readonly). "
                            "Für das Setzen von Labels wird die Berechtigung 'gmail.modify' benötigt. "
                            "Bitte den Token mit '--modify' neu autorisieren."
                        )
                        continue

                    with AuthorizedSession(creds) as session:
                        session.trust_env = False
                        profile = gmail_get(session, "/profile")
                        if str(profile.get("emailAddress", "")).casefold() != email.casefold():
                            errors.append(f"Die Google-Anmeldung gehört nicht zu {email}; keine Labels geändert.")
                            continue
                        label_cache = get_or_create_labels(session)
                        for msg in account_messages:
                            msg_id = msg.get("message_id")
                            if not msg_id:
                                continue

                            fingerprint = snapshot_id(msg, input_version)
                            effective = None
                            item = self.feedback["items"].get(self.review_key({**msg, "account_email": email}))
                            if (item
                                    and item.get("action") in {"accepted", "corrected"}
                                    and fingerprint is not None
                                    and item.get("snapshot_id") == fingerprint
                                    and valid_labels(item.get("labels"))):
                                effective = item["labels"]
                            elif msg.get("status") == "suggestion" and valid_labels(msg.get("suggestions")):
                                effective = msg["suggestions"]

                            if not effective:
                                continue

                            applied = apply_labels(session, msg_id, effective, label_cache, archive=False)
                            msg["applied_labels"] = applied
                            msg.pop("label_error", None)
                            applied_count += 1
                except Exception:
                    errors.append(f"Labels für {email} konnten nicht vollständig aktualisiert werden. Bitte Anmeldung und Verbindung prüfen.")

            with self.lock:
                if self.report_path != target_path:
                    errors.append("Der Berichtspfad hat sich waehrend des Labelns geaendert; Bericht nicht ueberschrieben.")
                else:
                    try:
                        save_report(target_path, report)
                    except Exception:
                        errors.append("Bericht konnte nicht gespeichert werden.")

            return {
                "status": "ok",
                "applied_count": applied_count,
                "total_messages": len(messages),
                "errors": errors,
            }
        finally:
            with self.lock:
                self.applying_labels = False


    def save_feedback(self, data):
        if (not isinstance(data, dict) or not isinstance(data.get("action"), str)
                or data["action"] not in {"accepted", "corrected", "skipped"}):
            raise FeedbackError("Bitte eine gültige Bewertungsaktion wählen.")
        with self.lock:
            report = self.report()
            if not self.report_path or data.get("report_id") != self.report_path.name:
                raise FeedbackError("Der Bericht hat sich geändert. Bitte die Seite neu laden und erneut prüfen.", 409)
            message_id = data.get("message_id")
            if not isinstance(message_id, str):
                raise FeedbackError("Die Nachricht fehlt.")
            account = data.get("account_email")
            if account is not None and not isinstance(account, str):
                raise FeedbackError("Ungültige Kontozuordnung.")
            matches = [row for row in report["messages"] if row.get("message_id") == message_id
                       and (account is None or row.get("account_email", self.account).casefold() == account.casefold())]
            if len(matches) != 1:
                raise FeedbackError("Diese Nachricht gehört nicht zum angezeigten Bericht.", 409)
            row = matches[0]
            fingerprint = snapshot_id(row, report.get("input_version"))
            if "snapshot_id" not in data or data["snapshot_id"] != fingerprint:
                raise FeedbackError("Die ausgewertete Texteingabe hat sich geändert. Bitte neu laden.", 409)
            action = data["action"]
            labels = None
            if action != "skipped":
                if fingerprint is None:
                    raise FeedbackError("Für diesen älteren Bericht fehlt der ausgewertete Text. Bitte einen neuen Lesetest starten.", 409)
                labels = row.get("suggestions") if action == "accepted" else data.get("labels")
                if action == "accepted" and row.get("status") != "suggestion":
                    raise FeedbackError("Eine fehlgeschlagene Auswertung kann nur manuell korrigiert werden.")
                if not valid_labels(labels):
                    raise FeedbackError("Bitte Inhalt, Wichtigkeit und Spamverdacht vollständig auswählen.")
            timestamp = now()
            item = {"action": action, "labels": labels, "updated_utc": timestamp,
                    "message_id": message_id, "account_email": row.get("account_email", self.account),
                    "snapshot_id": fingerprint, "model_input": row.get("model_input"),
                    "input_version": report.get("input_version"), "source_report_id": self.report_path.name,
                    "original_suggestions": row.get("suggestions"),
                    "truncated": row.get("truncated", False), "content_omitted": row.get("content_omitted", False),
                    "attachment_names": row.get("attachment_names", [])}
            pending = {"revision": timestamp, "items": {**self.feedback["items"], self.review_key(row): item}}
            save_report(self.directory / "feedback.json", pending)
            self.feedback = pending
            return self.review_report()

    def training_export(self, split=None):
        with self.lock:
            examples = []
            for message_id, item in self.feedback["items"].items():
                if not training_eligible(item):
                    continue
                # Deterministic train/val assignment: 80% train, 20% val
                h = int(hashlib.sha256(item["model_input"]["state"].encode()).hexdigest()[:8], 16)
                item_split = "val" if (h % 100) < 20 else "train"
                if split and split in ("train", "val") and item_split != split:
                    continue
                payload = item["model_input"]
                gold = {name: {"type": "choice", "probabilities": {
                    option: float(option == item["labels"][name]) for option in question["criteria"]}}
                    for name, question in payload["questions"].items()}
                metadata = {name: item.get(name) for name in ("snapshot_id", "input_version", "source_report_id",
                            "updated_utc", "action", "original_suggestions", "truncated", "content_omitted", "attachment_names")}
                metadata["message_id"] = item.get("message_id") or message_id.rsplit(":", 1)[-1]
                metadata["account_email"] = item.get("account_email", self.account)
                metadata["split"] = item_split
                examples.append(json.dumps({**payload, "gold": gold, "metadata": metadata}, ensure_ascii=False))
            return ("\n".join(examples) + ("\n" if examples else "")).encode()

    def status(self):
        with self.lock:
            state = dict(self.state)
            state["applying_labels"] = self.applying_labels
            state["feedback_revision"] = self.feedback["revision"]
            if state["phase"] == "running":
                try:
                    progress = json.loads((self.directory / "progress.json").read_text())
                    state.update(processed=state.get("progress_offset", 0) + int(progress["processed"]))
                    if state.get("batch_mode") == "all":
                        state["total"] = (None if progress.get("total") is None else
                                          state.get("progress_offset", 0) + int(progress["total"]))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        state.update(read_only=not (self.settings.get("sort_inbox", False)
                                   or (state["phase"] == "running" and state.get("batch_mode") in {"all", "rescan"})))
        state["accounts"] = self.accounts_mgr.get_state()
        state["settings"] = self.get_settings()
        state["gpu"] = self.gpu_status()
        return state

    def select_backend(self):
        cpu = (self.laya_url, self.directory / "laya-api-key",
               {"name": "Laya multilingual", "device": "cpu", "host": "Mac mini"}, None)
        config_path = self.directory / "gpu-backend.json"
        if not config_path.exists():
            return cpu
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            endpoint = urlsplit(config["url"])
            address = ipaddress.ip_address(endpoint.hostname)
            # Same local-address policy as the pilot; never probe an arbitrary public URL.
            if (endpoint.scheme not in {"http", "https"}
                    or not (address.is_loopback or address in ipaddress.ip_network("10.0.0.0/8")
                            or address in ipaddress.ip_network("172.16.0.0/12")
                            or address in ipaddress.ip_network("192.168.0.0/16"))
                    or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                    or endpoint.path != "/v1/systemone" or config.get("key_file") != "gpu-api-key"):
                raise ValueError("Invalid local GPU configuration")
            key_path = self.directory / "gpu-api-key"
            if key_path.is_symlink() or not key_path.is_file() or not key_path.read_text().strip():
                raise ValueError("GPU key unavailable")
            with requests.Session() as probe:
                probe.trust_env = False
                response = probe.get(endpoint._replace(path="/health").geturl(), timeout=2, allow_redirects=False)
                health = response.json() if response.status_code == 200 else None
            if (isinstance(health, dict) and health.get("status") == "ok"
                    and health.get("device") in {"cuda", "cuda:0"}
                    and isinstance(health.get("loaded"), list) and "multilingual" in health["loaded"]):
                return config["url"], key_path, {"name": "Laya multilingual", "device": "cuda", "host": "Hauptrechner"}, None
        except (OSError, ValueError, TypeError, KeyError, requests.RequestException):
            pass
        return (*cpu[:3], "Hauptrechner nicht bereit; dieser Stapel läuft auf dem Mac mini.")

    def start(self, mode="latest", limit=20, live=False):
        if not isinstance(mode, str) or mode not in {"latest", "next", "rescan", "all"}:
            raise FeedbackError("Bitte eine gültige Ladeaktion wählen.")
        if mode != "all" and (type(limit) is not int or not 1 <= limit <= 100):
            raise FeedbackError("Bitte eine ganze Anzahl zwischen 1 und 100 wählen.")
        if mode == "all":
            limit = None
        with self.lock:
            if self.state["phase"] == "running" or self.applying_labels or self.connecting_account:
                return False
            targets = self.accounts_mgr.get_active_targets()
            if not targets:
                raise FeedbackError("Kein ausgewähltes, aktives Postfach mit Anmeldung verfügbar.", 409)
            if mode == "next":
                try:
                    for target in targets:
                        save_report(self.seen_path(target["email"]), sorted(self.seen_ids(target["email"])))
                except (ValueError, TypeError, KeyError) as error:
                    raise FeedbackError("Die bisherigen Berichte konnten nicht gelesen werden. Der nächste Stapel wurde nicht gestartet.", 503) from error
            (self.directory / "stop-request.json").unlink(missing_ok=True)
            pending = dict(self.state, phase="running", processed=0, total=limit, started_at=now(),
                           finished_at=None, error=None, batch_mode=mode, requested_limit=limit, progress_offset=0,
                           stop_requested=False, accounts_total=len(targets), account_index=1)
            save_report(self.directory / "progress.json", {"processed": 0, "total": limit})
            save_report(self.directory / "status.json", pending)
            self.state = pending
            self._run_targets = targets
            self._run_sort = bool(self.settings.get("sort_inbox", False)) or mode in {"rescan", "all"}
            self._run_live = live
            try:
                threading.Thread(target=self.run, args=(mode, limit), daemon=True).start()
            except RuntimeError as error:
                self.state.update(phase="error", finished_at=now(), error="Der Lesetest konnte nicht gestartet werden.")
                save_report(self.directory / "status.json", self.state)
                raise OSError("Worker unavailable") from error
        return True

    def stop_run(self):
        with self.lock:
            if self.state["phase"] != "running" or self.state.get("batch_mode") != "all":
                raise FeedbackError("Es läuft gerade keine vollständige Posteingang-Sortierung.", 409)
            save_report(self.directory / "stop-request.json", {"requested_at": now()})
            self.state["stop_requested"] = True
            save_report(self.directory / "status.json", self.state)
        return {"status": "stopping"}

    def run(self, mode="latest", limit=20):
        # No dashboard lock is held while probing the optional GPU host.
        laya_url, key_path, model, note = self.select_backend()
        targets = getattr(self, "_run_targets", None) or self.accounts_mgr.get_active_targets()
        sort_inbox = mode in {"rescan", "all"} or getattr(self, "_run_sort", bool(self.settings.get("sort_inbox", False)))
        all_inbox = mode == "all"
        stopped = False
        all_rows = []
        errors = []
        report_path = self.directory / ("pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        report = {"created_utc": now(), "input_version": INPUT_VERSION, "batch_mode": mode,
                  "read_only": not sort_inbox, "applied_labels": sort_inbox, "limit": None if all_inbox else limit,
                  "category_definitions": QUESTIONS, "processing_backend": model}

        for index, target in enumerate(targets):
            if all_inbox and (self.directory / "stop-request.json").exists():
                stopped = True
                break
            if all_inbox:
                quota = 100  # The pilot ignores the bounded-batch limit with --all-inbox.
            else:
                remaining = limit - len(all_rows)
                if remaining <= 0:
                    break
                remaining_accounts = len(targets) - index
                quota = max(1, (remaining + remaining_accounts - 1) // remaining_accounts)
            email = target["email"]
            token = target["token_path"]
            account_report_path = self.directory / f".{report_path.stem}-{index}.json"
            reviews_path = self.directory / f".{report_path.stem}-{index}-reviews.json"
            command = [
                sys.executable, str(Path(__file__).with_name("pilot_gmail.py")),
                "--token", str(token),
                "--laya-key-file", str(key_path),
                "--expected-account", email,
                "--laya-url", laya_url,
                "--output", str(account_report_path),
                "--progress", str(self.directory / "progress.json"),
                "--limit", str(quota),
            ]
            if mode == "next":
                command.extend(["--exclude-ids-file", str(self.seen_path(email))])
            if sort_inbox:
                command.append("--apply-labels")
                command.extend(["--reviews-file", str(reviews_path)])
            if all_inbox:
                command.extend(["--all-inbox", "--stop-file", str(self.directory / "stop-request.json")])

            try:
                with self.lock:
                    self.state.update(model=model, processing_note=note, progress_offset=len(all_rows),
                                      account_index=index + 1, accounts_total=len(targets))
                    save_report(self.directory / "progress.json", {"processed": 0, "total": None if all_inbox else quota})
                    if sort_inbox:
                        reviews = {}
                        for key, item in self.feedback["items"].items():
                            owner = item.get("account_email") or (key.rsplit(":", 1)[0] if ":" in key else self.account)
                            if owner.casefold() == email.casefold() and training_eligible(item):
                                message_id = item.get("message_id") or key.rsplit(":", 1)[-1]
                                reviews[message_id] = {name: item[name] for name in ("action", "labels", "snapshot_id")}
                        save_report(reviews_path, reviews)
                    save_report(self.directory / "status.json", self.state)
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=None if all_inbox else max(900, quota * 20))
                pilot_error = None
                if result.returncode:
                    pilot_error = pilot_failure_message(getattr(result, "stderr", None))
                    errors.append(pilot_error or "Ein Postfach konnte nicht vollständig ausgewertet oder sortiert werden.")
                if account_report_path.exists():
                    rep = json.loads(account_report_path.read_text(encoding="utf-8"))
                    report["limitations"] = rep.get("limitations", [])
                    for msg in rep.get("messages", []):
                        msg["account_email"] = email
                        all_rows.append(msg)
                    if rep.get("status") == "error":
                        errors.append("Ein Postfach meldet Fehler; bitte die Nachrichtenhinweise prüfen.")
                    stopped = stopped or rep.get("status") == "stopped"
                    account_report_path.unlink(missing_ok=True)
                elif not pilot_error:
                    errors.append("Für ein Postfach wurde kein Auswertungsbericht erstellt.")
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
                errors.append("Ein Postfachlauf wurde abgebrochen oder lieferte einen ungültigen Bericht.")
            finally:
                try:
                    reviews_path.unlink(missing_ok=True)
                except OSError:
                    errors.append("Die temporären Bewertungen konnten nicht aufgeräumt werden.")
            if all_inbox and (stopped or errors):
                break

        with self.lock:
            if not targets:
                errors.append("Kein aktives Postfach verfügbar.")
            if any(row.get("status") == "error" or row.get("label_error") for row in all_rows):
                errors.append("Einzelne Nachrichten konnten nicht vollständig verarbeitet werden.")
            error = " ".join(dict.fromkeys(errors)) or None
            count = len(all_rows)
            stopped = stopped or (all_inbox and self.state.get("stop_requested", False))
            phase = "error" if error else "stopped" if stopped else "complete"
            report.update(status=phase, messages=all_rows)
            try:
                if count or (not error and not stopped and not getattr(self, "_run_live", False)):
                    save_report(report_path, report)
                    self.report_path = report_path
            except OSError:
                error = "Der neue Auswertungsbericht konnte nicht gespeichert werden."

            self.state.update(phase="error" if error else phase, processed=count,
                              total=count, finished_at=now(), error=error, progress_offset=0)
            try:
                save_report(self.directory / "status.json", self.state)
            except OSError:
                self.state.update(phase="error", error="Der Abschlussstatus konnte nicht gespeichert werden.")

    def cookie(self):
        value = str(int(time.time()) + 43200) + "." + secrets.token_hex(16)
        return value + "." + hmac.new(self.key.encode(), value.encode(), hashlib.sha256).hexdigest()

    def authorized(self, header):
        try:
            cookies = SimpleCookie()
            cookies.load(header or "")
            value, signature = cookies["laya_session"].value.rsplit(".", 1)
            expected = hmac.new(self.key.encode(), value.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature.encode(), expected.encode()) and int(value.split(".", 1)[0]) > time.time()
        except (CookieError, KeyError, ValueError):
            return False


LOGIN = '''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laya Mail · Anmeldung</title><style>
body{font:16px/1.6 system-ui;background:#f3f5f4;color:#17272a;margin:0;display:grid;min-height:100vh;place-items:center}
main{max-width:430px;padding:36px;background:white;border:1px solid #dae3e0;border-radius:18px;margin:20px}
h1{font-size:32px;margin:0 0 10px}p{color:#506263}label{display:block;font-weight:650;margin-top:24px}
input,button{box-sizing:border-box;width:100%;padding:13px;border-radius:8px;font:inherit;margin-top:8px}
input{border:1px solid #a6b7b3}button{background:#126e64;border:0;color:white;cursor:pointer}small{display:block;margin-top:18px;color:#506263}
</style><main><h1>Laya Mail</h1><p>Deine lokale Übersicht für Gmail-Sortiervorschläge.</p>
<form method="post" action="/login"><label for="code">Zugangscode</label><input id="code" name="code" type="password" autocomplete="off" required autofocus>
<button>Anmelden</button></form><small>Der Zugangscode liegt in deiner lokalen Zugangsdatei. Diese Oberfläche verändert keine E-Mails.</small></main></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, app, origin, **kwargs):
        self.app = app
        self.origin = origin
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass  # Request paths can contain user-provided data; no request logging.

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def respond(self, code, body=b"", content_type="application/json; charset=utf-8", cookie=None, location=None, download=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        # Keep Origin on same-site form POSTs so the login passes CSRF checks.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        if location:
            self.send_header("Location", location)
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def json(self, code, value):
        self.respond(code, json.dumps(value, ensure_ascii=False).encode())

    def valid_host(self):
        return self.headers.get("Host") == urlsplit(self.origin).netloc

    def do_GET(self):
        if not self.valid_host():
            self.json(400, {"error": "Ungültige Serveradresse."})
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self.json(200, {"status": "ok"})
            return
        if not self.app.authorized(self.headers.get("Cookie")):
            if path == "/":
                self.respond(200, LOGIN.encode(), "text/html; charset=utf-8")
            else:
                self.json(401, {"error": "Bitte anmelden."})
            return
        if path == "/api/status":
            self.json(200, self.app.status())
        elif path == "/api/gpu-status":
            self.json(200, self.app.gpu_status())
        elif path == "/api/accounts":
            with self.app.lock:
                accounts = self.app.accounts_mgr.get_state()
            self.json(200, accounts)
        elif path == "/api/accounts/setup":
            self.json(200, account_setup.setup_info(self.app.directory))
        elif path == "/api/settings":
            self.json(200, self.app.get_settings())
        elif path == "/api/report":
            with self.app.lock:
                report = self.app.review_report()
            self.json(200, report)
        elif path == "/api/training-export":
            query = parse_qs(urlsplit(self.path).query)
            split = query.get("split", [None])[0]
            if split not in {"train", "val"}:
                split = None
            filename = f"laya-mail-{split}.jsonl" if split else "laya-mail-training.jsonl"
            self.respond(200, self.app.training_export(split=split), "application/x-ndjson; charset=utf-8", download=filename)
        elif path in {"/", "/app.js", "/style.css", "/accounts/new", "/onboarding.js", "/onboarding.css"}:
            filename, kind = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                              "/style.css": ("style.css", "text/css"), "/accounts/new": ("onboarding.html", "text/html"),
                              "/onboarding.js": ("onboarding.js", "text/javascript"), "/onboarding.css": ("onboarding.css", "text/css")}[path]
            self.respond(200, (Path(__file__).parent / "web" / filename).read_bytes(), kind + "; charset=utf-8")
        else:
            self.json(404, {"error": "Nicht gefunden."})

    def do_POST(self):
        # Host checking blocks DNS rebinding; Origin checking prevents cross-site actions.
        if not self.valid_host() or self.headers.get("Origin") != self.origin:
            self.json(403, {"error": "Diese Anfrage stammt nicht von der lokalen Oberfläche."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        path = urlsplit(self.path).path
        maximum = account_setup.MAX_UPLOAD if path == "/api/accounts/connect" else 4096
        if not 0 <= length <= maximum:
            self.json(400, {"error": "Ungültige Anfrage."})
            return
        data = self.rfile.read(length)
        if path == "/login":
            code = parse_qs(data.decode("utf-8", errors="replace")).get("code", [""])[0]
            if hmac.compare_digest(code.encode(), self.app.key.encode()):
                cookie = "laya_session=" + self.app.cookie() + "; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200"
                self.respond(303, cookie=cookie, location="/")
            else:
                self.json(401, {"error": "Der Zugangscode stimmt nicht. Bitte zurückgehen und erneut versuchen."})
            return
        if not self.app.authorized(self.headers.get("Cookie")):
            self.json(401, {"error": "Bitte anmelden."})
        elif path == "/api/logout":
            self.respond(200, b"{}", cookie="laya_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
        elif path == "/api/feedback":
            try:
                value = json.loads(data)
                self.json(200, self.app.save_feedback(value))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Die Bewertung ist kein gültiges JSON."})
            except FeedbackError as error:
                self.json(error.status, {"error": str(error)})
            except OSError:
                self.json(503, {"error": "Die Bewertung konnte nicht gespeichert werden. Bitte erneut versuchen."})
        elif path == "/api/settings":
            try:
                value = json.loads(data or b"{}")
                self.json(200, self.app.update_settings(value))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Die Einstellungen sind kein gültiges JSON."})
            except FeedbackError as error:
                self.json(error.status, {"error": str(error)})
        elif path in {"/api/accounts/setup", "/api/accounts/connect"}:
            try:
                value = json.loads(data or b"{}")
                result = self.app.account_setup(value) if path.endswith("/setup") else self.app.connect_account(value)
                self.json(200, result)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Bitte eine gültige JSON-Anmeldedatei auswählen."})
            except AccountError as error:
                self.json(error.status, {"error": str(error)})
            except OSError:
                self.json(503, {"error": "Das Konto konnte nicht gespeichert werden. Bitte erneut versuchen."})
        elif path == "/api/accounts/select":
            try:
                value = json.loads(data or b"{}")
                if not isinstance(value, dict):
                    raise AccountError("Ungültige Kontoauswahl.")
                self.json(200, self.app.change_account("select", value.get("selected", "all")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Ungültiges JSON."})
            except AccountError as error:
                self.json(error.status, {"error": str(error)})
        elif path == "/api/accounts/add":
            try:
                value = json.loads(data or b"{}")
                if not isinstance(value, dict):
                    raise AccountError("Ungültige Kontodaten.")
                self.json(200, self.app.change_account("add_account", value.get("email")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Ungültiges JSON."})
            except AccountError as error:
                self.json(error.status, {"error": str(error)})
        elif path == "/api/accounts/toggle":
            try:
                value = json.loads(data or b"{}")
                if not isinstance(value, dict):
                    raise AccountError("Ungültige Kontodaten.")
                self.json(200, self.app.change_account("toggle_account", value.get("id"), value.get("enabled")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Ungültiges JSON."})
            except AccountError as error:
                self.json(error.status, {"error": str(error)})
        elif path == "/api/accounts/remove":
            try:
                value = json.loads(data or b"{}")
                if not isinstance(value, dict):
                    raise AccountError("Ungültige Kontodaten.")
                self.json(200, self.app.change_account("remove_account", value.get("id")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Ungültiges JSON."})
            except AccountError as error:
                self.json(error.status, {"error": str(error)})
        elif path == "/api/labels/apply-current":
            try:
                self.json(200, self.app.apply_current_labels())
            except FeedbackError as error:
                self.json(error.status, {"error": str(error)})
            except Exception:
                self.json(500, {"error": "Labels konnten nicht aktualisiert werden."})
        elif path == "/api/run/stop":
            try:
                self.json(200, self.app.stop_run())
            except FeedbackError as error:
                self.json(error.status, {"error": str(error)})
            except OSError:
                self.json(503, {"error": "Der Stopp konnte nicht gespeichert werden. Bitte erneut versuchen."})
        elif path == "/api/run":
            try:
                value = json.loads(data or b"{}")
                if not isinstance(value, dict):
                    raise FeedbackError("Ungültige Ladeanfrage.")
                if self.app.start(value.get("mode", "latest"), value.get("limit", 20)):
                    self.json(202, {"status": "started"})
                else:
                    self.json(409, {"error": "Ein Lesetest läuft bereits."})
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.json(400, {"error": "Die Ladeanfrage ist kein gültiges JSON."})
            except FeedbackError as error:
                self.json(error.status, {"error": str(error)})
            except OSError:
                self.json(503, {"error": "Der Lesetest kann gerade nicht gestartet werden. Bitte Speicherplatz und Dienst prüfen."})
        else:
            self.json(404, {"error": "Nicht gefunden."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--laya-url", default="http://127.0.0.1:8000/v1/systemone")
    args = parser.parse_args()
    app = Dashboard(args.data, args.account, args.laya_url)
    handler = partial(Handler, app=app, origin=f"http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.serve_forever()
