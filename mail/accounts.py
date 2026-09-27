"""Multi-account management for Laya Mail."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_email(value):
    return isinstance(value, str) and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()) is not None


class AccountError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class AccountManager:
    def __init__(self, directory, default_email=None):
        self.directory = Path(directory)
        self.config_path = self.directory / "accounts.json"
        self.default_email = (default_email or "").strip().lower()
        self.data = self._load()

    def _load(self):
        if self.config_path.is_file():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("accounts"), list):
                    return data
            except (OSError, ValueError, TypeError):
                pass

        # Bootstrap initial account from default_email and token.json if present
        accounts = []
        token_path = self.directory / "token.json"
        if self.default_email and valid_email(self.default_email):
            accounts.append({
                "id": self.default_email,
                "email": self.default_email,
                "token_file": "token.json",
                "enabled": True,
                "added_utc": now(),
            })

        initial = {"selected": "all", "accounts": accounts}
        self._save(initial)
        return initial

    def _save(self, data):
        try:
            temp_path = self.config_path.with_name(".accounts-tmp.json")
            temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            if os.name != "nt":
                temp_path.chmod(0o600)
            os.replace(temp_path, self.config_path)
        except OSError as err:
            raise AccountError("Kontodaten konnten nicht gespeichert werden.", 503) from err

    def list_accounts(self):
        accounts = []
        for acc in self.data.get("accounts", []):
            token_path = self.directory / acc.get("token_file", "token.json")
            accounts.append({
                "id": acc["id"],
                "email": acc["email"],
                "token_file": acc.get("token_file", "token.json"),
                "enabled": bool(acc.get("enabled", True)),
                "token_exists": token_path.is_file() and not token_path.is_symlink(),
                "added_utc": acc.get("added_utc"),
            })
        return accounts

    def get_state(self):
        return {
            "selected": self.data.get("selected", "all"),
            "accounts": self.list_accounts(),
        }

    def select(self, target):
        target = (target or "").strip().lower()
        if target != "all" and not any(acc["id"].lower() == target for acc in self.list_accounts()):
            raise AccountError("Das gewählte Konto existiert nicht.", 404)
        self.data["selected"] = target
        self._save(self.data)
        return self.get_state()

    def add_account(self, email, token_filename=None, enabled=True):
        email = (email or "").strip().lower()
        if not valid_email(email):
            raise AccountError("Bitte eine gültige E-Mail-Adresse angeben.")

        if token_filename is None:
            safe_name = re.sub(r"[^a-z0-9_-]", "_", email.split("@")[0])
            token_filename = f"token_{safe_name}.json"

        accounts = [acc for acc in self.data.get("accounts", []) if acc["email"].lower() != email]
        accounts.append({
            "id": email,
            "email": email,
            "token_file": token_filename,
            "enabled": bool(enabled),
            "added_utc": now(),
        })
        self.data["accounts"] = accounts
        self._save(self.data)
        return self.get_state()

    def remove_account(self, account_id):
        account_id = (account_id or "").strip().lower()
        accounts = [acc for acc in self.data.get("accounts", []) if acc["id"].lower() != account_id]
        if len(accounts) == len(self.data.get("accounts", [])):
            raise AccountError("Konto nicht gefunden.", 404)
        self.data["accounts"] = accounts
        if self.data.get("selected", "").lower() == account_id:
            self.data["selected"] = "all"
        self._save(self.data)
        return self.get_state()

    def toggle_account(self, account_id, enabled=None):
        account_id = (account_id or "").strip().lower()
        found = False
        for acc in self.data.get("accounts", []):
            if acc["id"].lower() == account_id:
                acc["enabled"] = not acc.get("enabled", True) if enabled is None else bool(enabled)
                found = True
                break
        if not found:
            raise AccountError("Konto nicht gefunden.", 404)
        self._save(self.data)
        return self.get_state()

    def get_active_targets(self):
        """Returns list of active targets (email, token_path)."""
        selected = self.data.get("selected", "all")
        active = []
        for acc in self.list_accounts():
            if not acc["enabled"] or not acc["token_exists"]:
                continue
            if selected == "all" or acc["id"].lower() == selected.lower():
                token_path = self.directory / acc["token_file"]
                active.append({"email": acc["email"], "token_path": str(token_path)})
        return active

    def get_token_path(self, email):
        """Returns token Path for the given email, or None."""
        email = (email or "").strip().lower()
        for acc in self.list_accounts():
            if acc["email"].lower() == email or acc["id"].lower() == email:
                return self.directory / acc["token_file"]
        return None

