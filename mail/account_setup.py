"""Prepare desktop OAuth commands and verify new Gmail grants before storing them."""

from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import secrets
import tempfile

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import requests

from accounts import AccountError, valid_email


SCOPES = {name: "https://www.googleapis.com/auth/gmail." + name for name in ("readonly", "modify")}
TOKEN_URI = "https://oauth2.googleapis.com/token"
PROFILE_URI = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
MAX_UPLOAD = 16 * 1024
HELP_URL = "https://developers.google.com/identity/protocols/oauth2/native-app"


def selection(value):
    if not isinstance(value, dict):
        raise AccountError("Bitte Adresse und Berechtigung angeben.")
    email, permission = value.get("email"), value.get("permission")
    if not valid_email(email) or len(email) > 254:
        raise AccountError("Bitte eine gültige E-Mail-Adresse angeben.")
    if not isinstance(permission, str) or permission not in SCOPES:
        raise AccountError("Bitte Lesen oder Gmail-Labels wählen.")
    return email.strip().lower(), permission


def read_config(directory):
    path = Path(directory) / "account-setup.json"
    if not path.exists():
        return None
    try:
        if path.is_symlink() or path.stat().st_size > MAX_UPLOAD:
            raise ValueError()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError()
        for key in ("python_path", "authorize_path", "client_path", "token_directory"):
            text = value.get(key)
            if (not isinstance(text, str) or not text or len(text) > 4096
                    or any(ord(char) < 32 or char in "‘’“”" for char in text)
                    or not PureWindowsPath(text).is_absolute()):
                raise ValueError()
        if not isinstance(value.get("project_id"), str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", value["project_id"]):
            raise ValueError()
        if value.get("client_id") is not None and not valid_client_id(value["client_id"]):
            raise ValueError()
        return value
    except (OSError, ValueError, TypeError):
        raise AccountError("Die lokale Konto-Einrichtung ist noch nicht korrekt konfiguriert. Bitte die README-Anleitung verwenden.", 503) from None


def setup_info(directory):
    try:
        config = read_config(directory)
    except AccountError as error:
        return {"available": False, "platform": "windows", "help_url": HELP_URL, "reason": str(error)}
    result = {"available": bool(config), "platform": "windows", "help_url": HELP_URL}
    if config:
        result["project_id"] = config["project_id"]
    else:
        result["reason"] = "Die Windows-Pfade fehlen. account-setup.json nach der README-Anleitung einrichten oder den OAuth-Helfer manuell starten."
    return result


def token_filename(email):
    return "token_" + hashlib.sha256(email.encode()).hexdigest()[:16] + ".json"


def require_new_account(manager, email):
    filename = token_filename(email)
    if any(account["email"].casefold() == email.casefold()
           or account.get("token_file", "").casefold() == filename.casefold()
           for account in manager.list_accounts()):
        raise AccountError("Dieses Konto ist bereits eingerichtet. Die bestehende Anmeldung wird nicht überschrieben.", 409)
    target = manager.directory / filename
    if target.exists() or target.is_symlink():
        raise AccountError("Für dieses Konto liegt bereits eine Anmeldedatei vor. Sie wird nicht überschrieben. Bitte die Wiederherstellung im README-Abschnitt „Ein weiteres Gmail-Konto verbinden“ prüfen.", 409)
    return target


def prepare(directory, email, permission):
    config = read_config(directory)
    if config is None:
        raise AccountError(setup_info(directory)["reason"], 503)
    filename = token_filename(email)[:-5] + "_" + secrets.token_hex(4) + ".json"
    output = str(PureWindowsPath(config["token_directory"]) / filename)
    # PowerShell single quotes preserve literal $, backticks and command substitutions.
    quote = lambda text: "'" + text.replace("'", "''") + "'"
    command = "& " + quote(config["python_path"]) + " " + quote(config["authorize_path"])
    for option, value in (("--client", config["client_path"]), ("--token", output), ("--project-id", config["project_id"])):
        command += " " + option + " " + quote(value)
    if permission == "modify":
        command += " --modify"
    return {"email": email, "permission": permission, "command": command,
            "token_filename": filename, "project_id": config["project_id"],
            "console_url": "https://console.cloud.google.com/auth/audience?project=" + config["project_id"]}


def valid_client_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+\.apps\.googleusercontent\.com", value) is not None


def verify_grant(token, email, permission, expected_client=None):
    allowed_fields = {"type", "token", "refresh_token", "token_uri", "client_id", "client_secret", "scopes",
                      "expiry", "rapt_token", "universe_domain", "account"}
    if not isinstance(token, dict) or set(token) - allowed_fields:
        raise AccountError("Bitte die JSON-Anmeldedatei des OAuth-Helfers auswählen.")
    if (len(json.dumps(token).encode()) > MAX_UPLOAD
            or token.get("type", "authorized_user") != "authorized_user"
            or token.get("token_uri") != TOKEN_URI
            or token.get("universe_domain", "googleapis.com") != "googleapis.com"
            or not valid_client_id(token.get("client_id"))):
        raise AccountError("Die Anmeldedatei enthält keine unterstützte Google-Anmeldung.")
    if expected_client and token["client_id"] != expected_client:
        raise AccountError("Die Anmeldung gehört nicht zum eingerichteten Google-Desktop-Client.")
    for key in ("refresh_token", "client_secret"):
        if not isinstance(token.get(key), str) or not token[key].strip() or len(token[key]) > 8192:
            raise AccountError("Die Anmeldung enthält keine vollständige dauerhafte Google-Freigabe.")
    claimed = token.get("scopes")
    if (not isinstance(claimed, list) or not claimed or any(not isinstance(scope, str) for scope in claimed)
            or not set(claimed).issubset(SCOPES.values())):
        raise AccountError("Die Anmeldedatei darf ausschließlich Gmail-Lese- oder Label-Berechtigungen enthalten.")
    # Ignore uploaded access token, expiry, account and scopes as proof. A fresh
    # Google response must establish the real grant; no uploaded endpoint is used.
    info = {key: token[key] for key in ("refresh_token", "client_id", "client_secret")}
    info["token_uri"] = TOKEN_URI
    try:
        credentials = Credentials.from_authorized_user_info(info, scopes=[SCOPES[permission]])
        with requests.Session() as session:
            session.trust_env = False
            credentials.refresh(partial(Request(session), timeout=15, allow_redirects=False))
            granted = credentials.granted_scopes
            if not granted:
                response = session.post("https://oauth2.googleapis.com/tokeninfo", data="",
                                        headers={"Authorization": "Bearer " + credentials.token,
                                                 "Content-Type": "application/x-www-form-urlencoded"},
                                        timeout=15, allow_redirects=False)
                if response.status_code != 200:
                    raise AccountError("Google konnte die tatsächlichen Berechtigungen nicht bestätigen. Bitte die Freigabe erneut erstellen.")
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("scope"), str):
                    raise ValueError()
                granted = payload["scope"].split()
            actual = set(granted)
            if (not actual or not actual.issubset(SCOPES.values()) or SCOPES[permission] not in actual
                    or (permission == "readonly" and actual != {SCOPES["readonly"]})):
                raise AccountError("Die bestätigte Google-Freigabe passt nicht zur gewählten Berechtigung. Bitte den passenden OAuth-Befehl erneut ausführen.")
            response = session.get(PROFILE_URI, headers={"Authorization": "Bearer " + credentials.token},
                                   timeout=15, allow_redirects=False)
            if response.status_code != 200:
                raise AccountError("Die Gmail-Verbindung konnte nicht bestätigt werden. Bitte Freigabe und Gmail-API prüfen.")
            profile = response.json()
            if not isinstance(profile, dict) or not isinstance(profile.get("emailAddress"), str):
                raise ValueError()
            if profile["emailAddress"].casefold() != email.casefold():
                raise AccountError("Die Google-Anmeldung gehört zu einer anderen E-Mail-Adresse. Bitte das angegebene Konto bei Google wählen.")
        saved = json.loads(credentials.to_json())
        saved.update(scopes=sorted(actual), account=email)
        return saved
    except AccountError:
        raise
    except Exception:
        # Google/requests exceptions may contain tokens, full URLs or bodies.
        raise AccountError("Die Google-Anmeldung konnte nicht geprüft werden. Bitte Verbindung prüfen oder die Freigabe erneut erstellen.", 502) from None


def store_new_account(manager, email, token):
    target = require_new_account(manager, email)
    directory = manager.directory
    if directory.is_symlink() or not directory.is_dir() or (os.name != "nt" and directory.stat().st_mode & 0o077):
        raise AccountError("Das private Datenverzeichnis ist nicht ausreichend geschützt.", 503)
    previous = deepcopy(manager.data)
    temporary = None
    installed = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=".account-", dir=directory)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(token, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        # A hard link publishes the complete private file without overwriting a
        # concurrent file or symlink. Both names stay on the same filesystem.
        os.link(temporary, target)
        installed = True
        state = manager.add_account(email, target.name, enabled=True)
        return state
    except (OSError, AccountError):
        manager.data = previous
        if installed:
            target.unlink(missing_ok=True)
        raise AccountError("Das Konto konnte nicht gespeichert werden. Bestehende Anmeldungen bleiben erhalten.", 503) from None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
