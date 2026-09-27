"""Gmail label management and automatic sorting for Laya Mail."""
import requests

GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"

# Gmail rejects its localized system-label name "Wichtig" as a custom label.
LABEL_MAP = {
    "rechnung": "Rechnung",
    "werbung": "Werbung",
    "info": "Info",
    "wichtig": "Laya/Wichtig",
    "spamverdacht": "Spamverdacht",
}


class LabelerError(Exception):
    """Raised when label operations fail."""


def managed_label_ids(label_cache):
    """Require a complete, unambiguous cache before reconciling any labels."""
    if not isinstance(label_cache, dict):
        raise LabelerError("Ungueltiger Label-Cache.")
    missing = [key for key in LABEL_MAP if not isinstance(label_cache.get(key), str)
               or not label_cache[key] or label_cache[key] != label_cache[key].strip()]
    if missing:
        raise LabelerError("Label-IDs fehlen fuer: " + ", ".join(LABEL_MAP[key] for key in missing))
    managed = {key: label_cache[key] for key in LABEL_MAP}
    system_ids = {"INBOX", "SPAM", "TRASH", "UNREAD", "STARRED", "IMPORTANT", "SENT", "DRAFT", "CHAT"}
    if (len(set(managed.values())) != len(LABEL_MAP)
            or any(value in system_ids or value.startswith("CATEGORY_") for value in managed.values())):
        raise LabelerError("Ungueltige oder mehrdeutige verwaltete Label-IDs.")
    return managed


def get_or_create_labels(session):
    """Fetch existing Gmail labels and create required Laya sorting labels if missing."""
    try:
        res = session.get(f"{GMAIL}/labels", timeout=15, allow_redirects=False)
        if res.status_code != 200:
            raise LabelerError(f"labels_list_failed_status_{res.status_code}")
        existing = {lbl.get("name", ""): lbl.get("id") for lbl in res.json().get("labels", [])}
    except LabelerError:
        raise
    except Exception:
        raise LabelerError("labels_fetch_failed") from None

    label_cache = {}
    for key, name in LABEL_MAP.items():
        if name in existing:
            label_cache[key] = existing[name]
            continue
        # Create missing label
        payload = {
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
        try:
            create_res = session.post(f"{GMAIL}/labels", json=payload, timeout=15, allow_redirects=False)
            if create_res.status_code in (200, 201):
                label_cache[key] = create_res.json()["id"]
            elif create_res.status_code == 409:  # Already exists
                # Re-fetch ID
                response = session.get(f"{GMAIL}/labels", timeout=15, allow_redirects=False)
                if response.status_code != 200:
                    raise LabelerError(f"labels_list_failed_status_{response.status_code}")
                re_fetch = response.json()
                for lbl in re_fetch.get("labels", []):
                    if lbl.get("name") == name:
                        label_cache[key] = lbl["id"]
                        break
            else:
                if create_res.status_code == 403:
                    raise LabelerError(
                        f"Keine Berechtigung zum Erstellen des Labels '{name}' (Status 403). "
                        "Der Gmail-Token benötigt Schreibrechte (gmail.modify)."
                    )
                raise LabelerError(
                    f"Label '{name}' konnte nicht erstellt werden (Status {create_res.status_code})."
                )
        except LabelerError:
            raise
        except Exception:
            raise LabelerError(f"Fehler beim Erstellen des Labels '{name}'.") from None

    return managed_label_ids(label_cache)


def labels_for_suggestions(suggestions):
    """Determine which label keys should be applied for given model suggestions."""
    choices = {"inhalt": {"rechnung", "werbung", "info", "sonstiges"},
               "wichtig": {"ja", "nein"}, "spamverdacht": {"ja", "nein"}}
    if (not isinstance(suggestions, dict) or set(suggestions) != set(choices)
            or any(not isinstance(suggestions[key], str) or suggestions[key] not in values
                   for key, values in choices.items())):
        raise LabelerError("Vollstaendige, gueltige Klassifizierung erforderlich.")
    keys = []
    inhalt = suggestions.get("inhalt")
    if inhalt in ("rechnung", "werbung", "info"):
        keys.append(inhalt)
    if suggestions.get("wichtig") == "ja":
        keys.append("wichtig")
    if suggestions.get("spamverdacht") == "ja":
        keys.append("spamverdacht")
    return keys


def apply_labels(session, message_id, suggestions, label_cache, archive=False):
    """Reconcile only the five managed labels; preserve INBOX and unrelated labels."""
    if archive is not False:
        raise LabelerError("Archivieren wird nicht unterstuetzt.")
    keys_to_add = labels_for_suggestions(suggestions)
    managed = managed_label_ids(label_cache)
    payload = {"addLabelIds": [managed[key] for key in keys_to_add],
               "removeLabelIds": [value for key, value in managed.items() if key not in keys_to_add]}
    try:
        res = session.post(f"{GMAIL}/messages/{message_id}/modify", json=payload, timeout=15, allow_redirects=False)
    except requests.RequestException:
        raise LabelerError("modify_request_failed") from None
    if res.status_code == 403:
        raise LabelerError(
            "Keine Berechtigung zum Ändern der E-Mail-Labels (Status 403). "
            "Der Gmail-Token benötigt Schreibrechte (gmail.modify)."
        )
    if res.status_code != 200:
        raise LabelerError(f"modify_failed_status_{res.status_code}")

    return [LABEL_MAP[key] for key in keys_to_add]

