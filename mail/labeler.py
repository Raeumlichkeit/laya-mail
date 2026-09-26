"""Gmail label management and automatic sorting for Laya Mail."""
import json
import logging
import requests

GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"

# User-approved label names without prefix
LABEL_MAP = {
    "rechnung": "Rechnung",
    "werbung": "Werbung",
    "info": "Info",
    "wichtig": "Wichtig",
    "spamverdacht": "Spamverdacht",
}


class LabelerError(Exception):
    """Raised when label operations fail."""


def get_or_create_labels(session):
    """Fetch existing Gmail labels and create required Laya sorting labels if missing."""
    try:
        res = session.get(f"{GMAIL}/labels", timeout=15)
        if res.status_code != 200:
            raise LabelerError(f"labels_list_failed_status_{res.status_code}")
        existing = {lbl.get("name", "").strip(): lbl.get("id") for lbl in res.json().get("labels", [])}
    except Exception as err:
        raise LabelerError(f"labels_fetch_failed: {err}") from err

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
            create_res = session.post(f"{GMAIL}/labels", json=payload, timeout=15)
            if create_res.status_code in (200, 201):
                label_cache[key] = create_res.json()["id"]
            elif create_res.status_code == 409:  # Already exists
                # Re-fetch ID
                re_fetch = session.get(f"{GMAIL}/labels", timeout=15).json()
                for lbl in re_fetch.get("labels", []):
                    if lbl.get("name") == name:
                        label_cache[key] = lbl["id"]
                        break
            else:
                logging.warning("Failed to create label %s: %s", name, create_res.text)
        except Exception as err:
            logging.warning("Error creating label %s: %s", name, err)

    return label_cache


def labels_for_suggestions(suggestions):
    """Determine which label keys should be applied for given model suggestions."""
    if not isinstance(suggestions, dict):
        return []
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
    """Apply sorting labels to a Gmail message via messages.modify."""
    keys_to_add = labels_for_suggestions(suggestions)
    add_ids = [label_cache[k] for k in keys_to_add if k in label_cache]
    if not add_ids and not archive:
        return []

    payload = {"addLabelIds": add_ids, "removeLabelIds": ["INBOX"] if archive else []}
    res = session.post(f"{GMAIL}/messages/{message_id}/modify", json=payload, timeout=15)
    if res.status_code != 200:
        raise LabelerError(f"modify_failed_status_{res.status_code}")

    return [LABEL_MAP[k] for k in keys_to_add if k in label_cache]
