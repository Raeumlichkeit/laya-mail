"""Read Gmail's spam folder as an explicitly separate source of training inputs."""

try:
    from pilot_gmail import (INPUT_VERSION, PilotError, gmail_get, next_entries,
                            prepare_message, valid_message_id)
except ImportError:
    from mail.pilot_gmail import (INPUT_VERSION, PilotError, gmail_get, next_entries,
                                 prepare_message, valid_message_id)


def collect_spam(session, expected_account, exclude_ids, limit=20):
    """Return private inputs and fixed error codes, without writes or inference.

    The caller owns persistence and provenance. Gmail's SPAM label is a weak
    training label, not a human review or independent evaluation ground truth.
    ``processed`` counts attempted message reads; ``total`` is the selected batch,
    not the size of the whole Gmail folder. Profile/list failures raise PilotError.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise PilotError("limit_invalid")
    if not isinstance(expected_account, str) or "@" not in expected_account.strip():
        raise PilotError("expected_account_required")
    if not isinstance(exclude_ids, (set, frozenset)) or not all(valid_message_id(value) for value in exclude_ids):
        raise PilotError("exclude_ids_invalid")
    account = expected_account.strip().casefold()
    profile = gmail_get(session, "/profile")
    if str(profile.get("emailAddress", "")).casefold() != account:
        raise PilotError("account_mismatch")
    entries = next_entries(session, exclude_ids, limit=limit, label_id="SPAM")
    result = {"items": [], "errors": [], "processed": 0, "total": len(entries)}
    for entry in entries:
        result["processed"] += 1
        message_id = entry["id"].lower()
        try:
            message = gmail_get(session, "/messages/" + message_id, format="full")
            if not valid_message_id(message.get("id")) or message["id"].lower() != message_id:
                raise PilotError("gmail_message_id_invalid")
            labels = message.get("labelIds")
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise PilotError("gmail_response_invalid")
            if "SPAM" not in labels or "TRASH" in labels:
                raise PilotError("gmail_spam_label_changed")
            prepared = prepare_message(message)
            result["items"].append({
                "message_id": message_id, "account_email": account,
                "model_input": prepared["model_input"], "input_version": INPUT_VERSION,
            })
        except PilotError as error:
            code = str(error)
            result["errors"].append(code)
            if code in {"gmail_rate_limited", "gmail_quota_exceeded",
                        "gmail_authentication_failed", "gmail_permission_denied"}:
                break
        except Exception:
            result["errors"].append("message_processing_failed")
    return result
