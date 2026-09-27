"""Bounded retries for Gmail reads and idempotent message-label updates."""
from datetime import timezone
from email.utils import parsedate_to_datetime
import random
import re
import time

import requests


MAX_ATTEMPTS = 7
MAX_DELAY = 64
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
RATE_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}
MODIFY_URL = re.compile(r"https://gmail\.googleapis\.com/gmail/v1/users/me/messages/[^/?#]+/modify")


def error_code(response):
    """Expose only fixed codes; never return Google's error text or metadata."""
    status = response.status_code
    if status == 429:
        return "gmail_rate_limited"
    if status == 403:
        try:
            data = response.json()
            error = data.get("error") if isinstance(data, dict) else None
            reasons = error.get("errors", []) if isinstance(error, dict) else []
            if isinstance(reasons, list) and any(isinstance(item, dict)
                    and isinstance(item.get("reason"), str) and item["reason"] in RATE_REASONS for item in reasons):
                return "gmail_rate_limited"
            if isinstance(reasons, list) and any(isinstance(item, dict)
                    and item.get("reason") == "dailyLimitExceeded" for item in reasons):
                return "gmail_quota_exceeded"
        except (ValueError, TypeError):
            pass
        return "gmail_permission_denied"
    if status in {500, 502, 503, 504}:
        return "gmail_temporary_error"
    if status == 404:
        return "gmail_message_not_found"
    if status == 401:
        return "gmail_authentication_failed"
    return "gmail_request_failed"


def retry_after_seconds(response):
    value = getattr(response, "headers", {}).get("Retry-After", "")
    if not isinstance(value, str) or not value.strip():
        return 0
    value = value.strip()
    if value.isdigit():
        return int(value) if len(value) < 6 else MAX_DELAY + 1
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0, when.timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        return 0


def request_with_retry(session, method, url, **kwargs):
    """Return the original response; only GET and messages.modify may retry."""
    method = method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError("gmail_method_invalid")
    retryable = method == "GET" or MODIFY_URL.fullmatch(url) is not None
    send = session.get if method == "GET" else session.post
    kwargs["allow_redirects"] = False
    kwargs.setdefault("timeout", 30)
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = send(url, **kwargs)
        except (requests.Timeout, requests.ConnectionError):
            if not retryable or attempt == MAX_ATTEMPTS - 1:
                raise requests.ConnectionError("gmail_temporary_error") from None
            retry_after = 0
        except requests.RequestException:
            raise requests.RequestException("gmail_request_failed") from None
        else:
            transient = response.status_code in TRANSIENT_STATUS or (
                response.status_code == 403 and error_code(response) == "gmail_rate_limited")
            if not retryable or not transient or attempt == MAX_ATTEMPTS - 1:
                return response
            retry_after = retry_after_seconds(response)
            if retry_after > MAX_DELAY:
                return response  # Never retry earlier than Google's requested delay.
        time.sleep(max(retry_after, min(2 ** attempt + random.random(), MAX_DELAY)))
