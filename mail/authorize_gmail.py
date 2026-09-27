"""Authorize Gmail read-only access; this script never reads messages.

Run on the computer whose browser will complete Google's consent screen.
On Windows, create the token's parent directory with a private ACL first.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile

from google_auth_oauthlib.flow import InstalledAppFlow


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
SCOPES = [READONLY_SCOPE]
MODIFY_SCOPES = [READONLY_SCOPE, MODIFY_SCOPE]


def authorize(client_path: Path, token_path: Path, expected_project: str, modify: bool = False) -> None:
    if not isinstance(expected_project, str) or not expected_project.strip():
        raise ValueError("Expected project ID is required")
    if token_path.exists() or not token_path.parent.is_dir():
        raise ValueError("Token destination unavailable")

    target_scopes = MODIFY_SCOPES if modify else SCOPES
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_path), scopes=target_scopes, autogenerate_code_verifier=True
    )
    config = flow.client_config
    if (
        flow.client_type != "installed"
        or config.get("project_id") != expected_project
        or config.get("auth_uri") not in {
            "https://accounts.google.com/o/oauth2/auth",
            "https://accounts.google.com/o/oauth2/v2/auth",
        }
        or config.get("token_uri") != "https://oauth2.googleapis.com/token"
    ):
        raise ValueError("Expected this project's Google desktop client")

    # OAuth libraries can log callback codes and tokens at debug/info level.
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        credentials = flow.run_local_server(
            host="127.0.0.1",
            bind_addr="127.0.0.1",
            port=0,
            open_browser=False,
            timeout_seconds=600,
            authorization_prompt_message="Google-Freigabe im Browser oeffnen:\n{url}",
            success_message="Google-Antwort empfangen. Bitte Status im Terminal pruefen.",
            access_type="offline",
            prompt="consent",
            include_granted_scopes="false",
        )
    finally:
        logging.disable(previous_logging)

    if not credentials.refresh_token or not credentials.has_scopes(target_scopes):
        raise ValueError("Required offline grant missing")
    if credentials.granted_scopes and set(credentials.granted_scopes) != set(target_scopes):
        raise ValueError("Unexpected granted scopes")

    # mkstemp creates mode 0600 on Unix and inherits the parent ACL on Windows.
    descriptor, temporary_name = tempfile.mkstemp(dir=token_path.parent, prefix=".oauth-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(credentials.to_json())
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, token_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True, type=Path, help="Downloaded desktop OAuth JSON")
    parser.add_argument("--token", required=True, type=Path, help="New token file in a private directory")
    parser.add_argument("--project-id", required=True, help="Expected Google Cloud project ID for this desktop client")
    parser.add_argument("--modify", action="store_true", help="Authorize write access to manage labels (gmail.modify)")
    args = parser.parse_args(argv)
    try:
        authorize(args.client, args.token, args.project_id, modify=args.modify)
    except KeyboardInterrupt:
        print("Freigabe abgebrochen; kein neuer Token gespeichert.", file=sys.stderr)
        return 1
    except Exception:
        # Exceptions can contain authorization codes, URLs or client secrets.
        print(
            "Freigabe fehlgeschlagen oder abgelaufen; kein neuer Token gespeichert. "
            "Projekt-ID, Desktop-Client, Freigabe und freien Token-Zielpfad pruefen.",
            file=sys.stderr,
        )
        return 1
    scope_text = "Gmail-Lese- und Schreibzugriff (Labels)" if args.modify else "Gmail-Lesezugriff"
    print(f"{scope_text} autorisiert; Token gespeichert. Keine E-Mails abgerufen.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
