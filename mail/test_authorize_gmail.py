"""Offline checks: python -m unittest discover -s mail."""

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import authorize_gmail as helper


class FakeCredentials:
    refresh_token = "TEST_REFRESH_SECRET"

    def __init__(self, scopes=None):
        self.granted_scopes = helper.SCOPES if scopes is None else scopes

    def has_scopes(self, scopes):
        return set(scopes) == set(self.granted_scopes)

    def to_json(self):
        return '{"refresh_token":"TEST_REFRESH_SECRET"}'


class FakeFlow:
    client_type = "installed"
    client_config = {
        "project_id": "sample-mail-project",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    def __init__(self, scopes=None):
        self.scopes = helper.SCOPES if scopes is None else scopes

    def run_local_server(self, **kwargs):
        assert kwargs["host"] == kwargs["bind_addr"] == "127.0.0.1"
        assert kwargs["port"] == 0 and kwargs["timeout_seconds"] == 600
        assert kwargs["open_browser"] is False
        assert kwargs["access_type"] == "offline"
        assert kwargs["include_granted_scopes"] == "false"
        assert logging.root.manager.disable >= logging.CRITICAL
        logging.getLogger("google_auth_oauthlib.flow").critical("CALLBACK_SECRET")
        print(kwargs["authorization_prompt_message"].format(url="https://accounts.google.com/fake"))
        return FakeCredentials(self.scopes)


class AuthorizationCheck(unittest.TestCase):
    def test_private_read_only_flow_and_redacted_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.json"
            output = io.StringIO()
            errors = io.StringIO()
            args = ["--client", "dummy.json", "--token", str(token), "--project-id", "sample-mail-project"]
            with patch.object(helper.InstalledAppFlow, "from_client_secrets_file", side_effect=lambda f, scopes, autogenerate_code_verifier: FakeFlow(scopes)) as factory:
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(helper.main(args), 0)
                factory.assert_called_once_with(
                    "dummy.json", scopes=helper.SCOPES, autogenerate_code_verifier=True
                )
            self.assertEqual(token.read_text(), FakeCredentials().to_json())
            if os.name != "nt":
                self.assertEqual(token.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(directory).iterdir()), [token])
            self.assertIn("https://accounts.google.com/fake", output.getvalue())
            self.assertNotIn("SECRET", output.getvalue() + errors.getvalue())

            # A repeat must preserve the existing token and never start OAuth.
            with patch.object(helper.InstalledAppFlow, "from_client_secrets_file") as factory:
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(helper.main(args), 1)
                factory.assert_not_called()
            self.assertEqual(token.read_text(), FakeCredentials().to_json())

            # A client from another project must not open an authorization flow.
            rejected_token = token.with_name("wrong-project.json")
            with patch.object(helper.InstalledAppFlow, "from_client_secrets_file", return_value=FakeFlow()), \
                    patch.object(FakeFlow, "run_local_server") as local_server:
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(helper.main(["--client", "dummy.json", "--token", str(rejected_token),
                                                  "--project-id", "another-mail-project"]), 1)
                local_server.assert_not_called()
            self.assertFalse(rejected_token.exists())

            with patch.object(helper.InstalledAppFlow, "from_client_secrets_file", side_effect=ValueError("CLIENT_SECRET")):
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(helper.main(["--client", "dummy.json", "--token", str(token.with_name("other.json")),
                                                  "--project-id", "sample-mail-project"]), 1)
            self.assertNotIn("SECRET", output.getvalue() + errors.getvalue())
            self.assertFalse(token.with_name("other.json").exists())

    def test_modify_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.json"
            output = io.StringIO()
            errors = io.StringIO()
            args = ["--client", "dummy.json", "--token", str(token), "--project-id", "sample-mail-project", "--modify"]
            with patch.object(helper.InstalledAppFlow, "from_client_secrets_file", side_effect=lambda f, scopes, autogenerate_code_verifier: FakeFlow(scopes)) as factory:
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(helper.main(args), 0)
                factory.assert_called_once_with(
                    "dummy.json", scopes=helper.MODIFY_SCOPES, autogenerate_code_verifier=True
                )
            self.assertIn("Schreibzugriff", output.getvalue())
            self.assertEqual(token.read_text(), FakeCredentials(helper.MODIFY_SCOPES).to_json())


if __name__ == "__main__":
    unittest.main()

