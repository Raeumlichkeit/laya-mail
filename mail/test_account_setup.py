"""Offline onboarding checks; Google responses and grants are entirely synthetic."""

from functools import partial
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import requests

import account_setup as setup
from accounts import AccountError, AccountManager
from dashboard import Dashboard, FeedbackError, Handler


EMAIL = "new@example.invalid"
CLIENT = "synthetic.apps.googleusercontent.com"
GRANT = {"refresh_token": "synthetic-refresh-secret", "client_id": CLIENT,
         "client_secret": "synthetic-client-secret", "token_uri": setup.TOKEN_URI,
         "scopes": [setup.SCOPES["modify"]]}
CONFIG = {"python_path": "C:\\Tools\\python.exe", "authorize_path": "C:\\Laya Mail\\mail\\authorize_gmail.py",
          "client_path": "C:\\Private\\client.json", "token_directory": "C:\\Private",
          "project_id": "synthetic-project", "client_id": CLIENT}


class GoogleSession:
    def __init__(self, scopes=None, fallback=False, email=EMAIL):
        self.scopes = scopes or setup.SCOPES["modify"]
        self.fallback = fallback
        self.email = email
        self.calls = []
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url == setup.TOKEN_URI:
            result = {"access_token": "synthetic-fresh-access", "expires_in": 3600, "token_type": "Bearer"}
            if not self.fallback:
                result["scope"] = self.scopes
        elif url == "https://oauth2.googleapis.com/tokeninfo":
            result = {"scope": self.scopes, "aud": CLIENT}
        elif url == setup.PROFILE_URI:
            result = {"emailAddress": self.email}
        else:
            raise AssertionError("Unexpected network destination")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(result).encode()
        return response

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


class AccountSetupCheck(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.directory.chmod(0o700)
        (self.directory / "account-setup.json").write_text(json.dumps(CONFIG))

    def test_command_is_private_unique_literal_and_does_not_add_accounts(self):
        config = {**CONFIG, "python_path": "C:\\O'Brien\\$(literal)`file.exe"}
        (self.directory / "account-setup.json").write_text(json.dumps(config))
        info = setup.setup_info(self.directory)
        self.assertEqual(info["available"], True)
        self.assertNotIn("client_path", info)
        self.assertNotIn("client_id", info)
        first = setup.prepare(self.directory, EMAIL, "readonly")
        second = setup.prepare(self.directory, EMAIL, "modify")
        self.assertIn("'C:\\O''Brien\\$(literal)`file.exe'", first["command"])
        self.assertNotEqual(first["token_filename"], second["token_filename"])
        self.assertNotIn("--modify", first["command"])
        self.assertTrue(second["command"].endswith(" --modify"))
        self.assertFalse((self.directory / "accounts.json").exists())
        for path in ("C:\\bad’; Get-Item; #.exe", "C:\\bad\nname.exe", "relative.exe"):
            (self.directory / "account-setup.json").write_text(json.dumps({**CONFIG, "python_path": path}))
            self.assertFalse(setup.setup_info(self.directory)["available"])

    def test_verified_scope_profile_and_tokeninfo_fallback_use_fixed_destinations(self):
        for fallback in (False, True):
            session = GoogleSession(fallback=fallback)
            upload = {**GRANT, "token": "do-not-use-uploaded-token", "expiry": "not-trusted", "account": "wrong"}
            with patch("account_setup.requests.Session", return_value=session):
                verified = setup.verify_grant(upload, EMAIL, "modify", CLIENT)
            self.assertFalse(session.trust_env)
            self.assertEqual(verified["token"], "synthetic-fresh-access")
            self.assertEqual(verified["account"], EMAIL)
            self.assertEqual(verified["scopes"], [setup.SCOPES["modify"]])
            self.assertEqual([item[:2] for item in session.calls],
                             [("POST", setup.TOKEN_URI)] +
                             ([("POST", "https://oauth2.googleapis.com/tokeninfo")] if fallback else []) +
                             [("GET", setup.PROFILE_URI)])
            for _, url, kwargs in session.calls:
                self.assertFalse(kwargs["allow_redirects"])
                self.assertEqual(kwargs["timeout"], 15)
                self.assertNotIn("?", url)
                self.assertNotIn("do-not-use-uploaded-token", str(kwargs))

    def test_forged_scopes_wrong_account_and_untrusted_endpoints_fail_closed(self):
        for scope, email in ((setup.SCOPES["readonly"], EMAIL),
                             (setup.SCOPES["modify"] + " https://www.googleapis.com/auth/drive", EMAIL),
                             (setup.SCOPES["modify"], "other@example.invalid")):
            with patch("account_setup.requests.Session", return_value=GoogleSession(scope, email=email)):
                with self.assertRaises(AccountError):
                    setup.verify_grant(GRANT, EMAIL, "modify", CLIENT)
        with patch("account_setup.requests.Session") as network:
            for token in ({**GRANT, "token_uri": "https://evil.invalid/token"},
                          {**GRANT, "type": "service_account"}, {**GRANT, "scopes": ["forged"]},
                          {**GRANT, "client_id": "other.apps.googleusercontent.com"}):
                with self.assertRaises(AccountError):
                    setup.verify_grant(token, EMAIL, "modify", CLIENT)
            network.assert_not_called()
        with patch("account_setup.requests.Session", side_effect=RuntimeError("PRIVATE TOKEN")):
            with self.assertRaises(AccountError) as error:
                setup.verify_grant(GRANT, EMAIL, "modify", CLIENT)
            self.assertNotIn("PRIVATE", str(error.exception))

    def test_storage_preserves_existing_files_and_rolls_back_failed_account_save(self):
        manager = AccountManager(self.directory)
        before = manager.get_state()
        with patch.object(manager, "_save", side_effect=AccountError("synthetic disk failure")):
            with self.assertRaises(AccountError):
                setup.store_new_account(manager, EMAIL, GRANT)
        self.assertEqual(manager.get_state(), before)
        self.assertFalse((self.directory / setup.token_filename(EMAIL)).exists())
        self.assertFalse(list(self.directory.glob(".account-*")))
        result = setup.store_new_account(manager, EMAIL, GRANT)
        target = manager.get_token_path(EMAIL)
        content = target.read_bytes()
        self.assertEqual(result["selected"], "all")
        self.assertTrue(result["accounts"][0]["enabled"])
        if os.name != "nt":
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(AccountError):
            setup.store_new_account(manager, EMAIL, {"replacement": True})
        self.assertEqual(target.read_bytes(), content)
        manager.remove_account(EMAIL)  # Removing an account intentionally keeps its grant file.
        with self.assertRaises(AccountError) as error:
            setup.require_new_account(manager, EMAIL)
        self.assertEqual(error.exception.status, 409)
        self.assertIn("README", str(error.exception))
        self.assertEqual(target.read_bytes(), content)

    def test_connect_reserves_worker_without_holding_lock_or_mutating_before_verify(self):
        (self.directory / "dashboard-key").write_text("synthetic-dashboard-key-at-least-24")
        with patch("dashboard.LiveWatcher.start"):
            app = Dashboard(self.directory, "", "http://127.0.0.1:8000/v1/systemone")
        request = {"email": EMAIL, "permission": "modify", "token": GRANT}

        def verify(*args):
            self.assertTrue(app.lock.acquire(blocking=False))
            app.lock.release()
            self.assertFalse(app.start())
            with self.assertRaises(AccountError):
                app.change_account("add_account", "other@example.invalid")
            with self.assertRaises(FeedbackError):
                app.apply_current_labels()
            with self.assertRaises(FeedbackError) as error:
                app.update_settings({"live_mode": True, "sort_inbox": True})
            self.assertEqual(error.exception.status, 409)
            self.assertFalse(app.settings["live_mode"])
            self.assertFalse(app.settings["sort_inbox"])
            self.assertEqual(app.accounts_mgr.list_accounts(), [])
            return GRANT

        with patch("account_setup.verify_grant", side_effect=verify):
            result = app.connect_account(request)
        self.assertEqual(result["status"], "connected")
        self.assertFalse(app.connecting_account)
        self.assertEqual(app.state["phase"], "idle")
        app.settings.update(live_mode=True, sort_inbox=True)
        before = (self.directory / "accounts.json").read_bytes()
        with patch("account_setup.verify_grant") as network:
            with self.assertRaises(AccountError) as error:
                app.connect_account({**request, "email": "other@example.invalid"})
            self.assertEqual(error.exception.status, 409)
            self.assertIn("Live-Modus", str(error.exception))
            network.assert_not_called()
        self.assertEqual((self.directory / "accounts.json").read_bytes(), before)
        self.assertEqual(app.accounts_mgr.get_state()["selected"], "all")
        self.assertTrue(app.settings["live_mode"])
        self.assertTrue(app.settings["sort_inbox"])
        app.state["phase"] = "running"
        with patch("account_setup.verify_grant") as network:
            with self.assertRaises(AccountError) as error:
                app.connect_account({**request, "email": "other@example.invalid"})
            self.assertEqual(error.exception.status, 409)
            network.assert_not_called()

    def test_http_auth_origin_size_and_setup_routes(self):
        (self.directory / "dashboard-key").write_text("synthetic-dashboard-key-at-least-24")
        with patch("dashboard.LiveWatcher.start"):
            app = Dashboard(self.directory, "", "http://127.0.0.1:8000/v1/systemone")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        origin = "http://127.0.0.1:" + str(server.server_port)
        server.RequestHandlerClass = partial(Handler, app=app, origin=origin)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with requests.Session() as client:
            client.trust_env = False
            self.assertEqual(client.get(origin + "/api/accounts/setup", timeout=5).status_code, 401)
            self.assertEqual(client.get(origin + "/accounts/new", timeout=5).status_code, 401)
            client.post(origin + "/login", data={"code": app.key}, headers={"Origin": origin}, timeout=5)
            headers = {"Origin": origin}
            self.assertTrue(client.get(origin + "/api/accounts/setup", timeout=5).json()["available"])
            payload = {"email": EMAIL, "permission": "modify"}
            self.assertEqual(client.post(origin + "/api/accounts/setup", json=payload, headers=headers, timeout=5).status_code, 200)
            self.assertEqual(client.post(origin + "/api/accounts/setup", json=[], headers=headers, timeout=5).status_code, 400)
            payload["token"] = {**GRANT, "token": "x" * 5000}
            with patch("account_setup.verify_grant", return_value=GRANT) as verify:
                self.assertEqual(client.post(origin + "/api/accounts/connect", json=payload,
                                             headers={"Origin": "http://evil.invalid"}, timeout=5).status_code, 403)
                self.assertEqual(client.post(origin + "/api/accounts/connect", data="x" * (setup.MAX_UPLOAD + 1),
                                             headers=headers, timeout=5).status_code, 400)
                self.assertEqual(client.post(origin + "/api/accounts/setup", json=payload, headers=headers, timeout=5).status_code, 400)
                verify.assert_not_called()
                response = client.post(origin + "/api/accounts/connect", json=payload, headers=headers, timeout=5)
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("synthetic-client-secret", response.text)
                self.assertNotIn("synthetic-refresh-secret", response.text)


if __name__ == "__main__":
    unittest.main()
