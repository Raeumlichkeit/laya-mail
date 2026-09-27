"""Offline GPU selection checks; no Gmail, Laya or subprocess is contacted."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests

from dashboard import Dashboard
from pilot_gmail import save_report


class BackendCheck(unittest.TestCase):
    def test_gpu_selection_cpu_fallback_and_batch_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "dashboard-key").write_text("synthetic-dashboard-key-for-offline-test")
            (directory / "gpu-api-key").write_text("SYNTHETIC_GPU_SECRET")
            (directory / "token.json").write_text("{}")
            gpu_url = "http://192.168.50.20:8002/v1/systemone"
            cpu_url = "http://192.168.50.10:8000/v1/systemone"
            config = {"url": gpu_url, "key_file": "gpu-api-key"}
            save_report(directory / "gpu-backend.json", config)
            app = Dashboard(directory, "test@example.invalid", cpu_url)
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                self.assertFalse(app.lock.locked())
                save_report(Path(command[command.index("--output") + 1]),
                            {"status": "complete", "messages": [], "batch_mode": "next", "limit": 7})
                return SimpleNamespace(returncode=0)

            with patch("dashboard.requests.Session") as session, patch("dashboard.subprocess.run", side_effect=run):
                probe = session.return_value.__enter__.return_value

                def healthy(url, **kwargs):
                    self.assertFalse(app.lock.locked())
                    self.assertEqual(url, "http://192.168.50.20:8002/health")
                    self.assertEqual(kwargs, {"timeout": 2, "allow_redirects": False})
                    self.assertFalse(probe.trust_env)
                    return SimpleNamespace(status_code=200, json=lambda: {"status": "ok", "device": "cuda", "loaded": ["multilingual"]})

                probe.get.side_effect = healthy
                app.run("next", 7)
                gpu_command = commands[-1]
                self.assertEqual(gpu_command[gpu_command.index("--laya-url") + 1], gpu_url)
                self.assertEqual(gpu_command[gpu_command.index("--laya-key-file") + 1], str(directory / "gpu-api-key"))
                self.assertEqual(gpu_command[gpu_command.index("--limit") + 1], "7")
                self.assertIn("--exclude-ids-file", gpu_command)
                self.assertEqual(app.status()["model"], {"name": "Laya multilingual", "device": "cuda", "host": "Hauptrechner"})
                self.assertEqual(app.report()["processing_backend"], app.status()["model"])
                self.assertEqual(json.loads((directory / "status.json").read_text())["model"], app.status()["model"])
                self.assertIsNone(app.status()["processing_note"])

                for health in ({"status": "ok", "device": "cpu", "loaded": ["multilingual"]},
                               {"status": "ok", "device": "cuda", "loaded": ["english"]}):
                    probe.get.side_effect = None
                    probe.get.return_value = SimpleNamespace(status_code=200, json=lambda: health)
                    self.assertEqual(app.select_backend()[0], cpu_url)
                probe.get.side_effect = requests.ConnectionError("synthetic offline")
                app.run()
                cpu_command = commands[-1]
                self.assertEqual(cpu_command[cpu_command.index("--laya-url") + 1], cpu_url)
                self.assertEqual(cpu_command[cpu_command.index("--laya-key-file") + 1], str(directory / "laya-api-key"))
                self.assertEqual(app.status()["model"]["device"], "cpu")
                self.assertEqual(app.report()["processing_backend"]["host"], "Mac mini")
                self.assertTrue(app.status()["processing_note"])
                self.assertNotIn("SYNTHETIC_GPU_SECRET", json.dumps(app.report()) + json.dumps(app.status()) + repr(commands))

                # A public target or path-traversing key must never even be probed.
                for invalid in ({**config, "url": "https://example.com/v1/systemone"}, {**config, "key_file": "../token.json"}):
                    save_report(directory / "gpu-backend.json", invalid)
                    probe.get.reset_mock()
                    self.assertEqual(app.select_backend()[0], cpu_url)
                    probe.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
