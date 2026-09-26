"""Offline checks: no Torch import, network, private data or running service."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from serve_gpu import ROOT, concrete_ip, load_config, read_key


class ConfigCheck(unittest.TestCase):
    def test_portable_defaults_and_restricted_addresses(self):
        with tempfile.TemporaryDirectory(prefix="laya config with spaces ") as folder:
            config_path = Path(folder) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config["host"], "127.0.0.1")
            self.assertEqual(config["port"], 8002)
            self.assertEqual(config["allowed_clients"], ["127.0.0.1"])
            self.assertEqual(Path(config["model_path"]), ROOT / "runtime/models/multilingual")
            self.assertEqual(Path(config["key_file"]), ROOT / "runtime/private/gpu-api-key")
            for invalid in ("Any", "*", "0.0.0.0", "::", "192.0.2.0/24", "255.255.255.255", "224.0.0.1"):
                config_path.write_text(json.dumps({"allowed_clients": [invalid]}), encoding="utf-8")
                with self.subTest(client=invalid), self.assertRaises(ValueError):
                    load_config(config_path)
            for invalid in (True, 0, 65536, "8002"):
                config_path.write_text(json.dumps({"port": invalid}), encoding="utf-8")
                with self.subTest(port=invalid), self.assertRaises(ValueError):
                    load_config(config_path)
            config_path.write_text(json.dumps({"model_path": "runtime/model with spaces", "allowed_clients": ["192.0.2.10", "::ffff:192.0.2.10"]}), encoding="utf-8-sig")
            config = load_config(config_path)
            self.assertEqual(config["allowed_clients"], ["192.0.2.10"])
            self.assertEqual(Path(config["model_path"]), ROOT / "runtime/model with spaces")
        self.assertNotIn("torch", sys.modules)

    def test_short_keys_are_rejected_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            key_path = Path(folder) / "test-key"
            key_path.write_text("short synthetic key", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_key(key_path)
            key_path.write_text("synthetic-test-key-at-least-24-characters\n", encoding="utf-8")
            self.assertEqual(read_key(key_path), "synthetic-test-key-at-least-24-characters")
        self.assertEqual(concrete_ip("::1"), "::1")


if __name__ == "__main__":
    unittest.main()
