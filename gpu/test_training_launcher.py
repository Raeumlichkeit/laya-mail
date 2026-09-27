"""Offline PowerShell launcher tests using only temporary scripts and model markers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


POWERSHELL = shutil.which("powershell.exe")
STUB = '''
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
args = sys.argv[1:]
stage = {"export_training_data": "export", "evaluate_model": "eval", "train_laya": "train"}[Path(__file__).stem]
if stage == "train" and "--dry-run" in args:
    stage = "validation"
with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"stage": stage, "args": args, "offline": os.environ.get("HF_HUB_OFFLINE")}) + "\\n")
scenario = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
if scenario.get("fail") == stage:
    sys.exit(19)
def arg(name):
    return Path(args[args.index(name) + 1])
if stage == "export":
    directory = arg("--output-dir")
    directory.mkdir(parents=True)
    for name in ("laya-mail-train.jsonl", "laya-mail-val.jsonl"):
        (directory / name).write_text("SYNTHETIC DATA", encoding="utf-8")
elif stage == "train":
    directory = arg("--output-dir")
    directory.mkdir()
    (directory / "model.safetensors").write_text("CANDIDATE", encoding="utf-8")
    (directory / "rl_agent_config.json").write_text("{}", encoding="utf-8")
elif stage == "eval":
    arg("--report").write_text(json.dumps({"improved": scenario.get("improved", True),
                                          "nonregressed": scenario.get("nonregressed", True)}), encoding="utf-8")
'''


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is not available")
class TrainingLauncherCheck(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="laya launcher ' ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "laya-mail"
        gpu = self.repo / "gpu"
        gpu.mkdir(parents=True)
        private = self.root / "work" / "gmail-private"
        private.mkdir(parents=True)
        (private / "dashboard-key").write_text("SYNTHETIC-KEY-NOT-A-CREDENTIAL", encoding="utf-8")
        (private / "dashboard-url.txt").write_text("http://127.0.0.1:9", encoding="utf-8")
        base = self.root / "work" / "laya-gpu" / "multilingual"
        self.active = self.repo / "runtime" / "models" / "laya-mail-custom"
        for folder, marker in ((base, "BASE"), (self.active, "OLD")):
            folder.mkdir(parents=True)
            (folder / "model.safetensors").write_text(marker, encoding="utf-8")
            (folder / "rl_agent_config.json").write_text("{}", encoding="utf-8")
        for name in ("export_training_data", "train_laya", "evaluate_model"):
            (gpu / (name + ".py")).write_text(STUB, encoding="utf-8")
        source = Path(__file__).with_name("Train-Laya-Model.ps1").read_text(encoding="utf-8-sig")
        python_line = "$pythonPath = Join-Path $taskRoot 'work\\laya-gpu-venv\\Scripts\\python.exe'"
        self.assertEqual(source.count(python_line), 1, "Update the isolated interpreter substitution if the launcher changes")
        source = source.replace(python_line, "$pythonPath = " + ps_quote(sys.executable))
        self.launcher = gpu / "Train-Laya-Model.ps1"
        self.launcher.write_text(source, encoding="utf-8-sig")

    def run_launcher(self, scenario=None, check_only=False, accept=False, fail_promotion=False):
        (self.root / "scenario.json").write_text(json.dumps(scenario or {}), encoding="utf-8")
        (self.root / "calls.jsonl").write_text("", encoding="utf-8")
        command = [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
        if accept:
            # Only the temporary candidate move can fail; native moves stay inside this fixture.
            script = "function global:Read-Host { return 'j' }; "
            if fail_promotion:
                script += """function Move-Item {
                    param([string]$LiteralPath, [string]$Destination)
                    if ((Split-Path -Leaf $LiteralPath) -eq 'model') { throw 'Synthetic promotion failure' }
                    Microsoft.PowerShell.Management\\Move-Item -LiteralPath $LiteralPath -Destination $Destination
                }; """
            command += ["-Command", script + "& " + ps_quote(self.launcher)]
        else:
            command += ["-File", str(self.launcher), "-NoPause"]
            if check_only:
                command.append("-CheckOnly")
        environment = dict(os.environ)
        environment.pop("LAYA_DASHBOARD_URL", None)
        result = subprocess.run(command, cwd=self.root, env=environment, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30)
        self.calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        return result

    def assert_stages(self, *stages):
        self.assertEqual([call["stage"] for call in self.calls], list(stages))

    def assert_old_model(self):
        self.assertEqual((self.active / "model.safetensors").read_text(), "OLD")
        self.assertEqual((self.active / "rl_agent_config.json").read_text(), "{}")

    def test_check_only_exports_and_validates_without_training(self):
        result = self.run_launcher(check_only=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_stages("export", "validation")
        self.assert_old_model()
        self.assertEqual(list((self.repo / "runtime" / "training").glob("*/model")), [])

    def test_each_failed_stage_stops_before_the_next_stage(self):
        stages = ["export", "validation", "train", "eval"]
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                result = self.run_launcher({"fail": stage}, check_only=stage == "export")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assert_stages(*stages[:index + 1])
                self.assert_old_model()

    def test_better_candidate_with_no_pause_does_not_replace_active_model(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_stages("export", "validation", "train", "eval")
        self.assert_old_model()
        evaluation = self.calls[-1]["args"]
        self.assertEqual(Path(evaluation[evaluation.index("--baseline") + 1]), self.active)
        self.assertEqual(self.calls[-1]["offline"], "1")
        candidate = Path(evaluation[evaluation.index("--model") + 1])
        self.assertEqual((candidate / "model.safetensors").read_text(), "CANDIDATE")
        self.assertFalse((candidate.parent / "previous-model").exists())

    def test_non_improving_or_regressed_candidate_is_never_offered_for_promotion(self):
        for scenario in ({"improved": False}, {"nonregressed": False}):
            with self.subTest(scenario=scenario):
                # Answering yes proves that a failed comparison cannot reach the promotion path.
                result = self.run_launcher(scenario, accept=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_stages("export", "validation", "train", "eval")
                self.assert_old_model()

    def test_explicit_acceptance_preserves_backup_and_promotes_candidate(self):
        result = self.run_launcher(accept=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_stages("export", "validation", "train", "eval")
        candidate = Path(self.calls[-1]["args"][self.calls[-1]["args"].index("--model") + 1])
        self.assertEqual((self.active / "model.safetensors").read_text(), "CANDIDATE")
        self.assertEqual((candidate.parent / "previous-model" / "model.safetensors").read_text(), "OLD")
        self.assertFalse(candidate.exists())

    def test_failed_promotion_restores_the_previous_model(self):
        result = self.run_launcher(accept=True, fail_promotion=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assert_stages("export", "validation", "train", "eval")
        self.assert_old_model()
        candidate = Path(self.calls[-1]["args"][self.calls[-1]["args"].index("--model") + 1])
        self.assertEqual((candidate / "model.safetensors").read_text(), "CANDIDATE")
        self.assertFalse((candidate.parent / "previous-model").exists())


if __name__ == "__main__":
    unittest.main()
