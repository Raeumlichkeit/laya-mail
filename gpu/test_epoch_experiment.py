"""Synthetic epoch-runner tests; no training, model imports, or network access."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import epoch_experiment as experiment


HAS_MSVCRT = importlib.util.find_spec("msvcrt") is not None
PRIVATE = "SYNTHETIC PRIVATE INPUT NEVER LOG"


def sample(split):
    return {"state": PRIVATE + " " + split, "metadata": {"split": split},
            "questions": {"inhalt": {"type": "choice", "instructions": "Synthetic question",
                                       "criteria": {"info": "Information", "rechnung": "Invoice"}}},
            "gold": {"inhalt": {"type": "choice", "probabilities": {"info": 1, "rechnung": 0}}}}


class ImportCheck(unittest.TestCase):
    def test_discovery_does_not_import_model_libraries(self):
        result = subprocess.run([sys.executable, "-c", "import sys, unittest; "
                                 "unittest.defaultTestLoader.discover('.', pattern='test_epoch_experiment.py'); "
                                 "assert 'torch' not in sys.modules and 'laya' not in sys.modules"],
                                cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(HAS_MSVCRT, "The runner uses Windows byte locking")
class EpochExperimentCheck(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="laya epoch synthetic ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "runtime" / "training" / "source-run"
        self.source.mkdir(parents=True)
        for split in ("train", "val"):
            (self.source / f"laya-mail-{split}.jsonl").write_text(json.dumps(sample(split)) + "\n", encoding="utf-8")
        (self.source / "evaluation.json").write_text('{"improved": true}', encoding="utf-8")
        self.base = self.root / "base-model"
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"SYNTHETIC BASE WEIGHTS")
        self.active = self.root / "runtime" / "models" / "laya-mail-custom" / "model.safetensors"
        self.active.parent.mkdir(parents=True)
        self.active.write_bytes(b"SYNTHETIC ACTIVE WEIGHTS")
        self.originals = {path: path.read_bytes() for path in [*self.source.iterdir(), self.active, self.base / "model.safetensors"]}
        self.args = SimpleNamespace(run_dir=self.source, output_dir=self.source.parent / "study", base_model=self.base, seed=42)
        root_patch = patch.object(experiment, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.calls, self.children = [], []
        self.fail_stage = None
        self.history = {"schema_version": 1, "seed": 42, "compare_epoch": 8, "patience": 3,
                        "min_epochs": 9, "requested_epochs": 20, "completed_epochs": 9,
                        "epochs": [{"epoch": number, "train_loss": .6, "val_loss": .5} for number in range(1, 10)],
                        "best_epoch": 9, "best_through_epoch": 8, "best_after_epoch": 9, "stopped_early": True}

    def fake_child(self, command, env):
        command = list(map(str, command))
        stage = "training" if Path(command[2]).name == "train_laya.py" else "evaluating"
        self.calls.append((stage, command, env))
        child = SimpleNamespace(pid=900 + len(self.calls), terminate=Mock(), kill=Mock(), poll=Mock(return_value=None))

        def arg(name):
            return Path(command[command.index(name) + 1])

        def finish(*args, **kwargs):
            state = json.loads((self.args.output_dir / "experiment-state.json").read_text())
            self.assertEqual(state["status"], stage)
            self.assertEqual(state["child_pid"], child.pid)
            if self.fail_stage == stage:
                return 17
            if stage == "training":
                for name in ("best-through-8", "best-after-8"):
                    checkpoint = arg("--output-dir") / name
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "model.safetensors").write_text(name)
                arg("--history").write_text(json.dumps(self.history), encoding="utf-8")
            else:
                self.assertEqual(arg("--model").name, "best-after-8")
                self.assertEqual(arg("--baseline").name, "best-through-8")
                self.assertEqual(arg("--data"), self.args.output_dir / "laya-mail-val.jsonl")
                arg("--report").write_text('{"improved": true, "nonregressed": true}', encoding="utf-8")
            return 0

        child.wait = Mock(side_effect=finish)
        self.children.append(child)
        return child

    def assert_originals_unchanged(self):
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original)

    def run_with_fake_children(self):
        with patch.object(experiment.subprocess, "Popen", side_effect=self.fake_child) as spawn, redirect_stdout(io.StringIO()) as output:
            result = experiment.run_experiment(self.args)
        self.assertNotIn(PRIVATE, output.getvalue())
        return result, spawn

    def test_complete_once_preserves_sources_active_model_and_provenance(self):
        result, spawn = self.run_with_fake_children()
        self.assertEqual(result, 0)
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual([stage for stage, _, _ in self.calls], ["training", "evaluating"])
        self.assert_originals_unchanged()
        for split in ("train", "val"):
            name = f"laya-mail-{split}.jsonl"
            self.assertEqual((self.args.output_dir / name).read_bytes(), (self.source / name).read_bytes())
        state_path = self.args.output_dir / "experiment-state.json"
        state = json.loads(state_path.read_text())
        summary = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["completed_epochs"], 9)
        self.assertNotIn("child_pid", state)
        self.assertEqual(state["data"]["train"]["samples"], 1)
        self.assertEqual(state["data"]["train"]["labels"], {"inhalt": {"info": 1}})
        self.assertEqual(state["dataset_sha256"]["val"], experiment.sha256(self.source / "laya-mail-val.jsonl"))
        self.assertEqual(state["base_model_sha256"], experiment.sha256(self.base / "model.safetensors"))
        self.assertTrue(state["active_model_unchanged"])
        self.assertTrue(summary["active_model_unchanged"])
        self.assertTrue(summary["validation_only"])
        self.assertEqual(summary["history"], self.history)
        self.assertNotIn(PRIVATE, json.dumps(state) + json.dumps(summary))
        for _, _, environment in self.calls:
            self.assertEqual(environment["PYTHONHASHSEED"], "42")
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        saved_state = state_path.read_bytes()
        with patch.object(experiment.subprocess, "Popen") as second_spawn, self.assertRaises(FileExistsError):
            experiment.run_experiment(self.args)
        second_spawn.assert_not_called()
        self.assertEqual(state_path.read_bytes(), saved_state)

    def test_existing_started_directory_cannot_start_a_second_child(self):
        self.args.output_dir.mkdir()
        state_path = self.args.output_dir / "experiment-state.json"
        state_path.write_text('{"status": "training", "child_pid": 123}', encoding="utf-8")
        before = state_path.read_bytes()
        with patch.object(experiment.subprocess, "Popen") as spawn, self.assertRaises(FileExistsError):
            experiment.run_experiment(self.args)
        spawn.assert_not_called()
        self.assertEqual(state_path.read_bytes(), before)
        self.assert_originals_unchanged()

    def test_child_failures_stop_without_success_summary(self):
        for stage, expected in (("training", ["training"]), ("evaluating", ["training", "evaluating"])):
            with self.subTest(stage=stage):
                self.fail_stage = stage
                self.calls.clear()
                self.args.output_dir = self.source.parent / ("failed-" + stage)
                with patch.object(experiment.subprocess, "Popen", side_effect=self.fake_child), self.assertRaises(RuntimeError):
                    experiment.run_experiment(self.args)
                self.assertEqual([name for name, _, _ in self.calls], expected)
                self.assertFalse((self.args.output_dir / "summary.json").exists())
                state = json.loads((self.args.output_dir / "experiment-state.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["error_type"], "RuntimeError")
                self.assertNotIn("child_pid", state)
                self.assert_originals_unchanged()

    def test_copy_or_lock_failure_cannot_launch_training(self):
        for target in ("epoch_experiment.shutil.copyfile", "msvcrt.locking"):
            with self.subTest(target=target):
                self.args.output_dir = self.source.parent / target.rsplit(".", 1)[-1]
                with patch(target, side_effect=OSError(PRIVATE)), patch.object(experiment.subprocess, "Popen") as spawn, self.assertRaises(OSError):
                    experiment.run_experiment(self.args)
                spawn.assert_not_called()
                self.assertFalse((self.args.output_dir / "summary.json").exists())
                state = json.loads((self.args.output_dir / "experiment-state.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertNotIn(PRIVATE, json.dumps(state))
                self.assert_originals_unchanged()

    def test_spawned_child_is_reaped_before_unlock_on_interruption_or_status_write_failure(self):
        import msvcrt
        original_write = experiment.write_json
        for stage in ("interruption", "status-write"):
            with self.subTest(stage=stage):
                self.args.output_dir = self.source.parent / stage
                failure = KeyboardInterrupt() if stage == "interruption" else OSError(PRIVATE)
                events = []
                triggered = False

                def write_state(path, value):
                    nonlocal triggered
                    if stage == "status-write" and value.get("child_pid") and not triggered:
                        triggered = True
                        raise failure
                    original_write(path, value)

                def wait():
                    nonlocal triggered
                    if stage == "interruption" and not triggered:
                        triggered = True
                        raise failure
                    events.append("reaped")
                    with (self.source.parent / ".training.lock").open("a+b") as probe:
                        probe.seek(0)
                        with self.assertRaises(OSError):
                            msvcrt.locking(probe.fileno(), msvcrt.LK_NBLCK, 1)
                    return 1

                child = SimpleNamespace(pid=9876, poll=Mock(return_value=None), wait=Mock(side_effect=wait))
                with patch.object(experiment.subprocess, "Popen", return_value=child) as spawn, \
                        patch.object(experiment.subprocess, "run", side_effect=lambda *args, **kwargs: events.append("terminated")) as terminate, \
                        patch.object(experiment, "write_json", side_effect=write_state), self.assertRaises(type(failure)) as caught:
                    experiment.run_experiment(self.args)
                self.assertIs(caught.exception, failure)
                spawn.assert_called_once()
                terminate.assert_called_once_with(["taskkill", "/PID", "9876", "/T", "/F"],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                self.assertEqual(events, ["terminated", "reaped"])
                with (self.source.parent / ".training.lock").open("a+b") as probe:
                    probe.seek(0)
                    msvcrt.locking(probe.fileno(), msvcrt.LK_NBLCK, 1)
                state = json.loads((self.args.output_dir / "experiment-state.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["error_type"], type(failure).__name__)
                self.assertFalse((self.args.output_dir / "summary.json").exists())
                self.assert_originals_unchanged()


if __name__ == "__main__":
    unittest.main()
