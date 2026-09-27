"""Synthetic evaluation checks; never imports real torch or loads model weights."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import evaluate_model as evaluation


CHOICES = {"inhalt": "info", "wichtig": "ja", "spamverdacht": "nein"}


def sample():
    options = {"inhalt": ["rechnung", "info", "werbung", "sonstiges"],
               "wichtig": ["ja", "nein"], "spamverdacht": ["ja", "nein"]}
    return {"state": "SYNTHETIC PRIVATE MESSAGE - NEVER PRINT",
            "questions": {qid: {"type": "choice", "instructions": "Synthetic question",
                                "criteria": {label: "Synthetic criterion" for label in labels}}
                          for qid, labels in options.items()},
            "gold": {qid: {"type": "choice", "probabilities": {
                label: float(label == CHOICES[qid]) for label in labels}} for qid, labels in options.items()}}


class FakeAgent:
    def __init__(self, choices):
        self.choices = choices

    def predict(self, state, questions):
        return {"answers": {qid: {"type": "choice", "choice": choice} for qid, choice in self.choices.items()}}


class EvaluationCheck(unittest.TestCase):
    def test_discovery_does_not_import_torch_or_laya(self):
        script = "import sys, unittest; unittest.defaultTestLoader.discover('.', pattern='test_evaluate.py'); assert 'torch' not in sys.modules; assert 'laya' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "Evaluation test discovery must stay free of model imports")

    def test_data_validation_never_skips_malformed_or_empty_rows(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.jsonl"
            valid = json.dumps(sample())
            for contents in ("", "\n", valid + "\n{invalid-json", valid + "\n[]", valid + '\n{"state":"invalid"}'):
                path.write_text(contents, encoding="utf-8")
                with self.subTest(contents_empty=not contents), self.assertRaises(ValueError):
                    evaluation.load_data(path)
            path.write_text("\n" + valid + "\n\n", encoding="utf-8")
            self.assertEqual(len(evaluation.load_data(path)), 1)
        broken = sample()
        broken["gold"].pop("wichtig")
        agent = Mock()
        for rows in ([], [sample(), broken]):
            with self.assertRaises(ValueError):
                evaluation.run_eval(agent, rows)
        agent.predict.assert_not_called()

    def test_missing_invalid_and_unexpected_predictions_fail_instead_of_inflating_accuracy(self):
        complete = FakeAgent(CHOICES).predict(None, None)
        missing = deepcopy(complete)
        missing["answers"].pop("wichtig")
        invalid = deepcopy(complete)
        invalid["answers"]["wichtig"] = {"choice": "not-a-label"}
        extra = deepcopy(complete)
        extra["answers"]["extra"] = {"choice": "ja"}
        for prediction in (None, {}, {"answers": []}, missing, invalid, extra):
            with self.subTest(prediction_shape=type(prediction).__name__), self.assertRaises(ValueError):
                evaluation.run_eval(Mock(predict=Mock(return_value=prediction)), [sample()])
        metrics = evaluation.run_eval(FakeAgent({**CHOICES, "wichtig": "nein"}), [sample(), sample()])
        self.assertEqual(metrics["overall_accuracy"], 2 / 3)
        self.assertEqual(metrics["question_totals"], {qid: 2 for qid in CHOICES})
        self.assertEqual(metrics["confusion_inhalt"], {"info": {"info": 2}})
        before = deepcopy(metrics)
        with redirect_stdout(io.StringIO()):
            evaluation.print_report("Synthetic", metrics)
        self.assertEqual(metrics, before)

    def test_comparison_releases_candidate_before_baseline_and_reports_per_question_regression(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            data, report = directory / "data.jsonl", directory / "report.json"
            data.write_text(json.dumps(sample()), encoding="utf-8")
            events, references = [], []

            def load(path, device):
                if references:
                    self.assertIsNone(references[-1](), "Previous model must be freed before loading another")
                    self.assertEqual(events[-1], "empty_cache")
                events.append(Path(path).name)
                choices = ({**CHOICES, "wichtig": "nein"} if Path(path).name == "candidate"
                           else {"inhalt": "rechnung", "wichtig": "ja", "spamverdacht": "ja"})
                agent = FakeAgent(choices)
                references.append(weakref.ref(agent))
                return agent

            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                         empty_cache=lambda: events.append("empty_cache")))
            output = io.StringIO()
            with patch.dict(sys.modules, {"torch": torch, "laya": SimpleNamespace(load=load)}), redirect_stdout(output):
                code = evaluation.main(["--data", str(data), "--model", "candidate", "--baseline", "baseline",
                                        "--device", "cuda", "--report", str(report)])
            self.assertEqual(code, 0)
            self.assertEqual(events, ["candidate", "empty_cache", "baseline", "empty_cache"])
            self.assertTrue(all(reference() is None for reference in references))
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["improved"])
            self.assertFalse(result["nonregressed"])
            self.assertEqual(result["candidate"]["sample_count"], 1)
            self.assertEqual(result["baseline"]["overall_accuracy"], 1 / 3)
            self.assertNotIn(sample()["state"], output.getvalue() + report.read_text(encoding="utf-8"))

    def test_failures_are_redacted_do_not_load_baseline_and_do_not_write_success_report(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            data, report = directory / "data.jsonl", directory / "report.json"
            data.write_text(json.dumps(sample()), encoding="utf-8")
            agent = Mock(predict=Mock(side_effect=RuntimeError(sample()["state"])))
            load = Mock(return_value=agent)
            empty_cache = Mock()
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache))
            output = io.StringIO()
            with patch.dict(sys.modules, {"torch": torch, "laya": SimpleNamespace(load=load)}), \
                    redirect_stdout(output), redirect_stderr(output):
                code = evaluation.main(["--data", str(data), "--model", "candidate", "--baseline", "baseline",
                                        "--device", "cuda", "--report", str(report)])
            self.assertEqual(code, 1)
            load.assert_called_once()
            empty_cache.assert_called_once()
            self.assertFalse(report.exists())
            self.assertNotIn(sample()["state"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
