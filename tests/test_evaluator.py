import threading
import time
from typing import cast
import unittest

from src.feafea import CompiledConfig, Evaluator, Exporter, FlagEvaluation, logger


class TestEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._valid_config = CompiledConfig.from_dict(
            {
                "flags": {
                    "a": {
                        "default": False,
                    },
                    "b": {
                        "variants": [1, 2, 3],
                        "default": 1,
                    },
                },
                "rules": {
                    "r0": {
                        "filter": "attr:some_attr = 5",
                        "variants": {
                            "a": True,
                        },
                    },
                    "r1": {
                        "filter": "attr:some_other_attr = 6",
                        "variants": {
                            "b": 3,
                        },
                    },
                },
            }
        )

    def test_invalid_evaluator_usage(self):
        evaluator = Evaluator()

        with self.assertRaisesRegex(RuntimeError, "config not loaded"):
            evaluator.evaluate("a", "")

        evaluator.load_config(self._valid_config)

        with self.assertRaisesRegex(ValueError, "does not exist in the config"):
            evaluator.evaluate("z", "")

        with self.assertRaisesRegex(TypeError, "must be a string"):
            evaluator.evaluate("a", cast(str, 1))

        invalid_attributes = [
            "non_dict",
            set([1, 2]),
            [],
            {99: 11},  # non str key
            {"b": (1, 2, 3)},  # tuple value
            {"c": {"a": 1}},  # dict value
            {"d": {(1, 2), (3, 4)}},  # set of tuples
            {"e": {None, 1}},  # invalid set element
            {"e": {2, "3"}},  # different set element types
        ]
        for attr in invalid_attributes:
            with self.assertRaises(TypeError):
                evaluator.evaluate("a", "", attr)

    def test_local_evaluator_evaluation(self):
        evaluator = Evaluator()
        evaluator.load_config(self._valid_config)

        evs = evaluator.detailed_evaluate_all(["a", "b"], "")
        a = evs["a"]
        b = evs["b"]
        self.assertEqual((a.reason, a.rule, a.variant), ("default", "", False))
        self.assertEqual((b.reason, b.rule, b.variant), ("default", "", 1))

        evs = evaluator.detailed_evaluate_all(["a", "b"], "", {"some_attr": 5, "some_other_attr": 6})
        a = evs["a"]
        b = evs["b"]
        self.assertEqual((a.reason, a.rule, a.variant), ("const_rule", "r0", True))
        self.assertEqual((b.reason, b.rule, b.variant), ("const_rule", "r1", 3))

        evs = evaluator.evaluate_all(["a", "b"], "")
        self.assertDictEqual(evs, {"a": False, "b": 1})
        evs = evaluator.evaluate_all(["a", "b"], "", {"some_attr": 5, "some_other_attr": 6})
        self.assertDictEqual(evs, {"a": True, "b": 3})

        self.assertEqual(evaluator.evaluate("a", ""), False)
        self.assertEqual(evaluator.evaluate("b", ""), 1)
        self.assertEqual(evaluator.evaluate("a", "", {"some_attr": 5, "some_other_attr": 6}), True)
        self.assertEqual(evaluator.evaluate("b", "", {"some_attr": 5, "some_other_attr": 6}), 3)

        evaluator.set_default_attributes({"some_attr": 5, "some_other_attr": 6})
        self.assertEqual(evaluator.evaluate("a", ""), True)
        self.assertEqual(evaluator.evaluate("b", ""), 3)

    def test_periodic_exporter(self):
        evals: list[FlagEvaluation] = []

        event = threading.Event()

        class PeriodicExporter(Exporter):
            def __init__(self):
                self.exported = []

            def export(self, entries: list[FlagEvaluation]) -> None:
                event.set()
                if any(e.timestamp == 8000 for e in entries):
                    raise Exception("A101")
                evals.extend(entries)

        evaluator = Evaluator(exporter=PeriodicExporter(), export_block_seconds=2)
        evaluator.load_config(self._valid_config)

        with self.assertLogs(logger, level="ERROR") as cm:
            evaluator.evaluate("a", "10", {"__now": 8000})  # Needed so export is called.
            # Wait until an export event occurs so we know with confidence that the exporter is
            # not going to export in the middle of us doing evaluations.
            # We also get a chance to check if the exporter thread is working.
            self.assertTrue(event.wait(2.5))

            evaluator.evaluate("a", "10", {"__now": 1000})
            evaluator.evaluate("a", "12", {"__now": 1001})
            evaluator.evaluate("a", "12", {"__now": 1001})
            evaluator.evaluate("b", "18", {"__now": 1001})
            evaluator.evaluate("a", "10", {"__now": 1002})
            evaluator.evaluate("a", "18", {"__now": 1002})
            evaluator.evaluate("a", "19", {"__now": 1002})
            evaluator.evaluate("b", "88", {"__now": 1003, "some_other_attr": 6})

            time.sleep(2.5)

        self.assertRegex(cm.output[0], "A101")

        evaluator.stop_exporter()

        evals.sort(key=lambda x: (x.timestamp, x.flag, x.target_id))
        eval_tuples = [(e.timestamp, e.flag, e.target_id, e.reason, e.variant) for e in evals]
        self.assertListEqual(
            eval_tuples,
            [
                (1000, "a", "10", "default", False),
                (1000, "a", "12", "default", False),
                (1000, "b", "18", "default", 1),
                (1002, "a", "10", "default", False),
                (1002, "a", "18", "default", False),
                (1002, "a", "19", "default", False),
                (1002, "b", "88", "const_rule", 3),
            ],
        )
