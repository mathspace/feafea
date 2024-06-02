from typing import cast
import unittest

from src.feafea import CompiledConfig, Evaluator


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

    def test_record_evaluations(self):
        evaluator = Evaluator(record_evaluations=True)
        evaluator.load_config(self._valid_config)

        evaluator.evaluate("a", "10", {"__now": 1000})
        evaluator.evaluate("a", "12", {"__now": 1001})
        evaluator.evaluate("a", "12", {"__now": 1002})
        evaluator.evaluate("b", "18", {"__now": 1003})
        evaluator.evaluate("a", "10", {"__now": 1004})
        evaluator.evaluate("a", "18", {"__now": 1005})
        evaluator.evaluate("a", "19", {"__now": 1006})
        evaluator.evaluate("b", "88", {"__now": 1007, "some_other_attr": 6})

        evals = evaluator.pop_evaluations()

        evals.sort(key=lambda x: (x.timestamp, x.flag, x.target_id))
        eval_tuples = [(e.timestamp, e.flag, e.target_id, e.reason, e.variant) for e in evals]
        self.assertListEqual(
            eval_tuples,
            [
                (1000, "a", "10", "default", False),
                (1001, "a", "12", "default", False),
                (1002, "a", "12", "default", False),
                (1003, "b", "18", "default", 1),
                (1004, "a", "10", "default", False),
                (1005, "a", "18", "default", False),
                (1006, "a", "19", "default", False),
                (1007, "b", "88", "const_rule", 3),
            ],
        )

        evals = evaluator.pop_evaluations()
        self.assertListEqual(evals, [])
