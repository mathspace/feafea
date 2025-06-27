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
                        "variants": [1, 2, 3, 4],
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
                    "r2": {
                        "filter": "88 in attr:some_set_attr",
                        "variants": {
                            "b": 4,
                        },
                    },
                },
            }
        )

    def test_incorrect_config_version(self):
        evaluator = Evaluator()
        config = CompiledConfig.from_dict({"flags": {"a": {"default": False}}, "rules": {}})
        config._feafea_checksum = "123"
        with self.assertRaisesRegex(AssertionError, "incompatible config"):
            evaluator.evaluate(config, "a", "")

    def test_valid_config_checksum_after_serialization(self):
        evaluator = Evaluator()
        config = CompiledConfig.from_bytes(self._valid_config.to_bytes())
        evaluator.evaluate(config, "a", "")

    def test_invalid_evaluator_usage(self):
        evaluator = Evaluator()

        with self.assertRaisesRegex(ValueError, "does not exist in the config"):
            evaluator.evaluate(self._valid_config, "z", "")

        with self.assertRaisesRegex(TypeError, "must be a string"):
            evaluator.evaluate(self._valid_config, "a", cast(str, 1))

        invalid_attributes = [
            ("non_dict", "must be a dict"),
            (set([1, 2]), "must be a dict"),
            ([], "must be a dict"),
            ({99: 11}, "attribute key must be a string"), # int key
            ({"b": (1, 2, 3)}, "attribute value must be"), # tuple
            ({"c": {"a": 1}}, "attribute value must be"), # dict key
            ({"d": {(1, 2), (3, 4)}}, "set values must be"), # set with tuple elements
            ({"e": {None, 1}}, "set values must be"), # set with None
            ({"e": {2, "3"}}, "set values must be of the same type"), # set with mixed int and string types
        ]
        for attr, err in invalid_attributes:
            with self.assertRaisesRegex(TypeError, err, msg=attr):
                evaluator.evaluate(self._valid_config, "a", "", attr)

    def test_additional_attribute_validation_errors(self):
        evaluator = Evaluator()
        config = CompiledConfig.from_dict({
            "flags": {"test_flag": {"default": False}},
            "rules": {}
        })

        # Test additional invalid attribute types to cover missing lines
        invalid_attributes = [
            # Non-string key
            {123: "value"},
            # Set with mixed types including None
            {"key": {None, 1}},
            # Set with tuple elements
            {"key": {(1, 2), (3, 4)}},
            # Set with float element (invalid - only str/int allowed)
            {"key": {1.5}},
            # Set with mixed int and string types
            {"key": {1, "string"}},
        ]

        for attrs in invalid_attributes:
            with self.subTest(attrs=attrs):
                with self.assertRaises(TypeError):
                    evaluator.evaluate(config, "test_flag", "test_id", attrs)

    def test_local_evaluator_evaluation(self):
        evaluator = Evaluator()
        cfg = self._valid_config

        evs = evaluator.detailed_evaluate_all(cfg, ["a", "b"], "")
        a = evs["a"]
        b = evs["b"]
        self.assertEqual((a.reason, a.rule, a.variant), ("default", "", False))
        self.assertEqual((b.reason, b.rule, b.variant), ("default", "", 1))

        evs = evaluator.detailed_evaluate_all(cfg, ["a", "b"], "", {"some_attr": 5, "some_other_attr": 6})
        a = evs["a"]
        b = evs["b"]
        self.assertEqual((a.reason, a.rule, a.variant), ("const_rule", "r0", True))
        self.assertEqual((b.reason, b.rule, b.variant), ("const_rule", "r1", 3))

        evs = evaluator.evaluate_all(cfg, ["a", "b"], "")
        self.assertDictEqual(evs, {"a": False, "b": 1})
        evs = evaluator.evaluate_all(cfg, ["a", "b"], "", {"some_attr": 5, "some_other_attr": 6})
        self.assertDictEqual(evs, {"a": True, "b": 3})

        self.assertEqual(evaluator.evaluate(cfg, "a", ""), False)
        self.assertEqual(evaluator.evaluate(cfg, "b", ""), 1)
        self.assertEqual(evaluator.evaluate(cfg, "a", "", {"some_attr": 5, "some_other_attr": 6}), True)
        self.assertEqual(evaluator.evaluate(cfg, "b", "", {"some_attr": 5, "some_other_attr": 6}), 3)
        self.assertEqual(evaluator.evaluate(cfg, "b", "", {"some_set_attr": {100, 88}}), 4)

    def test_record_evaluations(self):
        evaluator = Evaluator(record_evaluations=True)
        cfg = self._valid_config

        evaluator.evaluate(cfg, "a", "10", {"__now": 1000})
        evaluator.evaluate(cfg, "a", "12", {"__now": 1001})
        evaluator.evaluate(cfg, "a", "12", {"__now": 1002})
        evaluator.evaluate(cfg, "b", "18", {"__now": 1003})
        evaluator.evaluate(cfg, "a", "10", {"__now": 1004})
        evaluator.evaluate(cfg, "a", "18", {"__now": 1005})
        evaluator.evaluate(cfg, "a", "19", {"__now": 1006})
        evaluator.evaluate(cfg, "b", "88", {"__now": 1007, "some_other_attr": 6})

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

    def test_pop_evaluations_when_disabled(self):
        """Test pop_evaluations when record_evaluations is False."""
        # Create evaluator with record_evaluations=False (default)
        evaluator = Evaluator(record_evaluations=False)
        cfg = self._valid_config

        # Perform some evaluations
        evaluator.evaluate(cfg, "a", "test_id", {})
        evaluator.evaluate(cfg, "a", "test_id2", {})

        # pop_evaluations should return empty list when recording is disabled
        evaluations = evaluator.pop_evaluations()
        self.assertEqual(evaluations, [])
