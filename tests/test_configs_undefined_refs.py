import unittest
import functools

from jsonschema.exceptions import ValidationError

from src.feafea import CompiledConfig

CompiledConfig_from_dict = functools.partial(CompiledConfig.from_dict, ignore_undefined_refs=True)


class TestValidConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._valid_config = CompiledConfig_from_dict(
            {
                "flags": {
                    "a": {
                        "default": True,
                        "metadata": {
                            "category": "simple",
                            "deprecated": False,
                            "revision": 22,
                        },
                    },
                    "b": {
                        "variants": [
                            1,
                            5,
                            9,
                            11,
                        ],
                        "default": 5,
                    },
                    "c": {
                        "variants": [
                            "low",
                            "med",
                            "high",
                        ],
                        "default": "high",
                    },
                    "cc": {
                        "alias": "c",
                    },
                    "d": {
                        "variants": [
                            "green",
                            "yellow",
                            "orange",
                            "red",
                            "white",
                        ],
                        "default": "white",
                    },
                    "e": {
                        "variants": [
                            "fast",
                            "slow",
                            "idle",
                        ],
                        "default": "idle",
                    },
                    "f": {
                        "variants": [
                            "up",
                            "down",
                            "middle",
                        ],
                        "default": "middle",
                    },
                    "g": {
                        "variants": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        "default": 0,
                    },
                    "gg": {
                        "alias": "g",
                    },
                },
                "filters": {
                    "f1": "attr:at1 in [5,6,7]",
                    "f2": "flag:b = 1 or flag:nonexistent = 55 or rule:nonexistent = true",
                    "f3": "filter:f1 and filter:f2 = true",
                    "f4": "rule:r1 != false",
                    "fbad": "filter:who and flag:what or rule:where or filter:f2",
                },
                "rules": {
                    "r0": {
                        "filter": "attr:at_dis",
                        "enabled": False,
                        "variants": {
                            "c": "med",
                        },
                    },
                    "r1": {
                        "filter": "filter:f1",
                        "variants": {
                            "a": False,
                            "b": 9,
                        },
                        "metadata": {
                            "desc": "rule 1",
                        },
                    },
                    "r2": {
                        "filter": "attr:at2 = 99",
                        "variants": {
                            "b": 1,
                        },
                    },
                    "r3": {
                        "filter": "attr:at3 = 0",
                        "variants": {
                            "b": 11,
                        },
                        "priority": 1000,
                    },
                    "r4": {
                        "filter": "filter:f2",
                        "variants": {
                            "c": "low",
                        },
                    },
                    "r5": {
                        "filter": "filter:f4",
                        "variants": {
                            "c": "med",
                        },
                    },
                    "r9": {
                        "variants": {
                            "gg": 9,
                        },
                    },
                    "r10": {
                        "variants": {
                            "nonexistent": "apply",
                        }
                    }
                },
            },
        )

    def test_valid_config_high_level(self):
        self.assertSetEqual(set(self._valid_config.flags), {"a", "b", "c", "cc", "d", "e", "f", "g", "gg"})
        self.assertDictEqual(self._valid_config.flags["a"].metadata, {"category": "simple", "deprecated": False, "revision": 22})
        self.assertEqual(self._valid_config.flags["a"].name, "a")
        self.assertTrue(self._valid_config.flags["a"].type is bool)
        self.assertEqual(self._valid_config.flags["a"].default, True)
        self.assertEqual(self._valid_config.flags["a"].variants, {True, False})

    def test_config_serialization(self):
        b = self._valid_config.to_bytes()
        cc = CompiledConfig.from_bytes(b)
        self.assertSetEqual(set(self._valid_config.flags), set(cc.flags))
        self.assertEqual(self._valid_config.flags["a"].eval({}).variant, cc.flags["a"].eval({}).variant)

    def test_valid_const_config(self):
        cases = [
            # flag, attrs, reason, rule, variant
            ("a", {}, "default", "", True),
            ("a", {"at1": 6}, "const_rule", "r1", False),
            ("b", {}, "default", "", 5),
            ("b", {"at1": 5}, "const_rule", "r1", 9),
            ("b", {"at1": 5, "at2": 99}, "const_rule", "r1", 9),
            ("b", {"at1": 9, "at2": 99}, "const_rule", "r2", 1),
            ("b", {"at1": 5, "at2": 99, "at3": 0}, "const_rule", "r3", 11),
            ("c", {"at_dis": True}, "default", "", "high"),
            ("c", {}, "default", "", "high"),
            ("c", {"at2": 99}, "const_rule", "r4", "low"),
            ("c", {"at1": 5}, "const_rule", "r5", "med"),
            ("cc", {"at_dis": True}, "default", "", "high"),  # alias of c
            ("cc", {}, "default", "", "high"),  # alias of c
            ("cc", {"at2": 99}, "const_rule", "r4", "low"),  # alias of c
            ("cc", {"at1": 5}, "const_rule", "r5", "med"),  # alias of c
            ("gg", {}, "const_rule", "r9", 9),
            ("g", {}, "const_rule", "r9", 9),  # setting rule against alias sets it against the original flag
        ]

        for flag, attrs, reason, rule, variant in cases:
            with self.subTest(f"flag={flag}, attrs={attrs}"):
                ev = self._valid_config.flags[flag].eval(attrs)
                self.assertEqual(ev.variant, variant)
                self.assertEqual(ev.reason, reason)
                self.assertEqual(ev.rule, rule)


class TestInvalidConfigs(unittest.TestCase):
    def test_invalid_flag_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig_from_dict({"flags": "not a dict"})

        cases = [
            ({}, ValidationError, "."),
            ({"variants": ["only"], "default": "only"}, ValidationError, "."),  # <2 variants
            ({"variants": [1, 2]}, ValidationError, "."),  # no default
            ({"variants": [1, 2], "default": 3}, ValueError, "must be one of the variants"),  # default not in variants
            ({"variants": [1, 2], "default": 1, "metadata": "not a dict"}, ValidationError, "."),
            ({"variants": [1, 2, "a"], "default": 1}, ValidationError, "."),
            ({"variants": [True, False], "default": True}, ValidationError, "."),  # bool with variants
            ({"variants": [1, 2, 2], "default": 1}, ValidationError, "."),  # duplicate variants
            ({"alias": "non_existing"}, ValueError, "unknown flag"),
        ]
        for case, expected_err, regex in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(expected_err, regex):
                    CompiledConfig_from_dict({"flags": {"a": case}})

        with self.assertRaisesRegex(ValueError, "already an alias"):
            CompiledConfig_from_dict(
                {
                    "flags": {
                        "a": {
                            "variants": [1, 2],
                            "default": 1,
                        },
                        "b": {"alias": "a"},
                        "c": {"alias": "b"},
                    },
                },
            )

    def test_invalid_filter_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig_from_dict({"filters": "not a dict"})

    def test_invalid_root_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig_from_dict({"": "not a dict"})

    def test_invalid_rules_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig_from_dict({"rules": "not a dict"})

        tpl_config = {
            "flags": {
                "a": {
                    "variants": [1, 2, 3],
                    "default": 1,
                },
                "z": {
                    "variants": ["a", "b", "c"],
                    "default": "a",
                },
            },
        }

        cases = [
            ({}, ValidationError, "."),
            ({"variants": {"a": 4}}, ValueError, "unknown flag/variant"),  # non-existing variant
            ({"split_group": 12}, ValidationError, "."),
            ({"variants": {"a": 4}, "split_group": 12}, ValidationError, "."),  # cannot combine split_group with variants
            ({"metadata": "abc"}, ValidationError, "."),
            ({"variants": {"a": 2}, "filter": "rule:r1"}, ValueError, "circular"),
            ({"variants": {"a": 2}, "filter": "flag:a = 1"}, ValueError, "circular"),
        ]

        for case, expected_err, regex in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(expected_err, regex, msg=f"case={case}"):
                    CompiledConfig_from_dict({**tpl_config, "rules": {"r1": case}})

        circular_config = {
            "flags": {
                "a": {
                    "variants": [1, 2, 3],
                    "default": 1,
                },
                "b": {
                    "variants": ["a", "b", "c"],
                    "default": "a",
                },
            },
            "rules": {
                "r1": {
                    "filter": "flag:a = 1",
                    "variants": {"b": "a"},
                },
                "r2": {
                    "filter": "flag:a = 2",
                    "variants": {"a": 3},
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "circular"):
            CompiledConfig_from_dict(circular_config)

    def test_ignore_undefined_refs_functionality(self):
        """Test that ignore_undefined_refs=True allows compilation with undefined references."""
        config_dict = {
            "flags": {
                "defined_flag": {"default": False}
            },
            "rules": {
                "rule_with_undefined_flag": {
                    "filter": "flag:undefined_flag = true",
                    "variants": {"defined_flag": True}
                },
                "rule_with_undefined_rule": {
                    "filter": "rule:undefined_rule",
                    "variants": {"defined_flag": True}
                }
            }
        }

        # Should compile successfully with ignore_undefined_refs=True
        config = CompiledConfig_from_dict(config_dict, ignore_undefined_refs=True)
        self.assertIsNotNone(config)

        # Should fail with ignore_undefined_refs=False (default)
        with self.assertRaisesRegex(ValueError, "unknown"):
            CompiledConfig.from_dict(config_dict, ignore_undefined_refs=False)
