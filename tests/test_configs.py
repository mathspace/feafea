import unittest

from jsonschema.exceptions import ValidationError

from src.feafea import CompiledConfig, _seconds_from_human_duration, merge_configs


class TestMergeConfig(unittest.TestCase):
    def test_merge_config(self):
        config1 = {
            "flags": {"a": {"variants": [True, False], "default": True}},
        }
        config2 = {
            "flags": {"b": {"variants": [1, 2], "default": 1}},
            "filters": {"f1": "flag:a = True"},
        }
        config3 = {
            "rules": {"r1": {"variants": {"a": False}}},
        }
        merged = {
            "flags": {
                "a": {"variants": [True, False], "default": True},
                "b": {"variants": [1, 2], "default": 1},
            },
            "filters": {"f1": "flag:a = True"},
            "rules": {"r1": {"variants": {"a": False}}},
        }
        m1 = merge_configs(config1, config2, config3)
        m2 = merge_configs(config2, config1, config3)
        m3 = merge_configs(config3, config2, config1)
        self.assertDictEqual(m1, merged)
        self.assertDictEqual(m2, merged)
        self.assertDictEqual(m3, merged)
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            merge_configs(config1, config1)


class TestHumanDuration(unittest.TestCase):
    def test_human_duration(self):
        cases = [
            # human duration, seconds
            ("0m", 0),
            ("0d", 0),
            ("0h", 0),
            ("1m", 60),
            ("1h", 3600),
            ("1d", 86400),
            ("48h 5m", 48 * 3600 + 5 * 60),
            ("48h5m", 48 * 3600 + 5 * 60),
            ("1d 3h", 86400 + 3 * 3600),
            ("1d 3h 5m", 86400 + 3 * 3600 + 5 * 60),
            ("1d3h5m", 86400 + 3 * 3600 + 5 * 60),
            ("2d 5m", 2 * 86400 + 5 * 60),  # no hours
        ]
        for dur, int in cases:
            with self.subTest(dur):
                self.assertEqual(_seconds_from_human_duration(dur), int)

    def test_invalid_human_duration(self):
        cases = [
            "1",
            "1d -3h",
            "-1d",
            "5s",
        ]
        for dur in cases:
            with self.subTest(dur):
                with self.assertRaises(ValueError):
                    _seconds_from_human_duration(dur)


class TestValidConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._valid_config = CompiledConfig.from_dict(
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
                    "f2": "flag:b = 1",
                    "f3": "filter:f1 and filter:f2",
                    "f4": "rule:r1",
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
                    "r6": {
                        "splits": [
                            {
                                "name": "A",
                                "percentage": 10,
                                "variants": {
                                    "d": "red",
                                },
                            },
                            {
                                "name": "B",
                                "percentage": 5,  # [10-15)
                                "variants": {
                                    "d": "orange",
                                },
                            },
                        ],
                    },
                    "r7": {
                        "split_group": "r6",
                        "splits": [
                            {
                                "name": "AB",
                                "percentage": 15,
                                "variants": {
                                    "e": "fast",
                                },
                            },
                        ],
                    },
                    "r8": {
                        "filter": "rule:r6/B",
                        "splits": [
                            {
                                "name": "B_1",
                                "percentage": 50,
                                "variants": {
                                    "f": "up",
                                },
                            },
                            {
                                "name": "B_2",
                                "percentage": 50,
                                "variants": {
                                    "f": "down",
                                },
                            },
                        ],
                    },
                    "r9": {
                        "variants": {
                            "gg": 9,
                        },
                    },
                },
            }
        )

    def test_valid_config_high_level(self):
        self.assertSetEqual(set(self._valid_config.flags), {"a", "b", "c", "cc", "d", "e", "f", "g", "gg"})
        self.assertDictEqual(self._valid_config.flags["a"].metadata, {"category": "simple", "deprecated": False, "revision": 22})
        self.assertEqual(self._valid_config.flags["a"].name, "a")
        self.assertTrue(self._valid_config.flags["a"].type == bool)
        self.assertEqual(self._valid_config.flags["a"].default, True)
        self.assertEqual(self._valid_config.flags["a"].variants, {True, False})

    def test_config_serialization(self):
        b = self._valid_config.to_bytes()
        cc = CompiledConfig.from_bytes(b)
        self.assertSetEqual(set(self._valid_config.flags), set(cc.flags))
        self.assertEqual(self._valid_config.flags["a"].eval("", {}).variant, cc.flags["a"].eval("", {}).variant)

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
                ev = self._valid_config.flags[flag].eval("", attrs)
                self.assertEqual(ev.variant, variant)
                self.assertEqual(ev.reason, reason)
                self.assertEqual(ev.rule, rule)

    def test_valid_split_config(self):
        # id values in the cases below are picked by running them through the
        # feafea._hash_percent() function finding out their percentage bucket.
        cases = [
            # flag, id, attrs, reason, rule, variant
            ("d", "16", {}, "default", "", "white"),  # 88%
            ("d", "22", {}, "default", "", "white"),  # 46%
            ("d", "18", {}, "default", "", "white"),  # 86%
            ("d", "20", {}, "split_rule", "r6/A", "red"),  # 9.4%
            ("d", "59", {}, "split_rule", "r6/A", "red"),  # 4%
            ("d", "15", {}, "split_rule", "r6/B", "orange"),  # 13%
            ("e", "23", {}, "default", "", "idle"),  # 67%
            ("e", "8", {}, "split_rule", "r7/AB", "fast"),  # 7.6%
            ("f", "13", {}, "default", "", "middle"),  # r6=98%, r8=40%
            ("f", "15", {}, "split_rule", "r8/B_1", "up"),  # r6=13%, r8=31%
        ]

        for flag, id, attrs, reason, rule, variant in cases:
            with self.subTest(f"flag={flag}, id={id}, attrs={attrs}"):
                ev = self._valid_config.flags[flag].eval(id, attrs)
                if rule:
                    rule_name, rule_split = rule.split("/")
                    self.assertEqual((ev.reason, ev.rule, ev.split, ev.variant), (reason, rule_name, rule_split, variant))
                else:
                    self.assertEqual((ev.reason, ev.rule, ev.variant), (reason, rule, variant))

    def test_valid_split_probability(self):
        config = {
            "flags": {
                "a": {"variants": [1, 2, 3], "default": 1},
            },
            "rules": {
                "r1": {
                    "splits": [
                        {"percentage": 10, "variants": {"a": 2}},
                        {"percentage": 60, "variants": {"a": 3}},
                    ],
                },
            },
        }
        variant_count = {1: 0, 2: 0, 3: 0}
        cc = CompiledConfig.from_dict(config)
        for id in range(100000):
            v = cc.flags["a"].eval(str(id), {}).variant
            assert isinstance(v, int)
            variant_count[v] += 1
        self.assertAlmostEqual(variant_count[2] / 100000, 0.1, delta=0.002)
        self.assertAlmostEqual(variant_count[3] / 100000, 0.6, delta=0.002)
        self.assertAlmostEqual(variant_count[1] / 100000, 0.3, delta=0.002)

    def test_valid_schedule(self):
        cases = [
            # schedule, ts, expected
            ({"start": "2024-03-12 03:38:20Z", "end": "2024-03-12 03:38:25Z"}, 1710214701, True),
            ({"start": "2024-03-12 03:38:20Z", "end": "2024-03-12 03:38:25Z"}, 1710214731, False),
        ]
        tpl_config = {
            "flags": {
                "a": {
                    "default": False,
                },
            },
        }
        tpl_rule = {"variants": {"a": True}}
        for schedule, ts, expected in cases:
            with self.subTest(f"{schedule}, {ts}"):
                cc = CompiledConfig.from_dict({**tpl_config, "rules": {"r1": {**tpl_rule, "schedule": schedule}}})
                ev = cc.flags["a"].eval("", {"__now": ts})
                self.assertEqual(ev.variant, expected)

    def test_valid_schedule_ramp(self):
        cc = CompiledConfig.from_dict(
            {
                "flags": {
                    "ramp": {
                        "variants": [1, 2],
                        "default": 1,
                    },
                },
                "rules": {
                    "r1": {
                        "variants": {"ramp": 2},
                        "schedule": {
                            "start": "2024-03-12 00:00:00Z",  # 1710201600
                            "end": "2024-03-12 10:00:00Z",  # 1710237600
                            "ramp_up": "1h",
                            "ramp_down": "2h",
                        },
                    },
                },
            }
        )

        non_default_count = 0
        for i in range(10000):
            v = cc.flags["ramp"].eval(str(i), {"__now": 1710201600 + 360}).variant  # 6m after start (or 10% into 1 hour)
            if v == 2:
                non_default_count += 1
        self.assertAlmostEqual(non_default_count / 10000, 0.1, delta=0.02)

        non_default_count = 0
        for i in range(10000):
            v = cc.flags["ramp"].eval(str(i), {"__now": 1710237600 - 7200 + 720}).variant  # 12m after start of ramp down
            if v == 2:
                non_default_count += 1
        self.assertAlmostEqual(non_default_count / 10000, 0.9, delta=0.02)


class TestInvalidConfigs(unittest.TestCase):
    def test_invalid_flag_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig.from_dict({"flags": "not a dict"})

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
                    CompiledConfig.from_dict({"flags": {"a": case}})

        with self.assertRaisesRegex(ValueError, "already an alias"):
            CompiledConfig.from_dict(
                {
                    "flags": {
                        "a": {
                            "variants": [1, 2],
                            "default": 1,
                        },
                        "b": {"alias": "a"},
                        "c": {"alias": "b"},
                    },
                }
            )

    def test_invalid_filter_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig.from_dict({"filters": "not a dict"})

    def test_invalid_root_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig.from_dict({"": "not a dict"})

    def test_invalid_rules_def(self):
        with self.assertRaisesRegex(ValidationError, "."):
            CompiledConfig.from_dict({"rules": "not a dict"})

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
            ({"splits": [{"percentage": 10, "variants": {"a": 4}}]}, ValueError, "unknown flag/variant"),  # non-existing variant
            ({"variants": {"b": True}}, ValueError, "unknown flag/variant"),  # non-existing flag
            ({"splits": [{"percentage": 10, "variants": {"b": True}}]}, ValueError, "unknown flag/variant"),  # non-existing flag
            ({"variants": {"a": 1}, "splits": [{"percentage": 10}]}, ValidationError, "."),  # both splits and const
            ({"splits": [{"percentage": 20}, {"percentage": 90}]}, ValueError, "must sum to 100 or less"),  # >100 percentage
            ({"splits": [{"percentage": -10}]}, ValidationError, "minimum"),  # <=0 percentage
            ({"splits": [{"percentage": 0}]}, ValidationError, "minimum"),  # <=0 percentage
            ({"splits": [{"percentage": 100}]}, ValidationError, "maximum"),  # <100 percentage (=100 is same as not using splits)
            ({"split_group": 12}, ValidationError, "."),
            ({"variants": {"a": 4}, "split_group": 12}, ValidationError, "."),  # cannot combine split_group with variants
            ({"metadata": "abc"}, ValidationError, "."),
            ({"variants": {"a": 2}, "filter": "flag:b = 4"}, ValueError, "unknown flag"),  # unknown flag in filter
            ({"variants": {"a": 2}, "filter": "rule:unknown"}, ValueError, "unknown rule"),  # unknown rule in filter
            ({"variants": {"a": 2}, "filter": "flag:z = 2"}, ValueError, "has inferred type"),  # mismatch inferred flag type
            ({"variants": {"a": 2}, "filter": "rule:r1"}, ValueError, "circular"),
            ({"variants": {"a": 2}, "filter": "flag:a = 1"}, ValueError, "circular"),
            ({"variants": {"a": 2}, "schedule": {"start": "2024-10-10 10:10:10", "end": "2024-10-10 20:20:20"}}, ValueError, "Timezone missing"),
            ({"variants": {"a": 2}, "schedule": {"start": "2024-10-10 20:10:10Z", "end": "2024-10-10 10:20:20Z"}}, ValueError, "start must be before end"),
            ({"variants": {"a": 2}, "schedule": {"start": "2024-10-10 10:10:10Z", "end": "2024-10-10 10:10:10Z"}}, ValueError, "start must be before end"),
            (
                {"variants": {"a": 2}, "schedule": {"start": "2024-10-10 10:10:00Z", "end": "2024-10-10 10:20:00Z", "ramp_up": "8m", "ramp_down": "8m"}},
                ValueError,
                r"ramp_up \+ ramp_down must be less than end - start",
            ),
        ]

        for case, expected_err, regex in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(expected_err, regex):
                    CompiledConfig.from_dict({**tpl_config, "rules": {"r1": case}})

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
            CompiledConfig.from_dict(circular_config)
