import unittest
from src.feafea import _FilterSet


class TestAttributeOnlyFilters(unittest.TestCase):
    def test_simple(self):
        cases = [
            # Single term
            ("attr:sid = 1", {"sid": 1}, True),
            ("attr:sid = 1", {"sid": 0}, False),
            ("attr:sid = 'apple'", {"sid": "apple"}, True),
            ("attr:sid = 'apple'", {"sid": "orange"}, False),
            ("attr:sid = 'apple'", {"sid": True}, False),
            ("attr:sid = 'apple'", {"sid": 4.5}, False),
            ("attr:sid = 4.5", {"sid": 4.5}, True),
            ("attr:sid = 4.5", {"sid": 4.6}, False),
            ("attr:sid", {"sid": True}, True),
            ("attr:sid", {"sid": False}, False),
            ("attr:sid > 8", {"sid": 9}, True),
            ("attr:sid < -8", {"sid": -9}, True),
            ("attr:sid < -8", {"sid": 88}, False),
            ("attr:sid >= 99", {"sid": 99}, True),
            ("attr:sid >= 99", {"sid": 100}, True),
            ("attr:sid >= 99", {"sid": 98}, False),
            ("attr:sid <= 99", {"sid": 98}, True),
            ("attr:sid <= 99", {"sid": 99}, True),
            ("attr:sid <= 99", {"sid": 100}, False),
            ("attr:sid < 99.5", {"sid": 99}, True),
            ("attr:sid < 99.5", {"sid": 99.5}, False),
            ("attr:sid != 50", {"sid": 51}, True),
            ("attr:sid != 50", {"sid": 49}, True),
            ("attr:sid != 50", {"sid": 50}, False),
            ("attr:sid in [5,6]", {"sid": 5}, True),
            ("attr:sid in [5,6]", {"sid": 9}, False),
            ("attr:sid in [  'apple', 'orange' \n,\n ]", {"sid": "apple"}, True),
            ("attr:sid in ['apple']", {"sid": "orang"}, False),
            ("attr:sid not in [5,6]", {"sid": 4}, True),
            ("attr:sid not in [5,6]", {"sid": 6}, False),
            ("2 in attr:sid", {"sid": [1, 2]}, True),
            ("2 in attr:sid", {"sid": [1, 4]}, False),
            ("2 not in attr:sid", {"sid": [1, 4]}, True),
            ("2 not in attr:sid", {"sid": [1, 2]}, False),
            ("'apple' in attr:sid", {"sid": ["apple", "orange"]}, True),
            ("'apple' in attr:sid", {"sid": ["walnut", "coconut"]}, False),
            # Multiple terms
            ("attr:sid = 1 and attr:age = 2", {"sid": 1, "age": 2}, True),
            ("attr:sid = 1 and attr:age = 2", {"sid": 1, "age": 3}, False),
            ("attr:sid = 1 and attr:age = 2", {"sid": 2, "age": 2}, False),
            ("attr:sid = 1 or attr:age = 2", {"sid": 1, "age": 2}, True),
            ("attr:sid = 1 or attr:age = 2", {"sid": 1, "age": 3}, True),
            ("attr:sid = 1 or attr:age = 2", {"sid": 4, "age": 3}, False),
            ("attr:a or attr:b or attr:c", {"a": False, "b": False, "c": True}, True),
            ("attr:a or attr:b or attr:c", {"a": True, "b": False, "c": False}, True),
            ("attr:a or attr:b or attr:c", {"a": False, "b": False, "c": False}, False),
            ("(attr:a and attr:b) or attr:c", {"a": False, "b": False, "c": False}, False),
            ("(attr:a and attr:b) or attr:c", {"a": True, "b": False, "c": False}, False),
            ("(attr:a and attr:b) or attr:c", {"a": True, "b": True, "c": False}, True),
            ("(attr:a and attr:b) or attr:c", {"a": False, "b": True, "c": False}, False),
            ("(attr:a and attr:b) or attr:c", {"a": False, "b": True, "c": True}, True),
            ("attr:a and attr:b or attr:c", {"a": False, "b": False, "c": False}, False),
            ("attr:a and attr:b or attr:c", {"a": True, "b": False, "c": False}, False),
            ("attr:a and attr:b or attr:c", {"a": True, "b": True, "c": False}, True),
            ("attr:a and attr:b or attr:c", {"a": False, "b": True, "c": False}, False),
            ("attr:a and attr:b or attr:c", {"a": False, "b": True, "c": True}, True),
            ("not attr:a = 2", {"a": 2}, False),
            ("not attr:a = 2", {"a": 3}, True),
        ]

        for filter, attr, expected in cases:
            with self.subTest(f"'{filter}', '{attr}'"):
                fs = _FilterSet()
                fs.parse("f", filter)
                c = fs.compile("f", {}, {})
                self.assertEqual(c.eval("", attr), expected)

        valid_syntax_cases = [
            "flag:a > 3",
            "flag:a = 3",
            "flag:a <= 'ABC'",
            "flag:a >= 9",
        ]
        for filter in valid_syntax_cases:
            with self.subTest(filter):
                fs = _FilterSet()
                fs.parse("f", filter)
                fs.compile("f", {}, {})

    def test_invalid_literals(self):
        cases = [
            ("attr:s = 'apple\"", ValueError, "unexpected character"),
            ("attr:s = 4e6", ValueError, "unexpected character"),
            ("attr:s in [1,'apple']", ValueError, "set values must all be of the same type"),
            ("attr:s not in [1,5.3]", ValueError, "unexpected character"),  # float in set
        ]
        for filter, err, regex in cases:
            with self.subTest(filter):
                fs = _FilterSet()
                with self.assertRaisesRegex(err, regex):
                    fs.parse("f", filter)
                    fs.compile("f", {}, {})

    def test_invalid_syntax(self):
        cases = [
            ("(attr:unbalanced_paren = 1", ValueError, r"expected '\)'"),
            ("attr:a == 3", ValueError, "expected number after 'EQ' not '='"),
            ("8 = attr:lhs_literal_without_set", ValueError, "expected in/not in"),
            ("8 in flag:non_attr_rhs_for_lhs_literal", ValueError, r"expected attr:\*"),
            ("@", ValueError, "unexpected character '@'"),  # invalid token
            ("3.2", ValueError, r"expected attr:\*/flag:\*"),  # float on LHS
            ("flag:flag_comp_with_float = 3.3", ValueError, "expected STR/INT for feature flag comparison"),
            ("attr:rhs_non_set in 3", ValueError, "expected a set"),
            ("attr:v > flag:rhs_flag_in_comp", ValueError, "expected number after 'GT'"),
            ("attr:v > 3.3 88", ValueError, "unexpected token '88'"),  # dangling literal
            ("flag:d >= 3.3", ValueError, "expected STR/INT for feature flag comparison"),
        ]
        for filter, err, regex in cases:
            with self.subTest(filter):
                fs = _FilterSet()
                with self.assertRaisesRegex(err, regex):
                    fs.parse("f", filter)
                    fs.compile("f", {}, {})

    def test_invalid_symbol_names(self):
        cases = [
            "naked",
            "attr:",
            "attr:_a",
            "attr:-io",
            "attr:12",
        ]
        for name in cases:
            with self.subTest(name):
                fs = _FilterSet()
                with self.assertRaisesRegex(ValueError, "unexpected character"):
                    fs.parse("f", name)
                    fs.compile("f", {}, {})

    def test_circular_ref(self):
        fs = _FilterSet()
        fs.parse("a", "attr:sid = 1 and filter:c")
        fs.parse("b", "attr:at6 in [9] and filter:a")
        fs.parse("c", "filter:b")
        with self.assertRaisesRegex(ValueError, "circular reference"):
            fs.compile("a", {}, {})

    def test_valid_symbol_names(self):
        cases = [
            "attr:abc",
            "attr:a23_23v",
            "flag:abc",
            "flag:_abc",
            "flag:a23_23v",
            "rule:abc",
            "rule:_abc",
            "rule:a23_23v",
            "filter:abc",
            "filter:_abc",
            "filter:a23_23v",
        ]
        for name in cases:
            with self.subTest(name):
                fs = _FilterSet()
                fs.parse("abc", "attr:d")
                fs.parse("_abc", "attr:d")
                fs.parse("a23_23v", "attr:d")
                fs.parse("f", name)
                fs.compile("f", {}, {})

    def test_inconsitent_inferred_flag_types(self):
        cases = [
            "flag:a = 1 and flag:a = 'apple'",
            "flag:a in [1,2] and flag:a in ['apple', 'orange']",
            "flag:a != 'apple' and flag:a in [2,3]",
            "flag:a and flag:a in [2,3]",
            "flag:a and flag:a = 'apple'",
        ]
        for filter in cases:
            with self.subTest(filter):
                fs = _FilterSet()
                with self.assertRaisesRegex(ValueError, "referenced feature flag a has conflicting inferred types"):
                    fs.parse("f", filter)
                    fs.compile("f", {}, {})

    def test_inferred_flag_types(self):
        cases = [
            ("flag:a = 'apple'", str),
            ("flag:a in ['apple', 'orange']", str),
            ("flag:a in [2,3]", int),
            ("flag:a != 'apple'", str),
            ("flag:a = 6", int),
            ("flag:a", bool),
        ]
        for filter, expected_type in cases:
            with self.subTest(filter):
                fs = _FilterSet()
                fs.parse("f", filter)
                c = fs.compile("f", {}, {})
                self.assertIs(c.flag_refs["a"], expected_type)

    def test_inlined_filters(self):
        fs = _FilterSet()
        fs.parse("a", "attr:sid = 1")
        fs.parse("b", "attr:age > 27")
        fs.parse("c", "filter:a and filter:b")
        fs.parse("d", "filter:a or filter:b")
        fs.parse("e", "not filter:a")
        fs.parse("f", "attr:name = 'apple' and filter:d")

        cases = [
            ("c", {"sid": 1, "age": 28}, True),
            ("c", {"sid": 2, "age": 28}, False),
            ("c", {"sid": 1, "age": 20}, False),
            ("d", {"sid": 1, "age": 28}, True),
            ("d", {"sid": 2, "age": 28}, True),
            ("d", {"sid": 2, "age": 20}, False),
            ("e", {"sid": 2}, True),
            ("f", {"sid": 1, "age": 28, "name": "apple"}, True),
            ("f", {"sid": 1, "age": 20, "name": "apple"}, True),
            ("f", {"sid": 2, "age": 20, "name": "apple"}, False),
            ("f", {"sid": 1, "age": 28, "name": "orange"}, False),
        ]

        for filter, attr, expected in cases:
            with self.subTest(f"{filter}, {attr}"):
                c = fs.compile(filter, {}, {})
                self.assertEqual(c.eval("", attr), expected)

    def test_missing_ref_filter(self):
        fs = _FilterSet()
        fs.parse("a", "filter:b")
        with self.assertRaisesRegex(ValueError, "unknown filter b"):
            fs.compile("a", {}, {})

    def test_rule_and_flag_refs(self):
        fs = _FilterSet()
        fs.parse("a", "rule:abc and rule:ghi/A or flag:xyz = '88'")
        fs.parse("b", "filter:a and rule:def or flag:uwu or flag:xyz != 'abc'")
        c = fs.compile("b", {}, {})
        self.assertDictEqual(c.flag_refs, {"xyz": str, "uwu": bool})
        self.assertSetEqual(c.rule_refs, {("abc", None), ("def", None), ("ghi", "A")})
