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
            ("attr:sid = true", {"sid": True}, True),
            ("attr:sid = false", {"sid": False}, True),
            ("attr:sid != true", {"sid": True}, False),
            ("attr:sid != false", {"sid": False}, False),
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
            ("attr:sid intersects [5,6]", {"sid": {6, 7}}, True),
            ("attr:sid intersect [5,6]", {"sid": {7, 8}}, False),
            ("2 in attr:sid", {"sid": {1, 2}}, True),
            ("2 in attr:sid", {"sid": {1, 4}}, False),
            ("2 not in attr:sid", {"sid": {1, 4}}, True),
            ("2 not in attr:sid", {"sid": {1, 2}}, False),
            ("'apple' in attr:sid", {"sid": {"apple", "orange"}}, True),
            ("'apple' in attr:sid", {"sid": {"walnut", "coconut"}}, False),
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
            ("not attr:a = true", {"a": True}, False),
            ("not attr:a = false", {"a": True}, True),
            ("not (attr:a in [1,2])", {"a": 1}, False),
            # Type coercion
            ("attr:pid = true", {"pid": 1}, False),
            # None
            ("attr:a = 2", {"a": None}, False),
            ("attr:a = 2 or attr:b = 3", {"a": None, "b": 3}, True),
            ("attr:a = 2 or attr:b = 3", {"b": 3}, True),
            ("attr:a", {}, False),
            ("not attr:a", {}, True),
            # insplit function (ignoring seed)
            ("insplit(attr:a, 0, 100) = true", {"a": 1}, True),  # int attr
            ("insplit(attr:a, 0, 100)", {"a": 2}, True),  # int attr without comparison
            ("insplit(attr:a, 0, 100)", {"a": "bat"}, True),  # str attr
            ("insplit(attr:a, 0, 100)", {"a": 4.5}, True),  # float attr
            ("insplit(attr:a, 0, 100)", {"a": False}, True),  # bool attr
            ("insplit(attr:a, 0, 100)", {}, False),  # missing attr
            ("insplit(attr:a, 0, 100)", {"a": {1, 2, 3}}, True),  # set of int attr
            ("insplit(attr:a, 0, 100)", {"a": {"x", "y", "z"}}, True),  # set of str attr
            ("insplit(attr:a, 0, 100)", {"a": {5.6, 8.2, 3.3}}, True),  # set of float attr
            ("insplit(attr:a, 0, 100)", {"a": set()}, False),  # empty set
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
            "flag:a in [1,9]",
            "flag:a not in [1,9]",
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
            ("flag:d in [1,'2']", ValueError, "set values must all be of the same type"),
            ("flag:d not in [1, '2']", ValueError, "set values must all be of the same type"),
            ("flag:d not in [1, 2.2]", ValueError, "unexpected character"),
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
            ("attr:a == 3", ValueError, "expected INT/STR/FLOAT after 'EQ' not '='"),
            ("8 = attr:lhs_literal_without_set", ValueError, "expected in/not in"),
            ("8 in flag:non_attr_rhs_for_lhs_literal", ValueError, r"expected attr:\*"),
            ("flag:non_attr_rhs_for_lhs_literal intersects [5,6]", ValueError, r"expected attr:\* on left-hand-side of INTERSECTS"),
            ("attr:a intersect 5", ValueError, r"expected set on right-hand-side of INTERSECTS"),
            ("@", ValueError, "unexpected character '@'"),  # invalid token
            ("3.2", ValueError, r"expected attr:\*/flag:\*"),  # float on LHS
            ("flag:flag_comp_with_float = 3.3", ValueError, "expected BOOL/STR/INT for feature flag comparison"),
            ("attr:rhs_non_set in 3", ValueError, "expected a set"),
            ("attr:v > flag:rhs_flag_in_comp", ValueError, "expected INT/STR/FLOAT after 'GT'"),
            ("attr:v > 3.3 88", ValueError, "unexpected token '88'"),  # dangling literal
            ("flag:d >= 3.3", ValueError, "expected BOOL/STR/INT for feature flag comparison"),
            ("attr:a > true", ValueError, "expected EQ/NE before boolean 'True' not GT"),
            ("attr:a < true", ValueError, "expected EQ/NE before boolean 'True' not LT"),
            ("attr:a >= true", ValueError, "expected EQ/NE before boolean 'True' not GE"),
            ("attr:a <= false", ValueError, "expected EQ/NE before boolean 'False' not LE"),
            ("truefalse", ValueError, r"unexpected character 't' at position 0"),  # combined bool literals
            ("falset", ValueError, r"unexpected character 'f' at position 0"),
            ("insplit(1,2)", ValueError, r"insplit func takes three arguments"),
            ("insplit(flag:a,1,2,3)", ValueError, r"insplit func takes three arguments"),
            ("insplit(flag:a,1,2)", ValueError, r"expected attr:\* as first argument to insplit"),
            ("insplit(attr:a, 1, 2) > 3", ValueError, r"unexpected token '>'"),
            ("insplit(attr:a, 1, 2) = '55'", ValueError, r"insplit can only be compared to a boolean"),
            ("insplit(attr:a, -1, 2)", ValueError, r"expected positive numbers as second and third argument to insplit"),
            ("insplit(attr:a, 50, 40)", ValueError, r"expected second argument to insplit to be less than third argument"),
            ("insplit(attr:a, 1, 101)", ValueError, r"expected numbers less than or equal to 100 as second and third argument to insplit"),
            ("insplit attr:a, 1, 2", ValueError, r"expected '\(' after function"),
            ("insplit(and, 1, 2)", ValueError, r"expected single value as func arg instead of"),
            ("insplit(attr:a and 1, 2)", ValueError, r"expected ',' or '\)' instead of"),
            ("insplit(attr:a, 1, 2) != 123", ValueError, r"insplit can only be compared to a boolean"),
            ("insplit(flag:a, 1, 2)", ValueError, r"expected attr:\* as first argument to insplit"),
            ("insplit(attr:a, 'string', 2)", ValueError, r"expected INT/FLOAT as second and third argument to insplit"),
            ("insplit(attr:a, true, 2)", ValueError, r"expected INT/FLOAT as second and third argument to insplit"),
        ]
        for filter, err, regex in cases:
            with self.subTest(filter):
                fs = _FilterSet()
                with self.assertRaisesRegex(err, regex):
                    fs.parse("f", filter)
                    fs.compile("f", {}, {})

    def test_filter_rule_comparison_errors(self):
        """Test error cases when comparing filter/rule to non-boolean values."""
        cases = [
            ("filter:some_filter = 'string'", ValueError, r"expected BOOL after 'EQ'"),
            ("rule:some_rule != 123", ValueError, r"expected BOOL after 'NE'"),
        ]

        for filter_str, expected_err, regex in cases:
            with self.subTest(filter_str):
                fs = _FilterSet()
                # First define the referenced filter/rule
                fs.parse("some_filter", "attr:test = 1")

                with self.assertRaisesRegex(expected_err, regex):
                    fs.parse("f", filter_str)

    def test_invalid_attributes(self):
        cases = [
            ("8 in attr:a", {"a": "not_a_set"}),
            ("attr:a intersects [5,6]", {"a": 5}),  # a must be a set
            ("attr:a > 8", {"a": set([1, 2, 3])}),
            ("attr:a > 8", {"a": "not_a_number"}),
            ("attr:a > 'not_a_str'", {"a": 8}),
            ("1 in attr:a", {"a": [1, 2, 3]}),  # not a set
            ("'1' in attr:a", {"a": ["1"]}),  # not a set
        ]
        for filter, attrs in cases:
            with self.subTest(filter):
                # Invalid attribute usage causes that particular clause to evaluate to False.
                fs = _FilterSet()
                fs.parse("f", f"({filter}) and attr:true")
                cf = fs.compile("f", {}, {})
                self.assertFalse(cf.eval("", {**attrs, "true": True}))

                fs = _FilterSet()
                fs.parse("f", f"({filter}) or attr:true")
                cf = fs.compile("f", {}, {})
                self.assertTrue(cf.eval("", {**attrs, "true": True}))

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
        self.assertSetEqual(c.flag_refs, {"xyz", "uwu"})
        self.assertSetEqual(c.rule_refs, {("abc", None), ("def", None), ("ghi", "A")})

    def test_insplit_function_distribution_and_exclusivity(self):
        fs = _FilterSet()
        fs.parse("a", "insplit(attr:a, 0, 10)")
        fs.parse("b", "insplit(attr:a, 10, 70)")
        fs.parse("c", "insplit(attr:a, 70, 100)")
        filter_a = fs.compile("a", {}, {})
        filter_b = fs.compile("b", {}, {})
        filter_c = fs.compile("c", {}, {})

        variant_count = [0, 0, 0]
        for id in range(100000):
            if filter_a.eval("", {"a": id}):
                variant_count[0] += 1
            if filter_b.eval("", {"a": id}):
                variant_count[1] += 1
            if filter_c.eval("", {"a": id}):
                variant_count[2] += 1

        self.assertEqual(sum(variant_count), 100000)

        self.assertAlmostEqual(variant_count[0] / 100000, 0.1, delta=0.002)
        self.assertAlmostEqual(variant_count[1] / 100000, 0.6, delta=0.002)
        self.assertAlmostEqual(variant_count[2] / 100000, 0.3, delta=0.002)

    def test_insplit_function_seed_spread(self):
        fs = _FilterSet()
        fs.parse("a", "insplit(attr:a, 0, 50)")
        filter = fs.compile("a", {}, {})

        true_count = 0
        false_count = 0
        for id in range(100000):
            # Here, we keep the attribute value constant and vary the seed and
            # we expect the same distribution as if we were varying the
            # attribute value.
            if filter.eval("", {"a": "constant", "__seed": id}):
                true_count += 1
            else:
                false_count += 1

        self.assertEqual(true_count + false_count, 100000)

        self.assertAlmostEqual(true_count / 100000, 0.5, delta=0.002)
        self.assertAlmostEqual(false_count / 100000, 0.5, delta=0.002)
