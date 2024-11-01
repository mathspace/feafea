from __future__ import annotations
import re
import logging
import datetime
import dill
import os
import time
import json
import jsonschema
import threading
from collections.abc import Callable
from collections import defaultdict
from typing import Any, Iterable, Literal, Mapping, Set
from hashlib import md5

from prometheus_client import Histogram


logger = logging.getLogger(__name__)

type Variant = str | bool | int
type AttributeSet = Set[str] | Set[int]
type AttributeValue = None | str | bool | int | float | AttributeSet
type Attributes = Mapping[str, AttributeValue]
type DictConfig = Mapping[str, Any]


# HACK: Use checksum of this file as the version for calculating the checksum
# of the compiled config. This is a hack to ensure that the compiled config is
# invalidated when the evaluator code changes.
with open(__file__, "rb") as f:
    _feafea_checksum = md5(f.read()).hexdigest()


def _hash_percent(s: Any, seed: str = "") -> float:
    """
    Hashes the given string and seed to a float in the range [0, 100).

    This is not the most efficient hash function but it's stable.
    Stability of this hash function is CRUCIAL. It's used to hash feature flag
    names and target IDs to ensure consistent evaluation of feature flags across
    time, different instances, different python versions, operating systems and
    so on because the distribution of target IDs depend on it.
    """
    return (
        int.from_bytes(
            md5(f"{seed}:{s}".encode("utf-8")).digest(),
            byteorder="big",  # Being explicit to survive future default changes.
            signed=False,  # Being explicit to survive future default changes.
        )
        / (1 << 128)  # md5 hash is 128 bits long.
        * 100
    )


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge the given configs into a single config. Order is not important. Config
    values are shallow copied.

    This merge function is naiive and does not check for validity of keys
    and types. The canonical way of ensuring valid config is by compiling it
    with CompiledConfig.from_dict.
    """
    merged = defaultdict(dict)
    for config in configs:
        for key, value in config.items():
            d = merged[key]
            intersection = d.keys() & value.keys()
            if intersection:
                raise ValueError(f"Duplicate keys: {intersection}")
            d.update(value)
    return merged


type _ParsedFilter = tuple[str, Any, Any]

_filter_token_re = re.compile(
    "|".join(
        f"(?P<{name}>{pattern})"
        # The order of the patterns is important. Generally, we want to match
        # the longest possible token first.
        for name, pattern in [
            # Literals
            ("BOOL", r"true\b|false\b"),
            ("STR", r'"[^"]*"|\'[^\']*\''),
            ("FLOAT", r"-?\d+\.\d+"),
            ("INT", r"-?\d+"),
            # Comparison
            ("EQ", r"="),
            ("NE", r"!="),
            ("LE", r"<="),
            ("GE", r">="),
            ("GT", r">"),
            ("LT", r"<"),
            # Set membership
            ("IN", r"in\b"),
            ("NOTIN", r"not\s+in\b"),
            ("INTERSECTS", r"intersects?\b"),
            ("CONTAINS", r"contains?\b"),
            ("NOTCONTAIN", r"not\s+contain\b"),
            # Boolean
            ("NOT", r"not\b"),
            ("AND", r"and\b"),
            ("OR", r"or\b"),
            # Sets of literal INTS and STRs
            # This can get big so we're going to parse the entire set as a single token
            # and leverage the C implementation of regex to do the heavy lifting.
            (
                "SET",
                r"""\[\s*(?:'[^']*'|"[^"]*"|-?\d+)(?:\s*,\s*(?:'[^']*'|"[^"]*"|-?\d+)\s*)*,?\s*\]""",
            ),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("COMMA", r","),
            ("ATTR", r"attr:[a-zA-Z][a-zA-Z0-9_]*"),
            ("FLAG", r"flag:[a-zA-Z_][a-zA-Z0-9_]*"),
            ("FILTER", r"filter:[a-zA-Z_][a-zA-Z0-9_]*"),
            # Format of Rule reference: rule-name/optional-split-name
            ("RULE", r"rule:[a-zA-Z_][a-zA-Z0-9_]*(?:/[a-zA-Z][a-zA-Z0-9_]*)?"),
            # Functions
            ("FUNC", r"insplit\b"),
            # Whitespace
            ("SPACE", r"\s+"),
            # Ensures we always have at least one token next in the list so we
            # won't have to check if token list is empty before indexing it in parse
            # stage.
            ("END", r"$"),
            # Needed so we always match and don't end up skipping over unexpected
            # characters.
            ("UNEXPECTED", r"."),
        ]
    ),
    flags=re.DOTALL,
)


def _parse_filter(f: str) -> _ParsedFilter:
    """
    Parse the filter string into a tokens and further into a parse tree.
    Root of the parse tree evaluates to a boolean. Each node in the parse tree
    is a tuple of three elements:

    1. The operator (AND, OR, NOT, EQ, NE, LE, GE, GT, LT, CONTAIN, NOTCONTAIN, IN, NOTIN)
    2. The left operand or list of operands in case of OR and AND
    3. The right operand (optional for NOT)

    The left operand is either a node tuple (as above) or a tuple of two elements representing
    a symbol. A symbol is either an attribute, a feature flag or a filter. The first element
    of the tuple is the type of the symbol (ATTR, FLAG, FILTER, RULE) and the
    second element is the name of the symbol.

    The right operand is either a node tuple or a literal value (INT, STR, FLOAT, SET).
    """

    # Tokenize

    tokens = []
    for match in _filter_token_re.finditer(f):
        kind = match.lastgroup
        assert kind is not None
        match match.lastgroup:
            case "SPACE":
                continue
            case "UNEXPECTED":
                raise ValueError(f"unexpected character {match.group()!r} at position {match.start()}")
            case _:
                value = match.group()
                match kind:
                    case "BOOL":
                        value = value == "true"
                    case "STR":
                        value = value[1:-1]
                    case "FLOAT":
                        value = float(value)
                    case "INT":
                        value = int(value)
                    case "SET":
                        # This eval is safe because it's matched by a regex that
                        # does not allow arbitrary code.
                        value = set(eval(value))
                    case "RULE":
                        value = value.split(":", 1)[1]
                        if "/" in value:
                            value = tuple(value.split("/", 1))
                        else:
                            value = (value, None)
                    case "ATTR" | "FLAG" | "FILTER":
                        value = value.split(":", 1)[1]
                    case _:
                        pass
                tokens.append((kind, value))

    # Parse

    def parse_expr(tokens: Any) -> Any:
        if tokens[0][0] == "LPAREN":
            tokens.pop(0)
            expr = parse_or(tokens)
            if tokens[0][0] != "RPAREN":
                raise ValueError("expected ')' instead of '{tokens[0][1]}'")
            tokens.pop(0)
            return expr
        return parse_value_comparison(tokens)

    def parse_or(tokens: Any) -> Any:
        expr = parse_and(tokens)
        if tokens[0][0] != "OR":
            return expr
        operands = [expr]
        while tokens[0][0] == "OR":
            tokens.pop(0)
            operands.append(parse_and(tokens))
        return ("OR", operands, None)

    def parse_and(tokens: Any) -> Any:
        expr = parse_not(tokens)
        if tokens[0][0] != "AND":
            return expr
        operands = [expr]
        while tokens[0][0] == "AND":
            tokens.pop(0)
            operands.append(parse_not(tokens))
        return ("AND", operands, None)

    def parse_not(tokens: Any) -> Any:
        if tokens[0][0] == "NOT":
            tokens.pop(0)
            expr = parse_expr(tokens)
            return ("NOT", expr, None)
        return parse_expr(tokens)

    def parse_func(tokens: Any) -> tuple[Literal["FUNC"], str, list[Any]] | None:
        """
        Parses a function in the form of `func_name(arg1, arg2, ...)` where each
        argument is a single value.
        """
        if tokens[0][0] != "FUNC":
            return None
        _, func_name = tokens.pop(0)
        if tokens[0][0] != "LPAREN":
            raise ValueError(f"expected '(' after function '{func_name}'")
        tokens.pop(0)
        args = []
        while tokens[0][0] != "RPAREN":
            arg = tokens.pop(0)
            if arg[0] not in {"BOOL", "STR", "FLOAT", "INT", "ATTR", "FLAG", "FILTER", "RULE"}:
                raise ValueError(f"expected single value as func arg instead of '{arg[1]}'")
            args.append(arg)
            if tokens[0][0] not in {"COMMA", "RPAREN"}:
                raise ValueError(f"expected ',' or ')' instead of '{tokens[0][1]}' after func arg")
            if tokens[0][0] == "COMMA":
                tokens.pop(0)
        tokens.pop(0)
        return "FUNC", func_name, args

    def parse_value_comparison(tokens: Any) -> Any:
        func_call = parse_func(tokens)
        if func_call is not None:
            _, func_name, func_args = func_call
            match func_name:
                # insplit(attr:*, int, int) -> bool
                # Check if the attribute value, after hashing it to a float
                # between 0 and 100, falls within the range of the two given
                # numbers.
                # If the attribute is a number, it's converted to a string. If
                # the attribute is a set, any match in the set is considered a
                # match.
                case "insplit":
                    if len(func_args) != 3:
                        raise ValueError("insplit func takes three arguments")
                    if func_args[0][0] != "ATTR":
                        raise ValueError("expected attr:* as first argument to insplit")
                    if not all(a[0] in {"INT", "FLOAT"} for a in func_args[1:]):
                        raise ValueError("expected INT/FLOAT as second and third argument to insplit")
                    if not all(float(a[1]) >= 0 for a in func_args[1:]):
                        raise ValueError("expected positive numbers as second and third argument to insplit")
                    if not all(float(a[1]) <= 100 for a in func_args[1:]):
                        raise ValueError("expected numbers less than or equal to 100 as second and third argument to insplit")
                    if not float(func_args[1][1]) < float(func_args[2][1]):
                        raise ValueError("expected second argument to insplit to be less than third argument")
                    if tokens[0][0] in {"EQ", "NE"}:
                        op = tokens.pop(0)[0]
                        if tokens[0][0] != "BOOL":
                            raise ValueError("insplit can only be compared to a boolean")
                        return (op, func_call, tokens.pop(0)[1])
                    return ("EQ", func_call, True)
                case _:
                    assert False, "unreachable"  # pragma: no cover

        if tokens[0][0] in {"FILTER", "RULE"}:
            sym_type, sym_name = tokens.pop(0)
            if tokens[0][0] in {"EQ", "NE"}:
                op = tokens.pop(0)[0]
                if tokens[0][0] != "BOOL":
                    raise ValueError(f"expected BOOL after '{op}' instead of '{tokens[0][1]}'")
                return (op, (sym_type, sym_name), tokens.pop(0)[1])
            return ("EQ", (sym_type, sym_name), True)

        if tokens[0][0] in {"STR", "INT"}:
            # The only time we can have a string or int as the left operand
            # is when we are testing whether it's a member of a list attribute.
            t, value = tokens.pop(0)
            if tokens[0][0] not in {"IN", "NOTIN"}:
                raise ValueError(f"expected in/not in, instead of '{tokens[0][1]}'")
            op = tokens.pop(0)[0]
            if tokens[0][0] != "ATTR":
                raise ValueError(f"expected attr:* instead of '{tokens[0][1]}'")
            attr = tokens.pop(0)[1]
            # In the parse tree, we distinguish between an attribute/flag value
            # being in a set vs a value being in an attribute by using different
            # operators. This simplifies the later stages of processing by not
            # having to check the type of the right operand since the operator
            # already tells us what it is.
            return ({"IN": "CONTAIN", "NOTIN": "NOTCONTAIN"}[op], attr, value)

        if tokens[0][0] not in {"ATTR", "FLAG"}:
            raise ValueError(f"expected attr:*/flag:* instead of '{tokens[0][1]}'")

        sym_type, sym_name = tokens.pop(0)

        if tokens[0][0] in ("LT", "LE", "GT", "GE", "EQ", "NE"):
            op = tokens.pop(0)[0]
            if tokens[0][0] == "BOOL":
                if op not in {"EQ", "NE"}:
                    raise ValueError(f"expected EQ/NE before boolean '{tokens[0][1]}' not {op}")
            elif tokens[0][0] not in {"INT", "STR", "FLOAT"}:
                raise ValueError(f"expected INT/STR/FLOAT after '{op}' not '{tokens[0][1]}'")
            t, value = tokens.pop(0)
            if sym_type == "FLAG" and t not in {"BOOL", "STR", "INT"}:
                raise ValueError("expected BOOL/STR/INT for feature flag comparison")
            return (op, (sym_type, sym_name), value)

        elif tokens[0][0] in {"IN", "NOTIN"}:
            op = tokens.pop(0)[0]
            if tokens[0][0] != "SET":
                raise ValueError(f"expected a set ([...]) after {op!r}")
            values = tokens.pop(0)[1]
            if len(set(type(v) for v in values)) > 1:
                # The token regex for set ensures that at least one element is
                # present so we don't have to check for empty set here.
                raise ValueError("set values must all be of the same type")
            return (op, (sym_type, sym_name), values)

        elif tokens[0][0] == "INTERSECTS":
            op = tokens.pop(0)[0]
            if sym_type != "ATTR":
                raise ValueError("expected attr:* on left-hand-side of INTERSECTS")
            t, value = tokens.pop(0)
            if t != "SET":
                raise ValueError("expected set on right-hand-side of INTERSECTS")
            return (op, (sym_type, sym_name), value)

        else:
            return ("EQ", (sym_type, sym_name), True)

    e = parse_or(tokens)
    if tokens[0][0] != "END":
        raise ValueError(f"unexpected token '{tokens[0][1]}'")
    return e


class _CompiledFilter:
    """
    The compiled filter function.
    """

    __slots__ = ("eval", "flag_refs", "rule_refs", "uses_insplit")

    # The compiled filter function.
    eval: Callable[[TargetID, Attributes], bool]
    # The set of flag names referenced in the filter mapped to the inferred type
    # of the flag. This is later used when compiling rules to ensure a flag has
    # the same type as the inferred type from the filter. It's also used to ensure
    # no circular references to feature flags occur.
    flag_refs: dict[str, type]
    # The set of rule names/splits referenced in the filter. This is used to ensure
    # no circular references to rules occur.
    rule_refs: set[tuple[str, str | None]]
    # Whether the filter uses the insplit function. This is used to disallow the
    # simultaneous use of insplit and split rules in the same rule.
    uses_insplit: bool

    # Filter references are not needed since filter references in a filter are
    # inlined during compilation and any circular references checked.


class _FilterSet:
    """
    The set of filters. This is used to parse and compile filters into a callable
    form. The set of filters is also responsible for inlining filter references
    and ensuring no circular references occur.
    """

    __slots__ = ("_filters", "_sets")

    _py_op_map = {
        "AND": "and",
        "OR": "or",
        "EQ": "==",
        "NE": "!=",
        "LE": "<=",
        "GE": ">=",
        "GT": ">",
        "LT": "<",
        "CONTAIN": "in",
        "NOTCONTAIN": "not in",
        "IN": "in",
        "NOTIN": "not in",
    }

    def __init__(self):
        self._filters: dict[str, _ParsedFilter] = {}
        self._sets: list[set] = []

    def _extract_set_literals(self, f):
        """
        Extract set literals in the parse tree and replace them with an integer index
        into the _sets list. This is used to separate the building of the set literals
        from the evaluation of the filter to improve evaluation performance.
        """
        f0, f1, f2 = f
        if f0 in {"IN", "NOTIN", "INTERSECTS"}:
            self._sets.append(f2)
            return (f0, f1, len(self._sets) - 1)
        if f0 in {"AND", "OR"}:
            return f0, [self._extract_set_literals(i) for i in f1], f2
        if f0 == "NOT":
            return f0, self._extract_set_literals(f1), f2
        return f0, f1, f2

    def parse(self, name: str, filter: str):
        pf = _parse_filter(filter)
        pf = self._extract_set_literals(pf)
        self._filters[name] = pf

    def _optimize(self, f: _ParsedFilter) -> _ParsedFilter:
        """
        Rewrites the parse tree to optimize it for evaluation.
        """

        def _reorder_bool_operands(n: _ParsedFilter) -> tuple[_ParsedFilter, int]:
            """
            Reorder the operands of a boolean operator to ensure that the left
            operand is the fastest to evaluate. This is done by counting the number
            of flag and rule references in each operand and ordering them by the
            number of references.
            """
            f0, f1, f2 = n
            if f0 in {"AND", "OR"}:
                weighted_f1 = [_reorder_bool_operands(i) for i in f1]
                weighted_f1.sort(key=lambda x: x[1])
                f1 = [i[0] for i in weighted_f1]
                return (f0, f1, f2), sum(i[1] for i in weighted_f1)
            if isinstance(f1, tuple):
                # TODO at this stage, our horizon of how deep the references go
                # stop at the first rule/flag reference. We can optimize this
                # further by looking deeper into the tree after all flags and
                # rules are parsed in config compilation stage.
                if f1[0] == "RULE":
                    return n, 1
                if f1[0] == "FLAG":
                    # Since flags tend to have multiple rules applied to them, we are
                    # going to heuristically assume that flags are more likely to be
                    # more expensive to evaluate than rules.
                    return n, 3
            return n, 0

        f = _reorder_bool_operands(f)[0]

        return f

    def _inline(
        self,
        f: _ParsedFilter,
        seen: set[str],
        sets: list[set],
        flag_refs: dict[str, type],
        rule_refs: set[tuple[str, str | None]],
    ) -> _ParsedFilter:
        """
        Inline filter references. This is an optimiztion step to avoid function
        calls and improve performance of filter evaluation. This method also
        replaces sets with integer indices into the _sets list, and keeps track
        of referenced flags and rules.
        """
        f0, f1, f2 = f

        # Keep track of referenced flags and rules.
        if isinstance(f1, tuple):
            if f1[0] == "RULE":
                rule_refs.add(f1[1])
            elif f1[0] == "FLAG":
                # Infer the type of the flag based on the type of the literal
                # it's being compared to.
                if f0 in {"IN", "NOTIN"}:
                    t = type(next(iter(self._sets[f2])))
                else:
                    t = type(f2)
                name = f1[1]
                existing_type = flag_refs.get(name)
                # Ensure all usage of a flag in a filter agree on the inferred type.
                if existing_type is not None and existing_type != t:
                    raise ValueError(f"referenced feature flag {name} has conflicting inferred types {existing_type} and {t}")
                flag_refs[name] = t

        if f0 == "EQ" and f1[0] == "FILTER":
            filter_name = f1[1]
            if filter_name in seen:
                raise ValueError(f"circular reference in filter {filter_name}")
            if filter_name not in self._filters:
                raise ValueError(f"unknown filter {filter_name}")
            assert isinstance(filter_name, str)
            seen.add(filter_name)
            return self._inline(self._filters[filter_name], seen, sets, flag_refs, rule_refs)

        if f0 in {"IN", "NOTIN", "INTERSECTS"}:
            assert isinstance(f2, int)
            sets.append(self._sets[f2])
            return (f0, f1, len(sets) - 1)

        if f0 in {"AND", "OR"}:
            return (
                f0,
                [self._inline(i, seen, sets, flag_refs, rule_refs) for i in f1],
                f2,
            )

        if f0 == "NOT":
            return (f0, self._inline(f1, seen, sets, flag_refs, rule_refs), f2)

        return (f0, f1, f2)

    def _pythonize(self, f: _ParsedFilter, visited_insplit: Callable[[], None]) -> str:
        """
        Convert the parse tree into a python expression.

        Enough type checks are done on attribute during runtime to ensure that
        the python expression never raises an exception.
        """
        op, arg1, arg2 = f
        match op:
            case "AND" | "OR":
                # We know for certain that lhs and rhs are booleans. AND/ORing them
                # together will always yield a boolean and will never raise an exception.
                return f" {self._py_op_map[op]} ".join(f"({self._pythonize(a, visited_insplit)})" for a in arg1)
            case "NOT":
                # We know for certain that the argument is a boolean. Negating it
                # will always yield a boolean and will never raise an exception.
                return f"(not ({self._pythonize(arg1, visited_insplit)}))"
            case "EQ" | "NE" | "LE" | "GE" | "GT" | "LT" | "IN" | "NOTIN" | "INTERSECTS":
                sym_type, sym_name, *sym_args = arg1
                match sym_type:
                    case "FUNC":
                        func_name, func_args = sym_name, sym_args[0]
                        if func_name == "insplit":
                            visited_insplit()
                            # We have two types of attributes we need to handle,
                            # sets and single values. All values and set values
                            # must be handled. For single values, we convert
                            # them to strings, hash them to a float between 0
                            # and 100 and check if they fall within the range of
                            # the two given numbers. For set values, we check if
                            # any value in the set after conversion to string
                            # and hashing falls within the range of the two
                            # given numbers.
                            #
                            # A __seed attribute is expected to exist in the
                            # attributes dict. This will be set by the rule
                            # evaluator.
                            attr_name, min_val, max_val = func_args

                            all_to_set = f"attributes[{attr_name[1]!r}] if isinstance(attributes.get({attr_name[1]!r}), set) else set([attributes.get({attr_name[1]!r})])"
                            any_in_set = f"any(v is not None and {min_val[1]!r} <= _hash_percent(str(v), attributes.get('__seed', '')) < {max_val[1]!r} for v in ({all_to_set}))"
                            return f"({any_in_set} {self._py_op_map[op]} {arg2!r})"
                        else:
                            assert False, "unreachable"  # pragma: no cover
                    case "FLAG":
                        # Further stages of compilation ensure that the referenced flag here
                        # exists and has the same type as the inferred type from the filter.
                        # This will therefore never raise an exception.
                        lhs = f"flags[{sym_name!r}].eval(target_id, attributes).variant"
                        if op in {"IN", "NOTIN"}:
                            return f"{lhs} {self._py_op_map[op]} sets[{arg2!r}]"
                        else:
                            return f"{lhs} {self._py_op_map[op]} {arg2!r}"
                    case "ATTR":
                        lhs = f"attributes[{sym_name!r}]"
                        if op in {"IN", "NOTIN"}:
                            # We check to ensure that the RHS is a str or int during runtime.
                            # Since `str/int in/not in set()` will always yield a boolean, the
                            # expression will never raise an exception.
                            return f"(isinstance(attributes.get({sym_name!r}), (str, int)) and {lhs} {self._py_op_map[op]} sets[{arg2!r}])"
                        elif op == "INTERSECTS":
                            # We check to ensure that the LHS is a set during runtime.
                            # Since `set() & set()` will always yield a set and will never raise
                            # an exception.
                            return f"(isinstance(attributes.get({sym_name!r}), set) and bool({lhs} & sets[{arg2!r}]))"
                        else:
                            # Not all types are comparable with each other. So we need to
                            # ensure that the types are compatible before comparing them.
                            if isinstance(arg2, str):
                                compat_types = "str"
                            elif type(arg2) is bool:
                                return f"((type(attributes.get({sym_name!r})) is bool) and {lhs} {self._py_op_map[op]} {arg2!r})"
                            elif isinstance(arg2, (int, float)):
                                compat_types = "(int, float)"
                            else:
                                assert False, "unreachable"  # pragma: no cover
                            return f"(isinstance(attributes.get({sym_name!r}), {compat_types}) and {lhs} {self._py_op_map[op]} {arg2!r})"
                    case "RULE":
                        # Further stages of compilation ensure that the referenced rule here
                        # exists. This will therefore never raise an exception.
                        rule_name, split_name = sym_name
                        if split_name is not None:
                            return f"rules[{rule_name!r}].eval(target_id, attributes)[0] is True"
                        else:
                            return f"rules[{rule_name!r}].eval(target_id, attributes) == (True, {split_name!r})"
                    case _:  # pragma: no cover
                        assert False, "unreachable"  # pragma: no cover
            case "CONTAIN" | "NOTCONTAIN":
                # We check to ensure that the RHS is a set during runtime.
                # Since arg2 can only be an int or a str, `arg2 in/not in set()`
                # will always yield a boolean and will never raise an exception.
                return f"(isinstance(attributes.get({arg1!r}), set) and {arg2!r} {self._py_op_map[op]} attributes[{arg1!r}])"
            case _:  # pragma: no cover
                assert False, "unreachable"  # pragma: no cover

    def compile(self, name: str, flags: dict[str, CompiledFlag], rules: dict[str, _CompiledRule]) -> _CompiledFilter:
        """
        Compile the filter into a callable form.
        """
        comp_filter = _CompiledFilter()
        comp_filter.uses_insplit = False

        seen = set()
        sets: list[set] = []
        flag_refs: dict[str, type] = {}
        rule_refs: set[tuple[str, str | None]] = set()
        inlined = self._inline(self._filters[name], seen, sets, flag_refs, rule_refs)
        optimized = self._optimize(inlined)

        def visited_insplit():
            comp_filter.uses_insplit = True

        py_expr = self._pythonize(optimized, visited_insplit)
        code_str = f"""
def a(target_id, attributes):
    return {py_expr}
"""
        py_code = compile(code_str, "", "exec")
        _locals = {}
        _globals = {
            "sets": sets,
            "flags": flags,
            "rules": rules,
            "_hash_percent": _hash_percent,
        }
        exec(py_code, _globals, _locals)
        comp_filter.eval = _locals["a"]
        comp_filter.flag_refs = flag_refs
        comp_filter.rule_refs = rule_refs
        return comp_filter


with open(os.path.join(os.path.dirname(__file__), "config_schema.json")) as f:
    _config_schema = json.load(f)


type TargetID = str


class _CompiledRule:
    """
    Flag independent compiled rule that evaluates to a boolean indicating
    whether the rule matched. It also returns split name if the rule is a split
    rule. If the rule does not apply, it returns (False, None).
    """

    __slots__ = ("name", "eval", "split_names", "_compiled_filter")
    name: str
    eval: Callable[[TargetID, Attributes], tuple[bool, str | None]]
    split_names: set[str]
    _compiled_filter: _CompiledFilter


class _CompiledFlagRule:
    """
    Flag specific compiled rule that evaluates to a variant or a tuple of
    variant and split name if the rule is a split rule. If the rule does not
    apply, it returns None.
    """

    __slots__ = ("rule_name", "eval")
    rule_name: str
    eval: Callable[[TargetID, Attributes], Variant | tuple[Variant, str, list[AttributeValue]] | None]


class CompiledFlag:
    __slots__ = (
        "name",
        "type",
        "variants",
        "default",
        "metadata",
        "_rules",
        "_all_rules",
        "_all_flags",
    )
    name: str
    type: type[Variant]
    variants: set[Variant]
    default: Variant
    metadata: dict[str, str]
    _rules: list[_CompiledFlagRule]
    _all_rules: dict[str, _CompiledRule]
    _all_flags: dict[str, CompiledFlag]

    def eval(self, target_id: TargetID, attributes: Attributes) -> FlagEvaluation:
        e = FlagEvaluation()
        e.default = self.default
        e.flag = self.name
        e.target_id = target_id
        e.split = ""
        e.split_attribute_values = []
        for rule in self._rules:
            v = rule.eval(target_id, attributes)
            if v is not None:
                e.rule = rule.rule_name
                if isinstance(v, tuple):
                    e.variant, e.split, e.split_attribute_values = v
                    e.reason = "split_rule"
                else:
                    e.variant = v
                    e.reason = "const_rule"
                break
        else:
            e.rule = ""
            e.reason = "default"
            e.variant = self.default
        return e


def _unix_seconds_from_tz_time(s) -> float:
    """
    Parse the given ISO 8601 time string and return the number of seconds since
    the unix epoch.
    """
    t = datetime.datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError("Timezone missing")
    return t.timestamp()


def _seconds_from_human_duration(s) -> float:
    """
    Parse the given human readable duration string and return the number of
    seconds.
    """
    m = re.match(r"^\s*(?:(?P<days>\d+)\s*d)?\s*(?:(?P<hours>\d+)\s*h)?\s*(?:(?P<minutes>\d+)\s*m)?\s*$", s)
    if not m:
        raise ValueError(f"Invalid timedelta {s}")
    return datetime.timedelta(
        days=int(m.group("days") or 0),
        hours=int(m.group("hours") or 0),
        minutes=int(m.group("minutes") or 0),
    ).total_seconds()


class CompiledConfig:
    """
    Compiled config that can be used to evaluate feature flags.
    """

    __slots__ = ("flags", "checksum", "_feafea_checksum")
    flags: dict[str, CompiledFlag]

    # Checksum of the config. This can be used as a hash key to cache the compiled
    # config. The checksum is derived from the dict config as well as the feafea
    # version.
    checksum: str

    # Checked by the evaluator to ensure that the compiled config is compatible
    # with the evaluator.
    _feafea_checksum: str

    @staticmethod
    def from_bytes(b: bytes) -> CompiledConfig:
        obj = dill.loads(b)
        assert isinstance(obj, CompiledConfig)
        return obj

    def to_bytes(self) -> bytes:
        return dill.dumps(self)

    @staticmethod
    def from_dict(c: DictConfig) -> CompiledConfig:
        """
        Compile the config into a format that can be loaded into the evaluator.
        """
        jsonschema.validate(c, _config_schema)

        compiled_rules: dict[str, _CompiledRule] = {}

        # Build compiled flags.

        flags: dict[str, CompiledFlag] = {}
        for flag_name, f in c.get("flags", {}).items():
            if "alias" in f:
                continue
            flag = CompiledFlag()
            flag.name = flag_name
            if isinstance(f["default"], bool):
                flag.variants = {True, False}
            else:
                if f["default"] not in f["variants"]:
                    raise ValueError("flag default value must be one of the variants")
                flag.variants = set(f["variants"])
            flag.type = type(f["default"])
            flag.default = f["default"]
            flag.metadata = f.get("metadata", {})
            flag._rules = []
            # Each flag holds a reference to all the flags and rules in the
            # config. This is necessary as filters may reference other flags and
            # rules.
            flag._all_flags = flags
            flag._all_rules = compiled_rules
            flags[flag_name] = flag

        # Build compiled flags for aliases.

        alias_names = set(n for n, f in c["flags"].items() if "alias" in f)
        for flag_name, f in c["flags"].items():
            if "alias" not in f:
                continue
            alias = f["alias"]
            if alias not in flags:
                raise ValueError(f"unknown flag {alias}")
            if alias in alias_names:
                raise ValueError(f"flag {alias} is already an alias")
            alias_flag = flags[alias]
            flag = CompiledFlag()
            # An alias shares everything with its referenced flag except the
            # name.
            for k in CompiledFlag.__slots__:
                setattr(flag, k, getattr(alias_flag, k))
            flag.name = flag_name
            flags[flag_name] = flag

        # Compile set of all flag,variant pairs for later validation of rule
        # references.

        all_variants = set((flag.name, v) for flag in flags.values() for v in flag.variants)

        # Parse all the named filters and validate usage of all flags and rules.

        filters = _FilterSet()
        for filter_name, f in c.get("filters", {}).items():
            filters.parse(filter_name, f)

        # Compile all the rules.

        rules = list(c.get("rules", {}).items())
        rules.sort(key=lambda x: (-x[1].get("priority", 0), x[0]))
        for rule_name, r in rules:
            # Flag independent rule.
            compiled_rule = _CompiledRule()

            # Rule validations

            referenced_variants = set((flag, variant) for flag, variant in r.get("variants", {}).items())
            referenced_variants.update((flag, variant) for s in r.get("splits", []) for flag, variant in s.get("variants", {}).items())
            if referenced_variants - all_variants:
                raise ValueError(f"unknown flag/variant in rule {rule_name}")
            if "splits" in r:
                percentage_sum = sum(v["percentage"] for v in r["splits"])
                if percentage_sum == 0 or percentage_sum > 100:
                    raise ValueError("split percentages must sum to 100 or less")

            # Rule compilation.

            py_flag = {f: [] for f in set(flag for flag, _ in referenced_variants)}
            py_common = []

            globals: dict[str, Any] = {
                "_hash_percent": _hash_percent,
                "flags": flags,
                "rules": compiled_rules,
            }

            if not r.get("enabled", True):
                py_common += ["return None"]

            if "filter" in r:
                filters.parse("rule:" + rule_name, r["filter"])
                cf = filters.compile("rule:" + rule_name, flags, compiled_rules)
                compiled_rule._compiled_filter = cf
                globals["match"] = cf.eval
                py_common += [f"attributes['__seed'] = {r.get('split_group', rule_name)!r}"]
                py_common += ["if not match(target_id, attributes): return None"]

            if "schedule" in r:
                start = _unix_seconds_from_tz_time(r["schedule"]["start"])
                end = _unix_seconds_from_tz_time(r["schedule"]["end"])
                if start >= end:
                    raise ValueError("start must be before end")
                ramp_up = _seconds_from_human_duration(r["schedule"].get("ramp_up", "0m"))
                ramp_down = _seconds_from_human_duration(r["schedule"].get("ramp_down", "0m"))
                if ramp_up + ramp_down > end - start:
                    raise ValueError("ramp_up + ramp_down must be less than end - start")
                py_common += [f"if attributes['__now'] < {start!r}: return None"]
                py_common += [f"if attributes['__now'] >= {end!r}: return None"]

                if ramp_up > 0 or ramp_down > 0:
                    schedule_seed = rule_name + "\0schedule"
                    py_common = [f"schedule_target_percent = _hash_percent(target_id, seed={schedule_seed!r})"]
                if ramp_up > 0:
                    py_common += [f"ramp_up_percent = 100 * (attributes['__now'] - {start!r}) / {ramp_up!r}"]
                    py_common += ["if schedule_target_percent >= ramp_up_percent: return None"]
                if ramp_down > 0:
                    py_common += [f"ramp_down_percent = 100 * ({end!r} - attributes['__now']) / {ramp_down!r}"]
                    py_common += ["if schedule_target_percent >= ramp_down_percent: return None"]

            compiled_rule.split_names = set()
            if "splits" in r:
                if hasattr(compiled_rule, "_compiled_filter") and compiled_rule._compiled_filter.uses_insplit:
                    raise ValueError("insplit filter and split rules cannot be used in the same rule")
                compiled_rule.split_names.update(s["name"] for s in r["splits"] if "name" in s)
                split_seed = r.get("split_group", rule_name)
                split_attribute = r.get("split_attribute")
                # Build a map of (attribute value -> percent) for each attribute value.
                if split_attribute is None:
                    py_common += [f"split_percents = {'{'}target_id:_hash_percent(target_id, seed={split_seed!r}){'}'}"]
                else:
                    py_common += [f"split_attribute = attributes[{split_attribute}] if isinstance(attributes.get({split_attribute}), set) else [attributes.get({split_attribute})]"]
                    py_common += [f"split_percents = {'{'}(v:_hash_percent(v, seed={split_seed!r}) for v in split_attribute)"]
                comulative_percentage_start = 0
                comulative_percentage_end = 0
                for split in r["splits"]:
                    comulative_percentage_end += split["percentage"]
                    matched_attr_values = f"matched_attr_values = [v for v, p in split_percents.items() if {comulative_percentage_start!r} <= p < {comulative_percentage_end!r}]"
                    py_common += ["#FIRULE " + matched_attr_values]
                    py_common += [f"#FIRULE if matched_attr_values: return (True, {split.get("name", "")!r})"]
                    for flag, variant in split.get("variants", {}).items():
                        py_flag[flag].append(matched_attr_values)
                        py_flag[flag].append(f"if matched_attr_values: return ({variant!r}, {split.get("name", "")!r}, matched_attr_values)")
                    comulative_percentage_start += split["percentage"]

            # Compile the flag independent rule.

            lines = ["def a(target_id, attributes):"]
            for line in py_common:
                line = re.sub(r"return None$", "return (False, None)", line)
                line = re.sub(r"^#FIRULE ", "", line)
                lines.append(f"  {line}")
            if "splits" not in r:
                lines.append("  return (True, None)")
            lines.append("  return (False, None)")
            code = compile("\n".join(lines), "", "exec")
            _locals = {}
            exec(code, globals, _locals)
            compiled_rule.name = rule_name
            compiled_rule.eval = _locals["a"]
            compiled_rules[rule_name] = compiled_rule

            # Compile the flag dependent rules.

            for flag, variant in r.get("variants", {}).items():
                py_flag[flag].append(f"return {variant!r}")

            for flag, py_lines in py_flag.items():
                if not py_lines:
                    assert False, "unreachable"  # pragma: no cover
                # Returns variant | (variant, split_name, split_attr_values) | None
                # - variant: The variant evaluated for non-split rule
                # - (variant, split_name, split_attr_values): The variant evaluated for split rule
                # - None: The rule does not apply
                lines = ["def a(target_id, attributes):"]
                lines += list(f"  {line}" for line in (py_common + py_lines))
                lines += ["  return None"]
                lines_str = "\n".join(lines)
                code = compile(lines_str, "", "exec")
                _locals = {}
                exec(code, globals, _locals)
                reval = _CompiledFlagRule()
                reval.rule_name = rule_name
                reval.eval = _locals["a"]
                flags[flag]._rules.append(reval)

        # Check all referenced rules and flags in all filters that are reachable
        # from rules, are valid.

        all_ref_rules = set(rname for r in compiled_rules.values() if hasattr(r, "_compiled_filter") for rname in r._compiled_filter.rule_refs)
        unknown_rules = all_ref_rules - (
            set((crname, None) for crname, cr in compiled_rules.items() if not cr.split_names) | set((crname, sname) for crname, cr in compiled_rules.items() for sname in cr.split_names)
        )
        if unknown_rules:
            raise ValueError(f"unknown rules {unknown_rules} referenced")
        all_ref_flags = set(fname for r in compiled_rules.values() if hasattr(r, "_compiled_filter") for fname in r._compiled_filter.flag_refs)
        unknown_flags = all_ref_flags - set(flags)
        if unknown_flags:
            raise ValueError(f"unknown flags {unknown_flags} referenced")

        # Ensure inferred type of flag references match the actual flag type.

        for r in compiled_rules.values():
            if not hasattr(r, "_compiled_filter"):
                continue
            for fname, ftype in r._compiled_filter.flag_refs.items():
                if flags[fname].type != ftype:
                    raise ValueError(f"referenced feature flag {fname} has inferred type {ftype} but actual type {flags[fname].type}")

        # Ensure no circular references to feature flags or rules from rule filters.

        def _check_circular_ref(
            entity: CompiledFlag | _CompiledFlagRule | _CompiledRule,
            must_exclude: tuple[set[str], set[str]],
        ):
            flag_refs, rule_refs = must_exclude
            if isinstance(entity, CompiledFlag):
                flag_refs = flag_refs | {entity.name}
                for r in entity._rules:
                    if r.rule_name in rule_refs:
                        raise ValueError(f"circular reference in flag {entity.name}")
                    _check_circular_ref(compiled_rules[r.rule_name], (flag_refs, rule_refs))
            elif isinstance(entity, _CompiledRule):
                if hasattr(entity, "_compiled_filter"):
                    rule_refs = rule_refs | {entity.name}
                    for fname in entity._compiled_filter.flag_refs:
                        if fname in flag_refs:
                            raise ValueError(f"circular reference in rule {entity.name}")
                        _check_circular_ref(flags[fname], (flag_refs, rule_refs))
                    for rname, _ in entity._compiled_filter.rule_refs:
                        if rname in rule_refs:
                            raise ValueError(f"circular reference in rule {entity.name}")
                        _check_circular_ref(compiled_rules[rname], (flag_refs, rule_refs))
            else:  # pragma: no cover
                assert False, "unreachable"  # pragma: no cover

        for r in compiled_rules.values():
            _check_circular_ref(r, (set(), set()))

        cc = CompiledConfig()
        cc.flags = flags
        cc._feafea_checksum = _feafea_checksum
        chk = md5()
        chk.update(cc._feafea_checksum.encode())
        chk.update(b"-")
        chk.update(json.dumps(c, sort_keys=True).encode())
        cc.checksum = chk.hexdigest()
        return cc


class FlagEvaluation:
    """
    The result of evaluating a flag.
    """

    __slots__ = (
        "flag",
        "variant",
        "default",
        "target_id",
        "rule",
        "reason",
        "split",
        "split_attribute_values",
        "timestamp",
    )
    flag: str
    variant: Variant
    default: Variant
    target_id: str
    rule: str | Literal[""]
    reason: Literal["split_rule", "const_rule", "default"]
    split: str | Literal[""]
    split_attribute_values: list[AttributeValue]
    timestamp: float


_prom_labels = ["flag", "variant", "default", "rule", "reason", "split"]
_prom_eval_duration = Histogram(
    "feafea_evaluation_duration_seconds",
    "Flag evaluation duration in seconds",
    buckets=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    labelnames=_prom_labels,
)


class Evaluator:
    """
    The evaluator is responsible for evaluating feature flags and rules. The
    evaluator is thread-safe.

    If record_evaluations is set to True, the evaluator will record all flag
    evaluations and stores them in memory. A call to pop_evaluations will
    return all recorded evaluations and clear the recorded evaluations.

    NOTE: There is no limit to how many evaluations are stored in memory. It's
    the responsibility of the caller to ensure that the memory usage is within
    acceptable limits and call pop_evaluations frequently to clear the recorded
    evaluations.
    """

    def __init__(self, record_evaluations: bool = False):
        self._record_evaluations = record_evaluations
        if record_evaluations:
            self._evaluations_mu = threading.Lock()
            self._evaluations: list[FlagEvaluation] = []

    @staticmethod
    def _validate_attributes_type(attributes: Attributes):
        if not isinstance(attributes, dict):
            raise TypeError(f"attributes must be a dict, not {type(attributes).__name__}")
        for k, v in attributes.items():
            if not isinstance(k, str):
                raise TypeError(f"attribute key must be a string, not {type(k).__name__}")
            if not isinstance(v, (str, int, float, bool, set, type(None))):
                raise TypeError(f"attribute value must be a string, int, float, bool, set, None, not {type(v).__name__}")
            if isinstance(v, set) and len(v) > 0:
                fel = next(iter(v))
                if not isinstance(fel, (str, int)):
                    raise TypeError(f"set values must be strings and ints not {type(fel).__name__}")
                if not all(isinstance(e, type(fel)) for e in v):
                    raise TypeError("set values must be of the same type")

    def _record_eval_metrics(self, e: FlagEvaluation, dur: float):
        labels = {
            "flag": e.flag,
            "variant": str(e.variant),
            "default": str(e.default),
            "rule": e.rule,
            "reason": e.reason,
            "split": e.split,
        }
        _prom_eval_duration.labels(**labels).observe(dur)

    def detailed_evaluate_all(self, config: CompiledConfig, names: Iterable[str], id: str, attributes: Attributes = {}) -> dict[str, FlagEvaluation]:
        """
        Evaluate all flags for the given id and attributes and return
        FlagEvaluations. detailed_evaluate_all is thread-safe.
        """
        assert config._feafea_checksum == _feafea_checksum, "incompatible config"
        if not isinstance(id, str):
            raise TypeError(f"id must be a string, not {type(id).__name__}")
        self._validate_attributes_type(attributes)
        merged_attributes = {"__now": time.time(), **attributes}
        now = merged_attributes["__now"]

        # Evaluate flags

        fe: dict[str, FlagEvaluation] = {}
        for name in set(names):
            flag = config.flags.get(name)
            if flag is None:
                raise ValueError(f"Flag {name} does not exist in the config")
            start = time.perf_counter()
            e = flag.eval(id, merged_attributes)
            e.timestamp = now
            dur = time.perf_counter() - start
            self._record_eval_metrics(e, dur)
            fe[name] = e

        # Record evaluations

        if self._record_evaluations:
            with self._evaluations_mu:
                self._evaluations.extend(fe.values())

        return fe

    def pop_evaluations(self) -> list[FlagEvaluation]:
        """
        Pop all recorded evaluations since the last call and return them.
        pop_evaluations is thread-safe.
        """
        if not self._record_evaluations:
            return []
        with self._evaluations_mu:
            es = self._evaluations
            self._evaluations = []
            return es

    def evaluate(self, config: CompiledConfig, name: str, id: str, attributes: Attributes = {}) -> Variant:
        """
        Evaluate the given flag. evaluate is thread-safe.

        name: The name of the flag.
        id: The id of the flag.
        attributes: The attributes to evaluate the flag with.
        """
        return next(iter(self.detailed_evaluate_all(config, [name], id, attributes).values())).variant

    def evaluate_all(self, config: CompiledConfig, names: Iterable[str], id: str, attributes: Attributes = {}) -> dict[str, Variant]:
        """
        Evaluate all flags for the given id and attributes. evaluate_all is
        thread-safe.
        """
        return {n: e.variant for n, e in self.detailed_evaluate_all(config, names, id, attributes).items()}
