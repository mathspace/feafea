# FeaFea

FeaFea is a simple but powerful feature flag system. In its most basic form,
given an some key/value pairs of attributes, it evaluates rules based on those
attributes to determine the value (aka variant) of a feature flag.

## Quick Start

Install the package:

```bash
pip install https://github.com/mathspace/feafea/archive/main.zip
```

```python
from feafea import Evaluator, CompiledConfig

compiled_config = CompiledConfig.from_dict({
  "flags": {
    "enable_feature_X": {
      "default": False
    },
    "dashboard_style": {
      "variants": ["A", "B", "C"],
      "default": "A",
      "metadata": {
        "deprecated": True,
        "descripition": "User dashboard style"
      }
    }
  },
  "rules": {
    "enable_feature_X_for_beta_users": {
      "filter": "attr:user_type = 'beta'",
      "variants": {
        "enable_feature_X": True
      }
    }
  }
})

ev = Evaluator()

assert ev.evaluate(compiled_config, "enable_feature_X", "user_1", {"user_type": "alpha"}) == False
assert ev.evaluate(compiled_config, "enable_feature_X", "user_2", {"user_type": "beta"}) == True
```

## Concepts

- **Feature Flag**: A named flag of either int, str or bool type. It has two or
  more variants and a default value defined. Bool flags always have two
  variants: true and false. A flag can optionally have metadata which is a set
  of custom key-value pairs.
- **Variant**: A possible value of a feature flag.
- **Attributes**: A set of key-value pairs. These are used in
  evaluation rules to determine the value of a feature flag. E.g. user type,
  country, etc.
- **Filter**: A boolean expression. It is used to match against the attributes,
  rules, other filters and flags. Both inline and named filters are supported.
- **Rule**: A rule is evaluated against a target to determine the variants of
  the feature flags. A rule is made up of an optional filter.
  The rule applies if all of its components match.
- **Evaluation**: The process of determining the value of a feature flag for a
  target based on the attributes and rules. Result of an evaluation is a
  variant. Detailed information about an evaluation can be retrieved by calling
  `Evaluator.detailed_evaluate_all`.

## Configuration

FeaFea doesn't define a configuration serialization format. Instead, it accepts
a python dictionary of a `DictConfig`. Where the dictionary comes from is
up to the user. The expected structure is JSON serializable (i.e. no functions,
classes, etc), thus it's common to use JSON or YAML files to define the
configuration.

The configuration must first be compiled into a `CompiledConfig` object. This
object is then loaded into an `Evaluator` object. The `Evaluator` object is used
to evaluate the feature flags for a target. The `Evaluator` object is
thread-safe and can be shared across multiple threads. Configurations can be
loaded into the `Evaluator` object at any time in a thread-safe manner.

The reason for compliation is to speed up the evaluation process. The compiled
config object is immutable and can be reused in multiple evaluators (albeit
uncommon). The compiled config can also be serialized and deserialized to avoid
recompiling the configuration every time the application starts.
`CompiledConfig.to_bytes` and `CompiledConfig.from_bytes` can be used for this
purpose.

A convenience function `merge_configs` is provided to merge multiple
`DictConfig`s into a single `DictConfig`. This is useful if the complete config
is composed of multiple files. A common pattern is to have flags and rules in
separate files.

## Flag Alias

A feature flag alias is an alternative name for a feature flag. It is useful for
backward compatibility when renaming a feature flag. Here's an example:

```json
{
  "flags": {
    "old_name": {
      "default": false
    },
    "new_name": {
      "alias": "old_name"
    }
  }
}
```

The alias flag behaves exactly like the original flag in every way. It can be
referenced in filters and rules. Any rule that applies to the original flag
also applies to all of its aliases and vice-versa. An alias can't reference
another alias.

## Filter

A filter can either be directly defined inside of a rule or it can be named and
referenced by the rule and other filters. Filters are boolean expressions that
are evaluated against the attributes of a target. Here is an example of a filter:

```
attr:user_type = 'beta' and attr:country in ['US', 'AU']
```

The above filter matches if the target has a user type of 'beta' and is in the
US or AU. The `attr:` prefix is used to reference the attributes of a target.
The attributes are passed in as a dictionary when evaluating the feature flags:

```python
evaluator.Evaluate("feature_flag", "target_id", {"user_type": "beta", "country": "US"})
```

Here is a rule with the filter defined directly:

```json
{
  "rules": {
    "enable_feature_X_for_beta_users": {
      "filter": "attr:user_type = 'beta'",
      "variants": {
        "enable_feature_X": true
      }
    }
}
```

Here is a the same rule referencing a named filter:

```json
{
  "filters": {
    "beta_users": "attr:user_type = 'beta'"
  },
  "rules": {
    "enable_feature_X_for_beta_users": {
      "filter": "filter:beta_users",
      "variants": {
        "enable_feature_X": true
      }
    }
}
```

### Syntax

TODO

## Rule

In process of evaluating a feature flag, all the rules referencing that flag are
evaluated. The first rule that matches (as defined below) defines the variant of
the feature flag. If no rule matches, the default variant (as defined in the
flag defintion) is returned.

Rules are evaluated in the lexical order of their names. It's possible to
override this by using the `priority` (integer) field in the rule definition.
The rule with the highest priority is evaluated first. Two rules with the same
priority are evaluated in the lexical order of their names.

### Splits

TODO
