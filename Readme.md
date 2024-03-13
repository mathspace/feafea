# FeaFea

The python module provides a simple but powerful solution for managing feature
flags in your application. It allows dynamic control of features via flexible
targetting filters and rules and provides a simple API to check if a feature is
enabled or not.

## Quick Start

Install the module via pip:

```bash
pip install https://github.com/mathspace/feafea/archive/main.zip
```

Define a feature flag config and store it in a file `features.json`:

```json
{
  "flags": {
    "enable_new_feature": {
      "variants": [true, false],
      "default": false
    }
  },
  "rules": {
    "new_feature_override": {
      "filter": "department_id in [1,2,3]",
      "variants": {
        "enable_new_feature": true
      }
    }
  }
}
```

Load the config and evaluate the feature flag:

```python
from feafea import CompiledConfig, Evaluator

with open('features.json') as f:
    plain_config = json.load(f)
compiled_config = CompiledConfig.from_dict(plain_config)
ev = Evaluator()
ev.load_config(compiled_config)

assert not ev.evaluate('enable_new_feature', "1", {})
assert ev.evaluate('enable_new_feature', "1", {"department_id":1})
```
