# xinterp

This package enables index to value mapping both in a forward and backward way.
Inverse retrieval of indices from given values can be done with different matching
rules (None, nearest, forward-fill, backward-fill). Results are exact even when
using big integer values (e.g., nanosecond datetime64).

## Installation

```
pip install xinterp
```

## Usage

```python
import numpy as np
from xinterp import forward, inverse

xp = np.array([0, 10, 20])
fp = np.array([0, 1000, 2000], dtype="datetime64[s]")

x = np.array([0, 5, 10, 15, 20])
result = forward(x, xp, fp)
expected = np.array([0, 500, 1000, 1500, 2000], dtype="datetime64[s]")
assert np.array_equal(result, expected)

x = np.array([1, 499, 1001, 1503, 1997], dtype="datetime64[s]")
result = inverse(x, xp, fp, method="nearest")
expected = np.array([0, 5, 10, 15, 20])
assert np.array_equal(result, expected)
```

## Development

The project is developed with [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/):

```sh
uv sync        # build the extension and install the dev dependencies
uv run pytest  # Python suite
cargo test     # Rust suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full toolchain.
