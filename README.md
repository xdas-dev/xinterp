# xinterp

This package enables index to value mapping both in a forward and backward way.
Inverse retrieval of indices from given values can be done with different matching
rules (None, nearest, forward-fill, backward-fill). Results are exact even when
using big integer values (e.g., nanosecond datetime64), and dtypes that are
integer, floating, `datetime64` or `timedelta64` are all supported for `fp`.

`method="nearest"` clamps without limit: a value beyond the first or last point
resolves to that boundary's index, however far outside the range it falls. Where
rounding to an integer is needed, ties round to even (banker's rounding), matching
the behaviour of `round()` and `numpy.round`.

## Guarantees

Values live on the tick grid of their dtype and every result is the exactly
rounded one, so the following hold at every sample rather than on average. They
hold identically for the points and the step family.

**Round trip, index to value and back.** `inverse(forward(x)) == x` whenever the
mapping is injective -- that is, whenever values advance by at least one tick per
index step (`|num| >= den` for the step family). Below that rate several indices
share a value, and `inverse` returns one of them.

**Round trip, value to index and back.** `forward(inverse(f)) == f` whenever
`inverse` returns. Under `method=None` a value the curve never attains raises
`KeyError` rather than resolving to a neighbour; the other methods resolve to a
neighbouring index by design, so the law does not apply to them.

**Simplification is lossless at zero.** `simplify(x, 0)` reproduces every sample
exactly.

**Simplification is bounded elsewhere.** No sample moves by more than `tol`.
`simplify_step` takes `tol` in ticks and applies it directly. `simplify_points`
measures its budget against the unrounded curve, so pass `en / ed = tol + 1/2`
to get the same bound on the reconstruction.

## Installation

```
pip install xinterp
```

## Usage

```python
import numpy as np
from xinterp import forward_points, inverse_points

xp = np.array([0, 10, 20])
fp = np.array([0, 1000, 2000], dtype="datetime64[s]")

x = np.array([0, 5, 10, 15, 20])
result = forward_points(x, xp, fp)
expected = np.array([0, 500, 1000, 1500, 2000], dtype="datetime64[s]")
assert np.array_equal(result, expected)

x = np.array([1, 499, 1001, 1503, 1997], dtype="datetime64[s]")
result = inverse_points(x, xp, fp, method="nearest")
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
