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

## The two families

An axis can be described in two ways, and the package mirrors the same operations
across both.

**Points** -- the knots are listed, and each piece implies its own slope. Use
`forward_points` / `inverse_points` when the axis is irregular and every knot
matters.

**Step** -- the knots are generated from a starting value at a fixed exact rate
`num / den`. Use `forward_step` / `inverse_step` when the axis is regular: a long
run of samples costs two integers rather than a knot each, and the rate stays
exact because it is never collapsed to a float.

| | points | step |
| --- | --- | --- |
| index to value | `forward_points` | `forward_step` |
| value to index | `inverse_points` | `inverse_step` |
| drop redundant knots | `simplify_points` | `simplify_step` |
| fit a rate | -- | `infer_step` |
| residual against a rate | -- | `deviation_step` |

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
exactly. The walk is conservative there: it drops a knot only when it can prove
the drop costs nothing, so at zero it may well keep everything.

**Simplification is bounded elsewhere.** No sample moves by more than `tol`.
`simplify_step` takes `tol` in ticks and applies it directly. `simplify_points`
measures its budget against the unrounded curve, so pass `en / ed = tol + 1/2`
to get the same bound on the reconstruction.

## Installation

```
pip install xinterp
```

## Usage

### Points

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

### Step

The rate is a ratio, so a 2.5 s sampling interval is `num, den = 5, 2` and stays
exact. Values landing on a half tick round to even:

```python
import numpy as np
from xinterp import forward_step, inverse_step

tie_indices = np.array([0, 4])
tie_values = np.array([0, 10], dtype="datetime64[s]")
num, den = 5, 2

x = np.array([0, 1, 2, 3, 4])
result = forward_step(x, tie_indices, tie_values, num, den)
expected = np.array([0, 2, 5, 8, 10], dtype="datetime64[s]")
assert np.array_equal(result, expected)

f = np.array([0, 2, 5, 8, 10], dtype="datetime64[s]")
result = inverse_step(f, tie_indices, tie_values, num, den)
assert np.array_equal(result, x)
```

`infer_step` recovers that ratio from an axis, and reports the worst per-segment
deviation from it, so you can tell whether a single rate is honest before
adopting one. `deviation_step` gives the residual segment by segment:

```python
import numpy as np
from xinterp import deviation_step, infer_step

x = np.array([0, 10, 20, 30])
f = np.array([0, 25, 50, 75], dtype="datetime64[s]")

assert infer_step(x, f) == (5, 2, 0)  # 2.5 s per index, fitting every segment
assert np.array_equal(deviation_step(x, f, 5, 2), [0, 0, 0])
```

### Simplification

`simplify_points` returns a mask of the knots worth keeping. Tolerance is the
exact ratio `en / ed`, measured against the unrounded curve -- to bound the
*reconstructed* value by `tol` ticks, pass `tol + 1/2`:

```python
import numpy as np
from xinterp import forward_points, simplify_points

xp = np.array([0, 10, 20, 30, 40])
fp = np.array([0, 1001, 1999, 3002, 4000], dtype="datetime64[s]")

keep = simplify_points(xp, fp, 5, 2)  # 5/2 == tol 2 ticks + 1/2
assert np.array_equal(keep, [True, False, False, False, True])

recon = forward_points(xp, xp[keep], fp[keep])
assert np.abs((recon - fp).astype("i8")).max() <= 2  # the promised bound
```

`simplify_step` is the twin for rate-described segments: it fuses consecutive
runs whose declared step agrees, and re-anchors each survivor. It returns the
keep mask alongside the re-anchored tie values:

```python
import numpy as np
from xinterp import simplify_step

tie_values = np.array([0, 1000, 2000])
tie_lengths = np.array([10, 10, 10])

keep, fused = simplify_step(tie_values, tie_lengths, [100], [1], 0)
assert np.array_equal(keep, [True, False, False])  # one run, not three
assert np.array_equal(fused, [0])
```

## Migrating from 0.1

`forward` and `inverse` are now `forward_points` and `inverse_points`. The old
names still work as aliases but emit a `DeprecationWarning`, and are scheduled
for removal in 0.5:

```python
from xinterp import forward_points as forward  # was: from xinterp import forward
```

Nothing else in the 0.1 surface changed. See [CHANGELOG.md](CHANGELOG.md) for the
full list.

## Development

The project is developed with [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/):

```sh
uv sync        # build the extension and install the dev dependencies
uv run pytest  # Python suite
cargo test     # Rust suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full toolchain.
