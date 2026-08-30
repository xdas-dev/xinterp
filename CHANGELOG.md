# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

The release that turns a two-function package into two families of them. The
explicit-knot pair `forward`/`inverse` becomes `forward_points`/`inverse_points`
and gains a constant-rate twin, `forward_step`/`inverse_step`, where knots are
generated at a fixed exact rate instead of listed. Both families ship a
simplifier, and the documented guarantees now hold at every sample rather than
on average.

### Added

- `forward_step`, `inverse_step` -- the constant-rate twins of the points family.
  Each segment advances at the exact rational rate `num / den`, so a long regular
  axis needs two integers rather than a knot per sample. `inverse_step` accepts
  the same `method` values as `inverse_points`, and tie values may run backwards
  (a distance axis need not increase).
- `infer_step` -- the single exact `(num, den)` best describing every consecutive
  segment of an `(x, f)` pair, as the length-weighted Chebyshev centre of the
  per-segment rates, gcd-reduced. Returns the worst per-segment deviation
  alongside, so a caller can decide whether one rate is honest before adopting it.
- `deviation_step` -- the per-segment residual between each tie value and what the
  declared step predicts. Zero means the segment fits its rate exactly.
- `simplify_points` -- drops tie points already described, to within `en / ed`, by
  their surviving neighbours. A one-pass greedy sleeve over the intersection of
  the dropped points' slope cones; kept points are original points, never moved.
- `simplify_step` -- fuses consecutive segments whose declared step agrees, within
  `tol` ticks, re-anchoring each surviving run to the Chebyshev centre of its
  samples' offsets.
- `timedelta64` joins `datetime64`, integer and floating dtypes as a supported
  value type.
- Documented guarantees, held at every sample rather than on average: both round
  trips, losslessness of `simplify(x, 0)`, and the bound on every other
  simplification. See the README.
- Packaging metadata -- license, authors, keywords, classifiers, project URLs --
  and a `xinterp.__version__` attribute.

### Changed

- `forward` and `inverse` are renamed `forward_points` and `inverse_points`. The
  old names remain as aliases (see Deprecated); no code breaks on upgrade.
- `method="nearest"` clamps without a distance limit: a value beyond either end
  resolves to that boundary's index however far outside the range it falls. This
  was already the behaviour; it is now stated and tested.
- Rounding to an integer tick breaks ties to even (banker's rounding), matching
  `round()` and `numpy.round`.
- `method=None` resolves against the *rounded* forward value. Because
  `forward_points` itself rounds to the nearest tick, the exact solve's index is
  not always the one that round-trips; the nearest candidate is now checked
  against `forward_points` and returned when it matches, instead of raising.
- The extended-precision core replaces the `astro-float` dependency with a
  hand-rolled double-double, so the crate builds with no arithmetic dependencies.
- Upgraded to pyo3 and rust-numpy 0.29.
- The Python-to-Rust boundary rejects unsafe casts instead of silently
  truncating them.
- Renamed the internal `piecewise` module to `points`, matching the public names.

### Deprecated

- `forward` and `inverse` emit a `DeprecationWarning` and are scheduled for
  removal in 0.5. Use `forward_points` and `inverse_points`.

### Removed

- The Douglas-Peucker simplification, superseded by `simplify_points`. It was
  never reachable from Python, so this is not a breaking change for Python
  callers.

### Fixed

- `simplify_step` bounds its drift per sample rather than per tie value, so no
  reconstructed sample moves by more than `tol`.
- `simplify_points` closes its slope cone at the candidate it is testing, so the
  emitted knots honour the tolerance at every dropped point.

## [0.1.3]

Initial published behaviour: `forward` and `inverse` over explicit knots, exact
for big integer values including nanosecond `datetime64`, with `None`, `nearest`,
`ffill` and `bfill` matching rules.

[Unreleased]: https://github.com/xdas-dev/xinterp/compare/0.2.0...HEAD
[0.2.0]: https://github.com/xdas-dev/xinterp/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/xdas-dev/xinterp/releases/tag/0.1.3
