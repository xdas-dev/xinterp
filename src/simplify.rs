//! Dropping and fusing redundant tie points: `simplify_points` and `simplify_step`.
//!
//! Both are one-pass, `O(n)`, `O(1)`-state greedy walks -- see `simplify_points`'s doc comment
//! for the sleeve and `simplify_step`'s for the degenerate one-dimensional form it takes when the
//! slope is fixed by a shared step. The algorithm is the sleeve; it belongs in this comment, not
//! in the function name.

use crate::step::{predict, rate_at};

/// A rational bound `num / den` on a slope, `den` always strictly positive.
#[derive(Clone, Copy)]
struct Bound {
    num: i128,
    den: i128,
}

/// Drops points from `(x, f)` that the surviving chords already describe to within `en / ed`.
///
/// One-pass greedy sleeve walk in exact `u128`/`i128` arithmetic: from the current anchor it
/// maintains the intersection of every dropped point's `±en/ed` slope cone, and emits a knot
/// exactly when a candidate leaves it. Knots are original points, so surviving values are never
/// moved. `en` and `ed` are the caller's already-computed epsilon numerator/denominator (`ed`
/// implicitly positive); this port keeps no shear, no magnitude pre-check and no
/// arbitrary-precision fallback tier -- those three tiers exist in the Python original only
/// because it lacks a vectorised wide integer, and `u128`/`i128` here has the headroom the
/// realistic magnitudes need.
///
/// # Panics
///
/// Panics if `x` and `f` do not have the same length.
pub fn simplify_points_int(x: &[u64], f: &[i64], en: i64, ed: i64) -> Vec<bool> {
    assert_eq!(x.len(), f.len(), "x and f must have the same length");
    let n = x.len();
    let mut keep = vec![false; n];
    if n < 3 {
        keep.fill(true);
        return keep;
    }
    keep[0] = true;
    keep[n - 1] = true;
    let (en, ed) = (en as i128, ed as i128);
    let (mut ax, mut ay) = (x[0], f[0]);
    let mut cone: Option<(Bound, Bound)> = None;
    for i in 1..n {
        let mut dx = (x[i] - ax) as i128;
        let mut dy = f[i] as i128 - ay as i128;
        let inside = match cone {
            None => true,
            // strict: a point sitting exactly on the cone's edge is the one case where the
            // reconstruction's own rounding can push it outside a non-strict `tol` budget (see
            // the module doc comment), so it must not be treated as already covered
            Some((lo, hi)) => lo.num * dx < dy * lo.den && dy * hi.den < hi.num * dx,
        };
        if !inside {
            keep[i - 1] = true;
            ax = x[i - 1];
            ay = f[i - 1];
            dx = (x[i] - ax) as i128;
            dy = f[i] as i128 - ay as i128;
            cone = None;
        }
        let (num_lo, num_hi, den) = (ed * dy - en, ed * dy + en, ed * dx);
        cone = Some(match cone {
            None => (Bound { num: num_lo, den }, Bound { num: num_hi, den }),
            Some((lo, hi)) => (
                if num_lo * lo.den > lo.num * den {
                    Bound { num: num_lo, den }
                } else {
                    lo
                },
                if num_hi * hi.den < hi.num * den {
                    Bound { num: num_hi, den }
                } else {
                    hi
                },
            ),
        });
    }
    keep
}

/// The floating-point twin of [`simplify_points_int`], for tie values with no exact tick
/// representation. Same walk, plain `f64` arithmetic for the cone.
///
/// # Panics
///
/// Panics if `x` and `f` do not have the same length.
pub fn simplify_points_float(x: &[u64], f: &[f64], en: f64, ed: f64) -> Vec<bool> {
    assert_eq!(x.len(), f.len(), "x and f must have the same length");
    let n = x.len();
    let mut keep = vec![false; n];
    if n < 3 {
        keep.fill(true);
        return keep;
    }
    keep[0] = true;
    keep[n - 1] = true;
    struct FBound {
        num: f64,
        den: f64,
    }
    let (mut ax, mut ay) = (x[0], f[0]);
    let mut cone: Option<(FBound, FBound)> = None;
    for i in 1..n {
        let mut dx = (x[i] - ax) as f64;
        let mut dy = f[i] - ay;
        let inside = match &cone {
            None => true,
            // strict, mirroring `simplify_points_int` -- see its comment
            Some((lo, hi)) => lo.num * dx < dy * lo.den && dy * hi.den < hi.num * dx,
        };
        if !inside {
            keep[i - 1] = true;
            ax = x[i - 1];
            ay = f[i - 1];
            dx = (x[i] - ax) as f64;
            dy = f[i] - ay;
            cone = None;
        }
        let (num_lo, num_hi, den) = (ed * dy - en, ed * dy + en, ed * dx);
        cone = Some(match cone {
            None => (FBound { num: num_lo, den }, FBound { num: num_hi, den }),
            Some((lo, hi)) => (
                if num_lo * lo.den > lo.num * den {
                    FBound { num: num_lo, den }
                } else {
                    lo
                },
                if num_hi * hi.den < hi.num * den {
                    FBound { num: num_hi, den }
                } else {
                    hi
                },
            ),
        });
    }
    keep
}

/// Bounds the re-rounding term `R(m) = round(a) + round(m) - round(a + m)` over every sample
/// `m` in `[0, length)` of a segment that starts `offset` ticks into the run, at rate
/// `num / den`.
///
/// Once a tie point is dropped, every sample of the segment it used to anchor is instead
/// predicted straight from the run's own anchor: `round(a) + round(m)` (the old prediction, one
/// step from the run anchor to the segment start, then one step into the segment) becomes
/// `round(a + m)` (the new, single-step prediction). Writing `round(k) = k*num/den + e(k)` with
/// `|e| <= 1/2`, that difference is exactly `R(m)`, which is nonzero only when `e(a) + e(m)`
/// spills out of `[-1/2, 1/2]` -- and since `|e(m)| <= 1/2`, that can only happen on the side
/// `e(a)` already sits on: a shift that rounded up can only ever push a sample up, and vice
/// versa. `O(1)`.
fn round_shift_bounds(offset: u64, length: u64, num: i64, den: u64) -> (i128, i128) {
    if length <= 1 {
        return (0, 0); // only `m == 0`, where `R` is identically zero
    }
    let (num, den) = crate::step::reduce(num, den);
    let magnitude = offset as u128 * num.unsigned_abs() as u128;
    let den = den as u128;
    let (quotient, residue) = (magnitude / den, magnitude % den);
    let (lo, hi): (i128, i128) = if residue == 0 {
        // an exact shift moves nothing, unless it flips the parity a half-tie reads
        if den % 2 == 1 || quotient % 2 == 0 {
            (0, 0)
        } else {
            (-1, 1)
        }
    } else if 2 * residue > den || (2 * residue == den && quotient % 2 == 1) {
        (0, 1) // the shift rounded up, so samples can only move up
    } else {
        (-1, 0) // the shift rounded down, so samples can only move down
    };
    // `round_step` rounds the magnitude and negates for a negative rate, mirroring the pair
    if num < 0 {
        (-hi, -lo)
    } else {
        (lo, hi)
    }
}

/// Fuses consecutive segments whose declared step agrees and whose reconstruction drift stays
/// bounded, returning the keep mask and the re-anchored tie value of each surviving run.
///
/// Segment `i` starts at `tie_values[i]` and spans `tie_lengths[i]` index ticks; `num`/`den` are
/// its declared step, shared (length 1) or per-segment. This is the degenerate, one-dimensional
/// form the sleeve takes when the slope is fixed rather than free (see the module doc comment of
/// `xinterp`'s simplify family): tie point `j` fuses away (folds into the run started at `s`)
/// when the segment immediately preceding it -- `j - 1`, the span being erased -- declares the
/// same step as the run's, and every sample of the segment `j` starts -- not just the junction at
/// `j` itself, but everything up to wherever the run next re-anchors -- stays within `tol` of the
/// run's prediction once fused. Each segment contributes an *interval* of offsets,
/// `[off + rmin, off + rmax]`, where `off_j = tie_values[j] - (tie_values[s] + round(cumulative
/// length * num_s / den_s))` is the junction offset and `rmin`/`rmax` (from
/// `round_shift_bounds`) bound how far re-rounding can additionally shift the segment's samples;
/// the run continues while `max(off + rmax) - min(off + rmin) <= 2 * tol` over its whole history.
/// One pass, `O(n)`, `O(1)` state: a running min and max. The fused tie value is the run's
/// Chebyshev centre, `tie_values[s] + round((min + max) / 2)`, so a surviving value may move by
/// up to `tol` -- the trade this makes for fusing twice the window an anchor-pinned walk would.
///
/// # Panics
///
/// Panics if `tie_values` and `tie_lengths` do not have the same length, if that length is zero,
/// or if `num`/`den` do not have the same length, 1 or `tie_values.len()`.
pub fn simplify_step(
    tie_values: &[i64],
    tie_lengths: &[u64],
    num: &[i64],
    den: &[u64],
    tol: i64,
) -> (Vec<bool>, Vec<i64>) {
    let n = tie_values.len();
    assert_eq!(
        n,
        tie_lengths.len(),
        "tie_values and tie_lengths must have the same length"
    );
    assert!(n > 0, "simplify_step needs at least one segment");
    assert_eq!(
        num.len(),
        den.len(),
        "num and den must have the same length"
    );
    assert!(
        num.len() == 1 || num.len() == n,
        "num/den must have length 1 or tie_values.len()"
    );

    let mut keep = vec![false; n];
    let mut fused = Vec::new();
    keep[0] = true;
    let mut run_start = 0usize;
    let mut cum_length: u64 = 0;
    let mut run_min: i128 = 0;
    let mut run_max: i128 = 0;

    // Halves `sum` with the same ties-to-even rule as every other rounding in this crate. `sum`
    // is always even-or-odd by exactly 1 (dividing by 2), so a remainder of 1 is always an exact
    // tie -- no need for `round_step`'s general machinery.
    let round_half_even = |sum: i128| -> i128 {
        let quotient = sum.div_euclid(2);
        let is_tie = sum.rem_euclid(2) != 0;
        if is_tie && quotient % 2 != 0 {
            quotient + 1
        } else {
            quotient
        }
    };
    let close_run = |run_start: usize, run_min: i128, run_max: i128| -> i64 {
        let centre = round_half_even(run_min + run_max);
        (tie_values[run_start] as i128 + centre) as i64
    };

    for j in 1..n {
        let (run_num, run_den) = rate_at(num, den, run_start);
        let (seg_num, seg_den) = rate_at(num, den, j - 1);
        let same_step = {
            let (a_num, a_den) = crate::step::reduce(run_num, run_den);
            let (b_num, b_den) = crate::step::reduce(seg_num, seg_den);
            a_num == b_num && a_den == b_den
        };
        let candidate_length = cum_length + tie_lengths[j - 1];
        let bounds = if same_step {
            let off = tie_values[j] as i128
                - predict(tie_values[run_start], candidate_length, run_num, run_den) as i128;
            let (rmin, rmax) =
                round_shift_bounds(candidate_length, tie_lengths[j], run_num, run_den);
            Some((run_min.min(off + rmin), run_max.max(off + rmax)))
        } else {
            None
        };
        match bounds {
            Some((lo, hi)) if hi - lo <= 2 * (tol as i128) => {
                cum_length = candidate_length;
                run_min = lo;
                run_max = hi;
            }
            _ => {
                fused.push(close_run(run_start, run_min, run_max));
                keep[j] = true;
                run_start = j;
                cum_length = 0;
                run_min = 0;
                run_max = 0;
            }
        }
    }
    fused.push(close_run(run_start, run_min, run_max));
    (keep, fused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_ties_pass_through() {
        let x = [0u64, 9];
        let f = [0i64, 9];
        assert_eq!(simplify_points_int(&x, &f, 0, 1), [true, true]);
    }

    #[test]
    fn test_zero_width_cone_drops_nothing() {
        // at en=0 the cone has zero width, so a point can only be "inside" it by matching the
        // chord's slope exactly -- the boundary case the strict membership test exists to
        // reject (see its comment). Even perfectly collinear points therefore survive.
        let x = [0u64, 1, 2];
        let f = [0i64, 1, 2];
        assert_eq!(simplify_points_int(&x, &f, 0, 1), [true, true, true]);
    }

    #[test]
    fn test_strict_cone_keeps_a_point_exactly_at_the_half_tick_budget() {
        // x=[0,1,2], f=[0,3,5]: the middle point sits exactly ½ tick off the chord from (0,0)
        // to (2,5). At the reconstruction-aware budget tol=0 -> en/ed = 1/2, a non-strict cone
        // drops it and the reconstruction rounds to 2 (ties to even), a full tick off; the
        // strict cone keeps it, giving an exact reconstruction.
        let x = [0u64, 1, 2];
        let f = [0i64, 3, 5];
        assert_eq!(simplify_points_int(&x, &f, 1, 2), [true, true, true]);
    }

    #[test]
    fn test_real_discontinuity_survives() {
        let x = [0u64, 500, 501, 1000];
        let f = [0i64, 50_000, 50_100 + 2, 100_000 + 2];
        assert_eq!(simplify_points_int(&x, &f, 0, 1), [true, true, true, true]);
    }

    #[test]
    fn test_conservative_walk_keeps_interior_points_a_global_chord_would_drop() {
        // xdas's `_sleeve` runs a whole-curve fast path before falling back to this walk (see
        // its own doc comment); the walk alone is conservative and may keep points a single
        // spanning chord would already fit within epsilon -- this is that case, ported from
        // `TestSleeve.test_seam_below_the_storage_resolution_is_fused`, which relies on the fast
        // path to collapse it to 2 points.
        let x = [0u64, 999, 1000, 1999];
        let f = [0i64, 999, 1001, 2000];
        assert_eq!(simplify_points_int(&x, &f, 0, 1), [true, true, true, true]);
    }

    #[test]
    fn test_float_values() {
        let x = [0u64, 9, 10, 19];
        let f = [0.0f64, 9.0, 10.5, 19.5];
        assert_eq!(
            simplify_points_float(&x, &f, 1.0, 1.0),
            [true, false, false, true]
        );
        assert_eq!(
            simplify_points_float(&x, &f, 0.1, 1.0),
            [true, true, true, true]
        );
    }

    #[test]
    fn test_simplify_step_fuses_within_budget() {
        // three segments, same declared step, drift within budget throughout
        let tie_values = [0i64, 100, 199];
        let tie_lengths = [10u64, 10, 10];
        let num = [10i64];
        let den = [1u64];
        let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, 1);
        assert_eq!(keep, [true, false, false]);
        assert_eq!(fused, vec![0]);
    }

    #[test]
    fn test_simplify_step_splits_on_large_drift() {
        let tie_values = [0i64, 100, 300];
        let tie_lengths = [10u64, 10, 10];
        let num = [10i64];
        let den = [1u64];
        let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, 1);
        assert_eq!(keep, [true, false, true]);
        assert_eq!(fused, vec![0, 300]);
    }

    #[test]
    fn test_simplify_step_requires_equal_steps() {
        // segment 1 (tie_values[1] -> tie_values[2]) declares a different step than the run's;
        // the boundary that surfaces the mismatch (tie_values[2]) must survive even though its
        // own drift against the run's prediction is small
        let tie_values = [0i64, 100, 250];
        let tie_lengths = [10u64, 10, 10];
        let num = [10i64, 15, 15];
        let den = [1u64, 1, 1];
        let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, 1000);
        assert_eq!(keep, [true, false, true]);
        assert_eq!(fused, vec![0, 250]);
    }

    #[test]
    fn test_round_shift_bounds_units() {
        // exact shift, denominator reduces to odd -> untouched
        assert_eq!(round_shift_bounds(10, 5, 3, 3), (0, 0));
        // exact shift, even denominator, odd quotient -> the shift flips the half-tie's parity
        assert_eq!(round_shift_bounds(2, 5, 1, 2), (-1, 1));
        // inexact shift that rounds up -> samples can only move up
        assert_eq!(round_shift_bounds(3, 5, 2, 4), (0, 1));
        // inexact shift that rounds down -> samples can only move down
        assert_eq!(round_shift_bounds(1, 5, 2, 4), (-1, 0));
        // length <= 1: only m == 0, where R is identically zero
        assert_eq!(round_shift_bounds(100, 1, 7, 3), (0, 0));
        assert_eq!(round_shift_bounds(100, 0, 7, 3), (0, 0));
        // negative num mirrors the positive-num pair
        assert_eq!(round_shift_bounds(3, 5, -2, 4), (-1, 0));
        assert_eq!(round_shift_bounds(1, 5, -2, 4), (0, 1));
    }

    #[test]
    fn test_simplify_step_lossless_at_tol_zero_with_a_perfectly_fusible_run() {
        // num/den = 1/2, lengths [1, 2], values [0, 0]: the junction offset alone is 0, but
        // round(1*1/2) + round(2*1/2) = 0 + 1 = 1 != round(3*1/2) = 2, so fusing on the
        // junction offset alone would move the last sample by 1 tick. The interval bound
        // must refuse to fuse here at tol=0.
        let tie_values = [0i64, 0];
        let tie_lengths = [1u64, 2];
        let num = [1i64];
        let den = [2u64];
        let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, 0);
        assert_eq!(keep, [true, true]);
        assert_eq!(fused, vec![0, 0]);
    }

    #[test]
    fn test_simplify_step_refuses_the_ties_to_even_shift_variance_regression() {
        // num/den = 14/16 (reduces to 7/8), junction at a = 104 (exact: 104*7/8 = 91). The
        // segment starting there has length 4: before fusing, sample 108 = 91 + round(4*7/8) =
        // 91 + round(3.5) = 95 (ties to even); fused from index 0 it would instead be
        // round(108*7/8) = round(94.5) = 94 -- ties-to-even is not shift-invariant, so fusing
        // on the junction offset alone (which is exactly 0 here) would still drift the sample
        // by 1 tick.
        let tie_values = [0i64, 91];
        let tie_lengths = [104u64, 4];
        let num = [14i64];
        let den = [16u64];
        assert_eq!(
            simplify_step(&tie_values, &tie_lengths, &num, &den, 0).0,
            [true, true],
            "must not fuse at tol=0: fusing here would drift a sample by 1 tick"
        );
        let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, 1);
        assert_eq!(keep, [true, false]);
        assert_eq!(fused, vec![0]);
    }

    #[test]
    fn test_den_one_fuses_a_long_run_end_to_end_at_every_tolerance() {
        // exactly representable steps (`den` reduces to 1) always have residue 0 against an
        // odd denominator, so `round_shift_bounds` is (0, 0) regardless of offset or
        // tolerance -- the interval bound must not over-tighten a case that never needed it.
        let n = 50;
        let tie_lengths = vec![10u64; n];
        let tie_values: Vec<i64> = (0..n as i64).map(|i| i * 70).collect();
        let num = [7i64];
        let den = [1u64];
        for tol in [0i64, 1, 5] {
            let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, tol);
            let mut expected_keep = vec![false; n];
            expected_keep[0] = true;
            assert_eq!(keep, expected_keep, "tol={tol}");
            assert_eq!(fused, vec![0], "tol={tol}");
        }
    }

    /// Deterministic splitmix64, mirroring `step.rs`'s copy -- no `rand` dependency needed.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
        fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo + 1)
        }
    }

    /// The contract test: no sample may drift by more than `tol`, for any run `simplify_step`
    /// chooses to fuse. Replaces the `<= 1 tick` allowance the drift bound used to need -- that
    /// allowance was the defect in disguise.
    #[test]
    fn test_simplify_step_never_drifts_a_sample_by_more_than_tol() {
        let mut rng = SplitMix64(123);
        for tol in [0i64, 1, 2, 3, 5] {
            for _ in 0..200 {
                let n = rng.range_u64(2, 8) as usize;
                let tie_lengths: Vec<u64> = (0..n).map(|_| rng.range_u64(1, 40)).collect();
                let num_v = rng.range_i64(1, 30);
                let den_v = rng.range_u64(1, 15);
                // build tie values near what the declared step already predicts, with up to
                // `tol` of jitter -- the drift a caller's own upstream fusing may have left
                let mut tie_values = vec![0i64];
                let mut cum = 0u64;
                for i in 1..n {
                    cum += tie_lengths[i - 1];
                    let predicted = predict(0, cum, num_v, den_v);
                    let jitter = if tol > 0 { rng.range_i64(-tol, tol) } else { 0 };
                    tie_values.push(predicted + jitter);
                }
                let num = [num_v];
                let den = [den_v];
                let (keep, fused) = simplify_step(&tie_values, &tie_lengths, &num, &den, tol);

                // original: every segment stepped from its own (possibly jittered) anchor
                let mut original = Vec::new();
                for i in 0..n {
                    for k in 0..tie_lengths[i] {
                        original.push(predict(tie_values[i], k, num_v, den_v));
                    }
                }
                // reconstructed: every surviving run stepped from its re-anchored fused value
                let mut run_lens: Vec<u64> = Vec::new();
                for i in 0..n {
                    if keep[i] {
                        run_lens.push(tie_lengths[i]);
                    } else {
                        *run_lens.last_mut().unwrap() += tie_lengths[i];
                    }
                }
                let mut reconstructed = Vec::new();
                for (anchor, len) in fused.iter().zip(run_lens.iter()) {
                    for k in 0..*len {
                        reconstructed.push(predict(*anchor, k, num_v, den_v));
                    }
                }

                assert_eq!(original.len(), reconstructed.len());
                for (o, r) in original.iter().zip(reconstructed.iter()) {
                    assert!(
                        (o - r).abs() <= tol,
                        "drift {} exceeds tol {tol}: tie_values={tie_values:?} \
                         tie_lengths={tie_lengths:?} num={num_v} den={den_v}",
                        o - r
                    );
                }
            }
        }
    }
}
