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
            Some((lo, hi)) => lo.num * dx <= dy * lo.den && dy * hi.den <= hi.num * dx,
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
            Some((lo, hi)) => lo.num * dx <= dy * lo.den && dy * hi.den <= hi.num * dx,
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

/// Fuses consecutive segments whose declared step agrees and whose reconstruction drift stays
/// bounded, returning the keep mask and the re-anchored tie value of each surviving run.
///
/// Segment `i` starts at `tie_values[i]` and spans `tie_lengths[i]` index ticks; `num`/`den` are
/// its declared step, shared (length 1) or per-segment. This is the degenerate, one-dimensional
/// form the sleeve takes when the slope is fixed rather than free (see the module doc comment of
/// `xinterp`'s simplify family): tie point `j` fuses away (folds into the run started at `s`)
/// when the segment immediately preceding it -- `j - 1`, the span being erased -- declares the
/// same step as the run's, and the run's offsets --
/// `off_j = tie_values[j] - (tie_values[s] + round(cumulative_length * num_s / den_s))` --
/// satisfy `max(off) - min(off) <= 2 * tol` (D5). One pass, `O(n)`, `O(1)` state: a running min
/// and max. The fused tie value is the run's Chebyshev centre,
/// `tie_values[s] + round((min(off) + max(off)) / 2)`, so a surviving value may move by up to
/// `tol` -- the trade D5 makes for fusing twice the window an anchor-pinned walk would.
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
        let offset = if same_step {
            Some(
                tie_values[j] as i128
                    - predict(tie_values[run_start], candidate_length, run_num, run_den) as i128,
            )
        } else {
            None
        };
        let fits = match offset {
            Some(off) => {
                let lo = run_min.min(off);
                let hi = run_max.max(off);
                hi - lo <= 2 * (tol as i128)
            }
            None => false,
        };
        if fits {
            cum_length = candidate_length;
            run_min = run_min.min(offset.unwrap());
            run_max = run_max.max(offset.unwrap());
        } else {
            fused.push(close_run(run_start, run_min, run_max));
            keep[j] = true;
            run_start = j;
            cum_length = 0;
            run_min = 0;
            run_max = 0;
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
    fn test_every_gap_survives() {
        // strictly collinear points always survive when epsilon is zero and there are only two
        let x = [0u64, 1, 2];
        let f = [0i64, 1, 2];
        assert_eq!(simplify_points_int(&x, &f, 0, 1), [true, false, true]);
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
}
