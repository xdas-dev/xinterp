//! Exact rational stepping arithmetic shared by the constant-step kernels: `simplify_step`,
//! `infer_step`, `forward_step`, `inverse_step` and `deviation_step`.
//!
//! A step is a rate `num / den` (`num: i64`, `den: u64`, `den > 0`), possibly negative -- a
//! distance axis can run backwards. Every forward-direction kernel here reduces to the same
//! primitive: `tie0 + round(k * num / den)` for `k: u64` ticks from an anchor tie point. It is
//! computed in `u128` after splitting sign from magnitude at the boundary, exactly as
//! `schemes.rs` does for values via `ToUnsigned`. `i128` alone is not enough for the inverse
//! direction: `inverse_step` needs `(f - tie0) * den`, and `f - tie0` is a difference of two
//! `i64`s, spanning the full `u64` range, so that product reaches `(2**64-1)**2`, twice
//! `i128::MAX`. It fits `u128` with `2**65 - 2` to spare.

use crate::divop::{DivOp, Method};
use crate::wide::cmp_frac;
use std::cmp::Ordering;

/// Rounds `k * num / den` to the nearest integer, ties to even, exactly.
///
/// `den` must be strictly positive; `num` may be negative.
pub fn round_step(k: u64, num: i64, den: u64) -> i128 {
    let magnitude = (k as u128) * (num.unsigned_abs() as u128);
    let rounded = magnitude.div(den as u128, Method::Nearest).unwrap();
    if num < 0 {
        -(rounded as i128)
    } else {
        rounded as i128
    }
}

/// Predicts the tie value `k` ticks after `tie0` at the given step.
pub fn predict(tie0: i64, k: u64, num: i64, den: u64) -> i64 {
    (tie0 as i128 + round_step(k, num, den)) as i64
}

/// The signed residual between an observed value and the value the step predicts `k` ticks
/// after `tie0`: `actual - predict(tie0, k, num, den)`.
pub fn deviation(actual: i64, tie0: i64, k: u64, num: i64, den: u64) -> i64 {
    (actual as i128 - predict(tie0, k, num, den) as i128) as i64
}

/// Signed division with the rounding contract of [`DivOp`], generalized to any sign of dividend
/// and divisor (`den` must be nonzero).
pub fn div_signed(num: i128, den: i128, method: Method) -> Option<i128> {
    let sign = num.signum() * den.signum();
    let method = if sign < 0 {
        match method {
            Method::ForwardFill => Method::BackwardFill,
            Method::BackwardFill => Method::ForwardFill,
            other => other,
        }
    } else {
        method
    };
    let magnitude = num.unsigned_abs().div(den.unsigned_abs(), method)?;
    Some(if sign < 0 {
        -(magnitude as i128)
    } else {
        magnitude as i128
    })
}

/// Greatest common divisor of two non-negative `u128`s.
fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Reduces `num / den` to an irreducible fraction, keeping the sign in `num` and `den` positive
/// (D2: always store an irreducible fraction -- it gives the smallest denominator for that exact
/// value and makes two rates comparable exactly).
///
/// `den` must be strictly positive. `num == 0` reduces to `(0, 1)`.
pub fn reduce(num: i64, den: u64) -> (i64, u64) {
    if num == 0 {
        return (0, 1);
    }
    let divisor = gcd(num.unsigned_abs() as u128, den as u128);
    // the magnitude alone can be `2**63` (e.g. reducing `i64::MIN`), which does not fit a
    // positive `i64` -- negate in `i128` first, where it does, then narrow
    let reduced_magnitude = num.unsigned_abs() as u128 / divisor;
    let reduced_den = (den as u128 / divisor) as u64;
    let reduced_num = if num < 0 {
        -(reduced_magnitude as i128) as i64
    } else {
        reduced_magnitude as i64
    };
    (reduced_num, reduced_den)
}

/// A per-segment value span (`num`) over an index span (`den`), `den` always strictly positive.
#[derive(Clone, Copy)]
struct Segment {
    num: i64,
    den: u64,
}

/// Finds the pair of segments binding the length-weighted Chebyshev centre of the per-segment
/// rates `num_i / den_i`, in `O(n log n)`.
///
/// The centre minimises `max_i |den_i * s - num_i|` over `s`; equivalently, it is the lowest
/// point of the upper envelope of the `2n` lines `den_i * s - num_i` and `num_i - den_i * s`.
/// That point is where a negative-slope and a positive-slope line cross, found with the
/// convex-hull trick rather than the `O(n^2)` pairwise scan. Comparisons are exact rational
/// arithmetic (see [`crate::wide`]), so the answer does not depend on where the lines happen to
/// sit numerically.
///
/// # Panics
///
/// Panics if `segments` is empty.
fn chebyshev_pair(segments: &[Segment]) -> (usize, usize) {
    assert!(
        !segments.is_empty(),
        "chebyshev_pair needs at least one segment"
    );
    // one line per segment per sign: (slope, intercept, segment index)
    let mut lines: Vec<(i128, i128, usize)> = Vec::with_capacity(segments.len() * 2);
    for (i, seg) in segments.iter().enumerate() {
        let den = seg.den as i128;
        let num = seg.num as i128;
        lines.push((den, -num, i)); // positive-slope line: den*s - num
        lines.push((-den, num, i)); // negative-slope line: num - den*s
    }
    // ascending slope; equal slopes broken by descending intercept, so the dominant one (which
    // alone can ever be on the upper envelope) is processed first and the rest skipped
    lines.sort_by(|(s1, b1, _), (s2, b2, _)| s1.cmp(s2).then_with(|| b2.cmp(b1)));

    let mut hull: Vec<(i128, i128, usize)> = Vec::with_capacity(lines.len());
    for (s, b, seg) in lines {
        if let Some(&(last_s, _, _)) = hull.last() {
            if last_s == s {
                continue;
            }
        }
        while hull.len() >= 2 {
            let (s1, b1, _) = hull[hull.len() - 2];
            let (s2, b2, _) = hull[hull.len() - 1];
            // pop the last hull line when it is dominated: (b-b1)/(s1-s) <= (b2-b1)/(s1-s2)
            if cmp_frac(b - b1, s1 - s, b2 - b1, s1 - s2) != Ordering::Greater {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push((s, b, seg));
    }

    // the envelope's slope increases monotonically along the hull; its minimum sits at the
    // negative-to-positive transition. `chebyshev_pair` always feeds in both `+den_i` and
    // `-den_i` for every segment, so at least one slope of each sign reaches the hull.
    let mut t = 0;
    while hull[t].0 < 0 {
        t += 1;
    }
    (hull[t].2, hull[t - 1].2)
}

/// The single step best describing every consecutive segment of `(x, f)`, in exact integers.
///
/// Returns the gcd-reduced `(num, den)` and the worst per-segment absolute deviation from it.
///
/// # Panics
///
/// Panics if `x` and `f` do not have the same length, or if fewer than two tie points are given.
pub fn infer(x: &[u64], f: &[i64]) -> (i64, u64, i64) {
    assert_eq!(x.len(), f.len(), "x and f must have the same length");
    assert!(x.len() >= 2, "infer needs at least two tie points");
    let segments: Vec<Segment> = x
        .windows(2)
        .zip(f.windows(2))
        .map(|(xw, fw)| Segment {
            num: fw[1] - fw[0],
            den: xw[1] - xw[0],
        })
        .collect();
    if segments.len() == 1 {
        let (num, den) = reduce(segments[0].num, segments[0].den);
        return (num, den, 0);
    }
    let (i, j) = chebyshev_pair(&segments);
    let combined_num = segments[i].num as i128 + segments[j].num as i128;
    let combined_den = segments[i].den as u128 + segments[j].den as u128;
    let divisor = gcd(combined_num.unsigned_abs(), combined_den);
    let (num, den) = if divisor == 0 {
        (0i64, 1u64)
    } else {
        (
            (combined_num / divisor as i128) as i64,
            (combined_den / divisor) as u64,
        )
    };
    let worst = segments
        .iter()
        .map(|seg| deviation(seg.num, 0, seg.den, num, den).unsigned_abs())
        .max()
        .unwrap();
    (num, den, worst as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_step_ties_to_even() {
        assert_eq!(round_step(1, 1, 2), 0);
        assert_eq!(round_step(3, 1, 2), 2);
        assert_eq!(round_step(1, -1, 2), 0);
        assert_eq!(round_step(3, -1, 2), -2);
    }

    #[test]
    fn test_round_step_near_u64_max() {
        // k*num close to the u128 product of two near-u64::MAX magnitudes
        let k = u64::MAX;
        let num = i64::MIN; // magnitude 2^63
        let den = 1u64;
        assert_eq!(round_step(k, num, den), -(k as i128) * (1i128 << 63));
    }

    #[test]
    fn test_predict_matches_manual_computation() {
        assert_eq!(predict(100, 5, 3, 2), 100 + 8); // round(7.5) = 8, ties to even
        assert_eq!(predict(100, 4, 3, 2), 100 + 6);
        assert_eq!(predict(-100, 5, -3, 2), -100 - 8);
    }

    #[test]
    fn test_deviation_zero_on_exact_fit() {
        assert_eq!(deviation(112, 100, 4, 3, 1), 0);
        assert_eq!(deviation(113, 100, 4, 3, 1), 1);
    }

    #[test]
    fn test_div_signed_matches_i128_semantics() {
        assert_eq!(div_signed(7, 2, Method::ForwardFill), Some(3));
        assert_eq!(div_signed(-7, 2, Method::ForwardFill), Some(-4));
        assert_eq!(div_signed(7, -2, Method::ForwardFill), Some(-4));
        assert_eq!(div_signed(-7, -2, Method::ForwardFill), Some(3));
        assert_eq!(div_signed(7, 2, Method::BackwardFill), Some(4));
        assert_eq!(div_signed(-7, 2, Method::BackwardFill), Some(-3));
        assert_eq!(div_signed(6, 2, Method::None), Some(3));
        assert_eq!(div_signed(7, 2, Method::None), None);
        assert_eq!(div_signed(7, 2, Method::Nearest), Some(4));
        assert_eq!(div_signed(-7, 2, Method::Nearest), Some(-4));
    }

    #[test]
    fn test_reduce_is_gcd_reduced_and_sign_in_numerator() {
        assert_eq!(reduce(6, 4), (3, 2));
        assert_eq!(reduce(-6, 4), (-3, 2));
        assert_eq!(reduce(0, 5), (0, 1));
        assert_eq!(reduce(5, 1), (5, 1));
        assert_eq!(reduce(i64::MIN, u64::MAX), (i64::MIN, u64::MAX));
    }

    #[test]
    fn test_infer_two_segments_exact_fit() {
        let x = [0u64, 10, 20];
        let f = [0i64, 100, 200];
        let (num, den, worst) = infer(&x, &f);
        assert_eq!((num, den), (10, 1));
        assert_eq!(worst, 0);
    }

    #[test]
    fn test_infer_picks_chebyshev_centre() {
        // two segments of equal length disagreeing on rate: centre is their average rate
        let x = [0u64, 10, 20];
        let f = [0i64, 100, 202]; // rates 10 and 10.2
        let (num, den, worst) = infer(&x, &f);
        // centre rate = (100+102)/(10+10) = 202/20 = 101/10
        assert_eq!((num, den), (101, 10));
        assert_eq!(worst, 1);
    }

    #[test]
    fn test_infer_single_segment() {
        let x = [0u64, 7];
        let f = [0i64, 21];
        assert_eq!(infer(&x, &f), (3, 1, 0));
    }

    /// Deterministic splitmix64, so the property tests below need no `rand` dependency.
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

    /// The Chebyshev pair minimises the worst per-segment deviation over every candidate pair --
    /// cross-check the hull trick's answer against a brute-force `O(n^2)` scan, which is
    /// obviously correct (if slow), on random small segment sets.
    #[test]
    fn test_chebyshev_pair_matches_brute_force_scan() {
        let mut rng = SplitMix64(42);
        for _ in 0..500 {
            let n = rng.range_u64(2, 8) as usize;
            let segments: Vec<Segment> = (0..n)
                .map(|_| Segment {
                    num: rng.range_i64(-1_000_000, 1_000_000),
                    den: rng.range_u64(1, 1_000),
                })
                .collect();
            let worst_of = |i: usize, j: usize| -> u64 {
                let (num, den) = reduce(
                    segments[i].num + segments[j].num,
                    segments[i].den + segments[j].den,
                );
                segments
                    .iter()
                    .map(|seg| deviation(seg.num, 0, seg.den, num, den).unsigned_abs())
                    .max()
                    .unwrap()
            };
            let (i, j) = chebyshev_pair(&segments);
            let found = worst_of(i, j);
            let brute_force_best = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| worst_of(i, j))
                .min()
                .unwrap();
            assert_eq!(
                found,
                brute_force_best,
                "segments = {:?}",
                segments.iter().map(|s| (s.num, s.den)).collect::<Vec<_>>()
            );
        }
    }

    /// `infer`'s `worst_deviation` must actually equal the max per-segment residual its own
    /// returned `(num, den)` produces.
    #[test]
    fn test_infer_worst_deviation_is_self_consistent() {
        let mut rng = SplitMix64(7);
        for _ in 0..200 {
            let n_ties = rng.range_u64(2, 10) as usize + 1;
            let mut x = vec![0u64];
            let mut f = vec![rng.range_i64(-1_000_000, 1_000_000)];
            for _ in 1..n_ties {
                let dx = rng.range_u64(1, 1_000);
                let df = rng.range_i64(-100_000, 100_000);
                x.push(x.last().unwrap() + dx);
                f.push(f.last().unwrap() + df);
            }
            let (num, den, worst) = infer(&x, &f);
            let recomputed = x
                .windows(2)
                .zip(f.windows(2))
                .map(|(xw, fw)| deviation(fw[1] - fw[0], 0, xw[1] - xw[0], num, den).unsigned_abs())
                .max()
                .unwrap();
            assert_eq!(worst as u64, recomputed, "x={x:?} f={f:?} -> ({num},{den})");
        }
    }

    #[test]
    fn test_infer_negative_rate() {
        let x = [0u64, 10, 20];
        let f = [100i64, 0, -100];
        let (num, den, worst) = infer(&x, &f);
        assert_eq!((num, den), (-10, 1));
        assert_eq!(worst, 0);
    }
}
