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
use crate::extended::F106;
use crate::points::InterpError;
use crate::wide::cmp_frac;
use std::cmp::Ordering;

/// Returns `num[i.min(num.len()-1)]` when `num` has length 1 (shared) and `num[i]` otherwise
/// (per-segment) -- the length-1-or-n_segments convention every step kernel shares, chosen so a
/// future per-segment sampling interval is a no-op at this ABI.
pub fn rate_at(num: &[i64], den: &[u64], i: usize) -> (i64, u64) {
    if num.len() == 1 {
        (num[0], den[0])
    } else {
        (num[i], den[i])
    }
}

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
/// -- an irreducible fraction gives the smallest denominator for that exact value and makes two
/// rates comparable exactly.
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

/// Which way `tie_values` runs -- a distance axis may run backwards, so unlike [`Points`],
/// [`StepSeries`] allows either direction as long as it is consistent across the whole series.
///
/// [`Points`]: crate::points::Points
#[derive(Clone, Copy, PartialEq, Debug)]
enum Direction {
    Increasing,
    Decreasing,
}

fn direction(tie_values: &[i64]) -> Option<Direction> {
    if tie_values.windows(2).all(|w| w[0] < w[1]) {
        Some(Direction::Increasing)
    } else if tie_values.windows(2).all(|w| w[0] > w[1]) {
        Some(Direction::Decreasing)
    } else {
        None
    }
}

/// A constant-step tie series: `tie_indices`/`tie_values` are the segment boundaries (points),
/// `num`/`den` the declared step of each segment, shared (length 1) or per-segment. The
/// generalisation of [`Points`](crate::points::Points) that reconstructs values by stepping
/// from the nearest anchor at an exact rational rate rather than by two-point interpolation.
pub struct StepSeries<'a> {
    tie_indices: &'a [u64],
    tie_values: &'a [i64],
    num: &'a [i64],
    den: &'a [u64],
    forwardable: bool,
    direction: Option<Direction>,
}

impl<'a> StepSeries<'a> {
    /// # Panics
    ///
    /// Panics if `tie_indices` and `tie_values` do not have the same length, or `num`/`den` do
    /// not have the same length, 1 or `tie_indices.len() - 1`.
    pub fn new(
        tie_indices: &'a [u64],
        tie_values: &'a [i64],
        num: &'a [i64],
        den: &'a [u64],
    ) -> Self {
        assert_eq!(
            tie_indices.len(),
            tie_values.len(),
            "tie_indices and tie_values must have the same length"
        );
        assert_eq!(
            num.len(),
            den.len(),
            "num and den must have the same length"
        );
        let n_segments = tie_indices.len().saturating_sub(1);
        assert!(
            num.len() == 1 || num.len() == n_segments,
            "num/den must have length 1 or tie_indices.len() - 1"
        );
        let forwardable = tie_indices.windows(2).all(|w| w[0] < w[1]);
        let direction = direction(tie_values);
        StepSeries {
            tie_indices,
            tie_values,
            num,
            den,
            forwardable,
            direction,
        }
    }

    fn rate_at(&self, seg: usize) -> (i64, u64) {
        rate_at(self.num, self.den, seg)
    }

    /// Predicts the value at index `x`.
    pub fn forward(&self, x: u64) -> Result<i64, InterpError> {
        if !self.forwardable {
            return Err(InterpError::NotStrictlyIncreasing);
        }
        match self.tie_indices.binary_search(&x) {
            Ok(i) => Ok(self.tie_values[i]),
            Err(0) => Err(InterpError::OutOfBounds),
            Err(len) if len == self.tie_indices.len() => Err(InterpError::OutOfBounds),
            Err(i) => {
                let seg = i - 1;
                let k = x - self.tie_indices[seg];
                let (num, den) = self.rate_at(seg);
                Ok(predict(self.tie_values[seg], k, num, den))
            }
        }
    }

    /// The per-segment residual: `tie_values[i+1] - predict(tie_values[i], length_i, num_i,
    /// den_i)`, one entry per segment.
    pub fn deviation(&self) -> Vec<i64> {
        (0..self.tie_indices.len().saturating_sub(1))
            .map(|seg| {
                let k = self.tie_indices[seg + 1] - self.tie_indices[seg];
                let (num, den) = self.rate_at(seg);
                deviation(self.tie_values[seg + 1], self.tie_values[seg], k, num, den)
            })
            .collect()
    }

    /// Finds the index whose predicted value is `f`, per `method` when no index matches
    /// exactly.
    pub fn inverse(&self, f: i64, method: Method) -> Result<u64, InterpError> {
        let direction = self.direction.ok_or(InterpError::NotStrictlyIncreasing)?;
        let found = self.tie_values.binary_search_by(|probe| match direction {
            Direction::Increasing => probe.cmp(&f),
            Direction::Decreasing => f.cmp(probe),
        });
        match found {
            Ok(i) => Ok(self.tie_indices[i]),
            Err(0) => match method {
                Method::None | Method::ForwardFill => Err(InterpError::OutOfBounds),
                Method::Nearest | Method::BackwardFill => Ok(self.tie_indices[0]),
            },
            Err(len) if len == self.tie_values.len() => match method {
                Method::None | Method::BackwardFill => Err(InterpError::OutOfBounds),
                Method::Nearest | Method::ForwardFill => {
                    Ok(self.tie_indices[self.tie_indices.len() - 1])
                }
            },
            Err(i) => {
                let seg = i - 1;
                let (num, den) = self.rate_at(seg);
                let offset = f as i128 - self.tie_values[seg] as i128;
                if let Method::None = method {
                    // `round` is not injective when `|num| < den`: several `k` can share the
                    // same predicted value. An exact match is one that `forward` maps back onto
                    // the very same value, not one whose *unrounded* ratio happens to be an
                    // integer -- so verify the nearest candidate round-trips, exactly as
                    // `Inverse<u64> for f64` does in schemes.rs, rather than asking whether
                    // `offset * den / num` divides evenly.
                    let nearest = div_signed(offset * den as i128, num as i128, Method::Nearest)
                        .expect("Method::Nearest never returns None");
                    let candidate = (self.tie_indices[seg] as i128 + nearest) as u64;
                    if self.forward(candidate) == Ok(f) {
                        Ok(candidate)
                    } else {
                        Err(InterpError::NotFound)
                    }
                } else {
                    let k = div_signed(offset * den as i128, num as i128, method)
                        .ok_or(InterpError::NotFound)?;
                    Ok((self.tie_indices[seg] as i128 + k) as u64)
                }
            }
        }
    }
}

fn rate_at_float(delta: &[f64], i: usize) -> f64 {
    if delta.len() == 1 {
        delta[0]
    } else {
        delta[i]
    }
}

/// The float twin of [`StepSeries`]: on the float side no exact rate exists (`den` is pinned to
/// 1), so `delta` replaces `num`/`den` and the arithmetic is `tie0 + k * delta` in [`F106`],
/// exactly the two-point float schemes in `schemes.rs` with the second point dropped.
pub struct FloatStepSeries<'a> {
    tie_indices: &'a [u64],
    tie_values: &'a [f64],
    delta: &'a [f64],
    forwardable: bool,
    direction: Option<Direction>,
}

fn float_direction(tie_values: &[f64]) -> Option<Direction> {
    if tie_values.windows(2).all(|w| w[0] < w[1]) {
        Some(Direction::Increasing)
    } else if tie_values.windows(2).all(|w| w[0] > w[1]) {
        Some(Direction::Decreasing)
    } else {
        None
    }
}

impl<'a> FloatStepSeries<'a> {
    /// # Panics
    ///
    /// Panics if `tie_indices` and `tie_values` do not have the same length, or `delta` does not
    /// have length 1 or `tie_indices.len() - 1`.
    pub fn new(tie_indices: &'a [u64], tie_values: &'a [f64], delta: &'a [f64]) -> Self {
        assert_eq!(
            tie_indices.len(),
            tie_values.len(),
            "tie_indices and tie_values must have the same length"
        );
        let n_segments = tie_indices.len().saturating_sub(1);
        assert!(
            delta.len() == 1 || delta.len() == n_segments,
            "delta must have length 1 or tie_indices.len() - 1"
        );
        let forwardable = tie_indices.windows(2).all(|w| w[0] < w[1]);
        let direction = float_direction(tie_values);
        FloatStepSeries {
            tie_indices,
            tie_values,
            delta,
            forwardable,
            direction,
        }
    }

    /// Predicts the value at index `x`.
    pub fn forward(&self, x: u64) -> Result<f64, InterpError> {
        if !self.forwardable {
            return Err(InterpError::NotStrictlyIncreasing);
        }
        match self.tie_indices.binary_search(&x) {
            Ok(i) => Ok(self.tie_values[i]),
            Err(0) => Err(InterpError::OutOfBounds),
            Err(len) if len == self.tie_indices.len() => Err(InterpError::OutOfBounds),
            Err(i) => {
                let seg = i - 1;
                let k = x - self.tie_indices[seg];
                let delta = rate_at_float(self.delta, seg);
                let value: f64 = F106::from(self.tie_values[seg])
                    .add(&F106::from(k).mul(&F106::from(delta)))
                    .into();
                Ok(value)
            }
        }
    }

    /// Finds the index whose predicted value is `f`, per `method` when no index matches
    /// exactly.
    pub fn inverse(&self, f: f64, method: Method) -> Result<u64, InterpError> {
        let direction = self.direction.ok_or(InterpError::NotStrictlyIncreasing)?;
        let found = self.tie_values.binary_search_by(|probe| {
            let order = probe.partial_cmp(&f).expect("nan or inf encountered");
            match direction {
                Direction::Increasing => order,
                Direction::Decreasing => order.reverse(),
            }
        });
        match found {
            Ok(i) => Ok(self.tie_indices[i]),
            Err(0) => match method {
                Method::None | Method::ForwardFill => Err(InterpError::OutOfBounds),
                Method::Nearest | Method::BackwardFill => Ok(self.tie_indices[0]),
            },
            Err(len) if len == self.tie_values.len() => match method {
                Method::None | Method::BackwardFill => Err(InterpError::OutOfBounds),
                Method::Nearest | Method::ForwardFill => {
                    Ok(self.tie_indices[self.tie_indices.len() - 1])
                }
            },
            Err(i) => {
                let seg = i - 1;
                let delta = rate_at_float(self.delta, seg);
                let tie0 = self.tie_values[seg];
                // both the numerator and denominator are exact in F106; the division is the
                // only rounding step, exactly as `Inverse<u64> for f64` in schemes.rs
                let w = F106::from_diff(f, tie0).div(&F106::from(delta));
                let x = F106::from(self.tie_indices[seg]).add(&w);
                match method {
                    Method::None => {
                        let candidate: u64 = x.round().into();
                        let candidate =
                            candidate.clamp(self.tie_indices[seg], self.tie_indices[seg + 1]);
                        if self.forward(candidate) == Ok(f) {
                            Ok(candidate)
                        } else {
                            Err(InterpError::NotFound)
                        }
                    }
                    Method::Nearest => Ok(x.round().into()),
                    Method::ForwardFill => Ok(x.floor().into()),
                    Method::BackwardFill => Ok(x.ceil().into()),
                }
            }
        }
    }
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

    #[test]
    fn test_forward_step_exact_and_boundaries() {
        let tie_indices = [0u64, 10, 20];
        let tie_values = [100i64, 200, 500];
        let num = [10i64, 30];
        let den = [1u64, 1];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.forward(0), Ok(100));
        assert_eq!(series.forward(5), Ok(150));
        assert_eq!(series.forward(10), Ok(200));
        assert_eq!(series.forward(15), Ok(350));
        assert_eq!(series.forward(20), Ok(500));
        assert_eq!(series.forward(21), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_forward_step_shared_rate() {
        let tie_indices = [0u64, 10, 20];
        let tie_values = [0i64, 100, 200];
        let num = [10i64];
        let den = [1u64];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.forward(7), Ok(70));
        assert_eq!(series.forward(17), Ok(170));
    }

    #[test]
    fn test_forward_step_negative_rate() {
        let tie_indices = [0u64, 10];
        let tie_values = [100i64, 0];
        let num = [-10i64];
        let den = [1u64];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.forward(3), Ok(70));
        assert_eq!(series.forward(10), Ok(0));
    }

    #[test]
    fn test_deviation_step_matches_manual() {
        let tie_indices = [0u64, 10, 20];
        let tie_values = [0i64, 101, 199];
        let num = [10i64];
        let den = [1u64];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.deviation(), vec![1, -2]);
    }

    #[test]
    fn test_inverse_step_round_trip_increasing() {
        let tie_indices = [0u64, 10, 20];
        let tie_values = [0i64, 100, 500];
        let num = [10i64, 40];
        let den = [1u64, 1];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        for x in 0..=20u64 {
            let f = series.forward(x).unwrap();
            assert_eq!(series.inverse(f, Method::None), Ok(x), "x={x}");
        }
    }

    #[test]
    fn test_inverse_step_round_trip_decreasing() {
        let tie_indices = [0u64, 10, 20];
        let tie_values = [500i64, 100, 0];
        let num = [-40i64, -10];
        let den = [1u64, 1];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        for x in 0..=20u64 {
            let f = series.forward(x).unwrap();
            assert_eq!(series.inverse(f, Method::None), Ok(x), "x={x}");
        }
    }

    #[test]
    fn test_inverse_step_methods_between_ticks() {
        // rate 3/2: index 0 -> 0, index 1 -> 2 (round(1.5)=2 ties to even), index 2 -> 3
        let tie_indices = [0u64, 2];
        let tie_values = [0i64, 3];
        let num = [3i64];
        let den = [2u64];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        // f=1 sits strictly between index 0 (f=0) and index 1 (f=2): exact k=2/3
        assert_eq!(series.inverse(1, Method::None), Err(InterpError::NotFound));
        assert_eq!(series.inverse(1, Method::ForwardFill), Ok(0));
        assert_eq!(series.inverse(1, Method::BackwardFill), Ok(1));
        assert_eq!(series.inverse(1, Method::Nearest), Ok(1));
    }

    #[test]
    fn test_inverse_step_out_of_bounds_and_clamping() {
        let tie_indices = [0u64, 10];
        let tie_values = [0i64, 100];
        let num = [10i64];
        let den = [1u64];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(
            series.inverse(-1, Method::None),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(series.inverse(-1, Method::Nearest), Ok(0));
        assert_eq!(
            series.inverse(-1, Method::ForwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(series.inverse(-1, Method::BackwardFill), Ok(0));
        assert_eq!(
            series.inverse(101, Method::None),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(series.inverse(101, Method::Nearest), Ok(10));
        assert_eq!(series.inverse(101, Method::ForwardFill), Ok(10));
        assert_eq!(
            series.inverse(101, Method::BackwardFill),
            Err(InterpError::OutOfBounds)
        );
    }

    #[test]
    fn test_forward_step_naive_i64_overflow_magnitudes() {
        // k*num close to u64::MAX * i64::MAX territory -- the arithmetic this module uses:
        // the barycentric-style product fits u128 (and i128) but a naive
        // i64 multiply would wrap many times over.
        let tie_indices = [0u64, u64::MAX];
        let tie_values = [i64::MIN, i64::MAX];
        let num = [i64::MAX];
        let den = [u64::MAX];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.forward(0), Ok(i64::MIN));
        assert_eq!(series.forward(u64::MAX), Ok(i64::MAX));
        // this rate is compressive (|num| < den, close to 1/2), so forward is not injective and
        // its rounding bias accumulates over large k -- several indices legitimately share a
        // rounded value, and the nearest nearby index is not always the original one (see
        // `test_forward_inverse_step_round_trip_property` for the round-trip guarantee, which
        // holds at the *construction* points of a step series). Here, just check the arithmetic
        // does not overflow or panic at these magnitudes, and stays within the declared range.
        for x in [1u64, u64::MAX / 4, u64::MAX / 3, u64::MAX / 2, u64::MAX - 1] {
            let f = series.forward(x).unwrap();
            assert!(
                (i64::MIN..i64::MAX).contains(&f),
                "f={f} out of range for x={x}"
            );
        }
    }

    #[test]
    fn test_inverse_step_naive_i64_overflow_magnitudes() {
        // (f - tie0) * den reaches (2**64-1)**2, twice i128::MAX -- exactly the case the
        // module doc comment calls out as needing u128, not i128.
        let tie_indices = [0u64, u64::MAX];
        let tie_values = [i64::MIN, i64::MAX];
        let num = [1i64];
        let den = [u64::MAX];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.inverse(i64::MIN, Method::None), Ok(0));
        assert_eq!(series.inverse(i64::MAX, Method::None), Ok(u64::MAX));
    }

    #[test]
    fn test_deviation_step_per_segment_den_array() {
        let tie_indices = [0u64, 10, 25];
        let tie_values = [0i64, 100, 250];
        let num = [10i64, 10];
        let den = [1u64, 1];
        let series = StepSeries::new(&tie_indices, &tie_values, &num, &den);
        assert_eq!(series.deviation(), vec![0, 0]);
    }

    /// `forward_step` then `inverse_step(..., Method::None)` must round-trip on random
    /// series, including negative rates and magnitudes overflowing a naive `i64` product.
    #[test]
    fn test_forward_inverse_step_round_trip_property() {
        let mut rng = SplitMix64(99);
        for _ in 0..300 {
            let n_segments = rng.range_u64(1, 6) as usize;
            let mut tie_indices = vec![0u64];
            for _ in 0..n_segments {
                tie_indices.push(tie_indices.last().unwrap() + rng.range_u64(1, 1_000_000_000));
            }
            let increasing = rng.next().is_multiple_of(2);
            let mut tie_values = vec![rng.range_i64(-1_000_000_000_000, 1_000_000_000_000)];
            let shared_den = rng.range_u64(1, 1000);
            let per_segment = n_segments > 1 && rng.next().is_multiple_of(2);
            let rate = |rng: &mut SplitMix64| {
                if increasing {
                    rng.range_i64(1, 1_000_000)
                } else {
                    rng.range_i64(-1_000_000, -1)
                }
            };
            // shared: one rate for every segment; per-segment: independently drawn
            let shared_rate = rate(&mut rng);
            let mut num = Vec::new();
            for i in 0..n_segments {
                let length = tie_indices[i + 1] - tie_indices[i];
                let this_rate = if per_segment {
                    rate(&mut rng)
                } else {
                    shared_rate
                };
                num.push(this_rate);
                let step = round_step(length, this_rate, shared_den);
                tie_values.push((tie_values[i] as i128 + step) as i64);
            }
            let (num_arr, den_arr) = if per_segment {
                (num, vec![shared_den; n_segments])
            } else {
                (vec![num[0]], vec![shared_den])
            };
            let series = StepSeries::new(&tie_indices, &tie_values, &num_arr, &den_arr);
            for i in 0..tie_indices.len() {
                let x = tie_indices[i];
                let f = series.forward(x).unwrap();
                assert_eq!(
                    series.inverse(f, Method::None),
                    Ok(x),
                    "tie_indices={tie_indices:?} tie_values={tie_values:?} num={num_arr:?} den={den_arr:?}"
                );
            }
        }
    }
}
