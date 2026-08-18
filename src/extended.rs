//! Extended precision floating-point format that can accurately represent 64 bits integers.
//!
//! [`F106`] is a *double-double*: an unevaluated sum `hi + lo` of two non-overlapping `f64`s,
//! which carries a 106-bit significand while keeping the `f64` exponent range. It is built only
//! from IEEE-754 double operations (Knuth's `two_sum` and Dekker's `two_prod`/`split`), so it
//! needs no external crate, allocates nothing, and gives bit-identical results on every platform.
//!
//! Why 106 bits: interpolating an index in `0..2^64` from `f64` values requires strictly more
//! than 64 significand bits. A 64-bit format rounds the quotient to the unit of the last
//! integer place, which is enough to push `floor`/`ceil`/`round` to the wrong integer. With 106
//! bits the worst-case absolute error on a `u64` index stays below `2^-40`.

use std::cmp::Ordering;

/// Dekker's splitting constant, `2^27 + 1`.
const SPLITTER: f64 = 134_217_729.0;

/// Operands above this magnitude are scaled down before splitting to avoid overflow.
const SPLIT_THRESHOLD: f64 = 6.696_928_794_914_171e299; // 2^996

/// Splits an f64 into two 26/27-bit halves whose sum is exact.
#[inline]
fn split(a: f64) -> (f64, f64) {
    if a.abs() > SPLIT_THRESHOLD {
        let a = a * 3.725_290_298_461_914e-9; // 2^-28
        let t = SPLITTER * a;
        let hi = t - (t - a);
        let lo = a - hi;
        (hi * 268_435_456.0, lo * 268_435_456.0) // 2^28
    } else {
        let t = SPLITTER * a;
        let hi = t - (t - a);
        (hi, a - hi)
    }
}

/// Exact sum of two f64s: returns `(s, e)` with `s = fl(a + b)` and `a + b = s + e` exactly.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// Exact sum of two f64s, valid only when `|a| >= |b|`. Cheaper than [`two_sum`].
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Exact product of two f64s: returns `(p, e)` with `p = fl(a * b)` and `a * b = p + e` exactly.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// Double-double floating-point value with a 106 bits significand. It implements total ordering
/// by only allowing finite values (no nan or inf). Use the From/Into traits to initialize some
/// instance of this struct from u64 or f64.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct F106 {
    hi: f64,
    lo: f64,
}
impl From<u64> for F106 {
    /// Converts a u64 into an F106. Always exact.
    fn from(value: u64) -> F106 {
        let hi = value as f64;
        // |value - hi| <= 2^11, so the residual is exactly representable.
        let lo = (value as i128 - hi as i128) as f64;
        F106 { hi, lo }
    }
}
impl From<f64> for F106 {
    /// Converts an f64 into an F106. Panics if the input is NaN or infinity.
    fn from(value: f64) -> F106 {
        assert!(value.is_finite());
        F106 { hi: value, lo: 0.0 }
    }
}
impl From<F106> for f64 {
    /// Converts an F106 into an f64.
    fn from(float: F106) -> f64 {
        float.hi + float.lo
    }
}
impl From<F106> for u64 {
    /// Converts an F106 into a u64, saturating at both ends and truncating towards zero.
    fn from(float: F106) -> u64 {
        if float.hi <= 0.0 {
            return 0;
        }
        // `floor` leaves both limbs integral, so each converts exactly.
        let floor = float.floor();
        let hi = floor.hi as i128;
        let lo = floor.lo as i128;
        hi.saturating_add(lo).clamp(0, u64::MAX as i128) as u64
    }
}
impl Eq for F106 {}
impl Ord for F106 {
    /// Compares two F106.
    fn cmp(&self, other: &F106) -> Ordering {
        self.hi
            .partial_cmp(&other.hi)
            .unwrap()
            .then_with(|| self.lo.partial_cmp(&other.lo).unwrap())
    }
}
impl PartialOrd for F106 {
    /// Compares two F106.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl F106 {
    /// Builds the exact difference of two f64s. Unlike [`F106::from`] followed by
    /// [`F106::sub`], this never rounds: `f0 - f1` is always representable in a double-double.
    pub fn from_diff(lhs: f64, rhs: f64) -> F106 {
        let (hi, lo) = two_sum(lhs, -rhs);
        F106 { hi, lo }
    }
    /// Returns whether the value is finite.
    pub fn is_finite(&self) -> bool {
        self.hi.is_finite()
    }
    /// Restores the non-overlapping invariant of a `hi`/`lo` pair.
    #[inline]
    fn renormalize(hi: f64, lo: f64) -> F106 {
        let (hi, lo) = quick_two_sum(hi, lo);
        F106 { hi, lo }
    }
    /// Adds two F106s.
    pub fn add(&self, rhs: &F106) -> F106 {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let (t, f) = two_sum(self.lo, rhs.lo);
        let (s, e) = quick_two_sum(s, e + t);
        F106::renormalize(s, e + f)
    }
    /// Subtracts two F106s.
    pub fn sub(&self, rhs: &F106) -> F106 {
        self.add(&F106 {
            hi: -rhs.hi,
            lo: -rhs.lo,
        })
    }
    /// Multiplies two F106s.
    pub fn mul(&self, rhs: &F106) -> F106 {
        let (p, e) = two_prod(self.hi, rhs.hi);
        F106::renormalize(p, e + (self.hi * rhs.lo + self.lo * rhs.hi))
    }
    /// Divides two F106s.
    pub fn div(&self, rhs: &F106) -> F106 {
        // Long division: a first quotient digit, then one Newton-style correction on the
        // exact remainder. Accurate to about 2^-104 relative.
        let q1 = self.hi / rhs.hi;
        let r = self.sub(&rhs.mul(&F106::from(q1)));
        let q2 = f64::from(r) / rhs.hi;
        F106::renormalize(q1, q2)
    }
    /// Rounds an F106 to its nearest integer using the round ties to even rule.
    pub fn round(&self) -> F106 {
        let hi = self.hi.round_ties_even();
        if hi == self.hi {
            // `hi` already carries the integer part; `lo` holds the fraction. A tie in `lo`
            // must be broken against the parity of `hi`, which is what decides evenness.
            let lo = self.lo.round_ties_even();
            let lo = if (self.lo - lo).abs() == 0.5 && hi % 2.0 != 0.0 {
                lo + (self.lo - lo).signum()
            } else {
                lo
            };
            return F106::renormalize(hi, lo);
        }
        // `self.hi` is not an integer, so it sits at least one ulp away from the neighbouring
        // integers while `|lo| <= ulp/2`. Only an exact tie on `self.hi` can be swung by `lo`.
        let shift = hi - self.hi;
        if shift.abs() == 0.5 && self.lo != 0.0 && shift.signum() != self.lo.signum() {
            F106 {
                hi: hi - shift.signum(),
                lo: 0.0,
            }
        } else {
            F106 { hi, lo: 0.0 }
        }
    }
    /// Floors an F106.
    pub fn floor(&self) -> F106 {
        let hi = self.hi.floor();
        if hi == self.hi {
            F106::renormalize(hi, self.lo.floor())
        } else {
            F106 { hi, lo: 0.0 }
        }
    }
    /// Ceils an F106.
    pub fn ceil(&self) -> F106 {
        let hi = self.hi.ceil();
        if hi == self.hi {
            F106::renormalize(hi, self.lo.ceil())
        } else {
            F106 { hi, lo: 0.0 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_conversion() {
        let cases: [u64; 7] = [0, 1, 2, u64::MAX / 2, u64::MAX - 2, u64::MAX - 1, u64::MAX];
        for expected in cases.iter() {
            let result: u64 = F106::from(*expected).into();
            assert_eq!(result, *expected);
        }
        let cases: [(f64, u64); 4] = [(-1.0, 0), (0.5, 0), (1.5, 1), (1e32, u64::MAX)];
        for (input, expected) in cases.iter() {
            let result: u64 = F106::from(*input).into();
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_f64_conversion() {
        let cases: [f64; 11] = [
            0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 1e307, -1e307, 1e-307, -1e-307,
        ];
        for expected in cases.iter() {
            let result: f64 = F106::from(*expected).into();
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_rounding() {
        let cases: [(f64, u64); 13] = [
            (0.0, 0),
            (0.1, 0),
            (0.4, 0),
            (0.5, 0),
            (0.6, 1),
            (0.9, 1),
            (1.0, 1),
            (1.1, 1),
            (1.4, 1),
            (1.5, 2),
            (1.6, 2),
            (1.9, 2),
            (2.0, 2),
        ];
        for (input, expected) in cases {
            let result: u64 = F106::from(input).round().into();
            assert_eq!(result, expected)
        }
    }

    #[test]
    fn test_exact_u64_arithmetic() {
        // every u64 round-trips through add/sub without loss
        let cases: [u64; 5] = [1, 1 << 53, (1 << 53) + 1, u64::MAX - 1, u64::MAX];
        for value in cases {
            let a = F106::from(value);
            let one = F106::from(1u64);
            let result: u64 = a.sub(&one).add(&one).into();
            assert_eq!(result, value);
        }
    }

    #[test]
    fn test_exact_difference() {
        // f0 - f1 is exact even when the exponents are far apart
        let a = F106::from_diff(1.0, 1e-300);
        assert_eq!(a.hi, 1.0);
        assert_eq!(a.lo, -1e-300);
        let b: f64 = a.add(&F106::from(1e-300)).into();
        assert_eq!(b, 1.0);
    }

    #[test]
    fn test_division_resolves_u64() {
        // (2^64 - 3) / 3 * 3 must land back on the exact integer
        let num = F106::from(u64::MAX - 2);
        let three = F106::from(3u64);
        let result: u64 = num.div(&three).mul(&three).round().into();
        assert_eq!(result, u64::MAX - 2);
    }

    #[test]
    fn test_ordering() {
        assert!(F106::from(1u64) < F106::from(2u64));
        assert!(F106::from(u64::MAX - 1) < F106::from(u64::MAX));
        assert_eq!(F106::from(1.0), F106::from(1u64));
    }
}
