//! Forward and backward linear interpolation schemes between two points (x0, f0) and (x1, f1) for
//! different data types (x is u64, f is either i64 or f64).
//!
//! When the values are integers, operations are performed with u128 integers to avoid overflow.
//! Signed integers are mapped on positive values to avoid potential subtraction overflows
//! (subtracting i64::MIN to i64::MAX overflows whereas it does not for u64).
//!
//! When the values are floats, operations are performed with extended-precision floats. Indices
//! span the whole u64 range while an f64 only holds 53 significand bits, so plain f64 arithmetic
//! would resolve a large index to the wrong integer; 64 bits are not enough either, since the
//! quotient of the inverse scheme itself needs 64 bits before it is rounded to an index.
//!
//! Both float schemes are written as `p0 + (p1 - p0) * w` with an interpolation weight `w` in
//! `[0, 1]`, rather than as the two-product form `(p0 * (q1 - q) + p1 * (q - q0)) / (q1 - q0)`.
//! The two forms agree in exact arithmetic, but the weighted one keeps every intermediate within
//! the range spanned by the end points, so nothing overflows when the values are large and the
//! index span is wide. It also leaves a single rounding step, the division that forms `w`: the
//! index differences are exact in u64, and the value difference `p1 - p0` is exact in `F106`.

use crate::divop::{DivOp, Method};
use crate::extended::F106;

/// Implements forward scheme from index to value.
pub trait Forward<F>: Copy + Ord {
    /// Estimate f at index x between two points (x0, f0) and (x1, f1)
    fn forward(self, x0: Self, x1: Self, f0: F, f1: F) -> F;
}
impl Forward<u64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: u64, f1: u64) -> u64 {
        let num = (f0 as u128) * ((x1 - self) as u128) + (f1 as u128) * ((self - x0) as u128);
        let den = (x1 - x0) as u128;
        num.div(den, Method::Nearest).unwrap() as u64
    }
}
impl Forward<i64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: i64, f1: i64) -> i64 {
        self.forward(x0, x1, f0.to_unsigned(), f1.to_unsigned())
            .to_signed()
    }
}
impl Forward<f64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: f64, f1: f64) -> f64 {
        // index differences are exact in u64, and exact again in F106
        let w = F106::from(self - x0).div(&F106::from(x1 - x0));
        let (scale, unscale) = halving_scale(f0, f1);
        let (g0, g1) = (f0 * scale, f1 * scale);
        // f1 - f0 is exact in F106, so the division above is the only rounding step
        let g: f64 = F106::from(g0).add(&F106::from_diff(g1, g0).mul(&w)).into();
        g * unscale
    }
}

/// Returns the factor the end points must be multiplied by before their difference is taken,
/// together with the factor undoing it.
///
/// The schemes rely on `f1 - f0`, which overflows only when the two end points nearly span the
/// whole f64 range. Interpolation is linear, so halving the end points and undoing it on the
/// result is exact and sidesteps the overflow. Both factors are powers of two, so neither
/// multiplication rounds.
fn halving_scale(f0: f64, f1: f64) -> (f64, f64) {
    if (f1 - f0).is_finite() {
        (1.0, 1.0)
    } else {
        (0.5, 2.0)
    }
}

/// Implements inverse scheme from value to index.
pub trait Inverse<X>: Copy + PartialOrd {
    /// Estimate x at values f between two points (x0, f0) and (x1, f1)
    fn inverse(self, x0: X, x1: X, f0: Self, f1: Self, method: Method) -> Option<X>;
}
impl Inverse<u64> for u64 {
    fn inverse(self, x0: u64, x1: u64, f0: u64, f1: u64, method: Method) -> Option<u64> {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div(den, method).map(|x| x as u64)
    }
}
impl Inverse<u64> for i64 {
    fn inverse(self, x0: u64, x1: u64, f0: i64, f1: i64, method: Method) -> Option<u64> {
        self.to_unsigned()
            .inverse(x0, x1, f0.to_unsigned(), f1.to_unsigned(), method)
    }
}
impl Inverse<u64> for f64 {
    fn inverse(self, x0: u64, x1: u64, f0: f64, f1: f64, method: Method) -> Option<u64> {
        // both value differences are exact in F106, so the only rounding is the division
        let (scale, _) = halving_scale(f0, f1);
        let w =
            F106::from_diff(self * scale, f0 * scale).div(&F106::from_diff(f1 * scale, f0 * scale));
        let x = F106::from(x0).add(&F106::from(x1 - x0).mul(&w));
        match method {
            // an exact match is one that `forward` maps back onto the very same value; asking
            // whether `x` looks integral instead would make the answer depend on how the
            // division happened to round
            Method::None => {
                let candidate: u64 = x.round().into();
                let candidate = candidate.clamp(x0, x1);
                if candidate.forward(x0, x1, f0, f1) == self {
                    Some(candidate)
                } else {
                    None
                }
            }
            Method::Nearest => Some(x.round().into()),
            Method::ForwardFill => Some(x.floor().into()),
            Method::BackwardFill => Some(x.ceil().into()),
        }
    }
}

/// Implements signed to unsinged translation. Used to apply schemes on unsigned integers where
/// no overflow can occur.
pub trait ToUnsigned<U> {
    /// Converts signed to unsinged by subtracting the minimum negative signed integer.  
    fn to_unsigned(self) -> U;
}
impl ToUnsigned<u64> for i64 {
    fn to_unsigned(self) -> u64 {
        self.wrapping_sub(i64::MIN) as u64
    }
}

/// Implements unsinged to singed translation. Used to retreive applied schemes on unsigned integers.
pub trait ToSigned<S> {
    /// Converts unsigned to singed by adding the minimum negative signed integer.  
    fn to_signed(self) -> S;
}
impl ToSigned<i64> for u64 {
    fn to_signed(self) -> i64 {
        self.wrapping_add(i64::MIN as u64) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_uses_exact_index_differences() {
        // near 2^60 an f64 index step is 256, so `x0 + 1` is not representable: converting the
        // indices to f64 before subtracting would collapse the whole span onto f0
        let (x0, x1) = (1u64 << 60, (1u64 << 60) + 4);
        assert_eq!((x0 + 1).forward(x0, x1, 0.0, 1.0), 0.25);
        assert_eq!((x0 + 2).forward(x0, x1, 0.0, 1.0), 0.5);
        assert_eq!((x0 + 3).forward(x0, x1, 0.0, 1.0), 0.75);
    }

    #[test]
    fn test_forward_is_monotonic_and_bounded() {
        let (x0, x1) = (0u64, 100_000);
        let (f0, f1) = (-3.0, 2.0);
        let mut previous = f64::NEG_INFINITY;
        for x in (x0..=x1).step_by(7) {
            let f = x.forward(x0, x1, f0, f1);
            assert!(f >= previous);
            assert!((f0..=f1).contains(&f));
            previous = f;
        }
    }

    #[test]
    fn test_inverse_resolves_indices_near_u64_max() {
        // 64 bits of mantissa are not enough here: the quotient itself needs 64 bits, so any
        // rounding lands on the wrong integer
        let (x0, x1) = (u64::MAX - 1000, u64::MAX);
        let (f0, f1) = (0.0, 1.0);
        for x in [x0, x0 + 1, x0 + 499, x0 + 500, x1 - 1, x1] {
            let f = x.forward(x0, x1, f0, f1);
            assert_eq!(
                f.inverse(x0, x1, f0, f1, Method::Nearest),
                Some(x),
                "x = {x}"
            );
            assert_eq!(f.inverse(x0, x1, f0, f1, Method::None), Some(x), "x = {x}");
            // `f` is itself rounded to an f64, so its exact preimage may fall just short of
            // `x`; the fills must still bracket it without ever straying further
            let ffill = f.inverse(x0, x1, f0, f1, Method::ForwardFill).unwrap();
            let bfill = f.inverse(x0, x1, f0, f1, Method::BackwardFill).unwrap();
            assert!(ffill == x || ffill == x - 1, "x = {x}, ffill = {ffill}");
            assert!(bfill == x || bfill == x + 1, "x = {x}, bfill = {bfill}");
        }
    }

    #[test]
    fn test_forward_inverse_round_trip() {
        let (x0, x1) = (3u64, 100_003);
        let (f0, f1) = (-1.5e-7, 7.25e11);
        for x in (x0..=x1).step_by(97) {
            let f = x.forward(x0, x1, f0, f1);
            assert_eq!(f.inverse(x0, x1, f0, f1, Method::None), Some(x), "x = {x}");
        }
    }

    #[test]
    fn test_exact_match_rejects_off_line_values() {
        let (x0, x1) = (0u64, 2);
        let (f0, f1) = (3.0, 7.0);
        assert_eq!(5.0f64.inverse(x0, x1, f0, f1, Method::None), Some(1));
        assert_eq!(5.5f64.inverse(x0, x1, f0, f1, Method::None), None);
        assert_eq!(4.0f64.inverse(x0, x1, f0, f1, Method::None), None);
        let one_ulp_off = f64::from_bits(5.0f64.to_bits() + 1);
        assert_eq!(one_ulp_off.inverse(x0, x1, f0, f1, Method::None), None);
    }

    #[test]
    fn test_extreme_value_span_does_not_overflow() {
        // f1 - f0 overflows an f64 here
        let (x0, x1) = (0u64, 4);
        let (f0, f1) = (-1.5e308, 1.5e308);
        for x in x0..=x1 {
            assert!(x.forward(x0, x1, f0, f1).is_finite());
        }
        assert_eq!(2u64.forward(x0, x1, f0, f1), 0.0);
        assert_eq!(0.0f64.inverse(x0, x1, f0, f1, Method::Nearest), Some(2));
        assert_eq!(0.0f64.inverse(x0, x1, f0, f1, Method::None), Some(2));
    }

    #[test]
    fn test_ties_round_to_even() {
        let (x0, x1) = (0u64, 4);
        let (f0, f1) = (0.0, 4.0);
        // 0.5 and 2.5 sit exactly halfway between two indices
        assert_eq!(0.5f64.inverse(x0, x1, f0, f1, Method::Nearest), Some(0));
        assert_eq!(1.5f64.inverse(x0, x1, f0, f1, Method::Nearest), Some(2));
        assert_eq!(2.5f64.inverse(x0, x1, f0, f1, Method::Nearest), Some(2));
        assert_eq!(3.5f64.inverse(x0, x1, f0, f1, Method::Nearest), Some(4));
    }
}
