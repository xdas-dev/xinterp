//! Forward and inverse integer interpolation on an explicit series of points.
//!
//! This module provides functionality for performing forward and inverse interpolation
//! on a piecewise linear function defined by explicit knots. Forward interpolation estimates
//! the value of the function at a given index within the range of known data points, while
//! inverse interpolation estimates the index corresponding to a given value of the function.
//!
//! # Examples
//!
//! ```
//! use xinterp::points::Points;
//! use xinterp::divop::Method;
//!
//! let xp = vec![0, 2, 4];
//! let fp = vec![0.0, 4.0, 16.0];
//!
//! let interp = Points::new(&xp, &fp);
//!
//! let result = interp.forward(3);
//! assert_eq!(result, Ok(10.0));
//!
//! let result = interp.inverse(10.1, Method::Nearest);
//! assert_eq!(result, Ok(3));
//! ```
//!
//! # Errors
//!
//! - `InterpError::OutOfBounds`: Indicates that the input value is outside the range of known
//!   data points.
//! - `InterpError::NotFound`: Indicates that the output value does not exist within the range of
//!   known data points.
//! - `InterpError::NotStrictlyIncreasing`: Indicates that the input or output values are not
//!   strictly increasing, which is required for interpolation.

use crate::divop::Method;
use crate::schemes::{Forward, Inverse};

// Interpolation Errors
#[derive(PartialEq, Debug)]
pub enum InterpError {
    OutOfBounds,
    NotFound,
    NotStrictlyIncreasing,
}

/// Structure for performing forward and inverse interpolation on an explicit series of points.
pub struct Points<'a, X, F> {
    xp: &'a [X],
    fp: &'a [F],
    forwardable: bool,
    inversable: bool,
}

impl<'a, X, F> Points<'a, X, F>
where
    X: Forward<F>,
    F: Inverse<X>,
{
    /// Constructs a new Points instance borrowing the given data points.
    ///
    /// # Arguments
    ///
    /// * `xp` - Slice of indices.
    /// * `fp` - Slice of corresponding values.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `xp` and `fp` are not equal.
    pub fn new(xp: &'a [X], fp: &'a [F]) -> Points<'a, X, F> {
        assert!(xp.len() == fp.len(), "xp and fp must have same length");
        let forwardable = xp.windows(2).all(|pair| pair[0] < pair[1]);
        let inversable = fp.windows(2).all(|pair| pair[0] < pair[1]);
        Points {
            xp,
            fp,
            forwardable,
            inversable,
        }
    }
    /// Performs forward interpolation at the given index.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The index for forward interpolation.
    ///
    /// # Returns
    ///
    /// If successful, returns the interpolated value.
    /// Otherwise, returns an error indicating the reason for failure.
    pub fn forward(&self, rhs: X) -> Result<F, InterpError> {
        if self.forwardable {
            match self.xp.binary_search(&rhs) {
                Ok(index) => Ok(self.fp[index]),
                Err(0) => Err(InterpError::OutOfBounds),
                Err(len) if len == self.xp.len() => Err(InterpError::OutOfBounds),
                Err(index) => Ok(rhs.forward(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    /// Performs inverse interpolation at the given value.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The value for inverse interpolation.
    /// * `method` - The rounding method to use in case of inexact matching.
    ///
    /// # Returns
    ///
    /// If successful, returns the interpolated input value.
    /// Otherwise, returns an error indicating the reason for failure.
    pub fn inverse(&self, rhs: F, method: Method) -> Result<X, InterpError> {
        if self.inversable {
            match self
                .fp
                .binary_search_by(|f| f.partial_cmp(&rhs).expect("nan or inf encountered"))
            {
                Ok(index) => Ok(self.xp[index]),
                Err(0) => match method {
                    Method::None | Method::ForwardFill => Err(InterpError::OutOfBounds),
                    Method::Nearest | Method::BackwardFill => Ok(self.xp[0]),
                },
                Err(len) if len == self.xp.len() => match method {
                    Method::None | Method::BackwardFill => Err(InterpError::OutOfBounds),
                    Method::Nearest | Method::ForwardFill => Ok(self.xp[len - 1]),
                },
                Err(index) => rhs
                    .inverse(
                        self.xp[index - 1],
                        self.xp[index],
                        self.fp[index - 1],
                        self.fp[index],
                        method,
                    )
                    .ok_or(InterpError::NotFound),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![20, 25];
        let interp = Points::new(&xp, &fp);
        assert!(interp.forwardable);
        assert!(interp.inversable);

        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![-20, -25];
        let interp = Points::new(&xp, &fp);
        assert!(interp.forwardable);
        assert!(!interp.inversable);
    }

    #[test]
    fn test_forward_unsigned() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<u64> = vec![20, 25];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.forward(0), Ok(20));
        assert_eq!(interp.forward(1), Ok(20));
        assert_eq!(interp.forward(2), Ok(21));
        assert_eq!(interp.forward(3), Ok(22));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_forward_signed() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![-20, -25];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.forward(0), Ok(-20));
        assert_eq!(interp.forward(1), Ok(-20));
        assert_eq!(interp.forward(2), Ok(-21));
        assert_eq!(interp.forward(3), Ok(-22));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_forward_float() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<f64> = vec![20.0, 25.0];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.forward(0), Ok(20.0));
        assert_eq!(interp.forward(1), Ok(20.5));
        assert_eq!(interp.forward(2), Ok(21.0));
        assert_eq!(interp.forward(3), Ok(21.5));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_exact_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Points::new(&xp, &fp);
        assert_eq!(
            interp.inverse(19, Method::None),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(20, Method::None), Ok(0));
        assert_eq!(interp.inverse(21, Method::None), Err(InterpError::NotFound));
        assert_eq!(interp.inverse(22, Method::None), Ok(1));
        assert_eq!(interp.inverse(23, Method::None), Err(InterpError::NotFound));
        assert_eq!(interp.inverse(24, Method::None), Ok(2));
        assert_eq!(interp.inverse(25, Method::None), Err(InterpError::NotFound));
        assert_eq!(interp.inverse(26, Method::None), Ok(3));
        assert_eq!(interp.inverse(30, Method::None), Ok(5));
        assert_eq!(
            interp.inverse(31, Method::None),
            Err(InterpError::OutOfBounds)
        );
    }

    #[test]
    fn test_inverse_round_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(19, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(20, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(21, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(22, Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(23, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(24, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(25, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(26, Method::Nearest), Ok(3));
        assert_eq!(interp.inverse(30, Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(31, Method::Nearest), Ok(5));
    }

    #[test]
    fn test_inverse_ffill_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Points::new(&xp, &fp);
        assert_eq!(
            interp.inverse(19, Method::ForwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(20, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(21, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(22, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(23, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(24, Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(25, Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(26, Method::ForwardFill), Ok(3));
        assert_eq!(interp.inverse(30, Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(31, Method::ForwardFill), Ok(5));
    }

    #[test]
    fn test_inverse_bfill_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(19, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(20, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(21, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(22, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(23, Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(24, Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(25, Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(26, Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(30, Method::BackwardFill), Ok(5));
        assert_eq!(
            interp.inverse(31, Method::BackwardFill),
            Err(InterpError::OutOfBounds)
        );
    }

    #[test]
    fn test_inverse_exact_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Points::new(&xp, &fp);
        assert_eq!(
            interp.inverse(-31, Method::None),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(-30, Method::None), Ok(0));
        assert_eq!(
            interp.inverse(-29, Method::None),
            Err(InterpError::NotFound)
        );
        assert_eq!(interp.inverse(-28, Method::None), Ok(1));
        assert_eq!(
            interp.inverse(-27, Method::None),
            Err(InterpError::NotFound)
        );
        assert_eq!(interp.inverse(-26, Method::None), Ok(2));
        assert_eq!(
            interp.inverse(-25, Method::None),
            Err(InterpError::NotFound)
        );
        assert_eq!(interp.inverse(-24, Method::None), Ok(3));
        assert_eq!(interp.inverse(-20, Method::None), Ok(5));
        assert_eq!(
            interp.inverse(-19, Method::None),
            Err(InterpError::OutOfBounds)
        );
    }

    #[test]
    fn test_inverse_round_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(-31, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(-30, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(-29, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(-28, Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(-27, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(-26, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(-25, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(-24, Method::Nearest), Ok(3));
        assert_eq!(interp.inverse(-20, Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(-19, Method::Nearest), Ok(5));
    }

    #[test]
    fn test_inverse_ffill_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Points::new(&xp, &fp);
        assert_eq!(
            interp.inverse(-31, Method::ForwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(-30, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(-29, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(-28, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(-27, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(-26, Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(-25, Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(-24, Method::ForwardFill), Ok(3));
        assert_eq!(interp.inverse(-20, Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(-19, Method::ForwardFill), Ok(5));
    }

    #[test]
    fn test_inverse_bfill_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(-31, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(-30, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(-29, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(-28, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(-27, Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(-26, Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(-25, Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(-24, Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(-20, Method::BackwardFill), Ok(5));
        assert_eq!(
            interp.inverse(-19, Method::BackwardFill),
            Err(InterpError::OutOfBounds)
        );
    }

    #[test]
    fn test_inverse_round_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<f64> = vec![20.0, 30.0];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(19.9, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(20.0, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(20.1, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(20.9, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(21.1, Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(22.0, Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(29.9, Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(30.0, Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(30.1, Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(21.0, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(23.0, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(25.0, Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(27.0, Method::Nearest), Ok(4));
        assert_eq!(interp.inverse(29.0, Method::Nearest), Ok(4));
    }

    #[test]
    fn test_inverse_ffill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<f64> = vec![20.0, 30.0];
        let interp = Points::new(&xp, &fp);
        assert_eq!(
            interp.inverse(19.9, Method::ForwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(20.0, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(20.1, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(20.9, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(21.1, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(22.0, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(29.9, Method::ForwardFill), Ok(4));
        assert_eq!(interp.inverse(30.0, Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(30.1, Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(21.0, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(23.0, Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(25.0, Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(27.0, Method::ForwardFill), Ok(3));
        assert_eq!(interp.inverse(29.0, Method::ForwardFill), Ok(4));
    }

    #[test]
    fn test_inverse_bfill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<f64> = vec![20.0, 30.0];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(19.9, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(20.0, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(20.1, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(20.9, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(21.1, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(22.0, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(29.9, Method::BackwardFill), Ok(5));
        assert_eq!(interp.inverse(30.0, Method::BackwardFill), Ok(5));
        assert_eq!(
            interp.inverse(30.1, Method::BackwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(21.0, Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(23.0, Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(25.0, Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(27.0, Method::BackwardFill), Ok(4));
        assert_eq!(interp.inverse(29.0, Method::BackwardFill), Ok(5));
    }

    #[test]
    fn test_forward_big_numbers() {
        let xp = vec![0, u64::MAX];
        let fp = vec![i64::MIN, i64::MAX];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.forward(0), Ok(i64::MIN));
        assert_eq!(interp.forward(u64::MAX), Ok(i64::MAX));
        assert_eq!(interp.forward(u64::MAX / 2 + 1), Ok(0));
    }

    #[test]
    fn test_inverse_exact_big_numbers() {
        let xp = vec![0, u64::MAX];
        let fp = vec![i64::MIN, i64::MAX];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(i64::MIN, Method::None), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::None), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::None), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_round_big_numbers() {
        let xp = vec![0, u64::MAX];
        let fp = vec![i64::MIN, i64::MAX];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(i64::MIN, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::Nearest), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::Nearest), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_ffill_big_numbers() {
        let xp = vec![0, u64::MAX];
        let fp = vec![i64::MIN, i64::MAX];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(i64::MIN, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::ForwardFill), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::ForwardFill), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_bfill_big_numbers() {
        let xp = vec![0, u64::MAX];
        let fp = vec![i64::MIN, i64::MAX];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(i64::MIN, Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::BackwardFill), Ok(u64::MAX));
        assert_eq!(
            interp.inverse(0, Method::BackwardFill),
            Ok(u64::MAX / 2 + 1)
        );
    }

    #[test]
    fn test_use_case() {
        let xp: Vec<u64> = vec![0, 8];
        let fp: Vec<f64> = vec![100.0, 900.0];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.inverse(175.0, Method::Nearest), Ok(1))
    }

    #[test]
    fn test_inverse_recovers_a_value_only_forward_rounding_produces() {
        // slope 3/2: forward(1) rounds 1.5 up to 2 (ties to even), so 2's exact preimage is
        // 4/3, not an integer -- the exact solve alone would reject it, even though `forward`
        // itself produced exactly this value at this index.
        let xp: Vec<u64> = vec![0, 2];
        let fp: Vec<i64> = vec![0, 3];
        let interp = Points::new(&xp, &fp);
        assert_eq!(interp.forward(1), Ok(2));
        assert_eq!(interp.inverse(2, Method::None), Ok(1));
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
        fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo + 1)
        }
    }

    /// `forward` then `inverse(.., Method::None)` must round-trip at every index, across
    /// fractional slopes -- not only at the tie points themselves. Slopes are drawn strictly
    /// above one tick per index step so `fp` stays injective (see the sub-tick case below,
    /// which cannot round-trip to the original index by construction).
    #[test]
    fn test_forward_inverse_round_trip_across_fractional_slopes() {
        let mut rng = SplitMix64(11);
        for _ in 0..300 {
            let n = rng.range_u64(2, 8) as usize;
            let mut xp = vec![0u64];
            for _ in 1..n {
                xp.push(xp.last().unwrap() + rng.range_u64(1, 20));
            }
            let mut fp = vec![0i64];
            for i in 1..n {
                let dx = xp[i] - xp[i - 1];
                // numerator/denominator both drawn small so |num/den| lands above 1
                let den = rng.range_u64(1, 6);
                let num = den + rng.range_u64(1, 8);
                let df = (dx * num).div_ceil(den) as i64; // keeps fp strictly increasing
                fp.push(fp[i - 1] + df.max(1));
            }
            let interp = Points::new(&xp, &fp);
            for x in xp[0]..=xp[n - 1] {
                let f = interp.forward(x).unwrap();
                assert_eq!(
                    interp.inverse(f, Method::None),
                    Ok(x),
                    "xp={xp:?} fp={fp:?} x={x} f={f}"
                );
            }
        }
    }

    /// The sub-tick counterpart of the property above: when the slope falls below one tick per
    /// index step, `forward` is not injective -- several indices share a value -- so `inverse`
    /// cannot recover *the* original index. It must still return *a* valid preimage: one that
    /// `forward` maps back onto the same value.
    #[test]
    fn test_inverse_returns_a_valid_preimage_at_sub_tick_slopes() {
        let mut rng = SplitMix64(23);
        for _ in 0..300 {
            let n = rng.range_u64(2, 8) as usize;
            let mut xp = vec![0u64];
            for _ in 1..n {
                xp.push(xp.last().unwrap() + rng.range_u64(2, 20));
            }
            let mut fp = vec![0i64];
            for i in 1..n {
                let dx = xp[i] - xp[i - 1];
                // strictly below one tick per index step, but still strictly increasing
                let df = (rng.range_u64(1, dx.max(2) - 1)).max(1) as i64;
                fp.push(fp[i - 1] + df);
            }
            let interp = Points::new(&xp, &fp);
            for x in xp[0]..=xp[n - 1] {
                let f = interp.forward(x).unwrap();
                let candidate = interp.inverse(f, Method::None).unwrap();
                assert_eq!(
                    interp.forward(candidate),
                    Ok(f),
                    "xp={xp:?} fp={fp:?} x={x} f={f} candidate={candidate}"
                );
            }
        }
    }
}
