//! Forward and inverse integer interpolation on piecewise linear functions.
//!
//! This module provides functionality for performing forward and inverse interpolation
//! on piecewise linear functions. Forward interpolation estimates the value of the function
//! at a given index within the range of known data points, while inverse interpolation
//! estimates the index corresponding to a given value of the function.
//!
//! # Examples
//!
//! ```
//! use xinterp::{Interp, InterpError};
//! use xinterp::divop::Method;
//!
//! let xp = vec![0, 2, 4];
//! let fp = vec![0.0, 4.0, 16.0];
//!
//! let interp = Interp::new(xp, fp);
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

use std::collections::VecDeque;

use crate::divop::Method;
use crate::schemes::{Distance, Forward, Inverse, Zero};

// Interpolation Errors
#[derive(PartialEq, Debug)]
pub enum InterpError {
    OutOfBounds,
    NotFound,
    NotStrictlyIncreasing,
}

/// Structure for performing forward and inverse interpolation on piecewise linear functions.
pub struct Interp<X, F> {
    pub xp: Vec<X>,
    pub fp: Vec<F>,
    forwardable: bool,
    inversable: bool,
}

impl<X, F> Interp<X, F>
where
    X: Forward<F>,
    F: Inverse<X>,
{
    /// Constructs a new Interp instance with the given data points.
    ///
    /// # Arguments
    ///
    /// * `xp` - Vector of indices.
    /// * `fp` - Vector of corresponding values.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `xp` and `fp` are not equal.
    pub fn new(xp: Vec<X>, fp: Vec<F>) -> Interp<X, F> {
        assert!(xp.len() == fp.len(), "xp and fp must have same length");
        let forwardable = xp.windows(2).all(|pair| pair[0] < pair[1]);
        let inversable = fp.windows(2).all(|pair| pair[0] < pair[1]);
        Interp {
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

    pub fn simplify(&self, epsilon: F) -> Interp<X, F>
    where
        F: Zero + Distance,
    {
        let n = self.xp.len();
        if n <= 2 {
            return Interp::new(self.xp.clone(), self.fp.clone());
        }

        let mut keep = vec![false; n];
        keep[0] = true;
        keep[n - 1] = true;

        let mut stack = VecDeque::new();
        stack.push_back((0, n - 1));

        while let Some((start, end)) = stack.pop_back() {
            let interp = Interp {
                xp: vec![self.xp[start], self.xp[end]],
                fp: vec![self.fp[start], self.fp[end]],
                forwardable: true,
                inversable: true,
            };

            let mut max_dist = F::zero();
            let mut index = 0;

            for i in start + 1..end {
                let dist = interp.forward(self.xp[i]).unwrap().distance(self.fp[i]);
                if dist > max_dist {
                    max_dist = dist;
                    index = i;
                }
            }

            if max_dist > epsilon {
                keep[index] = true;
                stack.push_back((start, index));
                stack.push_back((index, end));
            }
        }

        let mut xp = Vec::new();
        let mut fp = Vec::new();
        for (i, value) in keep.iter().enumerate().take(n) {
            if *value {
                xp.push(self.xp[i]);
                fp.push(self.fp[i]);
            }
        }
        Interp::new(xp, fp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![20, 25];
        let interp = Interp::new(xp, fp);
        assert!(interp.forwardable);
        assert!(interp.inversable);

        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![-20, -25];
        let interp = Interp::new(xp, fp);
        assert!(interp.forwardable);
        assert!(!interp.inversable);
    }

    #[test]
    fn test_forward_unsigned() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<u64> = vec![20, 25];
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(xp, fp);
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
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.forward(0), Ok(i64::MIN));
        assert_eq!(interp.forward(u64::MAX), Ok(i64::MAX));
        assert_eq!(interp.forward(u64::MAX / 2 + 1), Ok(0));
    }

    #[test]
    fn test_inverse_exact_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse(i64::MIN, Method::None), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::None), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::None), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_round_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse(i64::MIN, Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::Nearest), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::Nearest), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_ffill_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse(i64::MIN, Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(i64::MAX, Method::ForwardFill), Ok(u64::MAX));
        assert_eq!(interp.inverse(0, Method::ForwardFill), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_bfill_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
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
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse(175.0, Method::Nearest), Ok(1))
    }

    #[test]
    fn test_simplify_noop() {
        // Already minimal: should not remove any points
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![20, 25];
        let interp = Interp::new(xp.clone(), fp.clone());
        let simplified = interp.simplify(0);
        assert_eq!(simplified.xp, xp);
        assert_eq!(simplified.fp, fp);
    }

    #[test]
    fn test_simplify_linear() {
        // All points are on a line, so only endpoints should remain
        let xp: Vec<u64> = vec![0, 5, 10];
        let fp: Vec<i64> = vec![20, 22, 24];
        let interp = Interp::new(xp, fp);
        let simplified = interp.simplify(0);
        assert_eq!(simplified.xp, vec![0, 10]);
        assert_eq!(simplified.fp, vec![20, 24]);
    }

    #[test]
    fn test_simplify_with_deviation() {
        // Middle point deviates enough to be kept
        let xp: Vec<u64> = vec![0, 5, 10];
        let fp: Vec<i64> = vec![20, 36, 40];
        let interp = Interp::new(xp, fp);
        let simplified = interp.simplify(4);
        // The deviation at x=5 is 6, so with epsilon=4, it should be kept
        assert_eq!(simplified.xp, vec![0, 5, 10]);
        assert_eq!(simplified.fp, vec![20, 36, 40]);
        // With a larger epsilon, the middle point can be dropped
        let simplified2 = interp.simplify(6);
        assert_eq!(simplified2.xp, vec![0, 10]);
        assert_eq!(simplified2.fp, vec![20, 40]);
    }

    #[test]
    fn test_simplify_multiple_points() {
        // Several points, some can be dropped, some must be kept
        let xp: Vec<u64> = vec![0, 2, 4, 6, 8];
        let fp: Vec<i64> = vec![0, 1, 10, 16, 20];
        let interp = Interp::new(xp, fp);
        // With epsilon=2, only the peak at x=4 should be kept
        let simplified = interp.simplify(2);
        assert_eq!(simplified.xp, vec![0, 2, 4, 8]);
        assert_eq!(simplified.fp, vec![0, 1, 10, 20]);
        // With epsilon=10, only endpoints remain
        let simplified2 = interp.simplify(10);
        assert_eq!(simplified2.xp, vec![0, 8]);
        assert_eq!(simplified2.fp, vec![0, 20]);
    }

    #[test]
    fn test_simplify_float() {
        let xp: Vec<u64> = vec![0, 1, 2, 3];
        let fp: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3];
        let interp = Interp::new(xp.clone(), fp.clone());
        // All points are on a line, so only endpoints should remain
        let simplified = interp.simplify(1e-6);
        assert_eq!(simplified.xp, vec![0, 3]);
        assert_eq!(simplified.fp, vec![0.0, 0.3]);
    }

    #[test]
    fn test_simplify_single_point() {
        let xp: Vec<u64> = vec![0];
        let fp: Vec<i64> = vec![42];
        let interp = Interp::new(xp.clone(), fp.clone());
        let simplified = interp.simplify(0);
        assert_eq!(simplified.xp, xp);
        assert_eq!(simplified.fp, fp);
    }

    #[test]
    fn test_simplify_two_points() {
        let xp: Vec<u64> = vec![0, 1];
        let fp: Vec<i64> = vec![10, 20];
        let interp = Interp::new(xp.clone(), fp.clone());
        let simplified = interp.simplify(0);
        assert_eq!(simplified.xp, xp);
        assert_eq!(simplified.fp, fp);
    }
}
