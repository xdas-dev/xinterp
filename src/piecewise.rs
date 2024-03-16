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
//! data points.
//! - `InterpError::NotFound`: Indicates that the output value does not exist within the range of
//! known data points.
//! - `InterpError::NotStrictlyIncreasing`: Indicates that the input or output values are not
//! strictly increasing, which is required for interpolation.

use crate::divop::Method;
use crate::schemes::{Forward, Inverse};

// Interpolation Errors
#[derive(PartialEq, Debug)]
pub enum InterpError {
    OutOfBounds,
    NotFound,
    NotStrictlyIncreasing,
}

/// Structure for performing forward and inverse interpolation on piecewise linear functions.
pub struct Interp<X, F> {
    xp: Vec<X>,
    fp: Vec<F>,
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
}
