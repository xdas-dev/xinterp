//! Forward and inverse interpolation on piecewise linear functions.

use crate::divop::Method;
use crate::schemes::{Forward, Inverse};

#[derive(PartialEq, Debug)]
pub enum InterpError {
    OutOfBounds,
    NotFound,
    NotStrictlyIncreasing,
}

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
    pub fn forward(&self, rhs: X) -> Result<F, InterpError> {
        if self.forwardable {
            match self.xp.binary_search(&rhs) {
                Ok(index) => Ok(self.fp[index].clone()),
                Err(0) => Err(InterpError::OutOfBounds),
                Err(len) if len == self.xp.len() => Err(InterpError::OutOfBounds),
                Err(index) => Ok(rhs.forward(
                    self.xp[index - 1].clone(),
                    self.xp[index].clone(),
                    self.fp[index - 1].clone(),
                    self.fp[index].clone(),
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    pub fn inverse(&self, rhs: F, method: Method) -> Result<X, InterpError> {
        if self.inversable {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Ok(self.xp[index].clone()),
                Err(0) => match method {
                    Method::None | Method::ForwardFill => Err(InterpError::OutOfBounds),
                    Method::Nearest | Method::BackwardFill => Ok(self.xp[0].clone()),
                },
                Err(len) if len == self.xp.len() => match method {
                    Method::None | Method::BackwardFill => Err(InterpError::OutOfBounds),
                    Method::Nearest | Method::ForwardFill => Ok(self.xp[len - 1].clone()),
                },
                Err(index) => rhs
                    .inverse(
                        self.xp[index - 1].clone(),
                        self.xp[index].clone(),
                        self.fp[index - 1].clone(),
                        self.fp[index].clone(),
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
    use crate::extended::F80;

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
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(25.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.forward(0), Ok(F80::from(20.0)));
        assert_eq!(interp.forward(1), Ok(F80::from(20.5)));
        assert_eq!(interp.forward(2), Ok(F80::from(21.0)));
        assert_eq!(interp.forward(3), Ok(F80::from(21.5)));
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
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse(F80::from(19.9), Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.0), Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.1), Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.9), Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(F80::from(21.1), Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(F80::from(22.0), Method::Nearest), Ok(1));
        assert_eq!(interp.inverse(F80::from(29.9), Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(F80::from(30.0), Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(F80::from(30.1), Method::Nearest), Ok(5));
        assert_eq!(interp.inverse(F80::from(21.0), Method::Nearest), Ok(0));
        assert_eq!(interp.inverse(F80::from(23.0), Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(F80::from(25.0), Method::Nearest), Ok(2));
        assert_eq!(interp.inverse(F80::from(27.0), Method::Nearest), Ok(4));
        assert_eq!(interp.inverse(F80::from(29.0), Method::Nearest), Ok(4));
    }

    #[test]
    fn test_inverse_ffill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(
            interp.inverse(F80::from(19.9), Method::ForwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(F80::from(20.0), Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.1), Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.9), Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(21.1), Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(22.0), Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(29.9), Method::ForwardFill), Ok(4));
        assert_eq!(interp.inverse(F80::from(30.0), Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(F80::from(30.1), Method::ForwardFill), Ok(5));
        assert_eq!(interp.inverse(F80::from(21.0), Method::ForwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(23.0), Method::ForwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(25.0), Method::ForwardFill), Ok(2));
        assert_eq!(interp.inverse(F80::from(27.0), Method::ForwardFill), Ok(3));
        assert_eq!(interp.inverse(F80::from(29.0), Method::ForwardFill), Ok(4));
    }

    #[test]
    fn test_inverse_bfill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse(F80::from(19.9), Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.0), Method::BackwardFill), Ok(0));
        assert_eq!(interp.inverse(F80::from(20.1), Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(20.9), Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(21.1), Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(22.0), Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(29.9), Method::BackwardFill), Ok(5));
        assert_eq!(interp.inverse(F80::from(30.0), Method::BackwardFill), Ok(5));
        assert_eq!(
            interp.inverse(F80::from(30.1), Method::BackwardFill),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse(F80::from(21.0), Method::BackwardFill), Ok(1));
        assert_eq!(interp.inverse(F80::from(23.0), Method::BackwardFill), Ok(2));
        assert_eq!(interp.inverse(F80::from(25.0), Method::BackwardFill), Ok(3));
        assert_eq!(interp.inverse(F80::from(27.0), Method::BackwardFill), Ok(4));
        assert_eq!(interp.inverse(F80::from(29.0), Method::BackwardFill), Ok(5));
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
        let fp: Vec<F80> = vec![F80::from(100.0), F80::from(900.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse(F80::from(175.0), Method::Nearest), Ok(1))
    }
}
