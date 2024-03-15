//! Forward and backward linear interpolation schemes between two points (x0, f0) and (x1, f1) for
//! different data types (x is u64, f is either i64 or f64).
//!
//! When the values are integers, operations are performed with u128 integers to avoid overflow.
//! Signed integers are mapped on positive values to avoid potential subtraction overflows
//! (subtracting i64::MIN to i64::MAX overflows whereas it does not for u64).
//!
//! When the values are floats, operations are performed with extended-precision floats to
//! avoid big numbers inacurracies (integers above 2^53 cannot accurately be represented by f64
//! which is problematic when using nanosecond datetime64 timestamps).

use crate::divop::{DivOp, Method};
use crate::extended::F80;

/// Implements forward scheme from index to value.
pub trait Forward<F>: Clone + Ord {
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
impl Forward<F80> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: F80, f1: F80) -> F80 {
        let x = F80::from(self);
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        f0.mul(&x1.sub(&x))
            .add(&f1.mul(&x.sub(&x0)))
            .div(&x1.sub(&x0))
            .into()
    }
}

/// Implements inverse scheme from value to index.
pub trait Inverse<X>: Clone + Ord {
    fn inverse(self, x0: X, x1: X, f0: Self, f1: Self, method: Method) -> Option<X>;

    /// Retrieve index x that corresonds to value f between two points (x0, f0) and (x1, f1).
    /// Returns x if f falls exactly on it or `None` otherwise.
    fn inverse_exact(self, x0: X, x1: X, f0: Self, f1: Self) -> Option<X>;

    /// Retrieve index x that corresonds to value f between two points (x0, f0) and (x1, f1).
    /// Rounds to the nearest x using round ties to even rule.
    fn inverse_round(self, x0: X, x1: X, f0: Self, f1: Self) -> X;

    /// Retrieve index x that corresonds to value f between two points (x0, f0) and (x1, f1).
    /// Rounds to the previous x using forward-fill rule.
    fn inverse_ffill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;

    /// Retrieve index x that corresonds to value f between two points (x0, f0) and (x1, f1).
    /// Rounds to the next x using backward-fill rule.
    fn inverse_bfill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
}
impl Inverse<u64> for u64 {
    fn inverse(self, x0: u64, x1: u64, f0: u64, f1: u64, method: Method) -> Option<u64> {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div(den, method).map(|x| x as u64)
    }
    fn inverse_exact(self, x0: u64, x1: u64, f0: u64, f1: u64) -> Option<u64> {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div_exact(den).map(|x| x as u64)
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: u64, f1: u64) -> u64 {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div_round(den) as u64
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: u64, f1: u64) -> u64 {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div_ffill(den) as u64
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: u64, f1: u64) -> u64 {
        let num = (x0 as u128) * ((f1 - self) as u128) + (x1 as u128) * ((self - f0) as u128);
        let den = (f1 - f0) as u128;
        num.div_bfill(den) as u64
    }
}
impl Inverse<u64> for i64 {
    fn inverse(self, x0: u64, x1: u64, f0: i64, f1: i64, method: Method) -> Option<u64> {
        self.to_unsigned()
            .inverse(x0, x1, f0.to_unsigned(), f1.to_unsigned(), method)
    }
    fn inverse_exact(self, x0: u64, x1: u64, f0: i64, f1: i64) -> Option<u64> {
        self.to_unsigned()
            .inverse_exact(x0, x1, f0.to_unsigned(), f1.to_unsigned())
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        self.to_unsigned()
            .inverse_round(x0, x1, f0.to_unsigned(), f1.to_unsigned())
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        self.to_unsigned()
            .inverse_ffill(x0, x1, f0.to_unsigned(), f1.to_unsigned())
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        self.to_unsigned()
            .inverse_bfill(x0, x1, f0.to_unsigned(), f1.to_unsigned())
    }
}
impl Inverse<u64> for F80 {
    fn inverse(self, x0: u64, x1: u64, f0: F80, f1: F80, method: Method) -> Option<u64> {
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        let x = x0
            .mul(&f1.sub(&self))
            .add(&x1.mul(&self.sub(&f0)))
            .div(&f1.sub(&f0));
        match method {
            Method::None => {
                let out = x.floor();
                if out == x {
                    Some(out.into())
                } else {
                    None
                }
            }
            Method::Nearest => Some(x.round().into()),
            Method::ForwardFill => Some(x.floor().into()),
            Method::BackwardFill => Some(x.ceil().into()),
        }
    }
    fn inverse_exact(self, x0: u64, x1: u64, f0: F80, f1: F80) -> Option<u64> {
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        let x = x0
            .mul(&f1.sub(&self))
            .add(&x1.mul(&self.sub(&f0)))
            .div(&f1.sub(&f0));
        let out = x.floor();
        if out == x {
            Some(out.into())
        } else {
            None
        }
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: F80, f1: F80) -> u64 {
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        let x = x0
            .mul(&f1.sub(&self))
            .add(&x1.mul(&self.sub(&f0)))
            .div(&f1.sub(&f0));
        x.round().into()
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: F80, f1: F80) -> u64 {
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        let x = x0
            .mul(&f1.sub(&self))
            .add(&x1.mul(&self.sub(&f0)))
            .div(&f1.sub(&f0));
        x.floor().into()
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: F80, f1: F80) -> u64 {
        let x0 = F80::from(x0);
        let x1 = F80::from(x1);
        let x = x0
            .mul(&f1.sub(&self))
            .add(&x1.mul(&self.sub(&f0)))
            .div(&f1.sub(&f0));
        x.ceil().into()
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
