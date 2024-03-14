use crate::divop::DivOp;
use std::cmp::Ordering;

pub trait Forward<F>: Copy + Ord {
    fn forward(self, x0: Self, x1: Self, f0: F, f1: F) -> F;
}

impl Forward<u64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: u64, f1: u64) -> u64 {
        let num = (f0 as u128) * ((x1 - self) as u128) + (f1 as u128) * ((self - x0) as u128);
        let den = (x1 - x0) as u128;
        num.div_round(den) as u64
    }
}

impl Forward<i64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: i64, f1: i64) -> i64 {
        self.forward(x0, x1, f0.to_unsigned(), f1.to_unsigned())
            .to_signed()
    }
}

impl Forward<F64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: F64, f1: F64) -> F64 {
        let t = ((self - x0) as f64) / ((x1 - x0) as f64);
        let value = (1.0 - t) * f0.value + t * f1.value;
        F64 { value }
    }
}

pub trait Inverse<X>: Copy + Ord {
    fn inverse_exact(self, x0: X, x1: X, f0: Self, f1: Self) -> Option<X>;
    fn inverse_round(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_ffill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_bfill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
}

impl Inverse<u64> for u64 {
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

impl Inverse<u64> for F64 {
    fn inverse_exact(self, x0: u64, x1: u64, f0: F64, f1: F64) -> Option<u64> {
        let t = (self.value - f0.value) / (f1.value - f0.value);
        let den = 2u64.pow(53);
        let num = (t * (den as f64)) as u64;
        let num = (x0 as u128) * (num as u128) + (x1 as u128) * ((den - num) as u128);
        let den = den as u128;
        num.div_exact(den).map(|x| x as u64)
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: F64, f1: F64) -> u64 {
        let t = (self.value - f0.value) / (f1.value - f0.value);
        let den = 2u64.pow(53);
        let num = (t * (den as f64)) as u64;
        let num = (x0 as u128) * (num as u128) + (x1 as u128) * ((den - num) as u128);
        let den = den as u128;
        num.div_round(den) as u64
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: F64, f1: F64) -> u64 {
        let t = (self.value - f0.value) / (f1.value - f0.value);
        let den = 2u64.pow(53);
        let num = (t * (den as f64)) as u64;
        let num = (x0 as u128) * (num as u128) + (x1 as u128) * ((den - num) as u128);
        let den = den as u128;
        num.div_ffill(den) as u64
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: F64, f1: F64) -> u64 {
        let t = (self.value - f0.value) / (f1.value - f0.value);
        let den = 2u64.pow(53);
        let num = (t * (den as f64)) as u64;
        let num = (x0 as u128) * (num as u128) + (x1 as u128) * ((den - num) as u128);
        let den = den as u128;
        num.div_bfill(den) as u64
    }
}

pub trait ToUnsigned<U> {
    fn to_unsigned(self) -> U;
}

impl ToUnsigned<u64> for i64 {
    fn to_unsigned(self) -> u64 {
        self.wrapping_sub(i64::MIN) as u64
    }
}

pub trait ToSigned<S> {
    fn to_signed(self) -> S;
}

impl ToSigned<i64> for u64 {
    fn to_signed(self) -> i64 {
        self.wrapping_add(i64::MIN as u64) as i64
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct F64 {
    value: f64,
}

impl F64 {
    fn new(value: f64) -> Option<F64> {
        if value.is_nan() {
            None
        } else {
            Some(F64 { value })
        }
    }
}

impl Eq for F64 {}

impl Ord for F64 {
    fn cmp(&self, other: &F64) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
