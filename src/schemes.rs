use crate::divop::DivOp;
use crate::extended::F80;

pub trait Forward<F>: Clone + Ord {
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

pub trait Inverse<X>: Clone + Ord {
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
impl Inverse<u64> for F80 {
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
