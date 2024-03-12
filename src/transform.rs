use crate::divop::DivOp;

pub trait Inverse<X>: Copy + Ord {
    fn inverse_exact(self, x0: X, x1: X, f0: Self, f1: Self) -> Option<X>;
    fn inverse_round(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_ffill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_bfill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
}

impl Inverse<u64> for i64 {
    fn inverse_exact(self, x0: u64, x1: u64, f0: i64, f1: i64) -> Option<u64> {
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_exact(den).map(|x| x as u64)
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_round(den) as u64
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_ffill(den) as u64
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_bfill(den) as u64
    }
}

pub trait Forward<F>: Copy + Ord {
    fn forward(self, x0: Self, x1: Self, f0: F, f1: F) -> F;
}

impl Forward<i64> for u64 {
    fn forward(self, x0: u64, x1: u64, f0: i64, f1: i64) -> i64 {
        assert!(x0 < x1, "x1 must greater than x0");
        assert!(x0 <= self && self <= x1, "x must be in [x0, x1]");
        let num = (f0 as i128) * ((x1 - self) as i128) + (f1 as i128) * ((self - x0) as i128);
        let den = (x1 - x0) as i128;
        num.div_round(den) as i64
    }
}
