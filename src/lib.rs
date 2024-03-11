 trait InterpValue {
    type Output;
    fn interp(self, xp: &[Self], fp: &[Self::Output]) -> Option<Self::Output>
    where
        Self: Sized;
}

impl InterpValue for u64 {
    type Output = i64;
    fn interp(self, xp: &[u64], fp: &[i64]) -> Option<i64> {
        match xp.binary_search(&self) {
            Ok(index) => Some(fp[index]),
            Err(0) => None,
            Err(len) if len == xp.len() => None,
            Err(index) => Some(self.forward(xp[index - 1], xp[index], fp[index - 1], fp[index])),
        }
    }
}

trait Inverse {
    type Output;
    fn inverse_exact(
        self,
        x0: Self::Output,
        x1: Self::Output,
        f0: Self,
        f1: Self,
    ) -> Option<Self::Output>;
    fn inverse_round(self, x0: Self::Output, x1: Self::Output, f0: Self, f1: Self) -> Self::Output;
    fn inverse_ffill(self, x0: Self::Output, x1: Self::Output, f0: Self, f1: Self) -> Self::Output;
    fn inverse_bfill(self, x0: Self::Output, x1: Self::Output, f0: Self, f1: Self) -> Self::Output;
}

impl Inverse for i64 {
    type Output = u64;
    fn inverse_exact(self, x0: u64, x1: u64, f0: i64, f1: i64) -> Option<u64> {
        assert!(f0 < f1, "f1 must greater than f0");
        assert!(f0 <= self && self <= f1, "f must be in [f0, f1]");
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_exact(den).map(|x| x as u64)
    }
    fn inverse_round(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        assert!(f0 < f1, "f1 must greater than f0");
        assert!(f0 <= self && self <= f1, "f must be in [f0, f1]");
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_round(den) as u64
    }
    fn inverse_ffill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        assert!(f0 < f1, "f1 must greater than f0");
        assert!(f0 <= self && self <= f1, "f must be in [f0, f1]");
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_ffill(den) as u64
    }
    fn inverse_bfill(self, x0: u64, x1: u64, f0: i64, f1: i64) -> u64 {
        assert!(f0 < f1, "f1 must greater than f0");
        assert!(f0 <= self && self <= f1, "f must be in [f0, f1]");
        f1.checked_sub(f0).expect("f range is to big");
        let num = (x0 as i128) * ((f1 - self) as i128) + (x1 as i128) * ((self - f0) as i128);
        let den = (f1 - f0) as i128;
        num.div_bfill(den) as u64
    }
}

trait Forward {
    type Output;
    fn forward(self, x0: Self, x1: Self, f0: Self::Output, f1: Self::Output) -> Self::Output;
}

impl Forward for u64 {
    type Output = i64;
    fn forward(self, x0: u64, x1: u64, f0: i64, f1: i64) -> i64 {
        assert!(x0 < x1, "x1 must greater than x0");
        assert!(x0 <= self && self <= x1, "x must be in [x0, x1]");
        let num = (f0 as i128) * ((x1 - self) as i128) + (f1 as i128) * ((self - x0) as i128);
        let den = (x1 - x0) as i128;
        num.div_round(den) as i64
    }
}

trait DivOp {
    type Output;
    fn div_exact(self, rhs: Self) -> Option<Self::Output>;
    fn div_round(self, rhs: Self) -> Self::Output;
    fn div_ffill(self, rhs: Self) -> Self::Output;
    fn div_bfill(self, rhs: Self) -> Self::Output;
}

impl DivOp for u128 {
    type Output = u128;
    fn div_exact(self, rhs: u128) -> Option<u128> {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        if rem == 0 {
            Some(div)
        } else {
            None
        }
    }
    fn div_round(self, rhs: u128) -> u128 {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        if rem * 2 < rhs {
            div
        } else if rem * 2 > rhs {
            div + 1
        } else {
            if div % 2 == 0 {
                div
            } else {
                div + 1
            }
        }
    }
    fn div_ffill(self, rhs: u128) -> u128 {
        self.div_euclid(rhs)
    }
    fn div_bfill(self, rhs: u128) -> u128 {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        if rem == 0 {
            div
        } else {
            div + 1
        }
    }
}

impl DivOp for i128 {
    type Output = i128;
    fn div_exact(self, rhs: i128) -> Option<i128> {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        if rem == 0 {
            Some(div)
        } else {
            None
        }
    }
    fn div_round(self, rhs: i128) -> i128 {
        let sgn = self.signum() * rhs.signum();
        sgn * ((self.abs() as u128).div_round(rhs.abs() as u128) as i128)
    }
    fn div_ffill(self, rhs: i128) -> i128 {
        self.div_euclid(rhs)
    }
    fn div_bfill(self, rhs: i128) -> i128 {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        if rem == 0 {
            div
        } else {
            div + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_exact() {
        assert_eq!(10.inverse_exact(0, 5, 10, 20), Some(0));
        assert_eq!(11.inverse_exact(0, 5, 10, 20), None);
        assert_eq!(12.inverse_exact(0, 5, 10, 20), Some(1));
        assert_eq!(13.inverse_exact(0, 5, 10, 20), None);
        assert_eq!(20.inverse_exact(0, 5, 10, 20), Some(5));

        assert_eq!((-20).inverse_exact(0, 5, -20, -10), Some(0));
        assert_eq!((-19).inverse_exact(0, 5, -20, -10), None);
        assert_eq!((-18).inverse_exact(0, 5, -20, -10), Some(1));
        assert_eq!((-17).inverse_exact(0, 5, -20, -10), None);
        assert_eq!((-10).inverse_exact(0, 5, -20, -10), Some(5));

        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        assert_eq!(0.inverse_exact(0, x1, 0, f1), Some(0));
        assert_eq!(f1.inverse_exact(0, x1, 0, f1), Some(x1));
        assert_eq!((f1 / 2).inverse_exact(0, x1, 0, f1), Some(x1 / 2));
    }

    #[test]
    fn test_inverse_round() {
        assert_eq!(10.inverse_round(0, 5, 10, 20), 0);
        assert_eq!(11.inverse_round(0, 5, 10, 20), 0);
        assert_eq!(12.inverse_round(0, 5, 10, 20), 1);
        assert_eq!(13.inverse_round(0, 5, 10, 20), 2);
        assert_eq!(20.inverse_round(0, 5, 10, 20), 5);

        assert_eq!((-20).inverse_round(0, 5, -20, -10), 0);
        assert_eq!((-19).inverse_round(0, 5, -20, -10), 0);
        assert_eq!((-18).inverse_round(0, 5, -20, -10), 1);
        assert_eq!((-17).inverse_round(0, 5, -20, -10), 2);
        assert_eq!((-10).inverse_round(0, 5, -20, -10), 5);

        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        assert_eq!(0.inverse_round(0, x1, 0, f1), 0);
        assert_eq!(f1.inverse_round(0, x1, 0, f1), x1);
        assert_eq!((f1 / 2).inverse_round(0, x1, 0, f1), x1 / 2);
    }

    #[test]
    fn test_inverse_ffill() {
        assert_eq!(10.inverse_ffill(0, 5, 10, 20), 0);
        assert_eq!(11.inverse_ffill(0, 5, 10, 20), 0);
        assert_eq!(12.inverse_ffill(0, 5, 10, 20), 1);
        assert_eq!(13.inverse_ffill(0, 5, 10, 20), 1);
        assert_eq!(20.inverse_ffill(0, 5, 10, 20), 5);

        assert_eq!((-20).inverse_ffill(0, 5, -20, -10), 0);
        assert_eq!((-19).inverse_ffill(0, 5, -20, -10), 0);
        assert_eq!((-18).inverse_ffill(0, 5, -20, -10), 1);
        assert_eq!((-17).inverse_ffill(0, 5, -20, -10), 1);
        assert_eq!((-10).inverse_ffill(0, 5, -20, -10), 5);

        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        assert_eq!(0.inverse_ffill(0, x1, 0, f1), 0);
        assert_eq!(f1.inverse_ffill(0, x1, 0, f1), x1);
        assert_eq!((f1 / 2).inverse_ffill(0, x1, 0, f1), x1 / 2);
    }

    #[test]
    fn test_inverse_bfill() {
        assert_eq!(10.inverse_bfill(0, 5, 10, 20), 0);
        assert_eq!(11.inverse_bfill(0, 5, 10, 20), 1);
        assert_eq!(12.inverse_bfill(0, 5, 10, 20), 1);
        assert_eq!(13.inverse_bfill(0, 5, 10, 20), 2);
        assert_eq!(20.inverse_bfill(0, 5, 10, 20), 5);

        assert_eq!((-20).inverse_bfill(0, 5, -20, -10), 0);
        assert_eq!((-19).inverse_bfill(0, 5, -20, -10), 1);
        assert_eq!((-18).inverse_bfill(0, 5, -20, -10), 1);
        assert_eq!((-17).inverse_bfill(0, 5, -20, -10), 2);
        assert_eq!((-10).inverse_bfill(0, 5, -20, -10), 5);

        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        assert_eq!(0.inverse_bfill(0, x1, 0, f1), 0);
        assert_eq!(f1.inverse_bfill(0, x1, 0, f1), x1);
        assert_eq!((f1 / 2).inverse_bfill(0, x1, 0, f1), x1 / 2);
    }

    #[test]
    fn test_forward() {
        assert_eq!(0.forward(0, 10, 20, 25), 20);
        assert_eq!(1.forward(0, 10, 20, 25), 20);
        assert_eq!(2.forward(0, 10, 20, 25), 21);
        assert_eq!(3.forward(0, 10, 20, 25), 22);
        assert_eq!(10.forward(0, 10, 20, 25), 25);
        assert_eq!(0.forward(0, 10, -20, -25), -20);
        assert_eq!(1.forward(0, 10, -20, -25), -20);
        assert_eq!(2.forward(0, 10, -20, -25), -21);
        assert_eq!(3.forward(0, 10, -20, -25), -22);
        assert_eq!(10.forward(0, 10, -20, -25), -25);
        assert_eq!(0.forward(0, u64::MAX, i64::MIN, i64::MAX), i64::MIN);
        assert_eq!(u64::MAX.forward(0, u64::MAX, i64::MIN, i64::MAX), i64::MAX);
        assert_eq!((u64::MAX / 2).forward(0, u64::MAX, i64::MIN, i64::MAX), -1);
    }

    #[test]
    #[should_panic(expected = "x1 must greater than x0")]
    fn test_forward_x1_less_than_x0() {
        let _ = 0.forward(10, 0, 20, 25);
    }

    #[test]
    #[should_panic(expected = "x must be in [x0, x1]")]
    fn test_forward_x_out_of_range() {
        let _ = 11.forward(0, 10, 20, 25);
    }

    #[test]
    fn test_div_exact() {
        assert_eq!(0u128.div_exact(2), Some(0));
        assert_eq!(1u128.div_exact(2), None);
        assert_eq!(2u128.div_exact(2), Some(1));
        assert_eq!(3u128.div_exact(2), None);
        assert_eq!(1u128.div_exact(3), None);
        assert_eq!(2u128.div_exact(3), None);
        assert_eq!((-1i128).div_exact(2), None);
        assert_eq!((-2i128).div_exact(2), Some(-1));
        assert_eq!((-3i128).div_exact(2), None);
        assert_eq!((-1i128).div_exact(3), None);
        assert_eq!((-2i128).div_exact(3), None);
    }

    #[test]
    fn test_div_round() {
        assert_eq!(0u128.div_round(2), 0);
        assert_eq!(1u128.div_round(2), 0);
        assert_eq!(2u128.div_round(2), 1);
        assert_eq!(3u128.div_round(2), 2);
        assert_eq!(1u128.div_round(3), 0);
        assert_eq!(2u128.div_round(3), 1);
        assert_eq!((-1i128).div_round(2), 0);
        assert_eq!((-2i128).div_round(2), -1);
        assert_eq!((-3i128).div_round(2), -2);
        assert_eq!((-1i128).div_round(3), 0);
        assert_eq!((-2i128).div_round(3), -1);
    }

    #[test]
    fn test_div_ffill() {
        assert_eq!(0u128.div_ffill(2), 0);
        assert_eq!(1u128.div_ffill(2), 0);
        assert_eq!(2u128.div_ffill(2), 1);
        assert_eq!(3u128.div_ffill(2), 1);
        assert_eq!(1u128.div_ffill(3), 0);
        assert_eq!(2u128.div_ffill(3), 0);
        assert_eq!((-1i128).div_ffill(2), -1);
        assert_eq!((-2i128).div_ffill(2), -1);
        assert_eq!((-3i128).div_ffill(2), -2);
        assert_eq!((-1i128).div_ffill(3), -1);
        assert_eq!((-2i128).div_ffill(3), -1);
    }

    #[test]
    fn test_div_bfill() {
        assert_eq!(0u128.div_bfill(2), 0);
        assert_eq!(1u128.div_bfill(2), 1);
        assert_eq!(2u128.div_bfill(2), 1);
        assert_eq!(3u128.div_bfill(2), 2);
        assert_eq!(1u128.div_bfill(3), 1);
        assert_eq!(2u128.div_bfill(3), 1);
        assert_eq!((-1i128).div_bfill(2), 0);
        assert_eq!((-2i128).div_bfill(2), -1);
        assert_eq!((-3i128).div_bfill(2), -1);
        assert_eq!((-1i128).div_bfill(3), 0);
        assert_eq!((-2i128).div_bfill(3), 0);
    }
}
