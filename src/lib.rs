struct Interp<X, F> {
    xp: Vec<X>,
    fp: Vec<F>,
    forward: bool,
    inverse: bool,
}

impl<X, F> Interp<X, F>
where
    X: Forward<F>,
    F: Inverse<X>,
{
    pub fn new(xp: Vec<X>, fp: Vec<F>) -> Interp<X, F> {
        let forward = xp.windows(2).all(|pair| pair[0] < pair[1]);
        let inverse = fp.windows(2).all(|pair| pair[0] < pair[1]);
        Interp {
            xp,
            fp,
            forward,
            inverse,
        }
    }
    pub fn forward(&self, rhs: X) -> Option<F> {
        if self.forward {
            match self.xp.binary_search(&rhs) {
                Ok(index) => Some(self.fp[index]),
                Err(0) => None,
                Err(len) if len == self.xp.len() => None,
                Err(index) => Some(rhs.forward(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                )),
            }
        } else {
            None
        }
    }
    pub fn inverse_exact(&self, rhs: F) -> Option<X> {
        if self.inverse {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Some(self.xp[index]),
                Err(0) => None,
                Err(len) if len == self.xp.len() => None,
                Err(index) => rhs.inverse_exact(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                ),
            }
        } else {
            None
        }
    }
    pub fn inverse_round(&self, rhs: F) -> Option<X> {
        if self.inverse {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Some(self.xp[index]),
                Err(0) => None,
                Err(len) if len == self.xp.len() => None,
                Err(index) => Some(rhs.inverse_round(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                )),
            }
        } else {
            None
        }
    }
    pub fn inverse_ffill(&self, rhs: F) -> Option<X> {
        if self.inverse {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Some(self.xp[index]),
                Err(0) => None,
                Err(len) if len == self.xp.len() => None,
                Err(index) => Some(rhs.inverse_ffill(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                )),
            }
        } else {
            None
        }
    }
    pub fn inverse_bfill(&self, rhs: F) -> Option<X> {
        if self.inverse {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Some(self.xp[index]),
                Err(0) => None,
                Err(len) if len == self.xp.len() => None,
                Err(index) => Some(rhs.inverse_bfill(
                    self.xp[index - 1],
                    self.xp[index],
                    self.fp[index - 1],
                    self.fp[index],
                )),
            }
        } else {
            None
        }
    }
}

trait Inverse<X>: Copy + Ord {
    fn inverse_exact(self, x0: X, x1: X, f0: Self, f1: Self) -> Option<X>;
    fn inverse_round(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_ffill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
    fn inverse_bfill(self, x0: X, x1: X, f0: Self, f1: Self) -> X;
}

impl Inverse<u64> for i64 {
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

trait Forward<F>: Copy + Ord {
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

trait DivOp: Sized {
    fn div_exact(self, rhs: Self) -> Option<Self>;
    fn div_round(self, rhs: Self) -> Self;
    fn div_ffill(self, rhs: Self) -> Self;
    fn div_bfill(self, rhs: Self) -> Self;
}

impl DivOp for u128 {
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
        sgn * ((self.unsigned_abs()).div_round(rhs.unsigned_abs()) as i128)
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
    fn test_coord() {
        let coord = Interp::new(vec![0, 10], vec![20, 25]);
        assert!(coord.forward);
        assert!(coord.inverse);

        assert_eq!(coord.forward(0), Some(20));
        assert_eq!(coord.forward(1), Some(20));
        assert_eq!(coord.forward(2), Some(21));
        assert_eq!(coord.forward(3), Some(22));
        assert_eq!(coord.forward(11), None);

        let coord = Interp::new(vec![0, 5], vec![20, 30]);
        assert!(coord.forward);
        assert!(coord.inverse);

        assert_eq!(coord.inverse_exact(19), None);
        assert_eq!(coord.inverse_exact(20), Some(0));
        assert_eq!(coord.inverse_exact(21), None);
        assert_eq!(coord.inverse_exact(22), Some(1));
        assert_eq!(coord.inverse_exact(23), None);
        assert_eq!(coord.inverse_exact(24), Some(2));
        assert_eq!(coord.inverse_exact(25), None);
        assert_eq!(coord.inverse_exact(26), Some(3));
        assert_eq!(coord.inverse_exact(30), Some(5));
        assert_eq!(coord.inverse_exact(31), None);

        assert_eq!(coord.inverse_round(19), None);
        assert_eq!(coord.inverse_round(20), Some(0));
        assert_eq!(coord.inverse_round(21), Some(0));
        assert_eq!(coord.inverse_round(22), Some(1));
        assert_eq!(coord.inverse_round(23), Some(2));
        assert_eq!(coord.inverse_round(24), Some(2));
        assert_eq!(coord.inverse_round(25), Some(2));
        assert_eq!(coord.inverse_round(26), Some(3));
        assert_eq!(coord.inverse_round(30), Some(5));
        assert_eq!(coord.inverse_round(31), None);

        assert_eq!(coord.inverse_ffill(19), None);
        assert_eq!(coord.inverse_ffill(20), Some(0));
        assert_eq!(coord.inverse_ffill(21), Some(0));
        assert_eq!(coord.inverse_ffill(22), Some(1));
        assert_eq!(coord.inverse_ffill(23), Some(1));
        assert_eq!(coord.inverse_ffill(24), Some(2));
        assert_eq!(coord.inverse_ffill(25), Some(2));
        assert_eq!(coord.inverse_ffill(26), Some(3));
        assert_eq!(coord.inverse_ffill(30), Some(5));
        assert_eq!(coord.inverse_ffill(31), None);

        assert_eq!(coord.inverse_bfill(19), None);
        assert_eq!(coord.inverse_bfill(20), Some(0));
        assert_eq!(coord.inverse_bfill(21), Some(1));
        assert_eq!(coord.inverse_bfill(22), Some(1));
        assert_eq!(coord.inverse_bfill(23), Some(2));
        assert_eq!(coord.inverse_bfill(24), Some(2));
        assert_eq!(coord.inverse_bfill(25), Some(3));
        assert_eq!(coord.inverse_bfill(26), Some(3));
        assert_eq!(coord.inverse_bfill(30), Some(5));
        assert_eq!(coord.inverse_bfill(31), None);
    }

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
