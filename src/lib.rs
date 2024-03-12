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
    fn test_initialization() {
        let interp = Interp::new(vec![0, 10], vec![20, 25]);
        assert!(interp.forward);
        assert!(interp.inverse);

        let interp = Interp::new(vec![0, 10], vec![-20, -25]);
        assert!(interp.forward);
        assert!(!interp.inverse);
    }

    #[test]
    fn test_forward() {
        let interp = Interp::new(vec![0, 10], vec![20, 25]);
        assert_eq!(interp.forward(0), Some(20));
        assert_eq!(interp.forward(1), Some(20));
        assert_eq!(interp.forward(2), Some(21));
        assert_eq!(interp.forward(3), Some(22));
        assert_eq!(interp.forward(11), None);
    }

    #[test]
    fn test_forward_negative() {
        let interp = Interp::new(vec![0, 10], vec![-20, -25]);
        assert_eq!(interp.forward(0), Some(-20));
        assert_eq!(interp.forward(1), Some(-20));
        assert_eq!(interp.forward(2), Some(-21));
        assert_eq!(interp.forward(3), Some(-22));
        assert_eq!(interp.forward(11), None);
    }

    #[test]
    fn test_inverse_exact() {
        let interp = Interp::new(vec![0, 5], vec![20, 30]);
        assert_eq!(interp.inverse_exact(19), None);
        assert_eq!(interp.inverse_exact(20), Some(0));
        assert_eq!(interp.inverse_exact(21), None);
        assert_eq!(interp.inverse_exact(22), Some(1));
        assert_eq!(interp.inverse_exact(23), None);
        assert_eq!(interp.inverse_exact(24), Some(2));
        assert_eq!(interp.inverse_exact(25), None);
        assert_eq!(interp.inverse_exact(26), Some(3));
        assert_eq!(interp.inverse_exact(30), Some(5));
        assert_eq!(interp.inverse_exact(31), None);
    }

    #[test]
    fn test_inverse_round() {
        let interp = Interp::new(vec![0, 5], vec![20, 30]);
        assert_eq!(interp.inverse_round(19), None);
        assert_eq!(interp.inverse_round(20), Some(0));
        assert_eq!(interp.inverse_round(21), Some(0));
        assert_eq!(interp.inverse_round(22), Some(1));
        assert_eq!(interp.inverse_round(23), Some(2));
        assert_eq!(interp.inverse_round(24), Some(2));
        assert_eq!(interp.inverse_round(25), Some(2));
        assert_eq!(interp.inverse_round(26), Some(3));
        assert_eq!(interp.inverse_round(30), Some(5));
        assert_eq!(interp.inverse_round(31), None);
    }

    #[test]
    fn test_inverse_ffill() {
        let interp = Interp::new(vec![0, 5], vec![20, 30]);
        assert_eq!(interp.inverse_ffill(19), None);
        assert_eq!(interp.inverse_ffill(20), Some(0));
        assert_eq!(interp.inverse_ffill(21), Some(0));
        assert_eq!(interp.inverse_ffill(22), Some(1));
        assert_eq!(interp.inverse_ffill(23), Some(1));
        assert_eq!(interp.inverse_ffill(24), Some(2));
        assert_eq!(interp.inverse_ffill(25), Some(2));
        assert_eq!(interp.inverse_ffill(26), Some(3));
        assert_eq!(interp.inverse_ffill(30), Some(5));
        assert_eq!(interp.inverse_ffill(31), None);
    }

    #[test]
    fn test_inverse_bfill() {
        let interp = Interp::new(vec![0, 5], vec![20, 30]);
        assert_eq!(interp.inverse_bfill(19), None);
        assert_eq!(interp.inverse_bfill(20), Some(0));
        assert_eq!(interp.inverse_bfill(21), Some(1));
        assert_eq!(interp.inverse_bfill(22), Some(1));
        assert_eq!(interp.inverse_bfill(23), Some(2));
        assert_eq!(interp.inverse_bfill(24), Some(2));
        assert_eq!(interp.inverse_bfill(25), Some(3));
        assert_eq!(interp.inverse_bfill(26), Some(3));
        assert_eq!(interp.inverse_bfill(30), Some(5));
        assert_eq!(interp.inverse_bfill(31), None);
    }

    #[test]
    fn test_inverse_exact_negative() {
        let interp = Interp::new(vec![0, 5], vec![-30, -20]);
        assert_eq!(interp.inverse_exact(-31), None);
        assert_eq!(interp.inverse_exact(-30), Some(0));
        assert_eq!(interp.inverse_exact(-29), None);
        assert_eq!(interp.inverse_exact(-28), Some(1));
        assert_eq!(interp.inverse_exact(-27), None);
        assert_eq!(interp.inverse_exact(-26), Some(2));
        assert_eq!(interp.inverse_exact(-25), None);
        assert_eq!(interp.inverse_exact(-24), Some(3));
        assert_eq!(interp.inverse_exact(-20), Some(5));
        assert_eq!(interp.inverse_exact(-19), None);
    }

    #[test]
    fn test_inverse_round_negative() {
        let interp = Interp::new(vec![0, 5], vec![-30, -20]);
        assert_eq!(interp.inverse_round(-31), None);
        assert_eq!(interp.inverse_round(-30), Some(0));
        assert_eq!(interp.inverse_round(-29), Some(0));
        assert_eq!(interp.inverse_round(-28), Some(1));
        assert_eq!(interp.inverse_round(-27), Some(2));
        assert_eq!(interp.inverse_round(-26), Some(2));
        assert_eq!(interp.inverse_round(-25), Some(2));
        assert_eq!(interp.inverse_round(-24), Some(3));
        assert_eq!(interp.inverse_round(-20), Some(5));
        assert_eq!(interp.inverse_round(-19), None);
    }

    #[test]
    fn test_inverse_ffill_negative() {
        let interp = Interp::new(vec![0, 5], vec![-30, -20]);
        assert_eq!(interp.inverse_ffill(-31), None);
        assert_eq!(interp.inverse_ffill(-30), Some(0));
        assert_eq!(interp.inverse_ffill(-29), Some(0));
        assert_eq!(interp.inverse_ffill(-28), Some(1));
        assert_eq!(interp.inverse_ffill(-27), Some(1));
        assert_eq!(interp.inverse_ffill(-26), Some(2));
        assert_eq!(interp.inverse_ffill(-25), Some(2));
        assert_eq!(interp.inverse_ffill(-24), Some(3));
        assert_eq!(interp.inverse_ffill(-20), Some(5));
        assert_eq!(interp.inverse_ffill(-19), None);
    }

    #[test]
    fn test_inverse_bfill_negative() {
        let interp = Interp::new(vec![0, 5], vec![-30, -20]);
        assert_eq!(interp.inverse_bfill(-31), None);
        assert_eq!(interp.inverse_bfill(-30), Some(0));
        assert_eq!(interp.inverse_bfill(-29), Some(1));
        assert_eq!(interp.inverse_bfill(-28), Some(1));
        assert_eq!(interp.inverse_bfill(-27), Some(2));
        assert_eq!(interp.inverse_bfill(-26), Some(2));
        assert_eq!(interp.inverse_bfill(-25), Some(3));
        assert_eq!(interp.inverse_bfill(-24), Some(3));
        assert_eq!(interp.inverse_bfill(-20), Some(5));
        assert_eq!(interp.inverse_bfill(-19), None);
    }

    #[test]
    fn test_inverse_exact_big_numbers() {
        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        let interp = Interp::new(vec![0, x1], vec![0, f1]);
        assert_eq!(interp.inverse_exact(0), Some(0));
        assert_eq!(interp.inverse_exact(f1), Some(x1));
        assert_eq!(interp.inverse_exact(f1 / 2), Some(x1 / 2));
    }

    #[test]
    fn test_forward_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.forward(0), Some(i64::MIN));
        assert_eq!(interp.forward(u64::MAX), Some(i64::MAX));
        assert_eq!(interp.forward((u64::MAX / 2)), Some(-1));
    }

    #[test]
    fn test_inverse_round_big_numbers() {
        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        let interp = Interp::new(vec![0, x1], vec![0, f1]);
        assert_eq!(interp.inverse_round(0), Some(0));
        assert_eq!(interp.inverse_round(f1), Some(x1));
        assert_eq!(interp.inverse_round(f1 / 2), Some(x1 / 2));
    }

    #[test]
    fn test_inverse_ffill_big_numbers() {
        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        let interp = Interp::new(vec![0, x1], vec![0, f1]);
        assert_eq!(interp.inverse_ffill(0), Some(0));
        assert_eq!(interp.inverse_ffill(f1), Some(x1));
        assert_eq!(interp.inverse_ffill(f1 / 2), Some(x1 / 2));
    }

    #[test]
    fn test_inverse_bfill_big_numbers() {
        let x1 = u64::MAX - 3;
        let f1 = (x1 / 2) as i64;
        let interp = Interp::new(vec![0, x1], vec![0, f1]);
        assert_eq!(interp.inverse_bfill(0), Some(0));
        assert_eq!(interp.inverse_bfill(f1), Some(x1));
        assert_eq!(interp.inverse_bfill(f1 / 2), Some(x1 / 2));
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
