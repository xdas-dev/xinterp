//! Integer division with different rounding rules

pub enum Method {
    None,
    Nearest,
    ForwardFill,
    BackwardFill,
}

/// Traits for performing division operations with different rounding rules.
pub trait DivOp: Sized {
    fn div(self, rhs: Self, method: Method) -> Option<Self>;

    /// Performs division and returns the quotien if the remainder is zero,
    /// otherwise returns `None`.
    fn div_exact(self, rhs: Self) -> Option<Self>;

    /// Performs rounding division, rounding the result to the nearest integer.
    /// If the remainder is exactly half of the divisor, rounds to the nearest even number.
    fn div_round(self, rhs: Self) -> Self;

    /// Performs rounding division, rounding to result to the previous integer (forward fill).
    fn div_ffill(self, rhs: Self) -> Self;

    /// Performs rounding division, rounding to result to the next integer (backward fill).
    fn div_bfill(self, rhs: Self) -> Self;
}

impl DivOp for u128 {
    fn div(self, rhs: u128, method: Method) -> Option<u128> {
        let div = self.div_euclid(rhs);
        let rem = self.rem_euclid(rhs);
        match method {
            Method::None => {
                if rem == 0 {
                    Some(div)
                } else {
                    None
                }
            }
            Method::Nearest => {
                if rem * 2 < rhs {
                    Some(div)
                } else if rem * 2 > rhs {
                    Some(div + 1)
                } else {
                    if div % 2 == 0 {
                        Some(div)
                    } else {
                        Some(div + 1)
                    }
                }
            }
            Method::ForwardFill => Some(div),
            Method::BackwardFill => {
                if rem == 0 {
                    Some(div)
                } else {
                    Some(div + 1)
                }
            }
        }
    }
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
    fn div(self, rhs: i128, method: Method) -> Option<i128> {
        todo!();
    }
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
    fn test_div_exact() {
        assert_eq!(0u128.div(2, Method::None), Some(0));
        assert_eq!(1u128.div(2, Method::None), None);
        assert_eq!(2u128.div(2, Method::None), Some(1));
        assert_eq!(3u128.div(2, Method::None), None);
        assert_eq!(1u128.div(3, Method::None), None);
        assert_eq!(2u128.div(3, Method::None), None);
        assert_eq!((-1i128).div_exact(2), None);
        assert_eq!((-2i128).div_exact(2), Some(-1));
        assert_eq!((-3i128).div_exact(2), None);
        assert_eq!((-1i128).div_exact(3), None);
        assert_eq!((-2i128).div_exact(3), None);
    }

    #[test]
    fn test_div_round() {
        assert_eq!(0u128.div(2, Method::Nearest), Some(0));
        assert_eq!(1u128.div(2, Method::Nearest), Some(0));
        assert_eq!(2u128.div(2, Method::Nearest), Some(1));
        assert_eq!(3u128.div(2, Method::Nearest), Some(2));
        assert_eq!(1u128.div(3, Method::Nearest), Some(0));
        assert_eq!(2u128.div(3, Method::Nearest), Some(1));
        assert_eq!((-1i128).div_round(2), 0);
        assert_eq!((-2i128).div_round(2), -1);
        assert_eq!((-3i128).div_round(2), -2);
        assert_eq!((-1i128).div_round(3), 0);
        assert_eq!((-2i128).div_round(3), -1);
    }

    #[test]
    fn test_div_ffill() {
        assert_eq!(0u128.div(2, Method::ForwardFill), Some(0));
        assert_eq!(1u128.div(2, Method::ForwardFill), Some(0));
        assert_eq!(2u128.div(2, Method::ForwardFill), Some(1));
        assert_eq!(3u128.div(2, Method::ForwardFill), Some(1));
        assert_eq!(1u128.div(3, Method::ForwardFill), Some(0));
        assert_eq!(2u128.div(3, Method::ForwardFill), Some(0));
        assert_eq!((-1i128).div_ffill(2), -1);
        assert_eq!((-2i128).div_ffill(2), -1);
        assert_eq!((-3i128).div_ffill(2), -2);
        assert_eq!((-1i128).div_ffill(3), -1);
        assert_eq!((-2i128).div_ffill(3), -1);
    }

    #[test]
    fn test_div_bfill() {
        assert_eq!(0u128.div(2, Method::BackwardFill), Some(0));
        assert_eq!(1u128.div(2, Method::BackwardFill), Some(1));
        assert_eq!(2u128.div(2, Method::BackwardFill), Some(1));
        assert_eq!(3u128.div(2, Method::BackwardFill), Some(2));
        assert_eq!(1u128.div(3, Method::BackwardFill), Some(1));
        assert_eq!(2u128.div(3, Method::BackwardFill), Some(1));
        assert_eq!((-1i128).div_bfill(2), 0);
        assert_eq!((-2i128).div_bfill(2), -1);
        assert_eq!((-3i128).div_bfill(2), -1);
        assert_eq!((-1i128).div_bfill(3), 0);
        assert_eq!((-2i128).div_bfill(3), 0);
    }
}
