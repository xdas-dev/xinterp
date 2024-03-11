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
        num.rounded_div(den) as i64
    }
}

trait RoundedDiv {
    fn rounded_div(self, rhs: Self) -> Self;
}

impl RoundedDiv for u128 {
    fn rounded_div(self, rhs: u128) -> u128 {
        let div = self / rhs;
        let rem = self % rhs;
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
}

impl RoundedDiv for i128 {
    fn rounded_div(self, rhs: i128) -> i128 {
        let sgn = self.signum() * rhs.signum();
        sgn * ((self.abs() as u128).rounded_div(rhs.abs() as u128) as i128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_rounded_div() {
        assert_eq!(0u128.rounded_div(2), 0);
        assert_eq!(1u128.rounded_div(2), 0);
        assert_eq!(2u128.rounded_div(2), 1);
        assert_eq!(3u128.rounded_div(2), 2);
        assert_eq!(1u128.rounded_div(3), 0);
        assert_eq!(2u128.rounded_div(3), 1);
        assert_eq!(-0i128.rounded_div(2), 0);
        assert_eq!(-1i128.rounded_div(2), 0);
        assert_eq!(-2i128.rounded_div(2), -1);
        assert_eq!(-3i128.rounded_div(2), -2);
        assert_eq!(-1i128.rounded_div(3), -0);
        assert_eq!(-2i128.rounded_div(3), -1);
    }
}
