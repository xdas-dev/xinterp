//! Integer division with different rounding rules

/// Rounding methods for integer division.
#[derive(Clone, Copy)]
pub enum Method {
    None,
    Nearest,
    ForwardFill,
    BackwardFill,
}

/// Traits for performing division operations with different rounding rules.
pub trait DivOp: Sized {
    /// Performs division with the specified rounding method.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The divisor.
    /// * `method` - The rounding method to use.
    ///
    /// # Returns
    ///
    /// It returns None if `None` rounding is chosen and the division is inexact. Otherwise, it
    /// returns the exact or rounded quotient.  
    fn div(self, rhs: Self, method: Method) -> Option<Self>;
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
                } else if div.is_multiple_of(2) {
                    Some(div)
                } else {
                    Some(div + 1)
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
    }

    #[test]
    fn test_div_round() {
        assert_eq!(0u128.div(2, Method::Nearest), Some(0));
        assert_eq!(1u128.div(2, Method::Nearest), Some(0));
        assert_eq!(2u128.div(2, Method::Nearest), Some(1));
        assert_eq!(3u128.div(2, Method::Nearest), Some(2));
        assert_eq!(1u128.div(3, Method::Nearest), Some(0));
        assert_eq!(2u128.div(3, Method::Nearest), Some(1));
    }

    #[test]
    fn test_div_ffill() {
        assert_eq!(0u128.div(2, Method::ForwardFill), Some(0));
        assert_eq!(1u128.div(2, Method::ForwardFill), Some(0));
        assert_eq!(2u128.div(2, Method::ForwardFill), Some(1));
        assert_eq!(3u128.div(2, Method::ForwardFill), Some(1));
        assert_eq!(1u128.div(3, Method::ForwardFill), Some(0));
        assert_eq!(2u128.div(3, Method::ForwardFill), Some(0));
    }

    #[test]
    fn test_div_bfill() {
        assert_eq!(0u128.div(2, Method::BackwardFill), Some(0));
        assert_eq!(1u128.div(2, Method::BackwardFill), Some(1));
        assert_eq!(2u128.div(2, Method::BackwardFill), Some(1));
        assert_eq!(3u128.div(2, Method::BackwardFill), Some(2));
        assert_eq!(1u128.div(3, Method::BackwardFill), Some(1));
        assert_eq!(2u128.div(3, Method::BackwardFill), Some(1));
    }
}
