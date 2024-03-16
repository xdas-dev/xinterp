//! Extended precision floating-point format that can accurately represent 64 bits integers.

use astro_float::{BigFloat, RoundingMode, Sign};
use std::cmp::Ordering;

/// f80 floating-point format with 64 bits mantissa. It wraps astro-float BigFloat struct with
/// imposed one word (64 bits) mantissa. It implements total ordering by only allowing finite
/// values (no nan or inf). It expose some basic methods of BigFloat. Use the From/Into traits
/// to initialize some instance of this struct from u64 or f64.  
#[derive(Clone, PartialEq, Debug)]
pub struct F80 {
    value: BigFloat,
}
impl From<u64> for F80 {
    /// Converts a u64 into an F80.
    fn from(value: u64) -> F80 {
        F80 {
            value: BigFloat::from_u64(value, 64),
        }
    }
}
impl From<f64> for F80 {
    /// Converts an f64 into an F80. Panics if the input is NaN or infinity.
    fn from(value: f64) -> F80 {
        assert!(value.is_finite());
        F80 {
            value: BigFloat::from_f64(value, 64),
        }
    }
}
impl From<F80> for f64 {
    /// Converts an F80 into an f64.
    fn from(float: F80) -> f64 {
        if float.value.is_zero() {
            return 0.0;
        }
        let sign = float.value.sign().unwrap();
        let exponent = float.value.exponent().unwrap();
        let mantissa = float.value.mantissa_digits().unwrap()[0];
        if mantissa == 0 {
            return 0.0;
        }
        let mut exponent: isize = exponent as isize + 0b1111111111;
        let mut ret = 0;
        if exponent >= 0b11111111111 {
            match sign {
                Sign::Pos => f64::INFINITY,
                Sign::Neg => f64::NEG_INFINITY,
            }
        } else if exponent <= 0 {
            let shift = -exponent;
            if shift < 52 {
                ret |= mantissa >> (shift + 12);
                if sign == Sign::Neg {
                    ret |= 0x8000000000000000u64;
                }
                f64::from_bits(ret)
            } else {
                0.0
            }
        } else {
            let mantissa = mantissa << 1;
            exponent -= 1;
            if sign == Sign::Neg {
                ret |= 1;
            }
            ret <<= 11;
            ret |= exponent as u64;
            ret <<= 52;
            ret |= mantissa >> 12;
            f64::from_bits(ret)
        }
    }
}
impl From<F80> for u64 {
    /// Converts an F80 into a u64.
    fn from(float: F80) -> u64 {
        if float.value.is_zero() {
            return 0;
        }
        let sign = float.value.sign().unwrap();
        let exponent = float.value.exponent().unwrap();
        let mantissa = float.value.mantissa_digits().unwrap()[0];
        match sign {
            Sign::Pos => {
                if exponent > 0 {
                    if exponent <= 64 {
                        let shift = (64 - exponent) as u64;
                        let ret = mantissa;
                        ret >> shift
                    } else {
                        u64::MAX
                    }
                } else {
                    0
                }
            }
            Sign::Neg => 0,
        }
    }
}
impl Eq for F80 {}
impl Ord for F80 {
    /// Compares two F80.
    fn cmp(&self, other: &F80) -> Ordering {
        self.value.partial_cmp(&other.value).unwrap()
    }
}
impl PartialOrd for F80 {
    /// Compares two F80.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl F80 {
    /// Adds two F80s.
    pub fn add(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.add(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    /// Subtracts two F80s.
    pub fn sub(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.sub(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    /// Multiplies two F80s.
    pub fn mul(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.mul(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    /// Divides two F80s.
    pub fn div(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.div(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    /// Computes the remainder of division of two F80s.
    pub fn rem(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.rem(&rhs.value),
        }
    }
    /// Rounds a F80  to its nearest integer using the round ties to even rule.
    pub fn round(&self) -> F80 {
        let floor = self.floor();
        let ceil = self.ceil();
        let mid = floor.add(&ceil).div(&F80::from(2));
        match self.cmp(&mid) {
            Ordering::Less => floor,
            Ordering::Equal => match floor.rem(&F80::from(2)).eq(&F80::from(0)) {
                true => floor,
                false => ceil,
            },
            Ordering::Greater => ceil,
        }
    }
    /// Floors a F80.
    pub fn floor(&self) -> F80 {
        F80 {
            value: self.value.floor(),
        }
    }
    /// Ceils a F80.
    pub fn ceil(&self) -> F80 {
        F80 {
            value: self.value.ceil(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_conversion() {
        let cases: [u64; 7] = [0, 1, 2, u64::MAX / 2, u64::MAX - 2, u64::MAX - 1, u64::MAX];
        for expected in cases.iter() {
            let result: u64 = F80::from(*expected).into();
            assert_eq!(result, *expected);
        }
        let cases: [(f64, u64); 4] = [(-1.0, 0), (0.5, 0), (1.5, 1), (1e32, u64::MAX)];
        for (input, expected) in cases.iter() {
            let result: u64 = F80::from(*input).into();
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_f64_conversion() {
        let cases: [f64; 11] = [
            0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 1e307, -1e307, 1e-307, -1e-307,
        ];
        for expected in cases.iter() {
            let result: f64 = F80::from(*expected).into();
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_rounding() {
        let cases: [(f64, u64); 13] = [
            (0.0, 0),
            (0.1, 0),
            (0.4, 0),
            (0.5, 0),
            (0.6, 1),
            (0.9, 1),
            (1.0, 1),
            (1.1, 1),
            (1.4, 1),
            (1.5, 2),
            (1.6, 2),
            (1.9, 2),
            (2.0, 2),
        ];
        for (input, expected) in cases {
            let result: u64 = F80::from(input).round().into();
            assert_eq!(result, expected)
        }
    }
}
