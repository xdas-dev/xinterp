use astro_float::{BigFloat, RoundingMode, Sign};

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct F80 {
    value: BigFloat,
}
impl From<u64> for F80 {
    fn from(value: u64) -> F80 {
        F80 {
            value: BigFloat::from_u64(value, 64),
        }
    }
}
impl From<f64> for F80 {
    fn from(value: f64) -> F80 {
        assert!(value.is_finite());
        F80 {
            value: BigFloat::from_f64(value, 64),
        }
    }
}
impl From<F80> for f64 {
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
    fn cmp(&self, other: &F80) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl F80 {
    pub fn add(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.add(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    pub fn sub(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.sub(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    pub fn mul(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.mul(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    pub fn div(&self, rhs: &F80) -> F80 {
        F80 {
            value: self.value.div(&rhs.value, 64, RoundingMode::ToEven),
        }
    }
    // pub fn round(&self) -> F80 {
    //     let abs = self.value.abs();
    //     let one = BigFloat::from_f64(1.0, 64);
    //     let cmp = abs.cmp(&one);
    //     match cmp {
    //         Some(Sign::Pos) => F80 {
    //             value: self.value.round(0, RoundingMode::ToEven),
    //         },
    //         Some(0) => self,
    //         Some(Sign::Neg) => F80 {
    //             value: self.value.round(0, RoundingMode::ToEven),
    //         },
    //         None => self,
    //     }
    //     // if .cmp(&BigFloat::from_f64(1.0, 64)).unwrap() {

    //     // }/
    //     // F80 {
    //     //     value: self.value,
    //     // }
    // }
    pub fn floor(&self) -> F80 {
        F80 {
            value: self.value.floor(),
        }
    }
    pub fn ceil(&self) -> F80 {
        F80 {
            value: self.value.ceil(),
        }
    }
}

// trait RoundTiesEven {
//     fn round_ties_even(self) -> f64;
// }
// impl RoundTiesEven for f64 {
//     fn round_ties_even(self) -> f64 {
//         let mut rounded = self.round();
//         if (self - rounded).abs() == 0.5 {
//             if rounded % 2.0 == 0.0 {
//                 return rounded;
//             } else {
//                 if self > 0.0 {
//                     rounded -= 1.0;
//                 } else {
//                     rounded += 1.0;
//                 }
//             }
//         }
//         rounded
//     }
// }

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
}
