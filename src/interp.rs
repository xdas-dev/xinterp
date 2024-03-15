use crate::schemes::{Forward, Inverse};

#[derive(PartialEq, Debug)]
pub enum InterpError {
    OutOfBounds,
    NotFound,
    NotStrictlyIncreasing,
}

pub struct Interp<X, F> {
    xp: Vec<X>,
    fp: Vec<F>,
    forwardable: bool,
    inversable: bool,
}

impl<X, F> Interp<X, F>
where
    X: Forward<F>,
    F: Inverse<X>,
{
    pub fn new(xp: Vec<X>, fp: Vec<F>) -> Interp<X, F> {
        assert!(xp.len() == fp.len(), "xp and fp must have same length");
        let forwardable = xp.windows(2).all(|pair| pair[0] < pair[1]);
        let inversable = fp.windows(2).all(|pair| pair[0] < pair[1]);
        Interp {
            xp,
            fp,
            forwardable,
            inversable,
        }
    }
    pub fn forward(&self, rhs: X) -> Result<F, InterpError> {
        if self.forwardable {
            match self.xp.binary_search(&rhs) {
                Ok(index) => Ok(self.fp[index].clone()),
                Err(0) => Err(InterpError::OutOfBounds),
                Err(len) if len == self.xp.len() => Err(InterpError::OutOfBounds),
                Err(index) => Ok(rhs.forward(
                    self.xp[index - 1].clone(),
                    self.xp[index].clone(),
                    self.fp[index - 1].clone(),
                    self.fp[index].clone(),
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    pub fn inverse_exact(&self, rhs: F) -> Result<X, InterpError> {
        if self.inversable {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Ok(self.xp[index].clone()),
                Err(0) => Err(InterpError::OutOfBounds),
                Err(len) if len == self.xp.len() => Err(InterpError::OutOfBounds),
                Err(index) => rhs
                    .inverse_exact(
                        self.xp[index - 1].clone(),
                        self.xp[index].clone(),
                        self.fp[index - 1].clone(),
                        self.fp[index].clone(),
                    )
                    .ok_or(InterpError::NotFound),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    pub fn inverse_round(&self, rhs: F) -> Result<X, InterpError> {
        if self.inversable {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Ok(self.xp[index].clone()),
                Err(0) => Ok(self.xp[0].clone()),
                Err(len) if len == self.xp.len() => Ok(self.xp[len - 1].clone()),
                Err(index) => Ok(rhs.inverse_round(
                    self.xp[index - 1].clone(),
                    self.xp[index].clone(),
                    self.fp[index - 1].clone(),
                    self.fp[index].clone(),
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    pub fn inverse_ffill(&self, rhs: F) -> Result<X, InterpError> {
        if self.inversable {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Ok(self.xp[index].clone()),
                Err(0) => Err(InterpError::OutOfBounds),
                Err(len) if len == self.xp.len() => Ok(self.xp[len - 1].clone()),
                Err(index) => Ok(rhs.inverse_ffill(
                    self.xp[index - 1].clone(),
                    self.xp[index].clone(),
                    self.fp[index - 1].clone(),
                    self.fp[index].clone(),
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
    pub fn inverse_bfill(&self, rhs: F) -> Result<X, InterpError> {
        if self.inversable {
            match self.fp.binary_search(&rhs) {
                Ok(index) => Ok(self.xp[index].clone()),
                Err(0) => Ok(self.xp[0].clone()),
                Err(len) if len == self.xp.len() => Err(InterpError::OutOfBounds),
                Err(index) => Ok(rhs.inverse_bfill(
                    self.xp[index - 1].clone(),
                    self.xp[index].clone(),
                    self.fp[index - 1].clone(),
                    self.fp[index].clone(),
                )),
            }
        } else {
            Err(InterpError::NotStrictlyIncreasing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::F80;

    #[test]
    fn test_initialization() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![20, 25];
        let interp = Interp::new(xp, fp);
        assert!(interp.forwardable);
        assert!(interp.inversable);

        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![-20, -25];
        let interp = Interp::new(xp, fp);
        assert!(interp.forwardable);
        assert!(!interp.inversable);
    }

    #[test]
    fn test_forward_unsigned() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<u64> = vec![20, 25];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.forward(0), Ok(20));
        assert_eq!(interp.forward(1), Ok(20));
        assert_eq!(interp.forward(2), Ok(21));
        assert_eq!(interp.forward(3), Ok(22));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_forward_signed() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<i64> = vec![-20, -25];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.forward(0), Ok(-20));
        assert_eq!(interp.forward(1), Ok(-20));
        assert_eq!(interp.forward(2), Ok(-21));
        assert_eq!(interp.forward(3), Ok(-22));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_forward_float() {
        let xp: Vec<u64> = vec![0, 10];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(25.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.forward(0), Ok(F80::from(20.0)));
        assert_eq!(interp.forward(1), Ok(F80::from(20.5)));
        assert_eq!(interp.forward(2), Ok(F80::from(21.0)));
        assert_eq!(interp.forward(3), Ok(F80::from(21.5)));
        assert_eq!(interp.forward(11), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_exact_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_exact(19), Err(InterpError::OutOfBounds));
        assert_eq!(interp.inverse_exact(20), Ok(0));
        assert_eq!(interp.inverse_exact(21), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(22), Ok(1));
        assert_eq!(interp.inverse_exact(23), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(24), Ok(2));
        assert_eq!(interp.inverse_exact(25), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(26), Ok(3));
        assert_eq!(interp.inverse_exact(30), Ok(5));
        assert_eq!(interp.inverse_exact(31), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_round_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_round(19), Ok(0));
        assert_eq!(interp.inverse_round(20), Ok(0));
        assert_eq!(interp.inverse_round(21), Ok(0));
        assert_eq!(interp.inverse_round(22), Ok(1));
        assert_eq!(interp.inverse_round(23), Ok(2));
        assert_eq!(interp.inverse_round(24), Ok(2));
        assert_eq!(interp.inverse_round(25), Ok(2));
        assert_eq!(interp.inverse_round(26), Ok(3));
        assert_eq!(interp.inverse_round(30), Ok(5));
        assert_eq!(interp.inverse_round(31), Ok(5));
    }

    #[test]
    fn test_inverse_ffill_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_ffill(19), Err(InterpError::OutOfBounds));
        assert_eq!(interp.inverse_ffill(20), Ok(0));
        assert_eq!(interp.inverse_ffill(21), Ok(0));
        assert_eq!(interp.inverse_ffill(22), Ok(1));
        assert_eq!(interp.inverse_ffill(23), Ok(1));
        assert_eq!(interp.inverse_ffill(24), Ok(2));
        assert_eq!(interp.inverse_ffill(25), Ok(2));
        assert_eq!(interp.inverse_ffill(26), Ok(3));
        assert_eq!(interp.inverse_ffill(30), Ok(5));
        assert_eq!(interp.inverse_ffill(31), Ok(5));
    }

    #[test]
    fn test_inverse_bfill_unsigned() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<u64> = vec![20, 30];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_bfill(19), Ok(0));
        assert_eq!(interp.inverse_bfill(20), Ok(0));
        assert_eq!(interp.inverse_bfill(21), Ok(1));
        assert_eq!(interp.inverse_bfill(22), Ok(1));
        assert_eq!(interp.inverse_bfill(23), Ok(2));
        assert_eq!(interp.inverse_bfill(24), Ok(2));
        assert_eq!(interp.inverse_bfill(25), Ok(3));
        assert_eq!(interp.inverse_bfill(26), Ok(3));
        assert_eq!(interp.inverse_bfill(30), Ok(5));
        assert_eq!(interp.inverse_bfill(31), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_exact_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_exact(-31), Err(InterpError::OutOfBounds));
        assert_eq!(interp.inverse_exact(-30), Ok(0));
        assert_eq!(interp.inverse_exact(-29), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(-28), Ok(1));
        assert_eq!(interp.inverse_exact(-27), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(-26), Ok(2));
        assert_eq!(interp.inverse_exact(-25), Err(InterpError::NotFound));
        assert_eq!(interp.inverse_exact(-24), Ok(3));
        assert_eq!(interp.inverse_exact(-20), Ok(5));
        assert_eq!(interp.inverse_exact(-19), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_round_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_round(-31), Ok(0));
        assert_eq!(interp.inverse_round(-30), Ok(0));
        assert_eq!(interp.inverse_round(-29), Ok(0));
        assert_eq!(interp.inverse_round(-28), Ok(1));
        assert_eq!(interp.inverse_round(-27), Ok(2));
        assert_eq!(interp.inverse_round(-26), Ok(2));
        assert_eq!(interp.inverse_round(-25), Ok(2));
        assert_eq!(interp.inverse_round(-24), Ok(3));
        assert_eq!(interp.inverse_round(-20), Ok(5));
        assert_eq!(interp.inverse_round(-19), Ok(5));
    }

    #[test]
    fn test_inverse_ffill_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_ffill(-31), Err(InterpError::OutOfBounds));
        assert_eq!(interp.inverse_ffill(-30), Ok(0));
        assert_eq!(interp.inverse_ffill(-29), Ok(0));
        assert_eq!(interp.inverse_ffill(-28), Ok(1));
        assert_eq!(interp.inverse_ffill(-27), Ok(1));
        assert_eq!(interp.inverse_ffill(-26), Ok(2));
        assert_eq!(interp.inverse_ffill(-25), Ok(2));
        assert_eq!(interp.inverse_ffill(-24), Ok(3));
        assert_eq!(interp.inverse_ffill(-20), Ok(5));
        assert_eq!(interp.inverse_ffill(-19), Ok(5));
    }

    #[test]
    fn test_inverse_bfill_signed() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<i64> = vec![-30, -20];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_bfill(-31), Ok(0));
        assert_eq!(interp.inverse_bfill(-30), Ok(0));
        assert_eq!(interp.inverse_bfill(-29), Ok(1));
        assert_eq!(interp.inverse_bfill(-28), Ok(1));
        assert_eq!(interp.inverse_bfill(-27), Ok(2));
        assert_eq!(interp.inverse_bfill(-26), Ok(2));
        assert_eq!(interp.inverse_bfill(-25), Ok(3));
        assert_eq!(interp.inverse_bfill(-24), Ok(3));
        assert_eq!(interp.inverse_bfill(-20), Ok(5));
        assert_eq!(interp.inverse_bfill(-19), Err(InterpError::OutOfBounds));
    }

    #[test]
    fn test_inverse_round_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_round(F80::from(19.9)), Ok(0));
        assert_eq!(interp.inverse_round(F80::from(20.0)), Ok(0));
        assert_eq!(interp.inverse_round(F80::from(20.1)), Ok(0));
        assert_eq!(interp.inverse_round(F80::from(20.9)), Ok(0));
        assert_eq!(interp.inverse_round(F80::from(21.1)), Ok(1));
        assert_eq!(interp.inverse_round(F80::from(22.0)), Ok(1));
        assert_eq!(interp.inverse_round(F80::from(29.9)), Ok(5));
        assert_eq!(interp.inverse_round(F80::from(30.0)), Ok(5));
        assert_eq!(interp.inverse_round(F80::from(30.1)), Ok(5));
        assert_eq!(interp.inverse_round(F80::from(21.0)), Ok(0));
        assert_eq!(interp.inverse_round(F80::from(23.0)), Ok(2));
        assert_eq!(interp.inverse_round(F80::from(25.0)), Ok(2));
        assert_eq!(interp.inverse_round(F80::from(27.0)), Ok(4));
        assert_eq!(interp.inverse_round(F80::from(29.0)), Ok(4));
    }

    #[test]
    fn test_inverse_ffill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(
            interp.inverse_ffill(F80::from(19.9)),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse_ffill(F80::from(20.0)), Ok(0));
        assert_eq!(interp.inverse_ffill(F80::from(20.1)), Ok(0));
        assert_eq!(interp.inverse_ffill(F80::from(20.9)), Ok(0));
        assert_eq!(interp.inverse_ffill(F80::from(21.1)), Ok(0));
        assert_eq!(interp.inverse_ffill(F80::from(22.0)), Ok(1));
        assert_eq!(interp.inverse_ffill(F80::from(29.9)), Ok(4));
        assert_eq!(interp.inverse_ffill(F80::from(30.0)), Ok(5));
        assert_eq!(interp.inverse_ffill(F80::from(30.1)), Ok(5));
        assert_eq!(interp.inverse_ffill(F80::from(21.0)), Ok(0));
        assert_eq!(interp.inverse_ffill(F80::from(23.0)), Ok(1));
        assert_eq!(interp.inverse_ffill(F80::from(25.0)), Ok(2));
        assert_eq!(interp.inverse_ffill(F80::from(27.0)), Ok(3));
        assert_eq!(interp.inverse_ffill(F80::from(29.0)), Ok(4));
    }

    #[test]
    fn test_inverse_bfill_float() {
        let xp: Vec<u64> = vec![0, 5];
        let fp: Vec<F80> = vec![F80::from(20.0), F80::from(30.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_bfill(F80::from(19.9)), Ok(0));
        assert_eq!(interp.inverse_bfill(F80::from(20.0)), Ok(0));
        assert_eq!(interp.inverse_bfill(F80::from(20.1)), Ok(1));
        assert_eq!(interp.inverse_bfill(F80::from(20.9)), Ok(1));
        assert_eq!(interp.inverse_bfill(F80::from(21.1)), Ok(1));
        assert_eq!(interp.inverse_bfill(F80::from(22.0)), Ok(1));
        assert_eq!(interp.inverse_bfill(F80::from(29.9)), Ok(5));
        assert_eq!(interp.inverse_bfill(F80::from(30.0)), Ok(5));
        assert_eq!(
            interp.inverse_bfill(F80::from(30.1)),
            Err(InterpError::OutOfBounds)
        );
        assert_eq!(interp.inverse_bfill(F80::from(21.0)), Ok(1));
        assert_eq!(interp.inverse_bfill(F80::from(23.0)), Ok(2));
        assert_eq!(interp.inverse_bfill(F80::from(25.0)), Ok(3));
        assert_eq!(interp.inverse_bfill(F80::from(27.0)), Ok(4));
        assert_eq!(interp.inverse_bfill(F80::from(29.0)), Ok(5));
    }

    #[test]
    fn test_forward_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.forward(0), Ok(i64::MIN));
        assert_eq!(interp.forward(u64::MAX), Ok(i64::MAX));
        assert_eq!(interp.forward(u64::MAX / 2 + 1), Ok(0));
    }

    #[test]
    fn test_inverse_exact_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse_exact(i64::MIN), Ok(0));
        assert_eq!(interp.inverse_exact(i64::MAX), Ok(u64::MAX));
        assert_eq!(interp.inverse_exact(0), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_round_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse_round(i64::MIN), Ok(0));
        assert_eq!(interp.inverse_round(i64::MAX), Ok(u64::MAX));
        assert_eq!(interp.inverse_round(0), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_ffill_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse_ffill(i64::MIN), Ok(0));
        assert_eq!(interp.inverse_ffill(i64::MAX), Ok(u64::MAX));
        assert_eq!(interp.inverse_ffill(0), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_inverse_bfill_big_numbers() {
        let interp = Interp::new(vec![0, u64::MAX], vec![i64::MIN, i64::MAX]);
        assert_eq!(interp.inverse_bfill(i64::MIN), Ok(0));
        assert_eq!(interp.inverse_bfill(i64::MAX), Ok(u64::MAX));
        assert_eq!(interp.inverse_bfill(0), Ok(u64::MAX / 2 + 1));
    }

    #[test]
    fn test_something() {
        let xp: Vec<u64> = vec![0, 8];
        let fp: Vec<F80> = vec![F80::from(100.0), F80::from(900.0)];
        let interp = Interp::new(xp, fp);
        assert_eq!(interp.inverse_round(F80::from(175.0)), Ok(1))
    }
}
