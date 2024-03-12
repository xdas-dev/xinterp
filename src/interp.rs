use crate::transform::{Forward, Inverse};

pub struct Interp<X, F> {
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
        assert!(xp.len() == fp.len(), "xp and fp must have same length");
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
        assert_eq!(interp.forward(u64::MAX / 2), Some(-1));
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
}
