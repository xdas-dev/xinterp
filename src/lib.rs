//! This is the main module.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// This is the python module.
#[pymodule]
fn rust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    use interp::interp_ndarray;

    #[pyfn(m)]
    fn interp_int64<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, i64>,
        fp: PyReadonlyArray1<'py, i64>,
        left: i64,
        right: i64,
    ) -> &'py PyArray1<i64> {
        let x = x.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let f = interp_ndarray(&x, &xp, &fp, left, right);
        f.into_pyarray(py)
    }

    Ok(())
}

/// This is the interpolation module.
mod interp {
    use numpy::ndarray::{Array1, ArrayView1};

    /// One-dimensional linear interpolation for monotonically increasing interger-valued samples.
    ///
    /// Returns the one-dimensional piecewise linear interpolant to a function with given discrete
    /// data points (xp, fp), evaluated at several x values. The `left` and `right` arguments
    /// specifies returned values ouside the data covered interval.
    pub fn interp_ndarray(
        x: &ArrayView1<i64>,
        xp: &ArrayView1<i64>,
        fp: &ArrayView1<i64>,
        left: i64,
        right: i64,
    ) -> Array1<i64> {
        x.map(|x| interp_value(*x, xp, fp, left, right))
    }

    /// One-dimensional linear interpolation for monotonically increasing interger-valued samples.
    ///
    /// Returns the one-dimensional piecewise linear interpolant to a function with given discrete
    /// data points (xp, fp), evaluated at a unique x value. The `left` and `right` arguments
    /// specifies returned values ouside the data covered interval.
    fn interp_value(
        x: i64,
        xp: &ArrayView1<i64>,
        fp: &ArrayView1<i64>,
        left: i64,
        right: i64,
    ) -> i64 {
        match xp.to_slice().unwrap().binary_search(&x) {
            Ok(index) => fp[index],
            Err(0) => left,
            Err(len) if len == xp.len() => right,
            Err(index) => linear(xp[index - 1], xp[index], fp[index - 1], fp[index], x),
        }
    }

    /// Linearly interpolates between two integer-valued points.
    ///
    /// The interpolation is performed with 128-bit integers and use tie to even rounding.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(linear(0, 10, 10, 20, 5), 15);
    /// ```
    fn linear(x0: i64, x1: i64, f0: i64, f1: i64, x: i64) -> i64 {
        let x0 = i128::from(x0);
        let x1 = i128::from(x1);
        let f0 = i128::from(f0);
        let f1 = i128::from(f1);
        let x = i128::from(x);
        let out = roundiv(f0 * (x1 - x) + f1 * (x - x0), x1 - x0);
        i64::try_from(out).expect("cannot convert to i64")
    }

    /// Computes the rounded division of n by d using ties to even rounding.
    ///
    /// Should pass https://cscx.org/roundiv tests.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(roundiv(7, 2), 4);
    /// ```
    fn roundiv(n: i128, d: i128) -> i128 {
        let s = n.signum() * d.signum();
        let (n, d) = (n.abs(), d.abs());
        let q = n / d;
        let r = n % d;
        if (r * 2) == d {
            if q % 2 == 0 {
                s * q
            } else {
                s * (q + 1)
            }
        } else if (r * 2) > d {
            s * (q + 1)
        } else {
            s * q
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use numpy::ndarray::array;

        #[test]
        fn test_interp_ndarray() {
            let xp = array![0, 2, 5];
            let fp = array![0, 20, 50];
            let x_values = array![-2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
            let result = interp_ndarray(&x_values.view(), &xp.view(), &fp.view(), -1, -1);
            let expected = array![-1, -1, 0, 10, 20, 30, 40, 50, -1, -1];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_interp_value() {
            // Define sample data points
            let xp = array![0, 1, 2, 3, 4, 5];
            let fp = array![0, 10, 20, 30, 40, 50];
            assert_eq!(interp_value(3, &xp.view(), &fp.view(), -1, -1), 30); // existing
            assert_eq!(interp_value(-1, &xp.view(), &fp.view(), -1, -1), -1); // before
            assert_eq!(interp_value(6, &xp.view(), &fp.view(), -1, -1), -1); // after
            assert_eq!(interp_value(2, &xp.view(), &fp.view(), -1, -1), 20); // between
            assert_eq!(interp_value(0, &xp.view(), &fp.view(), -1, -1), 0); // left edge
            assert_eq!(interp_value(5, &xp.view(), &fp.view(), -1, -1), 50); // righ edge
        }

        #[test]
        fn test_linear_interpolation() {
            let test_cases = [
                // x0, x1, f0, f1, x, expected
                (0, 10, 10, 20, 5, 15),
                (0, 10, 20, 30, 5, 25),
                (0, 100, 0, 200, 50, 100),
            ];
            for (x0, x1, f0, f1, x, expected) in &test_cases {
                let result = linear(*x0, *x1, *f0, *f1, *x);
                assert_eq!(result, *expected);
            }
        }

        #[test]
        fn test_roundiv() {
            assert_eq!(roundiv(7, 2), 4);
            assert_eq!(roundiv(10, 3), 3);
            assert_eq!(roundiv(-7, 2), -4);
            assert_eq!(roundiv(-10, 3), -3);
            assert_eq!(roundiv(0, 5), 0);
            assert_eq!(roundiv(7, -2), -4);
            assert_eq!(roundiv(-7, -2), 4);
        }
    }
}
