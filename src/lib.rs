//! This is the main module.

use num::{NumCast, PrimInt};
use numpy::{dtype, Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArray};
use pyo3::prelude::*;

/// This is the python module.
#[pymodule]
fn rust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    use interp::interp_ndarray_int;

    #[pyfn(m)]
    fn interp_int64<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, i64>,
        fp: PyReadonlyArray1<'py, i64>,
        left: i64,
        right: i64,
    ) -> &'py PyArray1<i64> {
        interp_int(py, x, xp, fp, left, right)
    }

    #[pyfn(m)]
    fn interp<'py>(
        py: Python<'py>,
        x: PyUntypedArray,
        xp: PyUntypedArray,
        fp: PyReadonlyArray1<'py, i64>,
        left: i64,
        right: i64,
    ) -> &'py PyArray1<i64> {
        let element_type = x.dtype();

        if element_type.is_equiv_to(dtype::<i32>(py)) {
            let x: &PyArray1<i32> = x.downcast()?;
            let xp: &PyArray1<i32> = xp.downcast()?;
            interp_int(py, x, xp, fp, left, right)
        } else if element_type.is_equiv_to(dtype::<i64>(py)) {
            let x: &PyReadonlyArray1<i64> = x.downcast()?;
            let xp:&PyReadonlyArray1<i64> = xp.downcast()?;
            interp_int(py, x, xp, fp, left, right)
        }
    }

    fn interp_int<'py, T, U>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, T>,
        xp: PyReadonlyArray1<'py, T>,
        fp: PyReadonlyArray1<'py, U>,
        left: U,
        right: U,
    ) -> &'py PyArray1<U>
    where
        T: PrimInt + NumCast + Element,
        U: PrimInt + NumCast + Element,
    {
        let x = x.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let f = interp_ndarray_int(&x, &xp, &fp, left, right);
        f.into_pyarray(py)
    }

    Ok(())
}

/// This is the interpolation module.
mod interp {
    use num::{Float, NumCast, PrimInt};
    use numpy::ndarray::{Array1, ArrayView1};

    /// One-dimensional linear interpolation for monotonically increasing interger-valued samples.
    ///
    /// Returns the one-dimensional piecewise linear interpolant to a function with given discrete
    /// data points (xp, fp), evaluated at several x values. The `left` and `right` arguments
    /// specifies returned values ouside the data covered interval.
    pub fn interp_ndarray_int<U, T>(
        x: &ArrayView1<T>,
        xp: &ArrayView1<T>,
        fp: &ArrayView1<U>,
        left: U,
        right: U,
    ) -> Array1<U>
    where
        T: PrimInt + NumCast,
        U: PrimInt + NumCast,
    {
        x.map(|x| interp_value_int(*x, xp, fp, left, right))
    }

    /// One-dimensional linear interpolation for monotonically increasing interger-valued samples.
    ///
    /// Returns the one-dimensional piecewise linear interpolant to a function with given discrete
    /// data points (xp, fp), evaluated at a unique x value. The `left` and `right` arguments
    /// specifies returned values ouside the data covered interval.
    fn interp_value_int<T, U>(x: T, xp: &ArrayView1<T>, fp: &ArrayView1<U>, left: U, right: U) -> U
    where
        T: PrimInt + NumCast,
        U: PrimInt + NumCast,
    {
        match xp.to_slice().unwrap().binary_search(&x) {
            Ok(index) => fp[index],
            Err(0) => left,
            Err(len) if len == xp.len() => right,
            Err(index) => forward_int(x, xp[index - 1], xp[index], fp[index - 1], fp[index]),
        }
    }

    // fn forward<T, U>(x: T, x0: T, x1: T, f0: U, f1: U) -> U;

    /// Linearly interpolates between two integer-valued points.
    ///
    /// The interpolation is performed with 128-bit integers and use tie to even rounding.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(linear(0, 10, 10, 20, 5), 15);
    /// ```
    fn forward_int<T, U>(x: T, x0: T, x1: T, f0: U, f1: U) -> U
    where
        T: PrimInt + NumCast,
        U: PrimInt + NumCast,
    {
        let x: i128 = NumCast::from(x).unwrap();
        let x0: i128 = NumCast::from(x0).unwrap();
        let x1: i128 = NumCast::from(x1).unwrap();
        let f0: i128 = NumCast::from(f0).unwrap();
        let f1: i128 = NumCast::from(f1).unwrap();
        let out = roundiv(f0 * (x1 - x) + f1 * (x - x0), x1 - x0);
        let out: U = NumCast::from(out).unwrap();
        out
    }

    fn forward_float<T, U>(x: T, x0: T, x1: T, f0: U, f1: U) -> U
    where
        T: PrimInt + NumCast,
        U: Float + NumCast,
    {
        let n: U = NumCast::from(x1 - x).unwrap();
        let d: U = NumCast::from(x1 - x0).unwrap();
        let w0 = n / d;
        let w1: U = NumCast::from(1.0).unwrap();
        let w1 = w1 - w0;
        f0 * w0 + f1 * w1
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
        fn test_forward() {
            assert_eq!(forward_int(1, 0, 2, 0, 20), 10);
            assert_eq!(forward_float(1, 0, 2, 0.0, 20.0), 10.0);
        }

        #[test]
        fn test_interp_ndarray() {
            let xp = array![0, 2, 5];
            let fp = array![0, 20, 50];
            let x_values = array![-2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
            let result = interp_ndarray_int(&x_values.view(), &xp.view(), &fp.view(), -1, -1);
            let expected = array![-1, -1, 0, 10, 20, 30, 40, 50, -1, -1];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_interp_value() {
            // Define sample data points
            let xp = array![0, 1, 2, 3, 4, 5];
            let fp = array![0, 10, 20, 30, 40, 50];
            assert_eq!(interp_value_int(3, &xp.view(), &fp.view(), -1, -1), 30); // existing
            assert_eq!(interp_value_int(-1, &xp.view(), &fp.view(), -1, -1), -1); // before
            assert_eq!(interp_value_int(6, &xp.view(), &fp.view(), -1, -1), -1); // after
            assert_eq!(interp_value_int(2, &xp.view(), &fp.view(), -1, -1), 20); // between
            assert_eq!(interp_value_int(0, &xp.view(), &fp.view(), -1, -1), 0); // left edge
            assert_eq!(interp_value_int(5, &xp.view(), &fp.view(), -1, -1), 50);
            // righ edge
        }

        #[test]
        fn test_linear_index_int64() {
            let test_cases = [
                // x0, x1, f0, f1, x, expected
                (0, 10, 10, 20, 5, 15),
                (0, 10, 20, 30, 5, 25),
                (0, 100, 0, 200, 50, 100),
            ];
            for (x0, x1, f0, f1, x, expected) in test_cases {
                let result = forward_int(x, x0, x1, f0, f1);
                assert_eq!(result, expected);
            }
        }
        #[test]
        fn test_linear_index_float64() {
            let test_cases = [
                // x0, x1, f0, f1, x, expected
                (0, 10, 10.0, 20.0, 5, 15.0),
                (0, 10, 20.0, 30.0, 5, 25.0),
                (0, 100, 0.0, 200.0, 50, 100.0),
            ];
            for (x0, x1, f0, f1, x, expected) in test_cases {
                let result = forward_float(x, x0, x1, f0, f1);
                assert_eq!(result, expected);
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
