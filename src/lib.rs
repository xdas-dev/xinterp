use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
        let f = interp_ndarray(x, xp, fp, left, right);
        f.into_pyarray(py)
    }

    Ok(())
}

mod interp {
    use numpy::ndarray::{Array1, ArrayView1};

    pub fn interp_ndarray(
        x: ArrayView1<i64>,
        xp: ArrayView1<i64>,
        fp: ArrayView1<i64>,
        left: i64,
        right: i64,
    ) -> Array1<i64> {
        x.map(|x| interp_value(*x, xp, fp, left, right))
    }

    fn interp_value(
        x: i64,
        xp: ArrayView1<i64>,
        fp: ArrayView1<i64>,
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

    fn linear(x0: i64, x1: i64, f0: i64, f1: i64, x: i64) -> i64 {
        let x0 = i128::from(x0);
        let x1 = i128::from(x1);
        let f0 = i128::from(f0);
        let f1 = i128::from(f1);
        let x = i128::from(x);
        let out = roundiv(f0 * (x1 - x) + f1 * (x - x0), x1 - x0);
        i64::try_from(out).expect("cannot convert to i64")
    }

    fn roundiv(n: i128, d: i128) -> i128 {
        let s = n.signum() * d.signum();
        let (n, d) = (n.abs(), d.abs());
        let q = n / d;
        let r = n % d;
        if (r * 2) == d {
            if q % 2 == 0 {
                return s * q;
            } else {
                return s * (q + 1);
            }
        } else if (r * 2) > d {
            return s * (q + 1);
        } else {
            return s * q;
        }
    }
}
