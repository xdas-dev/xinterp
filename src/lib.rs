pub mod divop;
pub mod interp;
pub mod transform;

use crate::interp::Interp;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pymodule]
fn rust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn forward<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> &'py PyArray1<i64> {
        let x = x.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let f = x.map(|x| interp.forward(*x).unwrap());
        f.into_pyarray(py)
    }

    Ok(())
}
