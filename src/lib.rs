pub mod divop;
pub mod interp;
pub mod schemes;

use crate::interp::{Interp, InterpError};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

#[pymodule]
fn rust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn forward<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<&'py PyArray1<i64>> {
        let x = x.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let mut f = Array1::zeros(x.len());

        for (index, value) in x.iter().zip(f.iter_mut()) {
            match interp.forward(*index) {
                Ok(result) => *value = result,
                Err(InterpError::NotStrictlyIncreasing) => {
                    return Err(PyIndexError::new_err("indices must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyIndexError::new_err("index out of bounds"))
                }
                Err(InterpError::NotFound) => {
                    return Err(PyIndexError::new_err("index note found"))
                }
            }
        }
        Ok(f.into_pyarray(py))
    }

    Ok(())
}
