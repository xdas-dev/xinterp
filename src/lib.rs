pub mod divop;
pub mod extended;
pub mod interp;
pub mod schemes;

use crate::interp::{Interp, InterpError};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
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
                    return Err(PyValueError::new_err("xp must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyValueError::new_err("x out of bounds"))
                }
                Err(InterpError::NotFound) => return Err(PyValueError::new_err("x not found")),
            }
        }
        Ok(f.into_pyarray(py))
    }
    #[pyfn(m)]
    fn inverse_exact<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<&'py PyArray1<u64>> {
        let f = f.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let mut x = Array1::zeros(f.len());
        for (value, index) in f.iter().zip(x.iter_mut()) {
            match interp.inverse_exact(*value) {
                Ok(result) => *index = result,
                Err(InterpError::NotStrictlyIncreasing) => {
                    return Err(PyValueError::new_err("fp must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyValueError::new_err("f out of bounds"))
                }
                Err(InterpError::NotFound) => return Err(PyValueError::new_err("f not found")),
            }
        }
        Ok(x.into_pyarray(py))
    }
    #[pyfn(m)]
    fn inverse_round<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<&'py PyArray1<u64>> {
        let f = f.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let mut x = Array1::zeros(f.len());
        for (value, index) in f.iter().zip(x.iter_mut()) {
            match interp.inverse_round(*value) {
                Ok(result) => *index = result,
                Err(InterpError::NotStrictlyIncreasing) => {
                    return Err(PyValueError::new_err("fp must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyValueError::new_err("f out of bounds"))
                }
                Err(InterpError::NotFound) => return Err(PyValueError::new_err("f not found")),
            }
        }
        Ok(x.into_pyarray(py))
    }
    #[pyfn(m)]
    fn inverse_ffill<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<&'py PyArray1<u64>> {
        let f = f.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let mut x = Array1::zeros(f.len());
        for (value, index) in f.iter().zip(x.iter_mut()) {
            match interp.inverse_ffill(*value) {
                Ok(result) => *index = result,
                Err(InterpError::NotStrictlyIncreasing) => {
                    return Err(PyValueError::new_err("fp must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyValueError::new_err("f out of bounds"))
                }
                Err(InterpError::NotFound) => return Err(PyValueError::new_err("f not found")),
            }
        }
        Ok(x.into_pyarray(py))
    }
    #[pyfn(m)]
    fn inverse_bfill<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<&'py PyArray1<u64>> {
        let f = f.as_array();
        let xp = xp.as_array();
        let fp = fp.as_array();
        let interp = Interp::new(xp.to_vec(), fp.to_vec());
        let mut x = Array1::zeros(f.len());
        for (value, index) in f.iter().zip(x.iter_mut()) {
            match interp.inverse_bfill(*value) {
                Ok(result) => *index = result,
                Err(InterpError::NotStrictlyIncreasing) => {
                    return Err(PyValueError::new_err("fp must be strictly increasing"))
                }
                Err(InterpError::OutOfBounds) => {
                    return Err(PyValueError::new_err("f out of bounds"))
                }
                Err(InterpError::NotFound) => return Err(PyValueError::new_err("f not found")),
            }
        }
        Ok(x.into_pyarray(py))
    }
    Ok(())
}
