pub mod divop;
pub mod extended;
pub mod piecewise;
pub mod schemes;

use crate::divop::Method;
use crate::piecewise::{InterpError, Points};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::{pyfunction, pymodule, Bound, PyResult, Python};

/// Parse the Python-facing method name into a rounding mode.
fn parse_method(method: Option<&str>) -> PyResult<Method> {
    match method {
        None => Ok(Method::None),
        Some("nearest") => Ok(Method::Nearest),
        Some("ffill") => Ok(Method::ForwardFill),
        Some("bfill") => Ok(Method::BackwardFill),
        Some(_) => Err(PyValueError::new_err(
            "method must be either None, 'nearest', 'ffill' or 'bfill'",
        )),
    }
}

#[pymodule]
mod rust {
    use super::*;

    #[pyfunction]
    fn forward_int<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let x = x.as_slice().expect("x must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let points = Points::new(xp, fp);
        let f = py.detach(|| -> PyResult<Array1<i64>> {
            let mut f = Array1::zeros(x.len());
            for (index, value) in x.iter().zip(f.iter_mut()) {
                match points.forward(*index) {
                    Ok(result) => *value = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("xp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyIndexError::new_err("x out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyIndexError::new_err("x not found")),
                }
            }
            Ok(f)
        })?;
        Ok(f.into_pyarray(py))
    }

    #[pyfunction]
    fn forward_uint<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let x = x.as_slice().expect("x must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let points = Points::new(xp, fp);
        let f = py.detach(|| -> PyResult<Array1<u64>> {
            let mut f = Array1::zeros(x.len());
            for (index, value) in x.iter().zip(f.iter_mut()) {
                match points.forward(*index) {
                    Ok(result) => *value = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("xp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyIndexError::new_err("x out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyIndexError::new_err("x not found")),
                }
            }
            Ok(f)
        })?;
        Ok(f.into_pyarray(py))
    }

    #[pyfunction]
    fn forward_float<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_slice().expect("x must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let points = Points::new(xp, fp);
        let f = py.detach(|| -> PyResult<Array1<f64>> {
            let mut f = Array1::zeros(x.len());
            for (index, value) in x.iter().zip(f.iter_mut()) {
                match points.forward(*index) {
                    Ok(result) => *value = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("xp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyIndexError::new_err("x out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyIndexError::new_err("x not found")),
                }
            }
            Ok(f)
        })?;
        Ok(f.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(signature = (f, xp, fp, method=None))]
    fn inverse_int<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, i64>,
        method: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let f = f.as_slice().expect("f must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let method = parse_method(method)?;
        let points = Points::new(xp, fp);
        let x = py.detach(|| -> PyResult<Array1<u64>> {
            let mut x = Array1::zeros(f.len());
            for (value, index) in f.iter().zip(x.iter_mut()) {
                match points.inverse(*value, method) {
                    Ok(result) => *index = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("fp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyKeyError::new_err("f out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyKeyError::new_err("f not found")),
                }
            }
            Ok(x)
        })?;
        Ok(x.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(signature = (f, xp, fp, method=None))]
    fn inverse_float<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, f64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, f64>,
        method: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let f = f.as_slice().expect("f must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let method = parse_method(method)?;
        let points = Points::new(xp, fp);
        let x = py.detach(|| -> PyResult<Array1<u64>> {
            let mut x = Array1::zeros(f.len());
            for (value, index) in f.iter().zip(x.iter_mut()) {
                match points.inverse(*value, method) {
                    Ok(result) => *index = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("fp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyKeyError::new_err("f out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyKeyError::new_err("f not found")),
                }
            }
            Ok(x)
        })?;
        Ok(x.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(signature = (f, xp, fp, method=None))]
    fn inverse_uint<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, u64>,
        xp: PyReadonlyArray1<'py, u64>,
        fp: PyReadonlyArray1<'py, u64>,
        method: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let f = f.as_slice().expect("f must be contiguous");
        let xp = xp.as_slice().expect("xp must be contiguous");
        let fp = fp.as_slice().expect("fp must be contiguous");
        let method = parse_method(method)?;
        let points = Points::new(xp, fp);
        let x = py.detach(|| -> PyResult<Array1<u64>> {
            let mut x = Array1::zeros(f.len());
            for (value, index) in f.iter().zip(x.iter_mut()) {
                match points.inverse(*value, method) {
                    Ok(result) => *index = result,
                    Err(InterpError::NotStrictlyIncreasing) => {
                        return Err(PyValueError::new_err("fp must be strictly increasing"))
                    }
                    Err(InterpError::OutOfBounds) => {
                        return Err(PyKeyError::new_err("f out of bounds"))
                    }
                    Err(InterpError::NotFound) => return Err(PyKeyError::new_err("f not found")),
                }
            }
            Ok(x)
        })?;
        Ok(x.into_pyarray(py))
    }
}
