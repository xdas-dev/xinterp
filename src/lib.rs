pub mod divop;
pub mod extended;
pub mod points;
pub mod schemes;
pub mod simplify;
pub mod step;
pub mod wide;

use crate::divop::Method;
use crate::points::{InterpError, Points};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
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

    #[pyfunction]
    fn simplify_points_int<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        f: PyReadonlyArray1<'py, i64>,
        en: i64,
        ed: i64,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let x = x.as_slice().expect("x must be contiguous");
        let f = f.as_slice().expect("f must be contiguous");
        let keep = py.detach(|| crate::simplify::simplify_points_int(x, f, en, ed));
        Ok(Array1::from_vec(keep).into_pyarray(py))
    }

    #[pyfunction]
    fn simplify_points_float<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        f: PyReadonlyArray1<'py, f64>,
        en: f64,
        ed: f64,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let x = x.as_slice().expect("x must be contiguous");
        let f = f.as_slice().expect("f must be contiguous");
        let keep = py.detach(|| crate::simplify::simplify_points_float(x, f, en, ed));
        Ok(Array1::from_vec(keep).into_pyarray(py))
    }

    type SimplifyStepResult<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>);

    #[pyfunction]
    fn simplify_step<'py>(
        py: Python<'py>,
        tie_values: PyReadonlyArray1<'py, i64>,
        tie_lengths: PyReadonlyArray1<'py, u64>,
        num: PyReadonlyArray1<'py, i64>,
        den: PyReadonlyArray1<'py, u64>,
        tol: i64,
    ) -> PyResult<SimplifyStepResult<'py>> {
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let tie_lengths = tie_lengths
            .as_slice()
            .expect("tie_lengths must be contiguous");
        let num = num.as_slice().expect("num must be contiguous");
        let den = den.as_slice().expect("den must be contiguous");
        if !(num.len() == 1 || num.len() == tie_values.len()) || num.len() != den.len() {
            return Err(PyValueError::new_err(
                "num and den must have the same length, either 1 or len(tie_values)",
            ));
        }
        let (keep, fused) =
            py.detach(|| crate::simplify::simplify_step(tie_values, tie_lengths, num, den, tol));
        Ok((
            Array1::from_vec(keep).into_pyarray(py),
            Array1::from_vec(fused).into_pyarray(py),
        ))
    }

    #[pyfunction]
    fn infer_step<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        f: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(i64, u64, i64)> {
        let x = x.as_slice().expect("x must be contiguous");
        let f = f.as_slice().expect("f must be contiguous");
        if x.len() != f.len() {
            return Err(PyValueError::new_err("x and f must have the same length"));
        }
        if x.len() < 2 {
            return Err(PyValueError::new_err(
                "infer_step needs at least two tie points",
            ));
        }
        Ok(py.detach(|| crate::step::infer(x, f)))
    }

    /// Validates that `num`/`den` have length 1 or `n_segments`, and that `den` is nonzero.
    fn check_rate<'py>(
        num: &PyReadonlyArray1<'py, i64>,
        den: &PyReadonlyArray1<'py, u64>,
        n_segments: usize,
    ) -> PyResult<()> {
        if num.len() != den.len() {
            return Err(PyValueError::new_err(
                "num and den must have the same length",
            ));
        }
        if !(num.len() == 1 || num.len() == n_segments) {
            return Err(PyValueError::new_err(
                "num/den must have length 1 or len(tie_indices) - 1",
            ));
        }
        if den.as_array().iter().any(|&d| d == 0) {
            return Err(PyValueError::new_err("den values must be positive"));
        }
        Ok(())
    }

    /// Error mapping for `forward_step`, matching `forward_int`/`forward_float`'s convention.
    fn forward_step_error(err: InterpError) -> pyo3::PyErr {
        match err {
            InterpError::NotStrictlyIncreasing => {
                PyValueError::new_err("tie_indices must be strictly increasing")
            }
            InterpError::OutOfBounds => PyIndexError::new_err("x out of bounds"),
            InterpError::NotFound => PyIndexError::new_err("x not found"),
        }
    }

    /// Error mapping for `inverse_step`, matching `inverse_int`/`inverse_float`'s convention.
    fn inverse_step_error(err: InterpError) -> pyo3::PyErr {
        match err {
            InterpError::NotStrictlyIncreasing => {
                PyValueError::new_err("tie_values must be strictly monotonic")
            }
            InterpError::OutOfBounds => PyKeyError::new_err("f out of bounds"),
            InterpError::NotFound => PyKeyError::new_err("f not found"),
        }
    }

    #[pyfunction]
    fn forward_step_int<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        tie_indices: PyReadonlyArray1<'py, u64>,
        tie_values: PyReadonlyArray1<'py, i64>,
        num: PyReadonlyArray1<'py, i64>,
        den: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        check_rate(&num, &den, tie_indices.len().saturating_sub(1))?;
        let x = x.as_slice().expect("x must be contiguous");
        let tie_indices = tie_indices
            .as_slice()
            .expect("tie_indices must be contiguous");
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let num = num.as_slice().expect("num must be contiguous");
        let den = den.as_slice().expect("den must be contiguous");
        let series = crate::step::StepSeries::new(tie_indices, tie_values, num, den);
        let f = py.detach(|| -> PyResult<Array1<i64>> {
            let mut f = Array1::zeros(x.len());
            for (index, value) in x.iter().zip(f.iter_mut()) {
                *value = series.forward(*index).map_err(forward_step_error)?;
            }
            Ok(f)
        })?;
        Ok(f.into_pyarray(py))
    }

    #[pyfunction]
    fn forward_step_float<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u64>,
        tie_indices: PyReadonlyArray1<'py, u64>,
        tie_values: PyReadonlyArray1<'py, f64>,
        delta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_segments = tie_indices.len().saturating_sub(1);
        if !(delta.len() == 1 || delta.len() == n_segments) {
            return Err(PyValueError::new_err(
                "delta must have length 1 or len(tie_indices) - 1",
            ));
        }
        let x = x.as_slice().expect("x must be contiguous");
        let tie_indices = tie_indices
            .as_slice()
            .expect("tie_indices must be contiguous");
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let delta = delta.as_slice().expect("delta must be contiguous");
        let series = crate::step::FloatStepSeries::new(tie_indices, tie_values, delta);
        let f = py.detach(|| -> PyResult<Array1<f64>> {
            let mut f = Array1::zeros(x.len());
            for (index, value) in x.iter().zip(f.iter_mut()) {
                *value = series.forward(*index).map_err(forward_step_error)?;
            }
            Ok(f)
        })?;
        Ok(f.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(signature = (f, tie_indices, tie_values, num, den, method=None))]
    fn inverse_step_int<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, i64>,
        tie_indices: PyReadonlyArray1<'py, u64>,
        tie_values: PyReadonlyArray1<'py, i64>,
        num: PyReadonlyArray1<'py, i64>,
        den: PyReadonlyArray1<'py, u64>,
        method: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        check_rate(&num, &den, tie_indices.len().saturating_sub(1))?;
        let method = parse_method(method)?;
        let f = f.as_slice().expect("f must be contiguous");
        let tie_indices = tie_indices
            .as_slice()
            .expect("tie_indices must be contiguous");
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let num = num.as_slice().expect("num must be contiguous");
        let den = den.as_slice().expect("den must be contiguous");
        let series = crate::step::StepSeries::new(tie_indices, tie_values, num, den);
        let x = py.detach(|| -> PyResult<Array1<u64>> {
            let mut x = Array1::zeros(f.len());
            for (value, index) in f.iter().zip(x.iter_mut()) {
                *index = series.inverse(*value, method).map_err(inverse_step_error)?;
            }
            Ok(x)
        })?;
        Ok(x.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(signature = (f, tie_indices, tie_values, delta, method=None))]
    fn inverse_step_float<'py>(
        py: Python<'py>,
        f: PyReadonlyArray1<'py, f64>,
        tie_indices: PyReadonlyArray1<'py, u64>,
        tie_values: PyReadonlyArray1<'py, f64>,
        delta: PyReadonlyArray1<'py, f64>,
        method: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n_segments = tie_indices.len().saturating_sub(1);
        if !(delta.len() == 1 || delta.len() == n_segments) {
            return Err(PyValueError::new_err(
                "delta must have length 1 or len(tie_indices) - 1",
            ));
        }
        let method = parse_method(method)?;
        let f = f.as_slice().expect("f must be contiguous");
        let tie_indices = tie_indices
            .as_slice()
            .expect("tie_indices must be contiguous");
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let delta = delta.as_slice().expect("delta must be contiguous");
        let series = crate::step::FloatStepSeries::new(tie_indices, tie_values, delta);
        let x = py.detach(|| -> PyResult<Array1<u64>> {
            let mut x = Array1::zeros(f.len());
            for (value, index) in f.iter().zip(x.iter_mut()) {
                *index = series.inverse(*value, method).map_err(inverse_step_error)?;
            }
            Ok(x)
        })?;
        Ok(x.into_pyarray(py))
    }

    #[pyfunction]
    fn deviation_step<'py>(
        py: Python<'py>,
        tie_indices: PyReadonlyArray1<'py, u64>,
        tie_values: PyReadonlyArray1<'py, i64>,
        num: PyReadonlyArray1<'py, i64>,
        den: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        check_rate(&num, &den, tie_indices.len().saturating_sub(1))?;
        let tie_indices = tie_indices
            .as_slice()
            .expect("tie_indices must be contiguous");
        let tie_values = tie_values
            .as_slice()
            .expect("tie_values must be contiguous");
        let num = num.as_slice().expect("num must be contiguous");
        let den = den.as_slice().expect("den must be contiguous");
        let series = crate::step::StepSeries::new(tie_indices, tie_values, num, den);
        let residual = py.detach(|| series.deviation());
        Ok(Array1::from_vec(residual).into_pyarray(py))
    }
}
