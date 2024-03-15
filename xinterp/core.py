import numpy as np

from . import rust


def forward(x, xp, fp):
    """
    One-dimensional linear interpolation from indices to values.

    Parameters
    ----------
    x : 1-D sequence of positive integers
        The indices at which to evaluate the interpolated values.
    xp : 1-D sequence of positive integers
        The indices of the data points, must be strictly increasing.
    fp : 1-D sequence of floats, integers or datetime64s
        The values of the data points, same length as `xp`.

    Returns
    -------
    1-D array of floats, integers or datetime64s.
        The interpolated values, same length as `x`.
    """
    return _forward(xp, fp, x=x)


def inverse(f, xp, fp, method=None):
    """
    One-dimensional linear interpolation from values to indices.

    Parameters
    ----------
    f : 1-D sequence of floats, integers or datetime64s
        The values at which to evaluate the interpolated indices.
    xp : 1-D sequence of positive integers
        The indices of the data points, same length as `fp`.
    fp : 1-D sequence of floats, integers or datetime64s
        The values of the data points, must be strictly increasing.
    method : str or None, optional
        The method to use for inexact mathces:
        - None (default): only exact matches, raises otherwise
        - "nearest": nearest matches
        - "ffill": propagate previous index forward
        - "bfill": propagate next index backward

    Returns
    -------
    1-D array of positive integers.
        The interpolated indices, same length as `f`.
    """
    if method is None:
        return _inverse_exact(xp, fp, f=f)
    elif method is "nearest":
        return _inverse_round(xp, fp, f=f)
    elif method is "ffill":
        return _inverse_ffill(xp, fp, f=f)
    elif method is "bfill":
        return _inverse_bfill(xp, fp, f=f)
    else:
        raise ValueError("method must be in [None, 'nearest', 'ffill', 'bfill']")


def wraps(func_int, func_float):
    def func(xp, fp, *, x=None, f=None):
        xp, fp, x, f = check(xp, fp, x, f)
        if np.issubdtype(fp.dtype, np.integer) or np.issubdtype(
            fp.dtype, np.datetime64
        ):
            if x is not None:
                return func_int(
                    x.astype("u8"), xp.astype("u8"), fp.astype("i8")
                ).astype(fp.dtype)
            if f is not None:
                return func_int(
                    f.astype("i8"), xp.astype("u8"), fp.astype("i8")
                ).astype(xp.dtype)
        elif np.issubdtype(fp.dtype, np.floating):
            if x is not None:
                out = func_float(
                    x.astype("u8"), xp.astype("u8"), fp.astype("f8")
                ).astype(fp.dtype)
            if f is not None:
                out = func_float(
                    f.astype("f8"), xp.astype("u8"), fp.astype("f8")
                ).astype(xp.dtype)
        else:
            raise ValueError("fp dtype must be either integer, floating or datetime")
        return out

    return func


def check(xp, fp, x=None, f=None):
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if not (xp.ndim == 1 and fp.ndim == 1):
        raise ValueError("xp and fp must be 1D")
    if not (xp.shape == fp.shape):
        raise ValueError("xp and fp must have the same shape")
    if not np.issubdtype(xp.dtype, np.integer):
        raise ValueError("xp must have integer dtype")
    if not np.all(xp >= 0):
        raise ValueError("xp values must be positive")
    if (x is None) == (f is None):
        raise ValueError("either x or f must be provided")
    if x is not None:
        x = np.asarray(x)
        if not (x.ndim == 1):
            raise ValueError("x must be 1D array")
        if not x.dtype == xp.dtype:
            raise ValueError("x and xp must have the same dtype")
        if not np.all(x >= 0):
            raise ValueError("x values must be positive")
        if not np.all(np.diff(xp) > 0):
            raise ValueError("xp must be strictly increasing")
    if f is not None:
        f = np.asarray(f)
        if not (f.ndim == 1):
            raise ValueError("f must be 1D array")
        if not f.dtype == fp.dtype:
            raise ValueError("f and fp must have the same dtype")
        if not np.all(np.diff(fp) > 0):
            raise ValueError("fp must be strictly increasing")
    return xp, fp, x, f


_forward = wraps(rust.forward_int, rust.forward_float)
_inverse_exact = wraps(rust.inverse_exact_int, rust.inverse_exact_float)
_inverse_round = wraps(rust.inverse_round_int, rust.inverse_round_float)
_inverse_ffill = wraps(rust.inverse_ffill_int, rust.inverse_ffill_float)
_inverse_bfill = wraps(rust.inverse_bfill_int, rust.inverse_bfill_float)
