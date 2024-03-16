import numpy as np

from . import rust


def forward(x, xp, fp):
    """
    One-dimensional linear interpolation from indices to values.

    Parameters
    ----------
    x : 1-D sequence or scalar of positive integers
        The indices at which to evaluate the interpolated values.
    xp : 1-D sequence of positive integers
        The indices of the data points, must be strictly increasing.
    fp : 1-D sequence of floats, integers or datetime64s
        The values of the data points, same length as `xp`.

    Returns
    -------
    1-D array or scalar of floats, integers or datetime64s.
        The interpolated values, same shape as `x`.

    Raises
    ------
    IndexError
        If any value of `x` is outside the `xp` range.
    """
    return _forward(xp, fp, x=x)


def inverse(f, xp, fp, method=None):
    """
    One-dimensional linear interpolation from values to indices.

    Parameters
    ----------
    f : 1-D sequence or scalar of floats, integers or datetime64s
        The values at which to evaluate the interpolated indices.
    xp : 1-D sequence of positive integers
        The indices of the data points, same length as `fp`.
    fp : 1-D sequence of floats, integers or datetime64s
        The values of the data points, must be strictly increasing.
    method : str or None, optional
        The method to use for inexact matches:
        - None (default): exact match, raises otherwise
        - "nearest": nearest match
        - "ffill": propagate previous index forward
        - "bfill": propagate next index backward

    Returns
    -------
    1-D array or scalar of positive integers.
        The interpolated indices, same shape as `f`.

    Raises
    ------
    KeyError
        If any value of `f` is outside the `fp` range.
    """
    return _inverse(xp, fp, f=f, method=method)


def wraps(func_int, func_float):
    def func(xp, fp, *, x=None, f=None, **kwargs):
        xp, fp, x, f, isscalar = check(xp, fp, x, f)
        if np.issubdtype(fp.dtype, np.integer) or np.issubdtype(
            fp.dtype, np.datetime64
        ):
            if x is not None:
                out = func_int(
                    x.astype("u8"), xp.astype("u8"), fp.astype("i8"), **kwargs
                ).astype(fp.dtype)
            if f is not None:
                out = func_int(
                    f.astype("i8"), xp.astype("u8"), fp.astype("i8"), **kwargs
                ).astype(xp.dtype)
        elif np.issubdtype(fp.dtype, np.floating):
            if x is not None:
                out = func_float(
                    x.astype("u8"), xp.astype("u8"), fp.astype("f8"), **kwargs
                ).astype(fp.dtype)
            if f is not None:
                out = func_float(
                    f.astype("f8"), xp.astype("u8"), fp.astype("f8"), **kwargs
                ).astype(xp.dtype)
        else:
            raise ValueError("fp dtype must be either integer, floating or datetime")
        if isscalar:
            return out[0]
        else:
            return out

    return func


def check(xp, fp, x=None, f=None):
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if not (xp.ndim == 1 and fp.ndim == 1):
        raise ValueError("xp and fp must be 1D")
    if not (len(xp) == len(fp)):
        raise ValueError("xp and fp must have the same length")
    if not (len(xp) > 1 and len(fp) > 1):
        raise ValueError("xp and fp must have at least two elements")
    if not np.issubdtype(xp.dtype, np.integer):
        raise ValueError("xp must have integer dtype")
    if not np.all(xp >= 0):
        raise ValueError("xp values must be positive")
    if not np.all(np.isfinite(fp)):
        raise ValueError("fp values must be finite")
    if (x is None) == (f is None):
        raise ValueError("either x or f must be provided")
    if x is not None:
        x = np.asarray(x).astype(xp.dtype)
        if x.ndim == 0:
            x = x.reshape(1)
            isscalar = True
        elif x.ndim == 1:
            isscalar = False
        else:
            raise ValueError("x must be 1D or scalar")
        if not x.dtype == xp.dtype:
            raise ValueError("x and xp must have the same dtype")
        if not np.all(x >= 0):
            raise ValueError("x values must be positive")
        if not np.all(xp[1:] > xp[:-1]):
            raise ValueError("xp must be strictly increasing")
    if f is not None:
        f = np.asarray(f).astype(fp.dtype)
        if f.ndim == 0:
            f = f.reshape(1)
            isscalar = True
        elif f.ndim == 1:
            isscalar = False
        else:
            raise ValueError("f must be 1D or scalar")
        if not f.dtype == fp.dtype:
            raise ValueError("f and fp must have the same dtype")
        if not np.all(np.isfinite(f)):
            raise ValueError("f values must be finite")
        if not np.all(fp[1:] > fp[:-1]):
            raise ValueError("fp must be strictly increasing")
    return xp, fp, x, f, isscalar


_forward = wraps(rust.forward_int, rust.forward_float)
_inverse = wraps(rust.inverse_int, rust.inverse_float)
