import numpy as np

from . import rust


def forward(x, xp, fp):
    """
    One-dimensional linear interpolation for monotonically increasing

    Parameters
    ----------
    x : 1-D sequence of integers
        The indices at which to evaluate the interpolated values.
    xp : 1-D sequence of integers
        The indices of the data points, must be increasing.
    fp : 1-D sequence of datetime64
        The values of the data points, same length as 'xp'.
    left : datetime64, optional
        Value to return for 'x < xp[0]', by default 'fp[0]'.
    right : float, optional
        Value to return for 'x > xp[-1]', by default 'fp[-1]'

    Returns
    -------
    1-D array of datetime64
        The interpolated values, same length as x.x
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if not (x.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise ValueError("all inputs must be 1D arrays")
    if not (xp.shape == fp.shape):
        raise ValueError("xp and fp must have the same shape")
    if not x.dtype == xp.dtype:
        raise ValueError("x and xp must have the same dtype")
    if not np.all(xp > 0):
        raise ValueError("xp values must be positive")
    if not np.all(np.diff(xp) > 0):
        raise ValueError("xp must be strictly increasing")
    return rust.forward(x.astype("u8"), xp.astype("u8"), fp.astype("i8")).astype(
        fp.dtype
    )


def inverse(f, xp, fp, method=None):
    f = np.asarray(f)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if not (f.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise ValueError("all inputs must be 1D arrays")
    if not (xp.shape == fp.shape):
        raise ValueError("xp and fp must have the same shape")
    if not f.dtype == fp.dtype:
        raise ValueError("f and fp must have the same dtype")
    if not np.all(xp > 0):
        raise ValueError("xp values must be positive")
    if not np.all(np.diff(fp) > 0):
        raise ValueError("fp must be strictly increasing")
    if method is None:
        return rust.inverse_exact(
            f.astype("i8"), xp.astype("u8"), fp.astype("i8")
        ).astype(xp.dtype)
    elif method is "nearest":
        return rust.inverse_round(
            f.astype("i8"), xp.astype("u8"), fp.astype("i8")
        ).astype(xp.dtype)
    elif method is "ffill":
        return rust.inverse_ffill(
            f.astype("i8"), xp.astype("u8"), fp.astype("i8")
        ).astype(xp.dtype)
    elif method is "bfill":
        return rust.inverse_bfill(
            f.astype("i8"), xp.astype("u8"), fp.astype("i8")
        ).astype(xp.dtype)
    else:
        raise ValueError("method must be in [None, 'nearest', 'ffill', 'bfill']")
