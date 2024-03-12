import numpy as np

from . import rust


def forward_intlike(x, xp, fp):
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
    if not x.dtype == xp.dtype:
        raise ValueError("x and xp must have the same dtype")
    return forward(x, xp, fp).astype(fp.dtype)


def forward(x, xp, fp):
    x = np.asarray(x, dtype="u8")
    xp = np.asarray(xp, dtype="u8")
    fp = np.asarray(fp, dtype="i8")
    if not np.all(np.diff(xp) > 0):
        raise ValueError("xp must be strictly increasing")
    if not (x.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise
    if not (xp.shape == fp.shape):
        raise
    return rust.forward(x, xp, fp)
