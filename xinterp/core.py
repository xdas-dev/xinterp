import numpy as np

from . import rust


def interp_datetime64(x, xp, fp, left=None, right=None):
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
    dtype = fp.dtype
    if not np.issubdtype(dtype, np.datetime64):
        raise
    return interp_int64(x, xp, fp, left, right).astype(dtype)


def interp_int64(x, xp, fp, left=None, right=None):
    x = np.asarray(x, dtype="int64")
    xp = np.asarray(xp, dtype="int64")
    fp = np.asarray(fp, dtype="int64")
    if left is None:
        left = fp[0]
    else:
        left = np.int64(left)
    if right is None:
        right = fp[-1]
    else:
        right = np.int64(right)
    if not (x.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise
    if not (xp.shape == fp.shape):
        raise
    return rust.interp_int64(x, xp, fp, left, right)
