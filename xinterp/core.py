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
    if not x.dtype == xp.dtype:
        raise ValueError("x and xp must have the same dtype")
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if not np.all(np.diff(xp) > 0):
        raise ValueError("xp must be strictly increasing")
    if not (x.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise
    if not (xp.shape == fp.shape):
        raise
    return rust.forward(x.astype("u8"), xp.astype("u8"), fp.astype("i8")).astype(
        fp.dtype
    )
