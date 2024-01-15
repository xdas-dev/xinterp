import numpy as np

from . import rust


def interp_datetime64(x, xp, fp):
    dtype = fp.dtype
    if not np.issubdtype(dtype, np.datetime64):
        raise
    return interp_int64(x, xp, fp).astype(dtype)


def interp_int64(x, xp, fp):
    x = np.asarray(x, dtype="int64")
    xp = np.asarray(xp, dtype="int64")
    fp = np.asarray(fp, dtype="int64")
    if not (x.ndim == 1 and xp.ndim == 1 and fp.ndim == 1):
        raise
    if not (xp.shape == fp.shape):
        raise
    return rust.interp_int64(x, xp, fp)
