import warnings

import numpy as np

from . import rust


def forward_points(x, xp, fp):
    """
    One-dimensional linear interpolation from indices to values.

    The knots are given explicitly, as points from which each piece implies its own
    slope. See :func:`forward_step` for the constant-rate twin, where the knots are
    generated at a fixed rate instead.

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


def inverse_points(f, xp, fp, method=None):
    """
    One-dimensional linear interpolation from values to indices.

    The knots are given explicitly, as points from which each piece implies its own
    slope. See :func:`inverse_step` for the constant-rate twin, where the knots are
    generated at a fixed rate instead.

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


def forward(x, xp, fp):
    """Deprecated alias for :func:`forward_points`.

    .. deprecated:: 0.2.1
        Use :func:`forward_points` instead. Scheduled for removal in 0.5.
    """
    warnings.warn(
        "forward is deprecated, use forward_points instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return forward_points(x, xp, fp)


def inverse(f, xp, fp, method=None):
    """Deprecated alias for :func:`inverse_points`.

    .. deprecated:: 0.2.1
        Use :func:`inverse_points` instead. Scheduled for removal in 0.5.
    """
    warnings.warn(
        "inverse is deprecated, use inverse_points instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return inverse_points(f, xp, fp, method)


def wraps(func_int, func_float, func_uint):
    def func(xp, fp, *, x=None, f=None, **kwargs):
        xp, fp, x, f, isscalar = check(xp, fp, x, f)
        if np.issubdtype(fp.dtype, np.unsignedinteger):
            if x is not None:
                out = func_uint(
                    x.astype("u8"), xp.astype("u8"), fp.astype("u8"), **kwargs
                ).astype(fp.dtype)
            elif f is not None:
                out = func_uint(
                    f.astype("u8"), xp.astype("u8"), fp.astype("u8"), **kwargs
                ).astype(xp.dtype)
        elif np.issubdtype(fp.dtype, np.integer) or np.issubdtype(
            fp.dtype, np.datetime64
        ):
            if x is not None:
                out = func_int(
                    x.astype("u8"), xp.astype("u8"), fp.astype("i8"), **kwargs
                ).astype(fp.dtype)
            elif f is not None:
                out = func_int(
                    f.astype("i8"), xp.astype("u8"), fp.astype("i8"), **kwargs
                ).astype(xp.dtype)
        elif np.issubdtype(fp.dtype, np.floating):
            if x is not None:
                out = func_float(
                    x.astype("u8"), xp.astype("u8"), fp.astype("f8"), **kwargs
                ).astype(fp.dtype)
            elif f is not None:
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
    if not (len(xp) > 0 and len(fp) > 0):
        raise ValueError("xp and fp must have at least one elements")
    if not np.issubdtype(xp.dtype, np.integer):
        raise ValueError("xp must have integer dtype")
    if not np.all(xp >= 0):
        raise ValueError("xp values must be positive")
    if not np.all(np.isfinite(fp)):
        raise ValueError("fp values must be finite")
    if (x is None) == (f is None):
        raise ValueError("either x or f must be provided")
    if x is not None:
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.datetime64):
            x = x.astype("i8")
        if x.ndim == 0:
            x = x.reshape(1)
            isscalar = True
        elif x.ndim == 1:
            isscalar = False
        else:
            raise ValueError("x must be 1D or scalar")
        if np.issubdtype(x.dtype, np.floating) and not np.all(x == np.floor(x)):
            raise ValueError("x values must be integral")
        if not np.all(x >= 0):
            raise ValueError("x values must be positive")
        if not np.all(xp[1:] > xp[:-1]):
            raise ValueError("xp must be strictly increasing")
    if f is not None:
        f_orig = np.asarray(f)
        if np.issubdtype(fp.dtype, np.unsignedinteger) and np.issubdtype(
            f_orig.dtype, np.signedinteger
        ):
            if not np.all(f_orig >= 0):
                raise ValueError("f values must be positive")
        f = f_orig.astype(fp.dtype)
        if np.issubdtype(f_orig.dtype, np.floating):
            lossless = np.array_equal(f_orig, f.astype(f_orig.dtype), equal_nan=True)
        else:
            lossless = np.array_equal(f_orig, f.astype(f_orig.dtype))
        if not lossless:
            raise ValueError("f values must fit fp's dtype without loss")
        if f.ndim == 0:
            f = f.reshape(1)
            isscalar = True
        elif f.ndim == 1:
            isscalar = False
        else:
            raise ValueError("f must be 1D or scalar")
        if not np.all(np.isfinite(f)):
            raise ValueError("f values must be finite")
        if not np.all(fp[1:] > fp[:-1]):
            raise ValueError("fp must be strictly increasing")
    return xp, fp, x, f, isscalar


_forward = wraps(rust.forward_int, rust.forward_float, rust.forward_uint)
_inverse = wraps(rust.inverse_int, rust.inverse_float, rust.inverse_uint)


def _check_points(x, f):
    x = np.asarray(x)
    f = np.asarray(f)
    if not (x.ndim == 1 and f.ndim == 1):
        raise ValueError("x and f must be 1D")
    if not len(x) == len(f):
        raise ValueError("x and f must have the same length")
    if not len(x) > 0:
        raise ValueError("x and f must have at least one element")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("x must have integer dtype")
    if not np.all(x >= 0):
        raise ValueError("x values must be positive")
    if not np.all(x[1:] > x[:-1]):
        raise ValueError("x must be strictly increasing")
    return x, f


def simplify_points(x, f, en, ed):
    """
    Drop tie points already described, to within `en / ed`, by their surviving neighbours.

    A one-pass greedy sleeve: from the current anchor it maintains the intersection of
    every dropped point's slope cone, and emits a knot exactly when a candidate leaves
    it. Kept points are original points, never moved. See :func:`simplify_step` for the
    constant-rate twin, where the slope is fixed rather than free.

    Parameters
    ----------
    x : 1-D sequence of positive integers
        The tie indices, must be strictly increasing.
    f : 1-D sequence of floats, integers or datetime64s
        The tie values, same length as `x`.
    en, ed : numbers
        The tolerance, as an exact ratio `en / ed` (`ed` strictly positive). For
        integers and datetimes, `en`/`ed` are counted in ticks; this function has no
        opinion on what the tolerance means, or on the half-tick slack a caller may
        want to fold in.

    Returns
    -------
    1-D boolean array
        Which of `x`/`f` survive, same length as `x`.
    """
    x, f = _check_points(x, f)
    if np.issubdtype(f.dtype, np.integer) or np.issubdtype(f.dtype, np.datetime64):
        return rust.simplify_points_int(
            x.astype("u8"), f.astype("i8"), int(en), int(ed)
        )
    elif np.issubdtype(f.dtype, np.floating):
        return rust.simplify_points_float(
            x.astype("u8"), f.astype("f8"), float(en), float(ed)
        )
    else:
        raise ValueError("f dtype must be either integer, floating or datetime")


def simplify_step(tie_values, tie_lengths, num, den, tol):
    """
    Fuse consecutive segments whose declared step agrees, within `tol`.

    Segment `i` starts at `tie_values[i]` and spans `tie_lengths[i]` index ticks, at
    rate `num[i] / den[i]` (or the single shared rate, when `num`/`den` have length 1).
    One-pass greedy walk: fuses while the run's steps agree and the spread of its
    junction offsets stays within `2 * tol`, then re-anchors the run's tie value to the
    Chebyshev centre of those offsets -- so a surviving value may move by up to `tol`.

    Parameters
    ----------
    tie_values : 1-D sequence of integers
        Start value of each segment.
    tie_lengths : 1-D sequence of positive integers
        Number of samples in each segment, same length as `tie_values`.
    num : 1-D sequence of integers
        Step numerator, length 1 (shared) or `len(tie_values)`.
    den : 1-D sequence of positive integers
        Step denominator, same length as `num`.
    tol : integer
        The tolerance budget, in tie-value ticks.

    Returns
    -------
    keep : 1-D boolean array
        Which of `tie_values` start a surviving run, same length as `tie_values`.
    fused : 1-D integer array
        The re-anchored tie value of each surviving run, length `keep.sum()`.
    """
    tie_values = np.asarray(tie_values)
    tie_lengths = np.asarray(tie_lengths)
    num = np.atleast_1d(np.asarray(num))
    den = np.atleast_1d(np.asarray(den))
    if not (tie_values.ndim == 1 and tie_lengths.ndim == 1):
        raise ValueError("tie_values and tie_lengths must be 1D")
    if not len(tie_values) == len(tie_lengths):
        raise ValueError("tie_values and tie_lengths must have the same length")
    if not len(tie_values) > 0:
        raise ValueError("tie_values and tie_lengths must have at least one element")
    if not (num.ndim == 1 and den.ndim == 1):
        raise ValueError("num and den must be 1D")
    if not len(num) == len(den):
        raise ValueError("num and den must have the same length")
    if not (len(num) == 1 or len(num) == len(tie_values)):
        raise ValueError("num and den must have length 1 or len(tie_values)")
    if not np.all(den > 0):
        raise ValueError("den values must be positive")
    keep, fused = rust.simplify_step(
        tie_values.astype("i8"),
        tie_lengths.astype("u8"),
        num.astype("i8"),
        den.astype("u8"),
        int(tol),
    )
    return keep, fused


def infer_step(x, f):
    """
    The single step (`num`, `den`) best describing every consecutive segment of `(x, f)`.

    The length-weighted Chebyshev centre of the per-segment rates, in exact integers,
    gcd-reduced (D2: an irreducible fraction is the canonical, comparable form). Returns
    the worst per-segment absolute deviation from it alongside.

    Parameters
    ----------
    x : 1-D sequence of positive integers
        The tie indices, strictly increasing, at least two.
    f : 1-D sequence of integers or datetime64s
        The tie values, same length as `x`.

    Returns
    -------
    num : int
    den : int
    worst_deviation : int
    """
    x, f = _check_points(x, f)
    if len(x) < 2:
        raise ValueError("infer_step needs at least two tie points")
    if not (
        np.issubdtype(f.dtype, np.integer) or np.issubdtype(f.dtype, np.datetime64)
    ):
        raise ValueError("f dtype must be either integer or datetime")
    return rust.infer_step(x.astype("u8"), f.astype("i8"))
