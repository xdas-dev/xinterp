import numpy as np
import pytest

import xinterp
from xinterp import forward_points as forward
from xinterp import inverse_points as inverse


class TestForward:
    def test_raises_not_1D(self):
        with pytest.raises(ValueError, match="x must be 1D or scalar"):
            forward([[1]], [0, 2], [3, 5])
        with pytest.raises(ValueError, match="xp and fp must be 1D"):
            forward([1], [[0, 2]], [3, 5])
        with pytest.raises(ValueError, match="xp and fp must be 1D"):
            forward([1], [0, 2], [[3, 5]])

    def test_raises_shape_mismatch(self):
        with pytest.raises(ValueError, match="xp and fp must have the same length"):
            forward([1], [0, 2, 5], [3, 5])

    def test_raises_at_no_element(self):
        with pytest.raises(
            ValueError, match="xp and fp must have at least one elements"
        ):
            forward([1], [], [])

    def test_raises_xp_not_integer(self):
        with pytest.raises(ValueError, match="xp must have integer dtype"):
            forward([1], [0.0, 2.0], [3, 5])

    def test_raises_not_positive(self):
        with pytest.raises(ValueError, match="xp values must be positive"):
            forward([1], [-1, 2], [3, 5])
        with pytest.raises(ValueError, match="x values must be positive"):
            forward([-1], [1, 2], [3, 5])

    def test_raises_not_finite(self):
        with pytest.raises(ValueError, match="fp values must be finite"):
            forward([1], [0, 2], [np.nan, np.nan])
        with pytest.raises(ValueError, match="fp values must be finite"):
            forward([1], [0, 2], [np.inf, np.inf])

    def test_dtype_matching(self):
        assert forward([1.0], [0, 2], [3, 5])[0] == 4
        assert forward([1], [0, 2], [3.0, 5.0])[0] == 4.0
        assert forward(np.array([1], dtype="M8[s]"), [0, 2], [3.0, 5.0])[0] == 4.0
        assert forward(np.array([1], dtype="M8[s]"), [0, 2], [3, 5])[0] == 4
        assert forward([1.0], [0, 2], np.array([3, 5], dtype="M8[s]"))[0] == np.array(
            4, dtype="M8[s]"
        )
        assert forward([1], [0, 2], np.array([3, 5], dtype="M8[s]"))[0] == np.array(
            4, dtype="M8[s]"
        )

    def test_raises_not_strictly_incresing(self):
        with pytest.raises(ValueError, match="xp must be strictly increasing"):
            forward([1], [2, 0], [3, 5])

    def test_raises_out_of_bounds(self):
        with pytest.raises(IndexError, match="x out of bounds"):
            forward([0], [1, 2], [3, 5])
        with pytest.raises(IndexError, match="x out of bounds"):
            forward([3], [1, 2], [3, 5])

    def test_type_handling(self):
        assert forward([1], [0, 2], [3, 5]) == 4
        assert forward([1], [0, 2], [3, 5]).dtype in ["i4", "i8"]
        assert forward([1], [0, 2], np.array([3, 5], "M8[s]")) == np.datetime64(4, "s")
        assert forward([1], [0, 2], np.array([3, 5], "M8[s]")).dtype == "M8[s]"
        assert forward([1], [0, 2], np.array([3, 5], "f4")) == 4
        assert forward([1], [0, 2], np.array([3, 5], "f4")).dtype == "f4"
        assert forward(np.array([1], "u2"), np.array([0, 2], "u2"), [3, 5])[0] == 4

    def test_scalar_handling(self):
        assert forward([1], [0, 2], [3, 5]).ndim == 1
        assert forward(1, [0, 2], [3, 5]).ndim == 0
        assert forward([1], [0, 2], np.array([3, 5], "M8[s]")).ndim == 1
        assert forward(1, [0, 2], np.array([3, 5], "M8[s]")).ndim == 0
        assert forward([1], [0, 2], np.array([3, 5], "f4")).ndim == 1
        assert forward(1, [0, 2], np.array([3, 5], "f4")).ndim == 0

    def test_empty_handling(self):
        assert forward([], [0, 2], [3, 5]).shape == (0,)
        assert forward([], [0, 2], np.array([3, 5], "M8[s]")).shape == (0,)
        assert forward([], [0, 2], np.array([3, 5], "f4")).shape == (0,)

    def test_singleton_handling(self):
        assert forward(0, [0], [3]) == 3
        assert forward(0, [0], [3.0]) == 3.0
        assert forward(0, [0], np.array([3], "M8[s]")) == np.array([3], "M8[s]")

    def test_interpolation_accuracy_int(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        result = forward(x, xp, fp)
        expected = np.rint(np.round(np.interp(x, xp, fp), 11)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_interpolation_accuracy_float(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = np.sort(rng.random(n))
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        result = forward(x, xp, fp)
        expected = np.interp(x, xp, fp)
        assert np.allclose(result, expected)
        assert result.dtype == expected.dtype


class TestInverse:
    def test_raises_not_1D(self):
        with pytest.raises(ValueError, match="f must be 1D or scalar"):
            inverse([[4]], [0, 2], [3, 5])
        with pytest.raises(ValueError, match="xp and fp must be 1D"):
            inverse([4], [[0, 2]], [3, 5])
        with pytest.raises(ValueError, match="xp and fp must be 1D"):
            inverse([4], [0, 2], [[3, 5]])

    def test_raises_shape_mismatch(self):
        with pytest.raises(ValueError, match="xp and fp must have the same length"):
            inverse([4], [0, 2, 5], [3, 5])

    def test_raises_no_element(self):
        with pytest.raises(
            ValueError, match="xp and fp must have at least one elements"
        ):
            inverse([4], [], [])

    def test_raises_xp_not_integer(self):
        with pytest.raises(ValueError, match="xp must have integer dtype"):
            inverse([4], [0.0, 2.0], [3, 5])

    def test_raises_not_positive(self):
        with pytest.raises(ValueError, match="xp values must be positive"):
            inverse([4], [-1, 2], [3, 5])

    def test_raises_not_finite(self):
        with pytest.raises(ValueError, match="fp values must be finite"):
            inverse([4.0], [0, 2], [np.nan, np.nan])
        with pytest.raises(ValueError, match="fp values must be finite"):
            inverse([4.0], [0, 2], [np.inf, np.inf])
        with pytest.raises(ValueError, match="f values must be finite"):
            inverse([np.nan], [0, 2], [3.0, 5.0])
        with pytest.raises(ValueError, match="f values must be finite"):
            inverse([np.inf], [0, 2], [3.0, 5.0])

    def test_dtype_matching(self):
        assert inverse([4.0], [0, 2], [3, 5])[0] == 1
        assert inverse([4], [0, 2], [3.0, 5.0])[0] == 1
        assert inverse(np.array([4], dtype="M8[s]"), [0, 2], [3.0, 5.0])[0] == 1
        assert inverse(np.array([4], dtype="M8[s]"), [0, 2], [3, 5])[0] == 1
        assert inverse([4.0], [0, 2], np.array([3, 5], dtype="M8[s]"))[0] == 1
        assert inverse([4], [0, 2], np.array([3, 5], dtype="M8[s]"))[0] == 1

    def test_raises_not_strictly_incresing(self):
        with pytest.raises(ValueError, match="fp must be strictly increasing"):
            inverse([4], [0, 2], [5, 3])

    def test_raises_out_of_bounds(self):
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([2], [1, 2], [3, 5])
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([6], [1, 2], [3, 5])
        assert inverse([2], [1, 2], [3, 5], method="nearest")[0] == 1
        assert inverse([6], [1, 2], [3, 5], method="nearest")[0] == 2
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([2], [1, 2], [3, 5], method="ffill")
        assert inverse([6], [1, 2], [3, 5], method="ffill")[0] == 2
        assert inverse([2], [1, 2], [3, 5], method="bfill")[0] == 1
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([6], [1, 2], [3, 5], method="bfill")

    def test_raises_not_found(self):
        assert inverse([5], [0, 2], [3, 7]) == 1
        with pytest.raises(KeyError, match="f not found"):
            inverse([4], [0, 2], [3, 7])
        assert inverse([5.0], [0, 2], [3.0, 7.0]) == 1
        with pytest.raises(KeyError, match="f not found"):
            inverse([4.0], [0, 2], [3.0, 7.0])
        with pytest.raises(KeyError, match="f not found"):
            inverse([5.5], [0, 2], [3.0, 7.0])
        assert inverse([5.0 + 1e-16], [0, 2], [3.0, 7.0])[0] == 1
        assert inverse([5.0 - 1e-16], [0, 2], [3.0, 7.0])[0] == 1

    def test_raises_wrong_method(self):
        with pytest.raises(
            ValueError,
            match="method must be either None, 'nearest', 'ffill' or 'bfill'",
        ):
            inverse([4], [0, 2], [3, 5], method="non_existing_method")

    def test_type_handling(self):
        assert inverse([4], [0, 2], [3, 5]) == 1
        assert inverse([4.0], [0, 2], [3.0, 5.0]) == 1
        assert inverse(np.array([4], "M8[s]"), [0, 2], np.array([3, 5], "M8[s]")) == 1
        assert inverse([4], np.array([0, 2], "u2"), [3, 5]) == 1
        assert inverse([4], np.array([0, 2], "u2"), [3, 5]).dtype == "u2"

    def test_scalar_handling(self):
        assert inverse([4], [0, 2], [3, 5]).ndim == 1
        assert inverse(4, [0, 2], [3, 5]).ndim == 0
        assert inverse([4.0], [0, 2], [3.0, 5.0]).ndim == 1
        assert inverse(4.0, [0, 2], [3.0, 5.0]).ndim == 0
        assert (
            inverse(np.array([4], "M8[s]"), [0, 2], np.array([3, 5], "M8[s]")).ndim == 1
        )
        assert (
            inverse(np.array(4, "M8[s]"), [0, 2], np.array([3, 5], "M8[s]")).ndim == 0
        )

    def test_empty_handling(self):
        assert inverse([], [0, 2], [3, 5]).shape == (0,)
        assert inverse([], [0, 2], [3.0, 5.0]).shape == (0,)
        assert inverse(
            np.array([], "M8[s]"), [0, 2], np.array([3, 5], "M8[s]")
        ).shape == (0,)

    def test_singleton_handling(self):
        assert inverse(3, [0], [3]) == 0
        assert inverse(3.0, [0], [3.0]) == 0
        assert inverse(np.array(3, "M8[s]"), [0], np.array([3], "M8[s]")) == 0
        assert inverse(4, [0], [3], method="nearest") == 0
        assert inverse(4.0, [0], [3.0], method="nearest") == 0
        assert (
            inverse(np.array(4, "M8[s]"), [0], np.array([3], "M8[s]"), method="nearest")
            == 0
        )

    def test_interpolation_accuracy_int(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = rng.integers(np.min(integers), np.max(integers), n)
        fp = np.sort(rng.choice(integers, n, replace=False))
        selected = np.arange(np.min(fp), np.max(fp) + 1)
        f = np.sort(rng.choice(selected, m, replace=False))
        result = inverse(f, xp, fp, method="nearest")
        expected = np.rint(np.round(np.interp(f, fp, xp), 11)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype
        result = inverse(f, xp, fp, method="ffill")
        expected = np.floor(np.round(np.interp(f, fp, xp), 11)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype
        result = inverse(f, xp, fp, method="bfill")
        expected = np.ceil(np.round(np.interp(f, fp, xp), 11)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_interpolation_accuracy_float(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = np.sort(rng.random(n))
        f = np.sort((np.max(fp) - np.min(fp)) * rng.random(m) + np.min(fp))
        result = inverse(f, xp, fp, method="nearest")
        expected = np.rint(np.interp(f, fp, xp)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype in ["i4", "i8"]
        result = inverse(f, xp, fp, method="ffill")
        expected = np.floor(np.interp(f, fp, xp)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype in ["i4", "i8"]
        result = inverse(f, xp, fp, method="bfill")
        expected = np.ceil(np.interp(f, fp, xp)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype in ["i4", "i8"]

    def test_use_case_integer(self):
        xp = np.array([0, 5, 15])
        fp = np.array([20, 30, 50])
        cases = [(x, forward([x], xp, fp)[0]) for x in range(16)]
        for x, f in cases:
            assert inverse([f], xp, fp)[0] == x
        for f in range(21, 50, 2):
            with pytest.raises(KeyError, match="f not found"):
                inverse([f], xp, fp)
        cases = [(0, 21), (2, 23), (2, 25), (4, 27), (4, 29), (6, 31), (6, 33), (8, 35)]
        for x, f in cases:
            assert inverse([f], xp, fp, method="nearest")[0] == x
        cases = [(0, 21), (1, 23), (2, 25), (3, 27), (4, 29), (5, 31), (6, 33), (7, 35)]
        for x, f in cases:
            assert inverse([f], xp, fp, method="ffill")[0] == x
        cases = [(1, 21), (2, 23), (3, 25), (4, 27), (5, 29), (6, 31), (7, 33), (8, 35)]
        for x, f in cases:
            assert inverse([f], xp, fp, method="bfill")[0] == x

    def test_use_case_float(self):
        xp = np.array([0, 5, 15])
        fp = np.array([20.0, 30.0, 50.0])
        cases = [(x, forward([x], xp, fp)[0]) for x in range(16)]
        for x, f in cases:
            assert inverse([float(f)], xp, fp)[0] == x
        for f in range(21, 50, 2):
            with pytest.raises(KeyError, match="f not found"):
                inverse([float(f)], xp, fp)
        cases = [(0, 21), (2, 23), (2, 25), (4, 27), (4, 29), (6, 31), (6, 33), (8, 35)]
        for x, f in cases:
            assert inverse([float(f)], xp, fp, method="nearest")[0] == x
        cases = [(0, 21), (1, 23), (2, 25), (3, 27), (4, 29), (5, 31), (6, 33), (7, 35)]
        for x, f in cases:
            assert inverse([float(f)], xp, fp, method="ffill")[0] == x
        cases = [(1, 21), (2, 23), (3, 25), (4, 27), (5, 29), (6, 31), (7, 33), (8, 35)]
        for x, f in cases:
            assert inverse([float(f)], xp, fp, method="bfill")[0] == x


class TestBoundaryHardening:
    """Regression tests for review.md's Correctness findings (C1-C4)."""

    def test_c1_uint64_fp_above_i64_max_no_longer_wraps(self):
        assert (
            forward(1, [0, 2], np.array([0, 2**63 + 10], dtype="u8"))
            == 4611686018427387909
        )
        assert forward(1, [0, 2], np.array([2**63 - 2, 2**63 + 2], dtype="u8")) == 2**63
        assert (
            inverse(
                np.uint64(2**63), [0, 2], np.array([2**63 - 2, 2**63 + 2], dtype="u8")
            )
            == 1
        )

    def test_c2_narrow_xp_no_longer_wraps_x(self):
        with pytest.raises(IndexError, match="x out of bounds"):
            forward(2**32, np.array([0, 2], dtype="i4"), [0, 100])
        with pytest.raises(IndexError, match="x out of bounds"):
            forward(300, np.array([0, 100], dtype="i1"), [0, 1000])

    def test_c2_narrow_xp_reports_out_of_bounds_not_positivity(self):
        with pytest.raises(IndexError, match="x out of bounds"):
            forward(2**31, np.array([0, 2], dtype="i4"), [0, 100])

    def test_c2_narrow_fp_rejects_lossy_f_cast(self):
        with pytest.raises(
            ValueError, match="f values must fit fp's dtype without loss"
        ):
            inverse(2**32 + 4, [0, 2], np.array([3, 5], dtype="i4"))

    def test_c2_f4_overflow_reports_truncation_not_finiteness(self):
        with pytest.raises(
            ValueError, match="f values must fit fp's dtype without loss"
        ):
            inverse(1e300, [0, 2], np.array([3.0, 5.0], dtype="f4"))

    def test_c2_narrow_xp_on_output_side_is_still_fine(self):
        assert inverse(4, np.array([0, 70000], dtype="u4"), [3, 5]) == 35000

    def test_c3_non_integral_x_raises(self):
        with pytest.raises(ValueError, match="x values must be integral"):
            forward(1.9, [0, 2], [0, 100])
        with pytest.raises(ValueError, match="x values must be integral"):
            forward(np.float32(1.9), [0, 2], [0, 100])

    def test_c3_integral_float_x_still_works(self):
        assert forward([1.0], [0, 2], [0, 100])[0] == 50

    def test_c4_dead_dtype_guards_removed(self):
        # x is no longer forced onto xp's dtype, so a mismatched dtype pair is
        # no longer silently coerced into passing the (now-deleted) guard.
        assert forward(np.array([1], dtype="u2"), [0, 2], [0, 100])[0] == 50

    def test_negative_f_against_unsigned_fp_raises_rather_than_wraps(self):
        with pytest.raises(ValueError, match="f values must be positive"):
            inverse(-1, [0, 2], np.array([3, 5], dtype="u8"))

    def test_rejects_unsupported_fp_dtype(self):
        with pytest.raises(
            ValueError, match="fp dtype must be either integer, floating or datetime"
        ):
            forward([1], [0, 2], np.array([1 + 2j, 3 + 4j]))

    def test_rejects_neither_x_nor_f_provided(self):
        with pytest.raises(ValueError, match="either x or f must be provided"):
            forward(None, [0, 2], [0, 100])


class TestDeprecatedPointsAliases:
    """`forward`/`inverse` stay as deprecated aliases for `forward_points`/
    `inverse_points`."""

    def test_forward_warns_and_matches_forward_points(self):
        with pytest.deprecated_call(match="use forward_points instead"):
            result = xinterp.forward(1, [0, 2], [3, 5])
        assert result == xinterp.forward_points(1, [0, 2], [3, 5])

    def test_inverse_warns_and_matches_inverse_points(self):
        with pytest.deprecated_call(match="use inverse_points instead"):
            result = xinterp.inverse(4, [0, 2], [3, 5])
        assert result == xinterp.inverse_points(4, [0, 2], [3, 5])
