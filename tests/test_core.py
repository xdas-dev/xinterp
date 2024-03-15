import numpy as np
import pytest

from xinterp import forward, inverse


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

    def test_raises_only_one_element(self):
        with pytest.raises(
            ValueError, match="xp and fp must have at least two elements"
        ):
            forward([1], [0], [3])

    def test_raises_xp_not_integer(self):
        with pytest.raises(ValueError, match="xp must have integer dtype"):
            forward([1], [0.0, 2.0], [3, 5])

    def test_raises_not_positive(self):
        with pytest.raises(ValueError, match="xp values must be positive"):
            forward([1], [-1, 2], [3, 5])
        with pytest.raises(ValueError, match="x values must be positive"):
            forward([-1], [1, 2], [3, 5])

    def test_dtype_matching(self):
        forward([1.0], [0, 2], [3, 5]) == 4
        forward([1], [0, 2], [3.0, 5.0]) == 4.0
        forward(np.array([1], dtype="M8[s]"), [0, 2], [3.0, 5.0]) == 4.0
        forward(np.array([1], dtype="M8[s]"), [0, 2], [3, 5]) == 4
        forward([1.0], [0, 2], np.array([3, 5], dtype="M8[s]")) == np.array(
            [4], dtype="M8[s]"
        )
        forward([1], [0, 2], np.array([3, 5], dtype="M8[s]")) == np.array(
            [4], dtype="M8[s]"
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
        assert forward([1], [0, 2], [3, 5]).dtype == "i8"
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

    def test_raises_only_one_element(self):
        with pytest.raises(
            ValueError, match="xp and fp must have at least two elements"
        ):
            inverse([4], [0], [3])

    def test_raises_xp_not_integer(self):
        with pytest.raises(ValueError, match="xp must have integer dtype"):
            inverse([4], [0.0, 2.0], [3, 5])

    def test_raises_not_positive(self):
        with pytest.raises(ValueError, match="xp values must be positive"):
            inverse([4], [-1, 2], [3, 5])

    def test_dtype_matching(self):
        inverse([4.0], [0, 2], [3, 5]) == 1
        inverse([4], [0, 2], [3.0, 5.0]) == 1
        inverse(np.array([4], dtype="M8[s]"), [0, 2], [3.0, 5.0]) == 1
        inverse(np.array([4], dtype="M8[s]"), [0, 2], [3, 5]) == 1
        inverse([4.0], [0, 2], np.array([3, 5], dtype="M8[s]")) == 1
        inverse([4], [0, 2], np.array([3, 5], dtype="M8[s]")) == 1

    def test_raises_not_strictly_incresing(self):
        with pytest.raises(ValueError, match="fp must be strictly increasing"):
            inverse([4], [0, 2], [5, 3])

    def test_raises_out_of_bounds(self):
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([2], [1, 2], [3, 5])
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([6], [1, 2], [3, 5])
        inverse([2], [1, 2], [3, 5], method="nearest") == 1
        inverse([6], [1, 2], [3, 5], method="nearest") == 2
        with pytest.raises(KeyError, match="f out of bounds"):
            inverse([2], [1, 2], [3, 5], method="ffill")
        inverse([6], [1, 2], [3, 5], method="ffill") == 2
        inverse([2], [1, 2], [3, 5], method="bfill") == 1
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
        inverse([5.0 + 1e-16], [0, 2], [3.0, 7.0])
        inverse([5.0 - 1e-16], [0, 2], [3.0, 7.0])

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
        assert result.dtype == expected.dtype
        result = inverse(f, xp, fp, method="ffill")
        expected = np.floor(np.interp(f, fp, xp)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype
        result = inverse(f, xp, fp, method="bfill")
        expected = np.ceil(np.interp(f, fp, xp)).astype("i8")
        assert np.array_equal(result, expected)
        assert result.dtype == expected.dtype

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


# class TestScaleOffset:
#     def test_init(self):
#         transform = ScaleOffset(10.0, 100.0)
#         assert transform.scale == 10.0
#         assert transform.offset == 100.0

#     def test_eq(self):
#         transform1 = ScaleOffset(10.0, 100.0)
#         transform2 = ScaleOffset(
#             np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
#         )
#         assert transform1 == transform1
#         assert transform2 == transform2
#         assert transform1 != transform2

#     def test_direct(self):
#         transform = ScaleOffset(10.0, 100.0)
#         assert transform.direct(150.0) == 5.0
#         assert np.allclose(
#             transform.direct(np.array([150.0, 160.0])), np.array([5.0, 6.0])
#         )
#         transform = ScaleOffset(
#             np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
#         )
#         assert transform.direct(np.datetime64("2000-01-01T00:00:10")) == 10.0

#     def test_inverse(self):
#         transform = ScaleOffset(10.0, 100.0)
#         assert transform.inverse(5.0) == 150
#         assert np.allclose(
#             transform.inverse(np.array([5.0, 6.0])), np.array([150.0, 160.0])
#         )
#         transform = ScaleOffset(
#             np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
#         )
#         assert transform.inverse(10.0) == np.datetime64("2000-01-01T00:00:10")
#         assert transform.inverse(10.3) == np.datetime64("2000-01-01T00:00:10")
#         assert transform.inverse(10.7) == np.datetime64("2000-01-01T00:00:11")
#         assert transform.inverse(11.0) == np.datetime64("2000-01-01T00:00:11")
#         transform = ScaleOffset(
#             np.timedelta64(1000, "us"), np.datetime64("2000-01-01T00:00:00")
#         )
#         assert transform.inverse(1000.0) == np.datetime64("2000-01-01T00:00:01")
#         assert transform.inverse(1000.3) == np.datetime64("2000-01-01T00:00:01")
#         assert transform.inverse(1000.7) == np.datetime64("2000-01-01T00:00:01.001")

#     def test_floatize(self):
#         assert ScaleOffset.floatize(np.linspace(0, 1)) == ScaleOffset(1.0, 0.0)
#         assert ScaleOffset.floatize(
#             np.array(
#                 [
#                     np.datetime64("2000-01-01T00:00:00"),
#                     np.datetime64("2000-01-02T00:00:00"),
#                 ]
#             )
#         ) == ScaleOffset(np.timedelta64(1, "s"), np.datetime64("2000-01-01T12:00:00"))

#     def test_accuracy(self):
#         x = np.random.rand(1000)
#         transform = ScaleOffset.floatize(x)
#         assert np.all(np.abs(x - transform.inverse(transform.direct(x))) < 1e-15)

#         t0 = np.datetime64("2000-01-01T00:00:00.000000000")
#         three_years_ns = t0 + np.random.randint(0, int(1e9) * 3600 * 24 * 1000, 1000)
#         three_month_ns = t0 + np.random.randint(0, int(1e9) * 3600 * 24 * 100, 1000)

#         t = three_years_ns.astype("datetime64[us]")
#         transform = ScaleOffset.floatize(t)
#         assert np.all(t == transform.inverse(transform.direct(t)))

#         t = three_month_ns
#         transform = ScaleOffset.floatize(t)
#         assert np.all(t == transform.inverse(transform.direct(t)))

#         t = three_years_ns
#         with pytest.warns(UserWarning):
#             transform = ScaleOffset.floatize(t)
#             error = np.abs(t - transform.inverse(transform.direct(t)))
#             assert np.all(error < np.timedelta64(1, "us"))
