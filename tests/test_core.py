import numpy as np
import pytest

from xinterp import forward, inverse


class TestForward:
    def test_raises_not_1D(self):
        with pytest.raises(ValueError, match="x must be 1D"):
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

    def test_raises_dtype_mismatch(self):
        xp = np.array([0, 10, 20], dtype="u4")
        fp = np.array([0, 1000, 2000])
        x = np.array([0, 5, 10, 15, 20])
        with pytest.raises(ValueError, match="x and xp must have the same dtype"):
            f = forward([1.0], [0, 2], [3, 5])

    def test_raises_not_strictly_incresing(self):
        with pytest.raises(ValueError, match="xp must be strictly increasing"):
            forward([1], [2, 0], [3, 5])

    def test_raises_out_of_bounds(self):
        with pytest.raises(ValueError, match="x out of bounds"):
            forward([0], [1, 2], [3, 5])
        with pytest.raises(ValueError, match="x out of bounds"):
            forward([3], [1, 2], [3, 5])

    def test_type_handling(self):
        assert forward([1], [0, 2], [3, 5]).dtype == "i8"
        assert forward([1], [0, 2], np.array([3, 5], "M8[s]")).dtype == "M8[s]"
        assert forward([1], [0, 2], np.array([3, 5], "f4")).dtype == "f4"
        assert forward(np.array([1], "u2"), np.array([0, 2], "u2"), [3, 5])[0] == 4

    def test_interpolation_accuracy_int(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        f = forward(x, xp, fp)
        f_expected = np.rint(np.round(np.interp(x, xp, fp), 6)).astype("i8")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_interpolation_accuracy_float(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65_535)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n).astype(np.float64)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        f = forward(x, xp, fp)
        f_expected = np.interp(x, xp, fp)
        assert np.allclose(f, f_expected)
        assert f.dtype == f_expected.dtype


class TestInverse:
    def test_(self):
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 3000])
        assert inverse([0], xp, fp) == 0
        assert inverse([100], xp, fp) == 1
        assert inverse([1000], xp, fp) == 10
        assert inverse([2000], xp, fp) == 15
        assert inverse([3000], xp, fp) == 20
        assert inverse([1], xp, fp, method="nearest") == 0
        assert inverse([1], xp, fp, method="ffill") == 0
        assert inverse([1], xp, fp, method="bfill") == 1
        assert inverse([4000], xp, fp, method="nearest") == 20
        assert inverse([-1000], xp, fp, method="nearest") == 0
        assert inverse([4000], xp, fp, method="ffill") == 20
        assert inverse([-1000], xp, fp, method="bfill") == 0
