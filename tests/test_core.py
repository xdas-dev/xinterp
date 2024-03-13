import numpy as np
import pytest

from xinterp import forward, inverse


class TestForward:
    def test_interpolation_accuracy(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(0, 65536)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        f = forward(x, xp, fp)
        f_expected = np.rint(np.round(np.interp(x, xp, fp), 6)).astype("i8")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_out_of_bound(self):
        xp = np.array([0, 10])
        fp = np.array([0, 1000])
        x = np.array([-1])
        with pytest.raises(ValueError):
            forward(x, xp, fp)
        x = np.array([11])
        with pytest.raises(ValueError):
            forward(x, xp, fp)

    def test_not_strictly_incresing(self):
        xp = np.array([10, 0])
        fp = np.array([0, 1000])
        x = np.array([5])
        with pytest.raises(ValueError):
            forward(x, xp, fp)

    def test_f_is_datetime(self):
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 2000], dtype="datetime64[s]")
        x = np.array([0, 5, 10, 15, 20])
        f = forward(x, xp, fp)
        f_expected = np.array([0, 500, 1000, 1500, 2000], dtype="datetime64[s]")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_dtype_mismatch(self):
        with pytest.raises(ValueError):
            xp = np.array([0, 10, 20], dtype="datetime64[s]")
            fp = np.array([0, 1000, 2000])
            x = np.array([0, 5, 10, 15, 20])
            f = forward(x, xp, fp)


class TestInverse:
    def test_(self):
        xp = np.array([0, 10])
        fp = np.array([0, 1000])
        assert inverse([0], xp, fp) == 0
        assert inverse([1], xp, fp, method="nearest") == 0
        assert inverse([1], xp, fp, method="ffill") == 0
        assert inverse([1], xp, fp, method="bfill") == 1
