import numpy as np
import pytest

from xinterp import interp_intlike, interp_int64


class TestInterpInt64:
    def test_interpolation_accuracy(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(-32768, 32767)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        f = interp_int64(x, xp, fp)
        f_expected = np.rint(np.round(np.interp(x, xp, fp), 6)).astype("int")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_out_of_bound(self):
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 2000])
        x = np.array([-10, -1, 0, 10, 20, 21, 30])

        f = interp_int64(x, xp, fp)
        f_expected = np.array([0, 0, 0, 1000, 2000, 2000, 2000])
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype
        f_expected = np.interp(x, xp, fp).astype("int")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

        f = interp_int64(x, xp, fp, left=-1, right=-2)
        f_expected = np.array([-1, -1, 0, 1000, 2000, -2, -2])
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype
        f_expected = np.interp(x, xp, fp, left=-1, right=-2).astype("int")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_raise_if_not_strictly_incresing(self):
        with pytest.raises(ValueError):
            xp = np.array([0, -10, 20])
            fp = np.array([0, 1000, 2000])
            x = np.array([0, 5, 10, 15, 20])
            f = interp_int64(x, xp, fp)


class TestInterpIntLike:
    def test_f_is_datetime(self):
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 2000], dtype="datetime64[s]")
        x = np.array([0, 5, 10, 15, 20])
        f = interp_intlike(x, xp, fp)
        f_expected = np.array([0, 500, 1000, 1500, 2000], dtype="datetime64[s]")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_x_is_datetime(self):
        xp = np.array([0, 10, 20], dtype="datetime64[s]")
        fp = np.array([0, 1000, 2000])
        x = np.array([0, 5, 10, 15, 20], dtype="datetime64[s]")
        f = interp_intlike(x, xp, fp)
        f_expected = np.array([0, 500, 1000, 1500, 2000])
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

    def test_out_of_bound(self):
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 2000], dtype="datetime64[s]")
        x = np.array([-10, -1, 0, 10, 20, 21, 30])

        f = interp_intlike(x, xp, fp)
        f_expected = np.array([0, 0, 0, 1000, 2000, 2000, 2000], dtype="datetime64[s]")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

        f = interp_intlike(
            x, xp, fp, left=np.datetime64(-1, "s"), right=np.datetime64(-2, "s")
        )
        f_expected = np.array([-1, -1, 0, 1000, 2000, -2, -2], dtype="datetime64[s]")
        assert np.array_equal(f, f_expected)
        assert f.dtype == f_expected.dtype

        nat = np.datetime64("NaT")
        f = interp_intlike(x, xp, fp, left=nat, right=nat)
        f_expected = np.array(
            [nat, nat, 0, 1000, 2000, nat, nat], dtype="datetime64[s]"
        )
        assert np.array_equal(f, f_expected, equal_nan=True)
        assert f.dtype == f_expected.dtype

    def test_raise_if_dtype_mismatch(self):
        with pytest.raises(ValueError):
            xp = np.array([0, 10, 20], dtype="datetime64[s]")
            fp = np.array([0, 1000, 2000])
            x = np.array([0, 5, 10, 15, 20])
            f = interp_intlike(x, xp, fp)
