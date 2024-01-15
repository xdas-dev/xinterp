import numpy as np

from xinterp import interp_datetime64, interp_int64


class TestCore:
    def test_interp_int64(self):
        rng = np.random.default_rng(42)
        n = 1_000
        m = 10_000
        integers = np.arange(-32768, 32767)
        xp = np.sort(rng.choice(integers, n, replace=False))
        fp = rng.integers(np.min(integers), np.max(integers), n)
        selected = np.arange(np.min(xp), np.max(xp) + 1)
        x = np.sort(rng.choice(selected, m, replace=False))
        f_int = interp_int64(x, xp, fp)
        f_float = np.rint(np.round(np.interp(x, xp, fp), 6)).astype("int")
        assert np.all(f_int == f_float)

    def test_interp_datetime64(self):
        x = np.array([0, 5, 10, 15, 20])
        xp = np.array([0, 10, 20])
        fp = np.array([0, 1000, 2000], dtype="datetime64[s]")
        f = interp_datetime64(x, xp, fp)
        assert np.all(f == np.array([0, 500, 1000, 1500, 2000], dtype="datetime64[s]"))
