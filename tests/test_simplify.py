import numpy as np
import pytest

from xinterp.core import forward_step, infer_step, simplify_points, simplify_step


def _sleeve_loop_reference(positions, values, en, ed, sn=None, sd=None):
    """Pure-Python, arbitrary-precision reference for the sleeve walk (mirrors xdas's
    ``_sleeve_loop``), used here as an independent oracle for `simplify_points`. With
    `sn`/`sd`, a chord that would drop a point must also honour the step band."""
    keep = [False] * len(positions)
    keep[0] = keep[-1] = True
    ax, ay = positions[0], values[0]
    lo = hi = None
    for i in range(1, len(positions)):
        dx = positions[i] - ax
        dy = values[i] - ay
        if not (
            lo is None
            or (
                lo[0] * dx < dy * lo[1]
                and dy * hi[1] < hi[0] * dx
                and (sn is None or abs(sd * ed * dy - sn * ed * dx) < 2 * en * sd)
            )
        ):
            keep[i - 1] = True
            ax, ay = positions[i - 1], values[i - 1]
            dx = positions[i] - ax
            dy = values[i] - ay
            lo = None
        num_lo, num_hi, den = ed * dy - en, ed * dy + en, ed * dx
        if lo is None:
            lo, hi = (num_lo, den), (num_hi, den)
        else:
            if num_lo * lo[1] > lo[0] * den:
                lo = (num_lo, den)
            if num_hi * hi[1] < hi[0] * den:
                hi = (num_hi, den)
    return keep


class TestSimplifyPoints:
    def test_two_ties_pass_through(self):
        result = simplify_points([0, 9], [0.0, 9.0], 0.0, 1.0)
        assert list(result) == [True, True]

    def test_float_values(self):
        x = [0, 9, 10, 19]
        f = [0.0, 9.0, 10.5, 19.5]
        assert list(simplify_points(x, f, 1.0, 1.0)) == [True, False, False, True]
        assert list(simplify_points(x, f, 0.1, 1.0)) == [True, True, True, True]

    def test_integer_values_exact_tick(self):
        x = [0, 999, 1000, 1999]
        f = [0, 999, 1001, 2000]
        # the raw walk is conservative (see xdas's `_sleeve` doc comment): a single
        # spanning chord fits these within epsilon 0, but the walk alone does not
        # collapse them -- this exercises the kernel, not the higher-level policy of
        # trying the global chord first
        assert list(simplify_points(x, f, 0, 1)) == [True, True, True, True]

    def test_datetime_values(self):
        x = np.array([0, 9, 10, 19], dtype="i8")
        f = np.array([0, 9000, 10500, 19500], dtype="M8[us]")
        result = simplify_points(x, f, 0, 1)
        assert list(result) == [True, True, True, True]

    def test_timedelta_values(self):
        x = np.array([0, 9, 10, 19], dtype="i8")
        f = np.array([0, 9000, 10500, 19500], dtype="m8[us]")
        result = simplify_points(x, f, 0, 1)
        assert list(result) == [True, True, True, True]

    def test_rejects_mismatched_length(self):
        with pytest.raises(ValueError, match="same length"):
            simplify_points([0, 1], [0.0], 0.0, 1.0)

    def test_rejects_non_increasing_x(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            simplify_points([0, 2, 1], [0.0, 1.0, 2.0], 0.0, 1.0)

    def test_rejects_not_1D(self):
        with pytest.raises(ValueError, match="x and f must be 1D"):
            simplify_points([[0, 1]], [[0.0, 1.0]], 0.0, 1.0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one element"):
            simplify_points(
                np.array([], dtype="i8"), np.array([], dtype="f8"), 0.0, 1.0
            )

    def test_rejects_non_integer_x_dtype(self):
        with pytest.raises(ValueError, match="x must have integer dtype"):
            simplify_points([0.0, 1.0], [0.0, 1.0], 0.0, 1.0)

    def test_rejects_negative_x(self):
        with pytest.raises(ValueError, match="x values must be positive"):
            simplify_points([-1, 2], [0.0, 1.0], 0.0, 1.0)

    def test_rejects_unsupported_f_dtype(self):
        with pytest.raises(
            ValueError, match="f dtype must be either integer, floating or datetime"
        ):
            simplify_points([0, 1], np.array([1 + 2j, 3 + 4j]), 0.0, 1.0)

    def test_matches_reference_oracle_on_random_curves(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            size = int(rng.integers(3, 30))
            positions = np.array(
                [
                    0,
                    *sorted(
                        rng.choice(np.arange(1, 3000), size - 1, replace=False).tolist()
                    ),
                ]
            )
            rate = int(rng.integers(1, 10**5))
            values = positions * rate + rng.integers(-4, 5, size)
            values = np.maximum.accumulate(values) + np.arange(size)
            en, ed = int(rng.integers(0, 5)), 2
            expected = _sleeve_loop_reference(
                positions.tolist(), values.tolist(), en, ed
            )
            result = simplify_points(positions, values.astype("i8"), en, ed)
            np.testing.assert_array_equal(result, expected)

    def test_deviation_bound_holds_on_random_curves(self):
        # every dropped point must be reconstructible, from the surviving chord, to
        # within the declared epsilon
        rng = np.random.default_rng(1)
        for _ in range(50):
            size = int(rng.integers(3, 20))
            x = np.array(
                [
                    0,
                    *sorted(
                        rng.choice(np.arange(1, 500), size - 1, replace=False).tolist()
                    ),
                ]
            )
            f = np.cumsum(rng.integers(-10, 100, size)).astype("i8")
            en, ed = 3, 2  # epsilon = 1.5
            keep = simplify_points(x, f, en, ed)
            xk, fk = x[keep], f[keep]
            from xinterp import forward_points

            reconstructed = forward_points(x, xk, fk)
            deviation = np.abs(reconstructed.astype("i8") - f.astype("i8"))
            assert np.all(
                2 * ed * deviation <= 2 * en + ed
            )  # |dev| <= en/ed rounded up


def _random_curve(rng, size, rate):
    positions = np.array(
        [0, *sorted(rng.choice(np.arange(1, 3000), size - 1, replace=False).tolist())]
    )
    values = positions * rate + rng.integers(-4, 5, size)
    return positions, np.maximum.accumulate(values) + np.arange(size)


def _band_drift(x, f, sn, sd):
    """`sd * (f_j - f_i) - sn * (x_j - x_i)` over every consecutive pair of `(x, f)`."""
    x = np.asarray(x).astype(object)
    f = np.asarray(f).astype("i8").astype(object)
    return sd * np.diff(f) - sn * np.diff(x)


def _hdas_like(seed):
    """Sync knots every `mtb` samples of a 250 Hz clock running fast by 125 ppm, the
    timestamps rounded to the millisecond with one of jitter and the odd late sync:
    every segment drifts by at most 3.5 ms against the nominal 4 ms step, so the
    nominal rate holds at T = 2 ms, while the drifts add up across segments."""
    rng = np.random.default_rng(seed)
    dt, mtb, n = 4_000_000, 1000, 400
    x = np.arange(n) * mtb
    f = np.round(x * dt * (1 + 1.25e-4) / 1e6).astype("i8")
    f += rng.integers(-1, 2, n) + (rng.random(n) < 0.02)
    return x, (f * 1_000_000).astype("M8[ns]"), dt, 2_000_000


class TestSimplifyPointsStep:
    def test_matches_reference_oracle_on_random_integer_curves(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            size = int(rng.integers(3, 30))
            rate = int(rng.integers(1, 10**5))
            positions, values = _random_curve(rng, size, rate)
            sd = int(rng.integers(1, 4))
            sn = rate * sd + int(rng.integers(-2, 3))
            en, ed = int(rng.integers(0, 12)), int(rng.integers(1, 4))
            expected = _sleeve_loop_reference(
                positions.tolist(), values.tolist(), en, ed, sn, sd
            )
            result = simplify_points(positions, values.astype("i8"), en, ed, sn, sd)
            np.testing.assert_array_equal(result, expected)

    def test_matches_reference_oracle_on_random_datetime_curves(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            size = int(rng.integers(3, 30))
            rate = int(rng.integers(1, 10**3))
            positions, values = _random_curve(rng, size, rate * 1000)
            en, ed = int(rng.integers(0, 12_000)), 2
            expected = _sleeve_loop_reference(
                positions.tolist(), values.tolist(), en, ed, rate * 1000, 1
            )
            result = simplify_points(
                positions,
                values.astype("M8[ns]"),
                en,
                ed,
                np.timedelta64(rate, "us"),
                1,
            )
            np.testing.assert_array_equal(result, expected)

    def test_matches_reference_oracle_on_random_float_curves(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            size = int(rng.integers(3, 30))
            positions = np.cumsum(rng.integers(1, 100, size)) - 1
            rate = float(rng.uniform(0.5, 2.0))
            values = positions * rate + rng.uniform(-2.0, 2.0, size)
            sn = rate + float(rng.uniform(-0.01, 0.01))
            en = float(rng.uniform(0.0, 3.0))
            expected = _sleeve_loop_reference(
                positions.tolist(), values.tolist(), en, 1.0, sn, 1.0
            )
            result = simplify_points(positions, values, en, 1.0, sn, 1)
            np.testing.assert_array_equal(result, expected)

    def test_without_step_is_unchanged(self):
        rng = np.random.default_rng(6)
        for _ in range(100):
            size = int(rng.integers(3, 30))
            positions, values = _random_curve(rng, size, int(rng.integers(1, 10**5)))
            en, ed = int(rng.integers(0, 12)), 2
            np.testing.assert_array_equal(
                simplify_points(positions, values, en, ed, None, None),
                _sleeve_loop_reference(positions.tolist(), values.tolist(), en, ed),
            )

    def test_kept_chords_honour_the_step_and_dropped_values_the_tolerance(self):
        from xinterp import forward_points

        rng = np.random.default_rng(7)
        for _ in range(200):
            size = int(rng.integers(3, 40))
            rate = int(rng.integers(1, 1000))
            positions, values = _random_curve(rng, size, rate)
            sd = int(rng.integers(1, 4))
            sn = rate * sd + int(rng.integers(-2, 3))
            en, ed = int(rng.integers(0, 12)), 2
            keep = simplify_points(positions, values, en, ed, sn, sd)
            xk, fk = positions[keep], values[keep]
            idx = np.flatnonzero(keep)
            thinned = np.diff(idx) > 1
            drift = _band_drift(xk, fk, sn, sd)[thinned]
            assert all(ed * abs(d) < 2 * en * sd for d in drift)
            reconstructed = forward_points(positions, xk, fk).astype("i8")
            deviation = np.abs(reconstructed - values.astype("i8"))
            assert np.all(2 * ed * deviation <= 2 * en + ed)

    def test_rate_holds_on_hdas_like_syncs_with_step_and_breaks_without(self):
        for seed in range(5):
            x, f, dt, T = _hdas_like(seed)
            # the input honours the nominal rate at T on every segment...
            assert np.all(np.abs(_band_drift(x, f, dt, 1)) <= 2 * T)
            en, ed = 2 * T + 1, 2  # T + 1/2
            keep = simplify_points(x, f, en, ed, np.timedelta64(dt, "ns"), 1)
            # ...and so does the thinned output, every chord of it
            assert np.all(np.abs(_band_drift(x[keep], f[keep], dt, 1)) <= 2 * T)
            assert keep.sum() < len(x) // 2
            # the plain walk thins further, but loses the rate
            plain = simplify_points(x, f, en, ed)
            assert plain.sum() < keep.sum()
            assert np.any(np.abs(_band_drift(x[plain], f[plain], dt, 1)) > 2 * T)

    def test_timedelta_sn_on_timedelta_values(self):
        x = [0, 10, 20, 30, 40]
        f = np.array([0, 102, 204, 306, 408], dtype="m8[ms]")
        keep = simplify_points(x, f, 5, 2, np.timedelta64(10, "ms"), 1)
        assert list(keep) == [True, False, True, False, True]

    def test_rejects_sn_without_sd(self):
        with pytest.raises(ValueError, match="sn and sd must be given together"):
            simplify_points([0, 1], [0, 1], 0, 1, sn=1)
        with pytest.raises(ValueError, match="sn and sd must be given together"):
            simplify_points([0, 1], [0, 1], 0, 1, sd=1)

    def test_rejects_non_positive_sd(self):
        with pytest.raises(ValueError, match="sd must be positive"):
            simplify_points([0, 1], [0, 1], 0, 1, 1, 0)

    def test_rejects_timedelta_sn_on_numeric_values(self):
        with pytest.raises(ValueError, match="timedelta64 sn needs datetime64"):
            simplify_points([0, 1], [0, 1], 0, 1, np.timedelta64(1, "ms"), 1)

    def test_rejects_timedelta_sn_finer_than_the_values_unit(self):
        f = np.array([0, 1], dtype="M8[ms]")
        with pytest.raises(ValueError, match="not a whole number of ms"):
            simplify_points([0, 1], f, 0, 1, np.timedelta64(1500, "us"), 1)

    def test_rejects_non_integer_step_on_integer_values(self):
        with pytest.raises(ValueError, match="must be integers for integer f"):
            simplify_points([0, 1], [0, 1], 0, 1, 1.5, 1)

    def test_rejects_non_unit_sd_on_float_values(self):
        with pytest.raises(ValueError, match="sd must be 1 for floating f"):
            simplify_points([0, 1], [0.0, 1.0], 0.0, 1.0, 3, 2)


class TestSimplifyStep:
    def test_fuses_within_budget(self):
        keep, fused = simplify_step([0, 100, 199], [10, 10, 10], [10], [1], 1)
        assert list(keep) == [True, False, False]
        assert list(fused) == [0]

    def test_splits_on_large_drift(self):
        keep, fused = simplify_step([0, 100, 300], [10, 10, 10], [10], [1], 1)
        assert list(keep) == [True, False, True]
        assert list(fused) == [0, 300]

    def test_rejects_bad_num_den_length(self):
        with pytest.raises(ValueError, match="length 1 or len"):
            simplify_step([0, 100, 300], [10, 10, 10], [10, 20], [1, 1], 1)

    def test_rejects_non_positive_den(self):
        with pytest.raises(ValueError, match="den values must be positive"):
            simplify_step([0, 100], [10, 10], [10], [0], 1)

    def test_rejects_not_1D(self):
        with pytest.raises(ValueError, match="tie_values and tie_lengths must be 1D"):
            simplify_step([[0]], [10], [1], [1], 0)

    def test_rejects_mismatched_length(self):
        with pytest.raises(
            ValueError, match="tie_values and tie_lengths must have the same length"
        ):
            simplify_step([0, 1], [10], [1], [1], 0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one element"):
            simplify_step([], [], [1], [1], 0)

    def test_rejects_num_den_not_1D(self):
        with pytest.raises(ValueError, match="num and den must be 1D"):
            simplify_step([0], [10], [[1]], [1], 0)

    def test_rejects_num_den_mismatched_length(self):
        with pytest.raises(ValueError, match="num and den must have the same length"):
            simplify_step([0], [10], [1, 2], [1], 0)

    def test_fused_values_stay_within_declared_tolerance_of_a_run(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            n = int(rng.integers(1, 15))
            lengths = rng.integers(1, 20, n).astype("u8")
            rate_num, rate_den = int(rng.integers(1, 50)), int(rng.integers(1, 5))
            tol = int(rng.integers(0, 5))
            values = [0]
            for length in lengths[:-1]:
                jitter = int(rng.integers(-tol, tol + 1)) if tol > 0 else 0
                values.append(values[-1] + round(length * rate_num / rate_den) + jitter)
            values = np.array(values, dtype="i8")
            keep, fused = simplify_step(values, lengths, [rate_num], [rate_den], tol)
            assert keep[0]
            assert len(fused) == int(np.sum(keep))

    def test_worst_sample_drift_never_exceeds_tol(self):
        # the contract: no SAMPLE may drift by more than tol, not just the fused tie
        # values
        rng = np.random.default_rng(2)
        for tol in (0, 1, 2, 3, 5):
            for _ in range(50):
                n = int(rng.integers(2, 15))
                lengths = rng.integers(1, 40, n).astype("u8")
                rate_num, rate_den = int(rng.integers(1, 50)), int(rng.integers(1, 15))
                values = [0]
                for length in lengths[:-1]:
                    jitter = int(rng.integers(-tol, tol + 1)) if tol > 0 else 0
                    values.append(
                        values[-1] + round(length * rate_num / rate_den) + jitter
                    )
                values = np.array(values, dtype="i8")
                tie_indices = np.concatenate([[0], np.cumsum(lengths)]).astype("u8")
                total = int(tie_indices[-1])
                original = forward_step(
                    np.arange(total, dtype="u8"),
                    tie_indices,
                    np.concatenate([values, [values[-1]]]).astype("i8"),
                    [rate_num],
                    [rate_den],
                )
                keep, fused = simplify_step(
                    values, lengths, [rate_num], [rate_den], tol
                )
                run_lengths = []
                for k, length in zip(keep, lengths, strict=True):
                    if k:
                        run_lengths.append(int(length))
                    else:
                        run_lengths[-1] += int(length)
                reconstructed = np.concatenate(
                    [
                        forward_step(
                            np.arange(run_length, dtype="u8"),
                            np.array([0, run_length], dtype="u8"),
                            np.array([anchor, anchor], dtype="i8"),
                            [rate_num],
                            [rate_den],
                        )
                        for anchor, run_length in zip(fused, run_lengths, strict=True)
                    ]
                )
                drift = np.abs(original.astype("i8") - reconstructed.astype("i8"))
                assert drift.max() <= tol


class TestInferStep:
    def test_exact_fit(self):
        num, den, worst = infer_step([0, 10, 20], [0, 100, 200])
        assert (num, den, worst) == (10, 1, 0)

    def test_gcd_reduced(self):
        num, den, _worst = infer_step([0, 4], [0, 20])
        assert (num, den) == (5, 1)

    def test_negative_rate(self):
        num, den, worst = infer_step([0, 10, 20], [100, 0, -100])
        assert (num, den, worst) == (-10, 1, 0)

    def test_requires_at_least_two_points(self):
        with pytest.raises(ValueError, match="at least two"):
            infer_step([0], [0])

    def test_rejects_unsupported_f_dtype(self):
        with pytest.raises(
            ValueError, match="f dtype must be either integer or datetime"
        ):
            infer_step([0, 10], [0.0, 10.0])

    def test_datetime_values(self):
        x = np.array([0, 999, 1998])
        f = np.array([0, 30000, 60000], dtype="M8[us]")
        num, den, worst = infer_step(x, f)
        # 30000/999 reduces by gcd 3 to 10000/333
        assert (num, den, worst) == (10000, 333, 0)

    def test_timedelta_values(self):
        x = np.array([0, 999, 1998])
        f = np.array([0, 30000, 60000], dtype="m8[us]")
        num, den, worst = infer_step(x, f)
        # 30000/999 reduces by gcd 3 to 10000/333
        assert (num, den, worst) == (10000, 333, 0)
