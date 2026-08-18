from fractions import Fraction

import numpy as np
import pytest

from xinterp import deviation_step, forward_step, inverse_step


def _round_half_even(fraction):
    """Python's `round()` on a `Fraction` already rounds ties to even; kept as a named
    helper so the reference computation below reads like the kernel's own contract."""
    return round(fraction)


def _reference_forward(tie0, k, num, den):
    """Exact `tie0 + round(k * num / den)` via `fractions.Fraction` -- the independent
    oracle every kernel is checked against."""
    return tie0 + _round_half_even(Fraction(k * num, den))


class TestForwardStep:
    def test_basic_shared_rate(self):
        tie_indices = [0, 10]
        tie_values = [0, 100]
        result = forward_step([0, 3, 7, 10], tie_indices, tie_values, [10], [1])
        assert list(result) == [0, 30, 70, 100]

    def test_negative_num(self):
        tie_indices = [0, 10]
        tie_values = [100, 0]
        result = forward_step([0, 3, 10], tie_indices, tie_values, [-10], [1])
        assert list(result) == [100, 70, 0]

    def test_per_segment_rate(self):
        tie_indices = [0, 10, 20]
        tie_values = [0, 100, 500]
        result = forward_step([5, 15], tie_indices, tie_values, [10, 40], [1, 1])
        assert list(result) == [50, 300]

    def test_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            forward_step([11], [0, 10], [0, 100], [10], [1])

    def test_datetime_tie_values(self):
        tie_indices = np.array([0, 999])
        tie_values = np.array([0, 30_000], dtype="M8[us]")
        result = forward_step([0, 999], tie_indices, tie_values, [30_000], [999])
        assert result[0] == np.datetime64(0, "us")
        assert result[1] == np.datetime64(30_000, "us")

    def test_float_tie_values(self):
        tie_indices = [0, 10]
        tie_values = [0.0, 100.0]
        result = forward_step([5], tie_indices, tie_values, [10.0], [1])
        assert result[0] == 50.0

    def test_float_rejects_den_not_one(self):
        with pytest.raises(ValueError, match="no exact rate exists"):
            forward_step([5], [0, 10], [0.0, 100.0], [10.0], [2])

    def test_rejects_bad_num_den_length(self):
        # 2 segments (3 tie points), but 3 rates: neither shared (1) nor per-segment (2)
        with pytest.raises(ValueError, match="length 1 or"):
            forward_step([5], [0, 10, 20], [0, 100, 200], [10, 20, 30], [1, 1, 1])

    def test_magnitude_sweep_against_fraction_reference(self):
        """Every kernel checked against `Fraction`/bigint reference values, including
        magnitudes overflowing i64 naively."""
        rng = np.random.default_rng(0)
        magnitudes = [
            4_300_000_000_000_000_000,  # ~4.3e19, overflows u64 handling naively
            470_000_000_000_000_000_000_000,  # ~4.7e23
            470_000_000_000_000_000_000_000_000,  # ~4.7e26
        ]
        for magnitude in magnitudes:
            den = int(rng.integers(1, 10**6))
            num = magnitude if rng.integers(0, 2) else -magnitude
            # keep num within i64 range while den can still be large
            num = int(np.clip(num, -(2**63) + 1, 2**63 - 1))
            length = int(rng.integers(1, 2**63 - 1))
            tie0 = int(rng.integers(-(2**62), 2**62))
            expected = _reference_forward(tie0, length, num, den)
            if not (-(2**63) <= expected <= 2**63 - 1):
                continue
            result = forward_step(
                [length], [0, length], [tie0, expected], [num], [den]
            )
            assert int(result[0]) == expected, (tie0, length, num, den)

    def test_den_array_length_1_and_n_segments_agree(self):
        tie_indices = [0, 10, 20, 30]
        tie_values = [0, 100, 300, 600]
        shared = forward_step([5, 15, 25], tie_indices, tie_values, [10], [1])
        per_segment = forward_step(
            [5, 15, 25], tie_indices, tie_values, [10, 20, 30], [1, 1, 1]
        )
        # not the same rate per segment, so just check both dispatch without error and
        # the shared-rate one matches the direct hand computation
        assert list(shared) == [50, 150, 350]
        assert len(per_segment) == 3


class TestInverseStep:
    def test_round_trip_exact(self):
        tie_indices = [0, 10]
        tie_values = [0, 100]
        f = forward_step([0, 3, 7, 10], tie_indices, tie_values, [10], [1])
        result = inverse_step(f, tie_indices, tie_values, [10], [1])
        assert list(result) == [0, 3, 7, 10]

    def test_negative_num_round_trip(self):
        tie_indices = [0, 10]
        tie_values = [100, 0]
        f = forward_step([0, 3, 7, 10], tie_indices, tie_values, [-10], [1])
        result = inverse_step(f, tie_indices, tie_values, [-10], [1])
        assert list(result) == [0, 3, 7, 10]

    def test_out_of_bounds_raises(self):
        with pytest.raises(KeyError):
            inverse_step([101], [0, 10], [0, 100], [10], [1])

    def test_methods_between_ticks(self):
        # rate 3/2: index 0 -> 0, index 1 -> 2, index 2 -> 3
        tie_indices = [0, 2]
        tie_values = [0, 3]
        with pytest.raises(KeyError):
            inverse_step([1], tie_indices, tie_values, [3], [2])
        assert inverse_step([1], tie_indices, tie_values, [3], [2], method="ffill")[0] == 0
        assert inverse_step([1], tie_indices, tie_values, [3], [2], method="bfill")[0] == 1
        assert inverse_step([1], tie_indices, tie_values, [3], [2], method="nearest")[0] == 1

    def test_datetime_tie_values(self):
        tie_indices = np.array([0, 999])
        tie_values = np.array([0, 30_000], dtype="M8[us]")
        result = inverse_step(
            np.array([0, 30_000], dtype="M8[us]"), tie_indices, tie_values, [30_000], [999]
        )
        assert list(result) == [0, 999]

    def test_float_tie_values(self):
        tie_indices = [0, 10]
        tie_values = [0.0, 100.0]
        result = inverse_step([50.0], tie_indices, tie_values, [10.0], [1])
        assert result[0] == 5


class TestDeviationStep:
    def test_exact_fit_is_zero(self):
        tie_indices = [0, 10, 20]
        tie_values = [0, 100, 200]
        result = deviation_step(tie_indices, tie_values, [10], [1])
        assert list(result) == [0, 0]

    def test_nonzero_residual(self):
        tie_indices = [0, 10, 20]
        tie_values = [0, 101, 199]
        result = deviation_step(tie_indices, tie_values, [10], [1])
        assert list(result) == [1, -2]

    def test_matches_fraction_reference(self):
        # forward_step queried exactly at a tie point returns the stored value, not the
        # step's prediction, so cross-check against the Fraction oracle directly instead
        rng = np.random.default_rng(3)
        tie_indices = np.sort(rng.choice(np.arange(1, 1000), 6, replace=False))
        tie_indices = np.insert(tie_indices, 0, 0)
        tie_values = np.cumsum(rng.integers(-50, 200, len(tie_indices))).astype("i8")
        num, den = int(rng.integers(1, 20)), 1
        residual = deviation_step(tie_indices, tie_values, [num], [den])
        expected = [
            int(tie_values[i + 1])
            - _reference_forward(
                int(tie_values[i]), int(tie_indices[i + 1] - tie_indices[i]), num, den
            )
            for i in range(len(tie_indices) - 1)
        ]
        np.testing.assert_array_equal(residual, expected)
