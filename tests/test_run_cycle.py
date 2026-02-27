"""
Tests for tap_testing.run_cycle (combine logic; no hardware or recording).
"""
import numpy as np
import pytest

from tap_testing.analyze import load_tap_csv
from tap_testing.run_cycle import _combine_tap_csvs


class TestCombineTapCsvs:
    def test_combine_two_files(self, two_tap_csvs):
        t, data, sr = _combine_tap_csvs(two_tap_csvs)
        assert data.shape[0] == 3
        assert data.shape[1] == 200  # 100 + 100
        assert sr > 0
        assert t[0] >= 0
        assert t[-1] > t[0]
        # Second segment should be time-offset so no overlap
        t1, _, _ = load_tap_csv(two_tap_csvs[0])
        t2, _, _ = load_tap_csv(two_tap_csvs[1])
        assert t[100] >= t[99] + (t1[1] - t1[0])  # continuity

    def test_combine_empty_raises(self):
        with pytest.raises(ValueError, match="No tap files"):
            _combine_tap_csvs([])
