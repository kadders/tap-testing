"""
Tests for tap_testing.run_cycle (combine logic; no hardware or recording).

We use several impact test cycles (multiple taps) and combine them so analysis
is based on averaged/combined data, not a single dataset.
"""
import numpy as np
import pytest

from tap_testing.analyze import analyze_tap_data, load_tap_csv
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

    def test_combine_three_taps_for_multi_cycle_workflow(self, three_tap_csvs):
        """Combine three impact cycles; combined data length is 3x single tap (no single-dataset analysis)."""
        t, data, sr = _combine_tap_csvs(three_tap_csvs)
        single_n = 200
        assert len(three_tap_csvs) == 3
        assert data.shape[0] == 3
        assert data.shape[1] == 3 * single_n
        assert sr > 0
        assert t[0] >= 0
        assert t[-1] > t[0]
        # Time is continuous across segments
        dt = np.median(np.diff(t))
        assert dt > 0

    def test_analysis_uses_combined_multi_tap_data(self, three_tap_csvs, default_flute_count, default_tool_diameter_mm):
        """Analysis is run on combined data from several impact cycles, not a single tap."""
        t_combined, data_combined, sr = _combine_tap_csvs(three_tap_csvs)
        result = analyze_tap_data(
            t_combined,
            data_combined,
            sr,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",  # fixture puts 100 Hz sine on x; magnitude would peak at 200 Hz
        )
        # Combined series length = 3 taps
        single_tap_samples = 200
        assert data_combined.shape[1] == 3 * single_tap_samples
        assert result is not None
        assert hasattr(result, "natural_freq_hz")
        # Dominant frequency from combined data should be near 100 Hz (three_tap_csvs fixture)
        assert 85 <= result.natural_freq_hz <= 115
