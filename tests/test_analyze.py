"""
Tests for tap_testing.analyze (no hardware required; uses synthetic data).
"""
from pathlib import Path

import numpy as np
import pytest

from tap_testing.analyze import (
    DEFAULT_MAX_RPM,
    DEFAULT_MIN_RPM,
    TapTestResult,
    analyze_tap,
    analyze_tap_data,
    dominant_frequency,
    feed_rate_mm_min,
    get_rpm_zones,
    load_tap_csv,
    plot_cycle_result_figure,
    plot_result_figure,
    rpm_to_avoid,
    suggested_rpm_range,
)


class TestLoadTapCsv:
    def test_load_synthetic_with_comment(self, synthetic_tap_csv, sample_rate_hz):
        t, data, sr = load_tap_csv(synthetic_tap_csv)
        assert sr == sample_rate_hz
        assert data.shape[0] == 3
        assert data.shape[1] == len(t)
        assert t[0] == 0.0
        assert np.all(np.diff(t) > 0)

    def test_load_without_comment_infers_rate(self, synthetic_tap_csv_no_comment, sample_rate_hz):
        t, data, sr = load_tap_csv(synthetic_tap_csv_no_comment)
        assert sr > 0
        assert abs(sr - sample_rate_hz) < 10.0  # inferred from dt
        assert data.shape[1] == len(t)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tap_csv(tmp_path / "nonexistent.csv")


class TestDominantFrequency:
    def test_sine_at_100_hz(self, sample_rate_hz):
        n = 2048
        t = np.arange(n) / sample_rate_hz
        signal = np.sin(2 * np.pi * 100.0 * t)
        f = dominant_frequency(signal, sample_rate_hz)
        assert 95 <= f <= 105

    def test_short_signal_returns_zero(self, sample_rate_hz):
        signal = np.array([1.0, 2.0])
        f = dominant_frequency(signal, sample_rate_hz)
        assert f == 0.0


class TestRpmToAvoid:
    def test_formula(self):
        # 100 Hz, 4 teeth => RPM = 60*100/4 = 1500 for k=1
        out = rpm_to_avoid(100.0, 4, harmonic_orders=3, rpm_min=100, rpm_max=10000)
        assert 1500 in out
        assert 750 in out  # k=2
        assert 500 in out  # k=3

    def test_respects_rpm_bounds(self):
        out = rpm_to_avoid(100.0, 4, rpm_min=4000, rpm_max=24000)
        for r in out:
            assert 4000 <= r <= 24000
        # 1500 is below 4000 so should not appear
        assert 1500 not in out


class TestSuggestedRpmRange:
    def test_empty_avoid_returns_full_range(self):
        lo, hi = suggested_rpm_range([], rpm_min=4000, rpm_max=24000)
        assert lo == 4000
        assert hi == 24000

    def test_picks_gap_between_avoid(self):
        avoid = [5000.0, 10000.0, 15000.0]  # gaps: 4k-5k, 5k-10k, 10k-15k, 15k-24k
        lo, hi = suggested_rpm_range(avoid, rpm_min=4000, rpm_max=24000)
        assert 4000 <= lo < hi <= 24000
        # Largest gap is 15k-24k (9k wide), so suggested should be in that range (with margin)
        assert lo >= 15000 or hi <= 15000  # at least one end in the big gap


class TestAnalyzeTap:
    def test_analyze_synthetic_csv_detects_near_100_hz(
        self, synthetic_tap_csv, default_flute_count, default_tool_diameter_mm
    ):
        # Default tooling: 6 mm, 3 flute endmill. Use x axis for FFT (magnitude would double freq).
        result = analyze_tap(
            synthetic_tap_csv,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            max_rpm=24000,
            min_rpm=4000,
            use_axis="x",
        )
        assert 90 <= result.natural_freq_hz <= 110
        assert result.n_teeth_used == default_flute_count
        assert result.tool_diameter_mm == default_tool_diameter_mm
        assert result.max_rpm == 24000
        assert result.min_rpm == 4000
        # With 100 Hz and 3 flutes, critical RPMs are 2000, 1000, ... all below min_rpm 4000, so avoid_rpm can be empty
        assert result.suggested_rpm_min < result.suggested_rpm_max


class TestAnalyzeTapData:
    def test_from_arrays(
        self, sample_rate_hz, default_flute_count, default_tool_diameter_mm
    ):
        n = 2048
        t = np.arange(n) / sample_rate_hz
        freq = 80.0
        ax = np.sin(2 * np.pi * freq * t)
        ay = np.zeros(n)
        az = np.zeros(n)
        data = np.stack([ax, ay, az])
        # Default tooling: 6 mm, 3 flute. use_axis="x" so we get 80 Hz.
        result = analyze_tap_data(
            t,
            data,
            sample_rate_hz,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            max_rpm=24000,
            use_axis="x",
        )
        assert 75 <= result.natural_freq_hz <= 85
        assert result.sample_rate_hz == sample_rate_hz
        assert result.n_teeth_used == default_flute_count
        assert result.tool_diameter_mm == default_tool_diameter_mm


class TestFeedRateMmMin:
    def test_formula(self):
        # 0.05 mm/tooth * 4 teeth * 3000 RPM = 600 mm/min
        f = feed_rate_mm_min(0.05, 4, 3000.0)
        assert f == 600.0

    def test_default_tooling_feed(self, default_flute_count):
        # Default: 3 flute. 0.05 mm/tooth * 3 flutes * 3000 RPM = 450 mm/min
        f = feed_rate_mm_min(0.05, default_flute_count, 3000.0)
        assert f == pytest.approx(450.0)


class TestGetRpmZones:
    def test_zones_alternate_optimal_avoid(self, default_flute_count, default_tool_diameter_mm):
        result = TapTestResult(
            sample_rate_hz=800,
            natural_freq_hz=100,
            magnitude_axis="magnitude",
            avoid_rpm=[6000.0],
            suggested_rpm_min=5000,
            suggested_rpm_max=7000,
            n_teeth_used=default_flute_count,
            harmonic_order_max=5,
            max_rpm=24000,
            min_rpm=4000,
            tool_diameter_mm=default_tool_diameter_mm,
        )
        zones = get_rpm_zones(result, avoid_width_fraction=0.1)
        kinds = [z[2] for z in zones]
        assert "optimal" in kinds
        assert "avoid" in kinds
        for i, (lo, hi, kind) in enumerate(zones):
            assert lo < hi
            if i > 0:
                assert zones[i][0] >= zones[i - 1][1] - 1e-6  # contiguous


class TestPlotResultFigure:
    def test_returns_figure_and_does_not_raise(
        self, synthetic_tap_csv, tmp_path, default_flute_count, default_tool_diameter_mm
    ):
        result = analyze_tap(
            synthetic_tap_csv,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
        )
        fig = plot_result_figure(result, output_path=tmp_path / "out.png", figsize=(6, 2))
        assert fig is not None
        assert (tmp_path / "out.png").exists()

    def test_plot_result_figure_structure_and_saved_file(
        self, synthetic_tap_csv, tmp_path, default_flute_count, default_tool_diameter_mm
    ):
        """Validate RPM band figure has expected axes and saves a valid PNG."""
        result = analyze_tap(
            synthetic_tap_csv,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
        )
        out = tmp_path / "rpm_chart.png"
        fig = plot_result_figure(result, output_path=out, figsize=(6, 2))
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[0] == result.min_rpm
        assert xlim[1] == result.max_rpm
        assert ax.get_xlabel() == "Spindle speed (RPM)"
        assert out.exists()
        assert out.stat().st_size > 500  # PNG has non-trivial content

    def test_plot_cycle_result_figure_structure_and_saved_file(
        self, synthetic_tap_csv, tmp_path, default_flute_count, default_tool_diameter_mm
    ):
        """Validate combined figure has RPM panel + traces panel and saves valid PNG."""
        result = analyze_tap(
            synthetic_tap_csv,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
        )
        # Three short synthetic tap series (same length)
        n = 100
        t = np.linspace(0, 0.1, n)
        tap_series_list = [
            (t, np.random.randn(3, n) * 0.1 + np.array([[0], [0], [1]])),  # small noise + 1g z
            (t, np.random.randn(3, n) * 0.1 + np.array([[0], [0], [1]])),
            (t, np.random.randn(3, n) * 0.1 + np.array([[0], [0], [1]])),
        ]
        out = tmp_path / "cycle_chart.png"
        fig = plot_cycle_result_figure(result, tap_series_list, output_path=out, figsize=(6, 5))
        assert len(fig.axes) == 2
        ax_rpm, ax_traces = fig.axes[0], fig.axes[1]
        assert ax_rpm.get_xlabel() == "Spindle speed (RPM)"
        assert ax_traces.get_xlabel() == "Time (s)"
        assert ax_traces.get_ylabel() == "Magnitude (g)"
        legend_labels = [t.get_text() for t in ax_traces.get_legend().get_texts()]
        assert "Average" in legend_labels
        assert "Tap 1" in legend_labels
        assert "Tap 2" in legend_labels
        assert "Tap 3" in legend_labels
        assert out.exists()
        assert out.stat().st_size > 500
