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
    ToolingConfiguration,
    analyze_tap,
    analyze_tap_data,
    compute_sdof_interpretation,
    dominant_frequency,
    feed_rate_mm_min,
    format_milling_guidance_for_cycle,
    get_rpm_zones,
    load_tap_csv,
    plot_cycle_result_figure,
    plot_milling_dynamics_figure,
    plot_result_figure,
    plot_tooling_configurations_figure,
    rpm_to_avoid,
    suggested_rpm_range,
)
from tap_testing.run_cycle import _combine_tap_csvs


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


class TestFormatMillingGuidanceForCycle:
    """format_milling_guidance_for_cycle for GUI / near real-time display."""

    def test_returns_string_with_fn_avoid_suggested_best(self):
        result = TapTestResult(
            sample_rate_hz=800.0,
            natural_freq_hz=500.0,
            magnitude_axis="magnitude",
            avoid_rpm=[6000.0, 12000.0],
            suggested_rpm_min=8000.0,
            suggested_rpm_max=10000.0,
            n_teeth_used=4,
            harmonic_order_max=5,
            max_rpm=24000.0,
            min_rpm=4000.0,
            tool_diameter_mm=6.0,
            natural_freq_hz_uncertainty=2.0,
        )
        text = format_milling_guidance_for_cycle(result, material_name="6061 aluminum")
        assert "500" in text
        assert "Natural frequency" in text
        assert "Avoid RPM" in text
        assert "Suggested RPM" in text
        assert "Best stability" in text
        assert "Process damping" in text
        assert "Cutting coefficients" in text
        assert "If chatter" in text or "chatter" in text
        assert "6061" in text or "aluminum" in text

    def test_n_lobes_controls_number_of_speeds(self):
        result = TapTestResult(
            sample_rate_hz=800.0,
            natural_freq_hz=500.0,
            magnitude_axis="magnitude",
            avoid_rpm=[],
            suggested_rpm_min=4000.0,
            suggested_rpm_max=24000.0,
            n_teeth_used=4,
            harmonic_order_max=5,
            max_rpm=24000.0,
            min_rpm=4000.0,
            tool_diameter_mm=None,
            natural_freq_hz_uncertainty=None,
        )
        text_3 = format_milling_guidance_for_cycle(result, n_lobes=3)
        text_2 = format_milling_guidance_for_cycle(result, n_lobes=2)
        assert "N0=" in text_3 and "N1=" in text_3 and "N2=" in text_3
        assert "N0=" in text_2 and "N1=" in text_2
        assert "N2=" not in text_2


class TestToolingConfiguration:
    def test_from_tap_result(self, default_flute_count, default_tool_diameter_mm):
        result = TapTestResult(
            sample_rate_hz=800.0,
            natural_freq_hz=500.0,
            magnitude_axis="magnitude",
            avoid_rpm=[6000.0],
            suggested_rpm_min=4000.0,
            suggested_rpm_max=8000.0,
            n_teeth_used=default_flute_count,
            harmonic_order_max=5,
            max_rpm=24000.0,
            min_rpm=4000.0,
            tool_diameter_mm=default_tool_diameter_mm,
            natural_freq_hz_uncertainty=1.0,
        )
        config = ToolingConfiguration.from_tap_result(result, material_name="6061 aluminum")
        assert config.n_teeth == default_flute_count
        assert config.tool_diameter_mm == default_tool_diameter_mm
        assert config.natural_freq_hz == 500.0
        assert config.natural_freq_hz_uncertainty == 1.0
        assert "6061" in config.material_name or "aluminum" in config.material_name

    def test_direct_construction(self):
        config = ToolingConfiguration(
            tool_diameter_mm=19.0,
            n_teeth=4,
            material_name="6061 aluminum",
            natural_freq_hz=900.0,
            label="19mm 4fl",
        )
        assert config.n_teeth == 4
        assert config.label == "19mm 4fl"


class TestPlotToolingConfigurationsFigure:
    def test_plots_one_config(self, tmp_path):
        configs = [
            ToolingConfiguration(
                tool_diameter_mm=6.0,
                n_teeth=3,
                material_name="6061 aluminum",
                natural_freq_hz=100.0,
            ),
        ]
        out = tmp_path / "tooling_one.png"
        fig = plot_tooling_configurations_figure(configs, output_path=out)
        assert len(fig.axes) >= 1
        assert out.exists()
        assert out.stat().st_size > 200

    def test_plots_three_configs_two_columns(self, tmp_path):
        configs = [
            ToolingConfiguration(6.0, 3, "6061 aluminum", 100.0),
            ToolingConfiguration(19.0, 4, "6061 aluminum", 900.0),
            ToolingConfiguration(6.0, 3, "7075 aluminum", 500.0),
        ]
        out = tmp_path / "tooling_three.png"
        fig = plot_tooling_configurations_figure(configs, output_path=out, max_cols=2)
        assert len(fig.axes) >= 3
        assert out.exists()

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError, match="At least one ToolingConfiguration"):
            plot_tooling_configurations_figure([])


class TestComputeSdofInterpretation:
    def test_given_mass_returns_effective_stiffness(self):
        result = TapTestResult(
            sample_rate_hz=800,
            natural_freq_hz=100.0,
            magnitude_axis="magnitude",
            avoid_rpm=[],
            suggested_rpm_min=4000,
            suggested_rpm_max=24000,
            n_teeth_used=4,
            harmonic_order_max=5,
            max_rpm=24000,
            min_rpm=4000,
            tool_diameter_mm=None,
        )
        interp = compute_sdof_interpretation(result, mass_kg=0.1)
        assert "effective_stiffness_n_per_m" in interp
        # k = m * (2π*f_n)²; f_n=100, m=0.1 => k = 0.1 * (200π)² ≈ 39478
        assert 35_000 <= interp["effective_stiffness_n_per_m"] <= 45_000
        assert "nutshell" in interp

    def test_given_stiffness_returns_effective_mass(self):
        result = TapTestResult(
            sample_rate_hz=800,
            natural_freq_hz=100.0,
            magnitude_axis="magnitude",
            avoid_rpm=[],
            suggested_rpm_min=4000,
            suggested_rpm_max=24000,
            n_teeth_used=4,
            harmonic_order_max=5,
            max_rpm=24000,
            min_rpm=4000,
            tool_diameter_mm=None,
        )
        interp = compute_sdof_interpretation(result, stiffness_n_per_m=40_000.0)
        assert "effective_mass_kg" in interp
        # m = k / (2π*f_n)² ≈ 40000 / 394784 ≈ 0.101
        assert 0.09 <= interp["effective_mass_kg"] <= 0.11
        assert "nutshell" in interp


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

    def test_analyze_synthetic_csv_includes_fft_uncertainty(
        self, synthetic_tap_csv, default_flute_count, default_tool_diameter_mm, sample_rate_hz
    ):
        """Result includes measurement uncertainty from FFT frequency resolution."""
        result = analyze_tap(
            synthetic_tap_csv,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",
        )
        assert result.natural_freq_hz_uncertainty is not None
        assert result.natural_freq_hz_uncertainty > 0
        # FFT resolution: df = sample_rate_hz / n; synthetic has 0.5 s * 800 = 400 samples => df=2, half-bin=1
        n = int(sample_rate_hz * 0.5)
        expected_u = 0.5 * (sample_rate_hz / n)
        assert 0.5 <= result.natural_freq_hz_uncertainty <= 2.0 * expected_u


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

    def test_analysis_on_combined_multi_tap_data_uses_several_impact_cycles(
        self, three_tap_csvs, default_flute_count, default_tool_diameter_mm
    ):
        """Analysis is validated on combined data from several impact cycles, not a single tap."""
        t_combined, data_combined, sr = _combine_tap_csvs(three_tap_csvs)
        assert data_combined.shape[1] == 3 * 200  # three taps, 200 samples each
        result = analyze_tap_data(
            t_combined,
            data_combined,
            sr,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",
        )
        # Dominant frequency from combined multi-tap data (fixture uses 100 Hz on x)
        assert 85 <= result.natural_freq_hz <= 115
        assert result.sample_rate_hz == sr

    def test_analyze_tap_data_includes_uncertainty(
        self, sample_rate_hz, default_flute_count, default_tool_diameter_mm
    ):
        """analyze_tap_data sets natural_freq_hz_uncertainty from FFT resolution."""
        n = 512
        t = np.arange(n) / sample_rate_hz
        data = np.stack([np.sin(2 * np.pi * 80 * t), np.zeros(n), np.zeros(n)])
        result = analyze_tap_data(
            t, data, sample_rate_hz,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",
        )
        assert result.natural_freq_hz_uncertainty is not None
        assert result.natural_freq_hz_uncertainty > 0

    def test_tap_spread_std_increases_combined_uncertainty(
        self, sample_rate_hz, default_flute_count, default_tool_diameter_mm
    ):
        """Passing tap_spread_std_hz yields combined uncertainty >= FFT-only."""
        n = 512
        t = np.arange(n) / sample_rate_hz
        data = np.stack([np.sin(2 * np.pi * 80 * t), np.zeros(n), np.zeros(n)])
        r_no_spread = analyze_tap_data(
            t, data, sample_rate_hz,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",
        )
        r_with_spread = analyze_tap_data(
            t, data, sample_rate_hz,
            flute_count=default_flute_count,
            tool_diameter_mm=default_tool_diameter_mm,
            use_axis="x",
            tap_spread_std_hz=0.5,
        )
        assert r_with_spread.natural_freq_hz_uncertainty is not None
        assert r_with_spread.natural_freq_hz_uncertainty >= r_no_spread.natural_freq_hz_uncertainty


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
        assert ax.get_xlabel() == "Spindle speed (rpm)"
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
        fig = plot_cycle_result_figure(result, tap_series_list, output_path=out, figsize=(6, 6))
        assert len(fig.axes) == 4
        ax_rpm, ax_traces, ax_spectrum, ax_milling = fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]
        assert ax_rpm.get_xlabel() == "Spindle speed (rpm)"
        assert ax_traces.get_xlabel() == "Time (s)"
        assert ax_traces.get_ylabel() == "Magnitude (g)"
        assert ax_spectrum.get_xlabel() == "Frequency (Hz)"
        legend_labels = [t.get_text() for t in ax_traces.get_legend().get_texts()]
        assert "Average" in legend_labels
        assert "Tap 1" in legend_labels
        assert "Tap 2" in legend_labels
        assert "Tap 3" in legend_labels
        # Milling dynamics panel is fourth axis
        assert len(ax_milling.get_children()) > 0
        assert out.exists()
        assert out.stat().st_size > 500

    def test_plot_milling_dynamics_figure_6mm_3fl(
        self, tmp_path, default_flute_count, default_tool_diameter_mm
    ):
        """Milling dynamics figure for 6 mm, 3-flute: structure and saved file."""
        result = TapTestResult(
            sample_rate_hz=800.0,
            natural_freq_hz=100.0,
            magnitude_axis="x",
            avoid_rpm=[6000.0],
            suggested_rpm_min=5000.0,
            suggested_rpm_max=7000.0,
            n_teeth_used=default_flute_count,
            harmonic_order_max=5,
            max_rpm=24000.0,
            min_rpm=4000.0,
            tool_diameter_mm=default_tool_diameter_mm,
            natural_freq_hz_uncertainty=0.5,
        )
        out = tmp_path / "milling_dynamics_6mm_3fl.png"
        fig = plot_milling_dynamics_figure(
            result,
            output_path=out,
            n_lobes=3,
            figsize=(8, 5),
        )
        assert len(fig.axes) == 1
        assert out.exists()
        assert out.stat().st_size > 500
