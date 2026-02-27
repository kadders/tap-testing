"""
Generate example chart images for viewing (synthetic data).
Run from tap-testing dir: python -m tap_testing.generate_example_chart
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analyze import (
    DEFAULT_MAX_RPM,
    DEFAULT_MIN_RPM,
    TapTestResult,
    analyze_tap_data,
    plot_cycle_result_figure,
    plot_result_figure,
)


def _synthetic_tap(sample_rate_hz: float = 800.0, duration_s: float = 0.5, freq_hz: float = 100.0):
    """Create (t, data) for a single tap with dominant frequency freq_hz."""
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    ax = np.sin(2 * np.pi * freq_hz * t) + 0.05 * np.random.randn(n)
    ay = 0.05 * np.random.randn(n)
    az = 0.05 * np.random.randn(n)
    data = np.stack([ax, ay, az])
    return t, data


def generate_example_chart(output_dir: str | Path | None = None) -> Path:
    """
    Write example RPM chart and cycle chart (with 3 taps + average) to output_dir.
    Uses synthetic data. Returns the output directory path.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "example_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = 800.0
    # One longer "combined" signal for analysis
    t1, d1 = _synthetic_tap(sample_rate_hz, 0.5, 100.0)
    t2, d2 = _synthetic_tap(sample_rate_hz, 0.5, 100.0)
    t3, d3 = _synthetic_tap(sample_rate_hz, 0.5, 100.0)
    offset = t1[-1] + 1.0 / sample_rate_hz
    t_combined = np.concatenate([t1, t2 + offset, t3 + 2 * offset])
    data_combined = np.concatenate([d1, d2, d3], axis=1)

    result = analyze_tap_data(
        t_combined,
        data_combined,
        sample_rate_hz,
        flute_count=3,
        max_rpm=DEFAULT_MAX_RPM,
        min_rpm=DEFAULT_MIN_RPM,
        tool_diameter_mm=6.0,
    )

    # RPM band only
    plot_result_figure(
        result,
        output_path=output_dir / "rpm_chart_example.png",
        figsize=(10, 2.5),
    )

    # RPM + 3 taps + average
    tap_series_list = [(t1, d1), (t2, d2), (t3, d3)]
    plot_cycle_result_figure(
        result,
        tap_series_list,
        output_path=output_dir / "cycle_chart_example.png",
        figsize=(10, 6),
    )

    print(f"Example charts written to: {output_dir.resolve()}")
    return output_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate example chart images (synthetic data).")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: tap-testing/example_output)",
    )
    args = parser.parse_args()
    generate_example_chart(args.output_dir)


if __name__ == "__main__":
    main()
