"""
Generate example chart images for viewing (synthetic data).
Run from tap-testing dir: python -m tap_testing.docs.generate_example_chart
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tap_testing.analyze import (
    DEFAULT_MAX_RPM,
    DEFAULT_MIN_RPM,
    TapTestResult,
    ToolingConfiguration,
    analyze_tap_data,
    plot_cycle_result_figure,
    plot_milling_dynamics_figure,
    plot_optimal_loads_figure,
    plot_result_figure,
    plot_spectrum_figure,
    plot_tooling_configurations_figure,
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


def _default_output_dir() -> Path:
    """Repo root example_output (this module lives in tap_testing/docs/)."""
    return Path(__file__).resolve().parent.parent.parent / "example_output"


def generate_example_chart(
    output_dir: str | Path | None = None,
    material_name: str | None = None,
) -> Path:
    """
    Write example RPM chart and cycle chart (with 3 taps + average) to output_dir.
    Uses synthetic data. material_name: workpiece material for labels (default: 6061 aluminum).
    Returns the output directory path.
    """
    if output_dir is None:
        output_dir = _default_output_dir()
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
        material_name=material_name,
    )

    # Standalone FFT spectrum (impact-test analysis: verify dominant peak)
    t_plot = t_combined
    data_plot = data_combined
    plot_spectrum_figure(
        result,
        t_plot,
        data_plot,
        sample_rate_hz,
        output_path=output_dir / "spectrum_example_6mm_3fl.png",
        figsize=(8, 3),
    )

    # RPM + 3 taps + average + spectrum + milling dynamics panel (6 mm, 3 flutes)
    tap_series_list = [(t1, d1), (t2, d2), (t3, d3)]
    cycle_path = output_dir / "cycle_chart_example_6mm_3fl.png"
    plot_cycle_result_figure(
        result,
        tap_series_list,
        output_path=cycle_path,
        figsize=(10, 7),
        material_name=material_name,
    )
    # Also save as legacy name
    import shutil
    shutil.copy(cycle_path, output_dir / "cycle_chart_example.png")

    # Standalone milling dynamics summary (fn, avoid RPM, best lobe speeds)
    plot_milling_dynamics_figure(
        result,
        output_path=output_dir / "milling_dynamics_example_6mm_3fl.png",
        n_lobes=4,
        figsize=(8, 5),
        material_name=material_name,
    )

    # Multi-configuration comparison (different tooling setups in one figure)
    configs = [
        ToolingConfiguration.from_tap_result(result, material_name, label="6 mm 3fl (tap result)"),
        ToolingConfiguration(
            tool_diameter_mm=19.0,
            n_teeth=4,
            material_name=material_name or "6061 aluminum",
            natural_freq_hz=900.0,
            natural_freq_hz_uncertainty=2.0,
            min_rpm=DEFAULT_MIN_RPM,
            max_rpm=DEFAULT_MAX_RPM,
            label="19 mm 4fl (fn=900 Hz)",
        ),
        ToolingConfiguration(
            tool_diameter_mm=6.0,
            n_teeth=3,
            material_name="7075 aluminum",
            natural_freq_hz=500.0,
            min_rpm=DEFAULT_MIN_RPM,
            max_rpm=DEFAULT_MAX_RPM,
            label="6 mm 3fl 7075",
        ),
    ]
    plot_tooling_configurations_figure(
        configs,
        output_path=output_dir / "tooling_configurations_example.png",
        n_lobes=3,
        max_cols=2,
    )

    # Optimal loads: all data points (avoid band, optimal band, best lobe speeds + table)
    plot_optimal_loads_figure(
        result,
        output_path=output_dir / "optimal_loads_example_6mm_3fl.png",
        chip_load_mm=0.05,
        n_lobes=6,
        figsize=(10, 6),
        material_name=material_name,
    )

    from tap_testing.material import get_material_or_default, list_material_names
    mat = get_material_or_default(material_name)
    print(f"Example charts written to: {output_dir.resolve()}")
    print(f"  Material: {mat.name}. All outputs in metric (mm, mm/min, Hz, rpm, g).")
    print(f"  Other materials: --material <name>  (choices: {', '.join(list_material_names())})")
    print("")
    print("  Charts include:")
    print("  · rpm_chart_example.png — RPM bands (avoid/optimal), tool, material, metric")
    print("  · spectrum_example_6mm_3fl.png — Standalone FFT magnitude spectrum (f_n marked)")
    print("  · cycle_chart_example.png / cycle_chart_example_6mm_3fl.png — RPM bands + 3 tap cycles + average + spectrum + milling panel (fn ± uncertainty, best stability speeds, material)")
    print("  · milling_dynamics_example_6mm_3fl.png — Tap result, measurement uncertainty, workpiece material (--material), tooth passing, best stability lobe speeds, metric")
    print("  · tooling_configurations_example.png — Side-by-side comparison of tooling configs (fn, avoid RPM, best stability, hints)")
    print("  · optimal_loads_example_6mm_3fl.png — All optimal-load data points: RPM bands, best lobe speeds (N=0,1,…), table (RPM, feed, f_tooth)")
    return output_dir


def generate_optimal_loads_chart(
    output_dir: str | Path | None = None,
    tool_diameter_mm: float = 6.0,
    flutes: int = 3,
    natural_freq_hz: float = 100.0,
    chip_load_mm: float = 0.05,
    material_name: str | None = None,
    min_rpm: float = DEFAULT_MIN_RPM,
    max_rpm: float = DEFAULT_MAX_RPM,
    n_lobes: int = 6,
) -> Path:
    """
    Generate only the optimal-loads chart for a given tool input (no tap data).

    Use this to see all recommended operating points for a specific tool:
    diameter, flute count, natural frequency, and optional chip load.
    """
    if output_dir is None:
        output_dir = _default_output_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from tap_testing.analyze import TapTestResult, rpm_to_avoid, suggested_rpm_range, plot_optimal_loads_figure
    from tap_testing.material import get_material_or_default

    get_material_or_default(material_name)
    avoid_rpm = rpm_to_avoid(
        natural_freq_hz, flutes, rpm_min=min_rpm, rpm_max=max_rpm
    )
    sug_min, sug_max = suggested_rpm_range(avoid_rpm, rpm_min=min_rpm, rpm_max=max_rpm)
    result = TapTestResult(
        sample_rate_hz=0.0,
        natural_freq_hz=natural_freq_hz,
        magnitude_axis="magnitude",
        avoid_rpm=avoid_rpm,
        suggested_rpm_min=sug_min,
        suggested_rpm_max=sug_max,
        n_teeth_used=flutes,
        harmonic_order_max=5,
        max_rpm=max_rpm,
        min_rpm=min_rpm,
        tool_diameter_mm=tool_diameter_mm,
        tool_material=None,  # use config default (carbide) for display
        natural_freq_hz_uncertainty=None,
    )
    filename = f"optimal_loads_{int(tool_diameter_mm)}mm_{flutes}fl.png"
    plot_optimal_loads_figure(
        result,
        output_path=output_dir / filename,
        chip_load_mm=chip_load_mm,
        n_lobes=n_lobes,
        material_name=material_name,
    )
    print(f"Optimal loads chart saved to {output_dir / filename}")
    return output_dir


def main() -> None:
    import argparse
    from tap_testing.material import list_material_names
    parser = argparse.ArgumentParser(description="Generate example chart images (synthetic data).")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: tap-testing/example_output)",
    )
    parser.add_argument(
        "--material",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Workpiece material for chart labels (default: 6061 aluminum). Choices: {', '.join(list_material_names())}",
    )
    parser.add_argument(
        "--optimal-loads-only",
        action="store_true",
        help="Generate only the optimal-loads chart for the given tool (use with --tool-diameter, --flutes, --natural-freq, --chip-load)",
    )
    parser.add_argument(
        "--tool-diameter",
        type=float,
        default=6.0,
        metavar="MM",
        help="Tool diameter in mm (for optimal-loads chart; default 6)",
    )
    parser.add_argument(
        "--flutes",
        type=int,
        default=3,
        metavar="N",
        help="Number of flutes (for optimal-loads chart; default 3)",
    )
    parser.add_argument(
        "--natural-freq",
        type=float,
        default=100.0,
        metavar="HZ",
        dest="natural_freq_hz",
        help="Natural frequency in Hz (for optimal-loads chart; default 100)",
    )
    parser.add_argument(
        "--chip-load",
        type=float,
        default=0.05,
        metavar="MM",
        dest="chip_load_mm",
        help="Chip load in mm/tooth for feed column (default 0.05)",
    )
    parser.add_argument(
        "--n-lobes",
        type=int,
        default=6,
        help="Number of stability lobes to show (default 6)",
    )
    args = parser.parse_args()

    if args.optimal_loads_only:
        generate_optimal_loads_chart(
            output_dir=args.output_dir,
            tool_diameter_mm=args.tool_diameter,
            flutes=args.flutes,
            natural_freq_hz=args.natural_freq_hz,
            chip_load_mm=args.chip_load_mm,
            material_name=args.material,
            n_lobes=args.n_lobes,
        )
    else:
        generate_example_chart(args.output_dir, material_name=args.material)


if __name__ == "__main__":
    main()
