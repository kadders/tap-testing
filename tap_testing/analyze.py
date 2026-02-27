"""
Analyze tap-test data: FFT → dominant natural frequency → spindle speed and feed guidance.

Tap testing excites the tool-holder-spindle structure; the dominant frequency in the
response is (approximately) the first natural frequency. Chatter occurs when tooth
passing frequency or its harmonics match this frequency, so we recommend avoiding
those spindle RPMs and suggest stable pockets between them. Feed is then chosen from
chip load and RPM once a safe speed is selected.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default RPM range for LDO Milo spindle (override max_rpm when testing a different spindle).
DEFAULT_MIN_RPM = 4000.0
DEFAULT_MAX_RPM = 24000.0


@dataclass
class TapTestResult:
    """Results from analyzing one tap-test recording."""

    sample_rate_hz: float
    natural_freq_hz: float
    magnitude_axis: str  # "x", "y", "z", or "magnitude"
    avoid_rpm: list[float]  # RPM values that can excite the mode (first few harmonics)
    suggested_rpm_min: float
    suggested_rpm_max: float
    n_teeth_used: int  # flute count
    harmonic_order_max: int
    max_rpm: float  # max spindle RPM (user-defined, default LDO Milo 24000)
    min_rpm: float  # min spindle RPM for range and chart (default 4000)
    tool_diameter_mm: float | None  # tool diameter in mm (for reference)


def load_tap_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load tap-test CSV with columns t_s, ax_g, ay_g, az_g and optional # sample_rate_hz.

    Returns:
        t: time vector (s)
        data: (3, N) array [ax, ay, az] in g
        sample_rate_hz: from file comment or 0 if missing
    """
    path = Path(path)
    rows = []
    sample_rate_hz = 0.0
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        second = next(reader)
        if str(second[0]).strip().startswith("#"):
            # second row is comment: # sample_rate_hz, value, ...
            try:
                sample_rate_hz = float(second[1])
            except (IndexError, ValueError):
                pass
        else:
            rows.append([float(x) for x in second])
        for row in reader:
            if row and not str(row[0]).strip().startswith("#"):
                rows.append([float(x) for x in row[:4]])
    if not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows)
    t = arr[:, 0]
    data = arr[:, 1:4].T  # (3, N)
    if sample_rate_hz <= 0:
        dt = np.diff(t)
        sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0
    return t, data, sample_rate_hz


def dominant_frequency(
    signal: np.ndarray,
    sample_rate_hz: float,
    axis_label: str = "magnitude",
) -> float:
    """
    One-sided FFT and return frequency (Hz) of the largest magnitude bin.

    Uses the first half of the signal if length is even to avoid DC/nyquist issues.
    """
    n = len(signal)
    if n < 4:
        return 0.0
    # Remove mean to reduce DC
    s = signal - np.mean(signal)
    window = np.hanning(n)
    s = s * window
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    # Skip DC (index 0) and very low frequency
    skip = max(1, int(5 / (sample_rate_hz / n)))
    idx = skip + np.argmax(np.abs(fft[skip:]))
    return float(freqs[idx])


def rpm_to_avoid(
    natural_freq_hz: float,
    n_teeth: int,
    harmonic_orders: int = 5,
    rpm_min: float = DEFAULT_MIN_RPM,
    rpm_max: float = DEFAULT_MAX_RPM,
) -> list[float]:
    """
    RPM values where tooth passing frequency (or harmonics) equals the natural frequency.

    Chatter risk is high when: (RPM/60) * N_teeth * k = natural_freq_hz
    => RPM = 60 * natural_freq_hz / (N_teeth * k)  for k = 1, 2, 3, ...
    Only returns RPMs within [rpm_min, rpm_max].
    """
    out = []
    for k in range(1, harmonic_orders + 1):
        rpm = 60.0 * natural_freq_hz / (n_teeth * k)
        if rpm_min <= rpm <= rpm_max:
            out.append(rpm)
    return sorted(set(round(r, 0) for r in out))


def suggested_rpm_range(
    avoid_rpm: list[float],
    rpm_min: float = DEFAULT_MIN_RPM,
    rpm_max: float = DEFAULT_MAX_RPM,
) -> tuple[float, float]:
    """
    Pick a stable pocket: midpoint of the largest gap between avoid_rpm in [rpm_min, rpm_max].
    """
    points = [r for r in avoid_rpm if rpm_min <= r <= rpm_max]
    points = sorted(set(points))
    if not points:
        return rpm_min, rpm_max
    gaps = []
    gaps.append((rpm_min, points[0], points[0] - rpm_min))
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        gaps.append((a, b, b - a))
    gaps.append((points[-1], rpm_max, rpm_max - points[-1]))
    best = max(gaps, key=lambda g: g[2])
    # Center of the gap, with some margin
    margin = best[2] * 0.2
    low = best[0] + margin
    high = best[1] - margin
    if high <= low:
        low, high = best[0], best[1]
    return round(low, 0), round(high, 0)


def analyze_tap(
    path: str | Path,
    flute_count: int = 4,
    harmonic_orders: int = 5,
    use_axis: str = "magnitude",
    max_rpm: float = DEFAULT_MAX_RPM,
    min_rpm: float = DEFAULT_MIN_RPM,
    tool_diameter_mm: float | None = None,
) -> TapTestResult:
    """
    Load a tap CSV, run FFT, and compute spindle speed guidance.

    Args:
        path: Path to tap-test CSV.
        flute_count: Number of flutes (teeth) for tooth-passing frequency.
        harmonic_orders: How many harmonics of tooth passing to consider for avoid RPM.
        use_axis: "x", "y", "z", or "magnitude" for FFT input.
        max_rpm: Maximum spindle RPM (defines range for avoid/suggested and chart; default 24000 for LDO Milo).
        min_rpm: Minimum spindle RPM for range and chart (default 4000).
        tool_diameter_mm: Tool diameter in mm (stored for reference, e.g. SFM calculations).

    Returns:
        TapTestResult with natural frequency, avoid RPM list, and suggested RPM range.
    """
    t, data, sample_rate_hz = load_tap_csv(path)
    if sample_rate_hz <= 0:
        sample_rate_hz = 1.0 / (float(t[1] - t[0]) if len(t) > 1 else 1.0)

    ax, ay, az = data[0], data[1], data[2]
    if use_axis == "magnitude":
        signal = np.sqrt(ax**2 + ay**2 + az**2)
        axis_label = "magnitude"
    elif use_axis == "x":
        signal = ax
        axis_label = "x"
    elif use_axis == "y":
        signal = ay
        axis_label = "y"
    elif use_axis == "z":
        signal = az
        axis_label = "z"
    else:
        signal = np.sqrt(ax**2 + ay**2 + az**2)
        axis_label = "magnitude"

    natural_freq_hz = dominant_frequency(signal, sample_rate_hz, axis_label)
    avoid_rpm = rpm_to_avoid(
        natural_freq_hz, flute_count, harmonic_orders,
        rpm_min=min_rpm, rpm_max=max_rpm,
    )
    suggested_min, suggested_max = suggested_rpm_range(
        avoid_rpm, rpm_min=min_rpm, rpm_max=max_rpm,
    )

    return TapTestResult(
        sample_rate_hz=sample_rate_hz,
        natural_freq_hz=natural_freq_hz,
        magnitude_axis=axis_label,
        avoid_rpm=avoid_rpm,
        suggested_rpm_min=suggested_min,
        suggested_rpm_max=suggested_max,
        n_teeth_used=flute_count,
        harmonic_order_max=harmonic_orders,
        max_rpm=max_rpm,
        min_rpm=min_rpm,
        tool_diameter_mm=tool_diameter_mm,
    )


def analyze_tap_data(
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    flute_count: int = 4,
    harmonic_orders: int = 5,
    use_axis: str = "magnitude",
    max_rpm: float = DEFAULT_MAX_RPM,
    min_rpm: float = DEFAULT_MIN_RPM,
    tool_diameter_mm: float | None = None,
) -> TapTestResult:
    """
    Run FFT and spindle speed guidance on in-memory tap data (e.g. combined from multiple recordings).

    Args:
        t: Time vector (s), shape (N,).
        data: (3, N) array [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        Other args: same as analyze_tap.

    Returns:
        TapTestResult.
    """
    if sample_rate_hz <= 0 and len(t) > 1:
        sample_rate_hz = 1.0 / float(np.median(np.diff(t)))
    ax, ay, az = data[0], data[1], data[2]
    if use_axis == "magnitude":
        signal = np.sqrt(ax**2 + ay**2 + az**2)
        axis_label = "magnitude"
    elif use_axis == "x":
        signal = ax
        axis_label = "x"
    elif use_axis == "y":
        signal = ay
        axis_label = "y"
    elif use_axis == "z":
        signal = az
        axis_label = "z"
    else:
        signal = np.sqrt(ax**2 + ay**2 + az**2)
        axis_label = "magnitude"

    natural_freq_hz = dominant_frequency(signal, sample_rate_hz, axis_label)
    avoid_rpm = rpm_to_avoid(
        natural_freq_hz, flute_count, harmonic_orders,
        rpm_min=min_rpm, rpm_max=max_rpm,
    )
    suggested_min, suggested_max = suggested_rpm_range(
        avoid_rpm, rpm_min=min_rpm, rpm_max=max_rpm,
    )

    return TapTestResult(
        sample_rate_hz=sample_rate_hz,
        natural_freq_hz=natural_freq_hz,
        magnitude_axis=axis_label,
        avoid_rpm=avoid_rpm,
        suggested_rpm_min=suggested_min,
        suggested_rpm_max=suggested_max,
        n_teeth_used=flute_count,
        harmonic_order_max=harmonic_orders,
        max_rpm=max_rpm,
        min_rpm=min_rpm,
        tool_diameter_mm=tool_diameter_mm,
    )


def feed_rate_mm_min(
    chip_load_mm: float,
    n_teeth: int,
    rpm: float,
) -> float:
    """
    Feed rate in mm/min from chip load (mm per tooth), number of teeth, and spindle RPM.

    F = fz * N_teeth * RPM  (mm/min when fz is mm/tooth).
    """
    return chip_load_mm * n_teeth * rpm


def get_rpm_zones(
    result: TapTestResult,
    avoid_width_fraction: float = 0.05,
    rpm_min: float | None = None,
) -> list[tuple[float, float, str]]:
    """
    Build a list of (rpm_low, rpm_high, 'avoid'|'optimal') bands from result.min_rpm to result.max_rpm.

    Avoid zones are ±avoid_width_fraction around each critical RPM (e.g. 0.05 = ±5%).
    Optimal zones are the gaps between avoid zones.
    """
    rpm_min = result.min_rpm if rpm_min is None else rpm_min
    rpm_max = result.max_rpm
    avoid_bands: list[tuple[float, float]] = []
    for r in result.avoid_rpm:
        half = r * avoid_width_fraction
        avoid_bands.append((max(rpm_min, r - half), min(rpm_max, r + half)))
    # Merge overlapping avoid bands
    avoid_bands.sort(key=lambda b: b[0])
    merged: list[tuple[float, float]] = []
    for a, b in avoid_bands:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    # Build alternating optimal / avoid segments
    zones: list[tuple[float, float, str]] = []
    left = rpm_min
    for a, b in merged:
        if left < a:
            zones.append((left, a, "optimal"))
        zones.append((a, b, "avoid"))
        left = b
    if left < rpm_max:
        zones.append((left, rpm_max, "optimal"))
    return zones


def plot_result(
    result: TapTestResult,
    output_path: str | Path | None = None,
    show: bool = True,
    avoid_width_fraction: float = 0.05,
) -> None:
    """
    Plot spindle RPM range with avoid (red) and optimal (green) bands.

    Optionally save to output_path (e.g. .png or .pdf).
    """
    fig = plot_result_figure(result, output_path=output_path, avoid_width_fraction=avoid_width_fraction)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    plt.close(fig)


def plot_result_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    avoid_width_fraction: float = 0.05,
    figsize: tuple[float, float] = (10, 2.5),
):
    """
    Build and return a matplotlib Figure for the RPM band chart (for embedding in a GUI).

    Caller is responsible for closing the figure if not embedding.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction)
    fig, ax = plt.subplots(figsize=figsize)
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"  # red / green
        alpha = 0.45 if kind == "avoid" else 0.35
        ax.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    ax.set_xlim(result.min_rpm, result.max_rpm)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Spindle speed (RPM)")
    title_parts = [f"Natural freq. {result.natural_freq_hz:.1f} Hz"]
    if result.tool_diameter_mm is not None:
        title_parts.append(f"Tool Ø{result.tool_diameter_mm} mm")
    title_parts.append(f"{result.n_teeth_used} flutes, max {result.max_rpm:.0f} RPM")
    ax.set_title("  ·  ".join(title_parts))
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (chatter risk)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal")
    ax.legend(handles=[avoid_patch, optimal_patch], loc="upper right")
    ax.set_ylabel(" ")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cycle_result_figure(
    result: TapTestResult,
    tap_series_list: list[tuple[np.ndarray, np.ndarray]],
    output_path: str | Path | None = None,
    avoid_width_fraction: float = 0.05,
    figsize: tuple[float, float] = (10, 6),
):
    """
    Build a figure with (1) RPM band chart and (2) three tap cycles + average magnitude vs time.

    tap_series_list: list of (t, data) with data shape (3, N) in g. Typically 3 taps.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax_rpm, ax_traces) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1.2])

    # Top: RPM band chart
    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction)
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"
        alpha = 0.45 if kind == "avoid" else 0.35
        ax_rpm.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    ax_rpm.set_xlim(result.min_rpm, result.max_rpm)
    ax_rpm.set_ylim(0, 1)
    ax_rpm.set_yticks([])
    ax_rpm.set_ylabel(" ")
    ax_rpm.set_xlabel("Spindle speed (RPM)")
    title_parts = [f"Natural freq. {result.natural_freq_hz:.1f} Hz"]
    if result.tool_diameter_mm is not None:
        title_parts.append(f"Tool Ø{result.tool_diameter_mm} mm")
    title_parts.append(f"{result.n_teeth_used} flutes, max {result.max_rpm:.0f} RPM")
    ax_rpm.set_title("  ·  ".join(title_parts))
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (chatter risk)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal")
    ax_rpm.legend(handles=[avoid_patch, optimal_patch], loc="upper right")

    # Bottom: three cycles + average magnitude
    if not tap_series_list:
        ax_traces.text(0.5, 0.5, "No tap series", ha="center", va="center", transform=ax_traces.transAxes)
    else:
        n_min = min(d.shape[1] for _, d in tap_series_list)
        magnitudes: list[np.ndarray] = []
        t_plot = None
        colors = ["#3498db", "#9b59b6", "#1abc9c"]  # blue, purple, teal
        for i, (t, data) in enumerate(tap_series_list):
            ax, ay, az = data[0], data[1], data[2]
            mag = np.sqrt(ax**2 + ay**2 + az**2)
            if mag.shape[0] > n_min:
                mag = mag[:n_min]
                t = t[:n_min]
            if t_plot is None:
                t_plot = t
            elif len(t) > n_min:
                t = t[:n_min]
            magnitudes.append(mag[:n_min])
            label = f"Tap {i + 1}"
            ax_traces.plot(t[:n_min], mag[:n_min], color=colors[i % len(colors)], alpha=0.6, label=label)

        if t_plot is not None and magnitudes:
            t_plot = t_plot[:n_min]
            avg_mag = np.mean(magnitudes, axis=0)
            ax_traces.plot(t_plot, avg_mag, color="black", linewidth=2, label="Average")
        ax_traces.set_xlabel("Time (s)")
        ax_traces.set_ylabel("Magnitude (g)")
        ax_traces.set_title("Tap cycles (independent) and average")
        ax_traces.legend(loc="upper right", fontsize=8)
        ax_traces.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze tap-test CSV and suggest spindle speeds and feed."
    )
    parser.add_argument("csv", type=Path, help="Tap-test CSV file")
    parser.add_argument(
        "-n", "--flutes",
        type=int,
        default=4,
        dest="flute_count",
        help="Flute (tooth) count (default 4)",
    )
    parser.add_argument(
        "--max-rpm",
        type=float,
        default=DEFAULT_MAX_RPM,
        help="Maximum spindle RPM for range and chart (default 24000, LDO Milo; set lower if testing different spindle)",
    )
    parser.add_argument(
        "--tool-diameter",
        type=float,
        default=None,
        metavar="MM",
        dest="tool_diameter_mm",
        help="Tool diameter in mm (for reference and chart title)",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z", "magnitude"),
        default="magnitude",
        help="Axis to use for FFT (default magnitude)",
    )
    parser.add_argument(
        "--chip-load",
        type=float,
        default=None,
        metavar="MM",
        help="Chip load in mm/tooth to print example feed at suggested RPM",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show RPM band chart (red = avoid, green = optimal)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save chart image to file (default: next to CSV as <name>_rpm_chart.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show chart window; image is still saved unless --no-save-chart",
    )
    parser.add_argument(
        "--no-save-chart",
        action="store_true",
        help="Do not save chart image to file",
    )
    args = parser.parse_args()

    r = analyze_tap(
        args.csv,
        flute_count=args.flute_count,
        use_axis=args.axis,
        max_rpm=args.max_rpm,
        tool_diameter_mm=args.tool_diameter_mm,
    )
    print(f"Natural frequency (dominant): {r.natural_freq_hz:.1f} Hz")
    tool_desc = f"Tool: Ø{r.tool_diameter_mm} mm, " if r.tool_diameter_mm is not None else ""
    print(f"{tool_desc}{r.n_teeth_used} flutes, max {r.max_rpm:.0f} RPM")
    print(f"Avoid spindle RPM (tooth-pass resonance): {r.avoid_rpm}")
    print(f"Suggested stable RPM range: {r.suggested_rpm_min:.0f} – {r.suggested_rpm_max:.0f} RPM")
    if args.chip_load is not None:
        rpm_mid = (r.suggested_rpm_min + r.suggested_rpm_max) / 2
        feed = feed_rate_mm_min(args.chip_load, r.n_teeth_used, rpm_mid)
        print(f"At {rpm_mid:.0f} RPM, chip load {args.chip_load} mm/tooth → feed ≈ {feed:.1f} mm/min")

    # Save chart image by default so it can be viewed later
    save_chart = not args.no_save_chart
    chart_path = args.plot_out
    if save_chart and chart_path is None:
        chart_path = args.csv.parent / f"{args.csv.stem}_rpm_chart.png"
    if save_chart and chart_path is not None:
        plot_result_figure(r, output_path=chart_path)
        print(f"Chart saved to {chart_path} (view anytime)")

    do_plot = args.plot and not args.no_plot
    if do_plot:
        plot_result(r, output_path=None, show=True)


if __name__ == "__main__":
    main()
