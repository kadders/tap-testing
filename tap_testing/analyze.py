"""
Analyze tap-test data: FFT → dominant natural frequency → spindle speed and feed guidance.

Theory (SDOF):
  • Free vibration: natural frequency f_n = (1/(2π))√(k/m); tap test identifies f_n.
  • Forced vibration: periodic excitation (e.g. tooth-passing) causes large response when
    forcing frequency ≈ f_n (resonance); we avoid those spindle RPMs.
  • Self-excited vibration: steady input modulated at a natural frequency (e.g. chatter in
    machining); avoiding forced resonance reduces the risk of exciting chatter.

We recommend avoiding RPMs where tooth-passing frequency or its harmonics match f_n, and
suggest stable pockets between them. Feed is then chosen from chip load and RPM.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tap_testing import sdof
from tap_testing.measurement_uncertainties import (
    combined_natural_freq_uncertainty_hz,
    fft_frequency_resolution_uncertainty_hz,
)

# Default RPM range for LDO Milo spindle (override max_rpm when testing a different spindle).
DEFAULT_MIN_RPM = 9000.0
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
    min_rpm: float  # min spindle RPM for range and chart (default 9000)
    tool_diameter_mm: float | None  # tool diameter in mm (for reference)
    tool_material: str | None = None  # cutter material (e.g. "carbide", "HSS"); None = use config default for display
    natural_freq_hz_uncertainty: float | None = None  # standard uncertainty from FFT resolution and/or tap spread
    spindle_operating_frequency_hz: float | None = None  # default spindle operating frequency (rev/s); rpm = this * 60


@dataclass
class ToolingConfiguration:
    """
    One tooling setup for comparison: tool geometry, material, and natural frequency.

    Used by plot_tooling_configurations_figure to visualize multiple configurations
    (e.g. different diameters, flute counts, materials) in one figure.
    """

    tool_diameter_mm: float | None
    n_teeth: int
    material_name: str
    natural_freq_hz: float
    natural_freq_hz_uncertainty: float | None = None
    min_rpm: float = DEFAULT_MIN_RPM
    max_rpm: float = DEFAULT_MAX_RPM
    harmonic_orders: int = 5
    label: str | None = None  # optional short label, e.g. "6mm 3fl 6061"

    @classmethod
    def from_tap_result(
        cls,
        result: TapTestResult,
        material_name: str | None = None,
        *,
        label: str | None = None,
    ) -> ToolingConfiguration:
        """Build a ToolingConfiguration from a TapTestResult (e.g. after a tap-test cycle)."""
        from tap_testing.config import get_config
        if material_name is None:
            material_name = get_config().material_name
        return cls(
            tool_diameter_mm=result.tool_diameter_mm,
            n_teeth=result.n_teeth_used,
            material_name=material_name,
            natural_freq_hz=result.natural_freq_hz,
            natural_freq_hz_uncertainty=result.natural_freq_hz_uncertainty,
            min_rpm=result.min_rpm,
            max_rpm=result.max_rpm,
            harmonic_orders=result.harmonic_order_max,
            label=label,
        )


def _result_tool_material_label(result: TapTestResult) -> str:
    """Resolve tool material label for display: result.tool_material or config default (e.g. 'Carbide')."""
    from tap_testing.config import get_config
    from tap_testing.material import get_tool_material_label, normalize_tool_material
    raw = result.tool_material or get_config().tool_material
    return get_tool_material_label(normalize_tool_material(raw) if raw else None)


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


def _tap_file_sort_key(path: Path) -> int:
    """Sort key for tap_1.csv, tap_2.csv, ... (numeric order)."""
    m = re.match(r"tap_(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def discover_cycle_directory(cycle_dir: Path) -> tuple[Path | None, list[Path], Path | None, Path]:
    """
    Discover tap, combined, and homing CSVs in a cycle run directory (e.g. data/cycle/20250303_120000).

    Looks for tap_1.csv, tap_2.csv, ... and combined.csv (as produced by run_cycle / cycle_gui),
    and homing.csv (as produced by homing_gui).

    Returns:
        combined_path: Path to combined.csv if present, else None.
        tap_paths: Sorted list of tap_*.csv paths (by numeric index).
        homing_path: Path to homing.csv if present, else None.
        output_base: Path with .parent = cycle_dir and .stem = cycle_dir.name, for output filenames.
    """
    if not cycle_dir.is_dir():
        raise ValueError(f"Not a directory: {cycle_dir}")
    tap_paths = sorted(cycle_dir.glob("tap_*.csv"), key=_tap_file_sort_key)
    combined_path = cycle_dir / "combined.csv"
    combined_path = combined_path if combined_path.exists() else None
    homing_path = cycle_dir / "homing.csv"
    homing_path = homing_path if homing_path.exists() else None
    output_base = cycle_dir / cycle_dir.name  # .parent = cycle_dir, .stem = cycle_dir.name
    return combined_path, tap_paths, homing_path, output_base


def combine_tap_csvs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load multiple tap CSVs and concatenate (t, ax, ay, az) with continuous time.

    Returns:
        t: combined time (s)
        data: (3, N) combined [ax, ay, az]
        sample_rate_hz
    """
    if not paths:
        raise ValueError("No tap files to combine")
    all_t: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    sample_rate_hz = 0.0
    offset = 0.0
    dt = 0.0
    for p in paths:
        t, data, sr = load_tap_csv(p)
        if sample_rate_hz <= 0:
            sample_rate_hz = sr
        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
        all_t.append(t + offset)
        all_data.append(data)
        offset = offset + float(t[-1]) - float(t[0]) + dt
    t_combined = np.concatenate(all_t)
    data_combined = np.concatenate(all_data, axis=1)
    return t_combined, data_combined, sample_rate_hz


def load_tap_csv_with_force(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """
    Load tap-test CSV; if force columns (Fx_N, Fy_N or F_N) exist, return them.

    Same as load_tap_csv for t, data, sample_rate_hz. If header contains Fx_N and/or
    Fy_N (or a single F_N), returns force_x, force_y (in N) for FRF computation.
    Otherwise force_x and force_y are None.

    Returns:
        t, data, sample_rate_hz: as in load_tap_csv.
        force_x: Force in x (N) or None.
        force_y: Force in y (N) or None (if F_N only, force_x = force_y = that column).
    """
    path = Path(path)
    rows: list[list[float]] = []
    sample_rate_hz = 0.0
    header: list[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = [c.strip().lower() for c in next(reader)]
        second = next(reader)
        if second and str(second[0]).strip().startswith("#"):
            try:
                sample_rate_hz = float(second[1])
            except (IndexError, ValueError):
                pass
        else:
            try:
                rows.append([float(x) for x in second[: max(len(header), 8)]])
            except (ValueError, TypeError):
                pass
        for row in reader:
            if row and not str(row[0]).strip().startswith("#"):
                try:
                    rows.append([float(x) for x in row[: max(len(header), 8)]])
                except (ValueError, TypeError):
                    pass
    if not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows)
    ncol = arr.shape[1]
    idx_t = header.index("t_s") if "t_s" in header else 0
    idx_ax = header.index("ax_g") if "ax_g" in header else 1
    idx_ay = header.index("ay_g") if "ay_g" in header else 2
    idx_az = header.index("az_g") if "az_g" in header else 3
    idx_fx = header.index("fx_n") if "fx_n" in header and ncol > 4 else None
    idx_fy = header.index("fy_n") if "fy_n" in header and ncol > 5 else None
    idx_f = header.index("f_n") if "f_n" in header and (idx_fx is None and idx_fy is None) and ncol > 4 else None
    if idx_fx is not None and (idx_fx >= ncol or (idx_fy is not None and idx_fy >= ncol)):
        idx_fx = idx_fy = None
    if idx_f is not None and idx_f >= ncol:
        idx_f = None
    # FRF needs at least one force column: Fx_N and Fy_N (both), single F_N, or Fx_N only
    has_force = (idx_fx is not None and idx_fy is not None) or idx_f is not None or idx_fx is not None
    t = arr[:, idx_t]
    data = np.stack([arr[:, idx_ax], arr[:, idx_ay], arr[:, idx_az]])
    if sample_rate_hz <= 0 and len(t) > 1:
        dt = np.diff(t)
        sample_rate_hz = float(1.0 / np.median(dt))
    force_x = force_y = None
    if has_force:
        if idx_fx is not None and idx_fy is not None:
            force_x = arr[:, idx_fx]
            force_y = arr[:, idx_fy]
        elif idx_f is not None:
            f_one = arr[:, idx_f]
            force_x = force_y = f_one
        elif idx_fx is not None:
            force_x = arr[:, idx_fx]
            force_y = None
    return t, data, sample_rate_hz, force_x, force_y


def preprocess_tap_data_for_analysis(
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    *,
    subtract_background: bool = True,
    tap_cycle_duration_s: float | None = None,
    background_duration_s: float | None = None,
    background_fraction: float | None = None,
    impact_threshold_g: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optionally extract tap cycle and subtract background vibration for cleaner analysis.

    When subtract_background is True, detects impact, defines ~1 s tap phase, estimates
    baseline from pre-tap segment, and subtracts it from the tap data. Returns the
    cleaned (t, data, sample_rate_hz) for the tap cycle only.

    Args:
        t: Time vector (s).
        data: (3, N) [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        subtract_background: If True, run tap-cycle extraction and baseline subtraction.
        tap_cycle_duration_s, background_duration_s, background_fraction, impact_threshold_g:
            Overrides for config; None = use config defaults.

    Returns:
        (t_processed, data_processed, sample_rate_hz) — either original or tap-cycle cleaned.
    """
    if not subtract_background:
        return t, data, sample_rate_hz
    from tap_testing.config import get_config
    from tap_testing.tap_cycle import extract_tap_cycle_and_subtract_background

    cfg = get_config()
    tap_dur = tap_cycle_duration_s if tap_cycle_duration_s is not None else cfg.tap_cycle_duration_s
    bg_dur = background_duration_s if background_duration_s is not None else cfg.background_duration_s
    bg_frac = background_fraction if background_fraction is not None else cfg.background_fraction
    thresh = impact_threshold_g if impact_threshold_g is not None else cfg.impact_threshold_g

    result = extract_tap_cycle_and_subtract_background(
        t,
        data,
        sample_rate_hz,
        tap_cycle_duration_s=tap_dur,
        background_duration_s=bg_dur,
        background_fraction=bg_frac,
        impact_threshold_g=thresh,
        use_impact_detection=True,
    )
    return result.t, result.data, result.sample_rate_hz


# Minimum signal level to consider tap data valid (g). Below this we treat as no signal.
_MIN_TAP_SIGNAL_STD_G = 1e-6
_MIN_TAP_SIGNAL_PEAK_G = 1e-5


def check_tap_signal(data: np.ndarray) -> tuple[bool, str]:
    """
    Check whether (3, N) tap data has usable signal (not all zeros or constant).

    Returns:
        (ok, message): ok True if at least one axis has non-negligible variation.
    """
    if data is None or data.size == 0:
        return False, "No data"
    ax, ay, az = data[0], data[1], data[2]
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    std_mag = float(np.std(mag))
    peak_mag = float(np.max(np.abs(mag)))
    if std_mag < _MIN_TAP_SIGNAL_STD_G and peak_mag < _MIN_TAP_SIGNAL_PEAK_G:
        return False, (
            "Tap data has no usable signal (all zeros or constant). "
            "Check accelerometer connection and wiring (I2C/SPI)."
        )
    return True, "OK"


def dominant_frequency(
    signal: np.ndarray,
    sample_rate_hz: float,
    axis_label: str = "magnitude",
) -> float:
    """
    One-sided FFT over the **entire** signal; return frequency (Hz) of the largest magnitude bin.

    The FFT is computed on the full-duration signal (after demean and Hanning window), so the
    result reflects the dominant frequency content over the whole record, not just the start.
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
    mag_fft = np.abs(fft[skip:])
    if np.max(mag_fft) < 1e-12:
        return 0.0  # No meaningful frequency content
    idx = skip + np.argmax(mag_fft)
    return float(freqs[idx])


def dominant_frequency_over_duration(
    signal: np.ndarray,
    sample_rate_hz: float,
    axis_label: str = "magnitude",
    n_windows: int = 5,
    min_samples_per_window: int = 256,
) -> float:
    """
    Dominant frequency representing the **overall common vibration** over the full dataset.

    Splits the signal into non-overlapping segments (windows), computes the dominant
    frequency in each segment, and returns the median. This avoids biasing the result
    toward the start of the record (e.g. initial tap transient) and emphasizes the
    frequency that persists across the duration. If the signal is too short to window
    meaningfully, falls back to a single FFT over the full signal.

    Args:
        signal: Time-series (e.g. magnitude or single axis).
        sample_rate_hz: Sample rate in Hz.
        axis_label: For logging only.
        n_windows: Number of segments to use when signal is long enough.
        min_samples_per_window: Minimum length per segment for FFT (need enough for resolution).

    Returns:
        Dominant frequency in Hz (median over windows, or single full-signal result if short).
    """
    n = len(signal)
    if n < 4:
        return 0.0
    total_samples_needed = n_windows * min_samples_per_window
    if n < total_samples_needed:
        # Signal too short to window; use full-duration FFT (already reflects whole record)
        return dominant_frequency(signal, sample_rate_hz, axis_label)
    # Non-overlapping segments
    segment_length = n // n_windows
    freqs: list[float] = []
    for i in range(n_windows):
        start = i * segment_length
        end = start + segment_length
        if end > n:
            break
        seg = signal[start:end]
        f = dominant_frequency(seg, sample_rate_hz, axis_label)
        if f > 0:
            freqs.append(f)
    if not freqs:
        return dominant_frequency(signal, sample_rate_hz, axis_label)
    return float(np.median(freqs))


def fft_magnitude_spectrum(
    signal: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided FFT magnitude spectrum for impact-test visualization.

    Same preprocessing as dominant_frequency (demean, Hanning). Returns
    frequency vector (Hz) and magnitude (same length as rfft output).

    Returns:
        freqs: Frequency in Hz, shape (n_fft,).
        magnitude: Magnitude of rfft(signal), shape (n_fft,).
    """
    n = len(signal)
    if n < 4:
        return np.array([]), np.array([])
    s = signal - np.mean(signal)
    window = np.hanning(n)
    s = s * window
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    magnitude = np.abs(fft)
    return freqs, magnitude


# ----- Chatter identification (cutting recording) -----

def _spectrum_peaks(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    n_peaks: int = 10,
    min_separation_hz: float = 3.0,
) -> list[tuple[float, float]]:
    """
    Find top frequency peaks (local maxima) with minimum separation.

    Returns list of (freq_hz, magnitude) sorted by magnitude descending.
    """
    if len(freqs) < 3 or n_peaks <= 0:
        return []
    # Local maxima: greater than both neighbors
    peak_idx: list[int] = []
    for i in range(1, len(freqs) - 1):
        if magnitude[i] >= magnitude[i - 1] and magnitude[i] >= magnitude[i + 1]:
            peak_idx.append(i)
    if not peak_idx:
        return []
    # Sort by magnitude descending, then enforce min separation (keep higher magnitude)
    by_mag = sorted(peak_idx, key=lambda j: magnitude[j], reverse=True)
    kept: list[int] = []
    for j in by_mag:
        f_j = float(freqs[j])
        if any(abs(f_j - float(freqs[k])) < min_separation_hz for k in kept):
            continue
        kept.append(j)
        if len(kept) >= n_peaks:
            break
    return [(float(freqs[k]), float(magnitude[k])) for k in kept]


@dataclass
class ChatterAssessmentResult:
    """Result of chatter assessment on a cutting recording."""

    is_chatter_likely: bool
    dominant_freq_hz: float
    peak_near_natural_hz: float | None  # strongest peak within fn band, or None
    natural_freq_hz: float
    natural_freq_band_hz: tuple[float, float]
    peaks: list[tuple[float, float]]  # (freq, magnitude) top peaks
    suggested_rpm: list[float]  # stability lobe speeds using detected or fn (if n_teeth set)


def assess_chatter(
    signal: np.ndarray,
    sample_rate_hz: float,
    natural_freq_hz: float,
    natural_freq_uncertainty_hz: float | None = None,
    n_teeth: int | None = None,
    rpm: float | None = None,
    band_fraction: float = 0.03,
) -> ChatterAssessmentResult:
    """
    Assess whether a cutting recording shows likely chatter (peak near natural frequency).

    Chatter is self-excited vibration at or near a structural natural frequency. If the
    spectrum of the cutting signal has a significant peak within the natural-frequency
    band (fn ± uncertainty or ± band_fraction×fn), chatter is likely. Tooth-passing
    content is not treated as chatter; use n_teeth and rpm to interpret peaks.

    Args:
        signal: Vibration signal (e.g. magnitude sqrt(ax²+ay²+az²)) from cutting.
        sample_rate_hz: Sample rate in Hz.
        natural_freq_hz: Natural frequency from tap test (Hz).
        natural_freq_uncertainty_hz: Optional ± band around fn (Hz). If None, use band_fraction×fn.
        n_teeth: Optional flute count for suggested stable RPM.
        rpm: Optional current spindle RPM (for context / tooth-passing check).
        band_fraction: Fraction of fn for band if natural_freq_uncertainty_hz not set (default 0.03).

    Returns:
        ChatterAssessmentResult with is_chatter_likely, dominant_freq_hz, peak_near_natural_hz, etc.
    """
    if natural_freq_hz <= 0 or sample_rate_hz <= 0 or len(signal) < 4:
        return ChatterAssessmentResult(
            is_chatter_likely=False,
            dominant_freq_hz=0.0,
            peak_near_natural_hz=None,
            natural_freq_hz=natural_freq_hz,
            natural_freq_band_hz=(natural_freq_hz, natural_freq_hz),
            peaks=[],
            suggested_rpm=[],
        )
    u_hz = natural_freq_uncertainty_hz
    if u_hz is None or u_hz <= 0:
        u_hz = natural_freq_hz * band_fraction
    fn_lo = natural_freq_hz - u_hz
    fn_hi = natural_freq_hz + u_hz

    freqs, magnitude = fft_magnitude_spectrum(signal, sample_rate_hz)
    if len(freqs) == 0:
        return ChatterAssessmentResult(
            is_chatter_likely=False,
            dominant_freq_hz=0.0,
            peak_near_natural_hz=None,
            natural_freq_hz=natural_freq_hz,
            natural_freq_band_hz=(fn_lo, fn_hi),
            peaks=[],
            suggested_rpm=[],
        )

    peaks = _spectrum_peaks(freqs, magnitude, n_peaks=10, min_separation_hz=min(3.0, sample_rate_hz / len(signal) * 2))
    dominant_freq_hz = float(freqs[np.argmax(magnitude[1:]) + 1]) if len(freqs) > 1 else 0.0

    # Strongest peak within fn band
    peak_near_natural_hz = None
    peak_near_mag = 0.0
    for f, mag in peaks:
        if fn_lo <= f <= fn_hi and mag > peak_near_mag:
            peak_near_natural_hz = f
            peak_near_mag = mag

    # Chatter likely if there is a substantial peak in the fn band (and optionally
    # it's not obviously tooth-passing when rpm/n_teeth given)
    f_tooth_hz = None
    if n_teeth is not None and rpm is not None and rpm > 0 and n_teeth > 0:
        f_tooth_hz = rpm * n_teeth / 60.0
    is_chatter = peak_near_natural_hz is not None
    if is_chatter and f_tooth_hz is not None and peak_near_natural_hz is not None:
        # If the peak is very close to tooth-passing or a harmonic, could be forced resonance
        for k in range(1, 6):
            if abs(peak_near_natural_hz - k * f_tooth_hz) < u_hz:
                # Peak coincides with tooth-pass harmonic; still treat as bad (resonance/chatter)
                break
        else:
            pass  # peak not at tooth-pass → clearer chatter

    # Suggested RPM: use detected chatter frequency (or fn) and stability lobe formula
    suggested_rpm = []
    if n_teeth is not None and n_teeth > 0:
        from tap_testing.milling_dynamics import stability_lobe_best_spindle_speed_rpm
        fc = peak_near_natural_hz if peak_near_natural_hz is not None else natural_freq_hz
        for n in range(4):
            rpm_n = stability_lobe_best_spindle_speed_rpm(fc, n_teeth, lobe_index_n=n)
            if rpm_n > 0:
                suggested_rpm.append(rpm_n)

    return ChatterAssessmentResult(
        is_chatter_likely=is_chatter,
        dominant_freq_hz=dominant_freq_hz,
        peak_near_natural_hz=peak_near_natural_hz,
        natural_freq_hz=natural_freq_hz,
        natural_freq_band_hz=(fn_lo, fn_hi),
        peaks=peaks,
        suggested_rpm=suggested_rpm,
    )


def plot_chatter_spectrum_figure(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    result: ChatterAssessmentResult,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 3),
    freq_max_hz: float | None = None,
) -> "matplotlib.figure.Figure":
    """Plot cutting spectrum with natural-frequency band and peaks marked (for chatter analysis)."""
    import matplotlib.pyplot as plt

    if len(freqs) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No spectrum data", ha="center", va="center", transform=ax.transAxes)
        return fig
    fn_lo, fn_hi = result.natural_freq_band_hz
    if freq_max_hz is None:
        freq_max_hz = max(float(freqs[-1]), 2.0 * result.natural_freq_hz)
    fig, ax = plt.subplots(figsize=figsize)
    mask = freqs <= freq_max_hz
    ax.plot(freqs[mask], magnitude[mask], color="#2980b9", linewidth=1.0, label="FFT magnitude")
    ax.axvspan(fn_lo, fn_hi, color="#c0392b", alpha=0.2, label=f"Natural freq. {result.natural_freq_hz:.1f} Hz ± band")
    ax.axvline(result.natural_freq_hz, color="#c0392b", linestyle="--", linewidth=1.0)
    for f, mag in result.peaks[:5]:
        if f <= freq_max_hz:
            ax.scatter([f], [mag], color="#e74c3c", s=40, zorder=5)
            ax.annotate(f"{f:.0f}", (f, mag), xytext=(0, 6), textcoords="offset points", fontsize=8, ha="center")
    ax.set_xlim(0, freq_max_hz)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Cutting recording — chatter assessment (peaks in red; band = natural freq.)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def decay_envelope_and_damping(
    t: np.ndarray,
    mag: np.ndarray,
    natural_freq_hz: float,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Estimate exponential decay envelope and damping ratio from free-decay magnitude.

    Uses peak picking after the first maximum, then fits ln(envelope) vs time to get
    decay rate; zeta = decay_rate / (2*pi*fn) for SDOF. Returns envelope on the same
    time base as input (envelope is NaN before first peak).

    Args:
        t: Time vector (s).
        mag: Magnitude signal (e.g. sqrt(ax^2+ay^2+az^2)).
        natural_freq_hz: Natural frequency (Hz) for period and omega_n.
        sample_rate_hz: Sample rate (Hz).

    Returns:
        t_out: Same as t (for plotting).
        envelope: Exponential envelope, same length as t (NaN before first peak).
        zeta_estimate: Damping ratio estimate, or None if fit failed.
    """
    n = len(mag)
    if n < 10 or natural_freq_hz <= 0 or sample_rate_hz <= 0:
        return t, np.full_like(t, np.nan), None
    peak0 = int(np.argmax(mag))
    if peak0 >= n - 5:
        return t, np.full_like(t, np.nan), None
    omega_n = 2.0 * np.pi * natural_freq_hz
    period_samples = max(4, int(sample_rate_hz / natural_freq_hz))
    half_period = max(2, period_samples // 2)
    envelope_vals: list[float] = []
    envelope_times: list[float] = []
    i = peak0
    while i + half_period < n:
        window = mag[i : i + half_period]
        if len(window) > 0:
            envelope_vals.append(float(np.max(window)))
            envelope_times.append(float(t[i]))
        i += half_period
    if len(envelope_vals) < 3:
        envelope = np.full_like(t, np.nan)
        envelope[peak0] = mag[peak0]
        return t, envelope, None
    envelope_vals_arr = np.array(envelope_vals)
    envelope_times_arr = np.array(envelope_times)
    valid = envelope_vals_arr > 1e-12
    if not np.any(valid):
        return t, np.full_like(t, np.nan), None
    log_env = np.log(envelope_vals_arr[valid])
    t_fit = envelope_times_arr[valid]
    t_fit = t_fit - t_fit[0]
    if t_fit[-1] - t_fit[0] < 1.0 / natural_freq_hz:
        return t, np.full_like(t, np.nan), None
    coeffs = np.polyfit(t_fit, log_env, 1)
    decay_rate = -coeffs[0]
    if decay_rate <= 0 or decay_rate > omega_n:
        zeta_est = None
    else:
        zeta_est = float(decay_rate / omega_n)
    t0 = float(t[peak0])
    A0 = float(mag[peak0])
    envelope = np.full_like(t, np.nan, dtype=float)
    for i in range(peak0, n):
        if decay_rate > 0:
            envelope[i] = A0 * np.exp(-decay_rate * (float(t[i]) - t0))
        else:
            envelope[i] = A0
    return t, envelope, zeta_est


# M593 (RRF input shaping) P parameter values per Duet3D docs; not case-sensitive in RRF.
# https://docs.duet3d.com/en/User_manual/Reference/Gcodes#m593-configure-input-shaping
M593_VALID_SHAPER_TYPES = frozenset(
    {"none", "zvd", "zvdd", "zvddd", "mzv", "ei2", "ei3", "custom"}
)


def format_m593_input_shaping(
    natural_freq_hz: float,
    damping_ratio: float | None = None,
    shaper_type: str = "zvd",
) -> str:
    """
    Build a RepRapFirmware M593 input-shaping line from tap-test natural frequency.

    Parameter mapping is from the official M593 spec (Duet3D GCode reference):
    - **F** = Centre frequency of ringing to cancel (Hz). RRF uses this to tune the shaper.
      The tap-test natural frequency is exactly this: the dominant frequency at which the
      structure rings when excited, so we use natural_freq_hz for F.
    - **S** = Damping factor of the ringing to be cancelled. RRF default is 0.1; we use
      that when damping_ratio is not provided (or clamp to (0,1) if provided).
    - **P** = Shaper type: one of "none", "zvd", "zvdd", "zvddd", "mzv", "ei2", "ei3",
      "custom" (RRF 3.6). "zvd" is a good default (balance of cancellation and duration).

    See: https://docs.duet3d.com/en/User_manual/Reference/Gcodes#m593-configure-input-shaping

    Args:
        natural_freq_hz: Natural frequency from tap test (Hz) — used as M593 F parameter.
        damping_ratio: Optional damping ratio ζ from decay fit; if None, S is set to 0.1.
        shaper_type: M593 P type (default "zvd"); must be in M593_VALID_SHAPER_TYPES.

    Returns:
        A single line suitable for config.g, e.g. 'M593 P"zvd" F40.50 S0.1'
    """
    if natural_freq_hz <= 0:
        return 'M593 P"none" ; invalid or missing natural frequency'
    # RRF 3.6 minimum F is 4 Hz; we still output the value but user can adjust if needed
    p_type = shaper_type.lower().strip()
    if p_type not in M593_VALID_SHAPER_TYPES:
        p_type = "zvd"
    s = f'M593 P"{p_type}" F{natural_freq_hz:.2f}'
    # S = damping factor; RRF default 0.1 per docs
    zeta = damping_ratio if damping_ratio is not None else 0.1
    if 0 < zeta < 1:
        s += f" S{zeta:.3f}"
    else:
        s += " S0.1"
    return s


def rpm_to_avoid(
    natural_freq_hz: float,
    n_teeth: int,
    harmonic_orders: int = 5,
    rpm_min: float = DEFAULT_MIN_RPM,
    rpm_max: float = DEFAULT_MAX_RPM,
) -> list[float]:
    """
    RPM values where tooth-passing frequency (or a harmonic) equals the natural frequency.

    Resonance when k·f_tooth = fn, with f_tooth = rpm·n_teeth/60 Hz, so:
    RPM = 60·natural_freq_hz / (n_teeth·k)  for k = 1, 2, 3, ...

    Same formula as stability_lobe_best_spindle_speed_rpm(..., lobe_index_n=k-1): avoid
    k=1 equals lobe N=0, avoid k=2 equals lobe N=1, etc. At the lobe peak the
    regenerative phase can still allow stable cutting; we mark the band as "avoid"
    for forced-vibration resonance. Only returns RPMs within [rpm_min, rpm_max].
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


def format_milling_guidance_for_cycle(
    result: TapTestResult,
    material_name: str | None = None,
    *,
    n_lobes: int = 3,
) -> str:
    """
    Format a short milling-dynamics guidance string for display in the cycle GUI.

    Uses tap-test result (fn, uncertainty, avoid RPM, suggested range), best
    stability lobe speeds, material label, and one-line hints for process damping
    and cutting coefficients. Suitable for a text area or the milling panel in the
    cycle chart.

    Args:
        result: TapTestResult from analyze_tap_data.
        material_name: Workpiece material (e.g. "6061 aluminum"). If None, uses config.
        n_lobes: Number of best-stability lobe speeds to list (N=0, 1, ...).

    Returns:
        Multi-line string for near real-time display after the test cycle.
    """
    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing.milling_dynamics import stability_lobe_best_spindle_speed_rpm

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)

    fn_str = f"{result.natural_freq_hz:.1f}"
    if result.natural_freq_hz_uncertainty is not None and result.natural_freq_hz_uncertainty > 0:
        fn_str += f" ± {result.natural_freq_hz_uncertainty:.2f}"
    fn_str += " Hz"

    tool_desc = f"Ø{result.tool_diameter_mm} mm" if result.tool_diameter_mm is not None else "—"
    tool_mat_label = _result_tool_material_label(result)
    mat_label = default_material_label(material_name)

    best_speeds = [
        stability_lobe_best_spindle_speed_rpm(result.natural_freq_hz, result.n_teeth_used, n)
        for n in range(n_lobes)
    ]
    best_str = ", ".join(f"N{n}={r:.0f}" for n, r in enumerate(best_speeds))

    avoid_str = ", ".join(f"{r:.0f}" for r in result.avoid_rpm[:5])
    if len(result.avoid_rpm) > 5:
        avoid_str += " …"

    cfg = get_config()
    lines = []
    if cfg.subtract_background:
        bg_desc = f"{cfg.background_duration_s:.1f} s" if cfg.background_duration_s is not None else "to impact"
        lines.append(
            f"Analysis: {cfg.tap_cycle_duration_s:.1f} s tap cycle from impact; "
            f"background subtracted (pre-tap {bg_desc})."
        )
    lines.extend([
        f"Natural frequency: {fn_str}  ·  Tool: {tool_mat_label} {tool_desc}, {result.n_teeth_used} fl  ·  {mat_label}",
        f"Avoid RPM (tooth-pass resonance): {avoid_str}",
        f"Suggested RPM: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f}",
        f"Best stability lobe speeds (rpm): {best_str}",
        "Ideal parameters: RPM from tap test; depth from stability lobe when FRF available; feed from chip load.",
        "Full stability lobe (blim vs Ω) requires x,y FRFs (impact with force) and cutting coefficients.",
        "Process damping: at low spindle speed, stable zone can increase (Sect. 4.8).",
        "Cutting coefficients: from slotting tests or inverse identification (Sect. 4.7).",
        "If chatter: record cutting signal → ignore tooth-pass content → detect chatter peak → set spindle speed so f_tooth (or harmonic) = chatter freq (special topics).",
    ])
    return "\n".join(lines)


def analyze_tap(
    path: str | Path,
    flute_count: int = 4,
    harmonic_orders: int = 5,
    use_axis: str = "magnitude",
    max_rpm: float = DEFAULT_MAX_RPM,
    min_rpm: float = DEFAULT_MIN_RPM,
    tool_diameter_mm: float | None = None,
    tool_material: str | None = None,
    subtract_background: bool | None = None,
    spindle_operating_frequency_hz: float | None = None,
) -> TapTestResult:
    """
    Load a tap CSV, run FFT, and compute spindle speed guidance.

    Uses the same pipeline as analyze_tap_data, including optional tap-cycle extraction
    and background subtraction (when subtract_background is True or config.subtract_background).

    Args:
        path: Path to tap-test CSV.
        flute_count: Number of flutes (teeth) for tooth-passing frequency.
        harmonic_orders: How many harmonics of tooth passing to consider for avoid RPM.
        use_axis: "x", "y", "z", or "magnitude" for FFT input.
        max_rpm: Maximum spindle RPM (defines range for avoid/suggested and chart; default 24000 for LDO Milo).
        min_rpm: Minimum spindle RPM for range and chart (default 9000).
        tool_diameter_mm: Tool diameter in mm (stored for reference, e.g. SFM calculations).
        tool_material: Cutter material (e.g. "carbide", "HSS"); stored for chart labels. If None, use config default when displaying.
        subtract_background: If True, extract tap cycle and subtract background before FFT.
            If None, use config.subtract_background.

    Returns:
        TapTestResult with natural frequency, avoid RPM list, and suggested RPM range.
    """
    t, data, sample_rate_hz = load_tap_csv(path)
    if sample_rate_hz <= 0:
        sample_rate_hz = 1.0 / (float(t[1] - t[0]) if len(t) > 1 else 1.0)
    return analyze_tap_data(
        t,
        data,
        sample_rate_hz,
        flute_count=flute_count,
        harmonic_orders=harmonic_orders,
        use_axis=use_axis,
        max_rpm=max_rpm,
        min_rpm=min_rpm,
        tool_diameter_mm=tool_diameter_mm,
        tool_material=tool_material,
        subtract_background=subtract_background,
        spindle_operating_frequency_hz=spindle_operating_frequency_hz,
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
    tool_material: str | None = None,
    tap_spread_std_hz: float | None = None,
    subtract_background: bool | None = None,
    spindle_operating_frequency_hz: float | None = None,
) -> TapTestResult:
    """
    Run FFT and spindle speed guidance on in-memory tap data (e.g. combined from multiple recordings).

    When subtract_background is True (or None and config.subtract_background), the data
    is preprocessed: impact is detected, the tap cycle (~1 s from impact) is extracted,
    and baseline (from pre-tap segment) is subtracted to reduce background vibration in
    the analyzed signal.

    Args:
        t: Time vector (s), shape (N,).
        data: (3, N) array [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        tap_spread_std_hz: If provided, standard deviation of natural frequency across
            multiple tap cycles; combined with FFT resolution for natural_freq_hz_uncertainty.
        subtract_background: If True, extract tap cycle and subtract background before FFT.
            If None, use config.subtract_background.
        Other args: same as analyze_tap.

    Returns:
        TapTestResult.
    """
    if sample_rate_hz <= 0 and len(t) > 1:
        sample_rate_hz = 1.0 / float(np.median(np.diff(t)))

    if subtract_background is None:
        from tap_testing.config import get_config
        subtract_background = get_config().subtract_background
    if subtract_background:
        t, data, sample_rate_hz = preprocess_tap_data_for_analysis(
            t, data, sample_rate_hz, subtract_background=True,
        )

    ax, ay, az = data[0], data[1], data[2]
    ok, msg = check_tap_signal(data)
    if not ok:
        raise ValueError(msg)
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

    natural_freq_hz = dominant_frequency_over_duration(signal, sample_rate_hz, axis_label)
    n_samples = len(signal)
    u_fft = fft_frequency_resolution_uncertainty_hz(sample_rate_hz, n_samples) if n_samples >= 2 else 0.0
    u_freq_hz = combined_natural_freq_uncertainty_hz(u_fft, tap_spread_std_hz)
    natural_freq_hz_uncertainty = u_freq_hz if u_freq_hz > 0 else None
    # Include all avoid RPMs up to max_rpm (use rpm_min=0) so the chart can show red zones
    # even when critical RPMs fall below the display min (e.g. 9000); suggested_rpm_range
    # still filters to [min_rpm, max_rpm] for the green band.
    avoid_rpm = rpm_to_avoid(
        natural_freq_hz, flute_count, harmonic_orders,
        rpm_min=0.0, rpm_max=max_rpm,
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
        tool_material=tool_material,
        natural_freq_hz_uncertainty=natural_freq_hz_uncertainty,
        spindle_operating_frequency_hz=spindle_operating_frequency_hz,
    )


def feed_rate_mm_min(
    chip_load_mm: float,
    n_teeth: int,
    rpm: float,
) -> float:
    """
    Feed rate in mm/min from chip load (mm per tooth), number of teeth, and spindle RPM.

    Formula: F = fz × N_teeth × RPM  (standard CNC relationship; fz in mm/tooth).
    Example: 0.05 mm/tooth, 3 flutes, 12 000 rpm → 0.05 × 3 × 12000 = 1800 mm/min.

    If your reference gives chip load in inches per tooth, convert first: mm/tooth = in/tooth × 25.4.
    """
    return chip_load_mm * n_teeth * rpm


def compute_sdof_interpretation(
    result: TapTestResult,
    mass_kg: float | None = None,
    stiffness_n_per_m: float | None = None,
) -> dict:
    """
    Interpret the tap-test result using SDOF free-vibration theory.

    Given the measured natural frequency (f_n), provide either an estimated effective
    mass or stiffness to get the other: k = m*(2π*f_n)², m = k/(2π*f_n)².

    Args:
        result: TapTestResult with natural_freq_hz from FFT.
        mass_kg: If set, compute effective stiffness k (N/m) for this mass.
        stiffness_n_per_m: If set, compute effective mass m (kg) for this stiffness.

    Returns:
        Dict with keys: effective_stiffness_n_per_m (if mass_kg was given),
        effective_mass_kg (if stiffness_n_per_m was given), nutshell (short summary).
    """
    out: dict = {"nutshell": sdof.sdof_summary_nutshell()}
    f_n = result.natural_freq_hz
    if mass_kg is not None and mass_kg > 0:
        out["effective_stiffness_n_per_m"] = sdof.stiffness_from_natural_frequency(
            f_n, mass_kg
        )
    if stiffness_n_per_m is not None and stiffness_n_per_m > 0:
        out["effective_mass_kg"] = sdof.mass_from_natural_frequency(
            f_n, stiffness_n_per_m
        )
    return out


def practical_suggested_range_at_spindle(
    suggested_min: float,
    suggested_max: float,
    spindle_rpm: float,
    tolerance: float = 0.10,
) -> tuple[float | None, float | None]:
    """
    Intersection of the theoretical suggested RPM range with a band around spindle operating speed.

    In application the spindle runs at a fixed frequency (e.g. 400 Hz → 24000 rpm), so the
    "practical" stable range is only the part of the suggested range near that speed.
    Returns (low, high) for the overlapping band, or (None, None) if no overlap.
    """
    if spindle_rpm <= 0:
        return None, None
    band_lo = spindle_rpm * (1 - tolerance)
    band_hi = spindle_rpm * (1 + tolerance)
    low = max(suggested_min, band_lo)
    high = min(suggested_max, band_hi)
    if low >= high:
        return None, None
    return low, high


def chart_rpm_range(result: TapTestResult, extend_to_show_avoid: bool = True) -> tuple[float, float]:
    """
    Return (rpm_min, rpm_max) for the RPM chart x-axis.

    When extend_to_show_avoid is True and result has avoid_rpm below result.min_rpm,
    the left limit is extended so red (avoid) zones are visible (e.g. for low natural
    frequency tools where critical RPMs fall below the default 9000 min).
    """
    lo, hi = result.min_rpm, result.max_rpm
    if extend_to_show_avoid and result.avoid_rpm:
        extend_left = min(result.avoid_rpm) - 500.0
        if extend_left < lo:
            lo = max(0.0, extend_left)
    return (lo, hi)


def get_rpm_zones(
    result: TapTestResult,
    avoid_width_fraction: float = 0.05,
    rpm_min: float | None = None,
) -> list[tuple[float, float, str]]:
    """
    Build a list of (rpm_low, rpm_high, 'avoid'|'optimal') bands from rpm_min to result.max_rpm.

    Avoid zones are ±avoid_width_fraction around each critical RPM (e.g. 0.05 = ±5%).
    Optimal zones are the gaps between avoid zones. Pass rpm_min from chart_rpm_range() to
    show red zones when critical RPMs lie below result.min_rpm.
    """
    rpm_min = result.min_rpm if rpm_min is None else rpm_min
    rpm_max = result.max_rpm
    avoid_bands: list[tuple[float, float]] = []
    for r in result.avoid_rpm:
        half = r * avoid_width_fraction
        a, b = max(rpm_min, r - half), min(rpm_max, r + half)
        if a < b:  # only add band if it has positive width (overlaps [rpm_min, rpm_max])
            avoid_bands.append((a, b))
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
    material_name: str | None = None,
    reference_rpm: float | None = None,
    reference_chip_load: float | None = None,
) -> None:
    """
    Plot spindle RPM range with avoid (red) and optimal (green) bands.

    Optionally save to output_path (e.g. .png or .pdf).
    material_name: Workpiece material for label; if None, uses config.
    reference_rpm: If set, mark this RPM on the chart (e.g. user's preferred 18k).
    """
    fig = plot_result_figure(
        result,
        output_path=output_path,
        avoid_width_fraction=avoid_width_fraction,
        material_name=material_name,
        reference_rpm=reference_rpm,
        reference_chip_load=reference_chip_load,
    )
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    plt.close(fig)


def plot_result_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    avoid_width_fraction: float = 0.05,
    figsize: tuple[float, float] = (10, 2.5),
    material_name: str | None = None,
    reference_rpm: float | None = None,
    reference_chip_load: float | None = None,
):
    """
    Build and return a matplotlib Figure for the RPM band chart (for embedding in a GUI).

    material_name: Workpiece material for label (e.g. "6061 aluminum"). If None, uses config.
    reference_rpm: If set, draw a vertical line at this RPM and label with feed (reference_chip_load used for feed).
    Caller is responsible for closing the figure if not embedding.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)  # validate
    rpm_lo_display, rpm_hi_display = chart_rpm_range(result)
    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction, rpm_min=rpm_lo_display)
    fig, ax = plt.subplots(figsize=figsize)
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"  # red / green
        alpha = 0.45 if kind == "avoid" else 0.35
        ax.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    # Highlight suggested RPM range (overall tooling RPM band)
    ax.axvline(result.suggested_rpm_min, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    ax.axvline(result.suggested_rpm_max, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    # Practical range at spindle: intersection of suggested with ±10% of spindle RPM (in application spindle is fixed)
    handles_legend = []
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None and rpm_lo_display <= prac_hi and prac_lo <= rpm_hi_display:
            ax.axvspan(
                max(prac_lo, rpm_lo_display), min(prac_hi, rpm_hi_display),
                color="#1e8449", alpha=0.35, zorder=0,
            )
            spindle_hz = result.spindle_operating_frequency_hz if result.spindle_operating_frequency_hz else (reference_rpm / 60.0)
            handles_legend.append(
                mpatches.Patch(facecolor="#1e8449", alpha=0.35, label=f"Practical at spindle ({spindle_hz:.0f} Hz)")
            )
    # Reference operating point (e.g. spindle 400 Hz → 24000 RPM)
    if reference_rpm is not None and rpm_lo_display <= reference_rpm <= rpm_hi_display:
        ref_cl = reference_chip_load if reference_chip_load is not None and reference_chip_load > 0 else 0.05
        ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
        ax.axvline(reference_rpm, color="#8e44ad", linewidth=2, linestyle="-", alpha=0.95)
        ax.scatter([reference_rpm], [0.5], color="#8e44ad", s=120, zorder=6, edgecolors="white", linewidths=2)
        ax.text(reference_rpm, 0.08, f"Ref\n{reference_rpm:.0f}\n{ref_feed:.0f} mm/min", fontsize=7, ha="center", va="bottom", color="#8e44ad", fontweight="bold")
        from matplotlib.lines import Line2D
        handles_legend.append(Line2D([0], [0], color="#8e44ad", linewidth=2, linestyle="-", label=f"Your ref: {reference_rpm:.0f} rpm ({ref_feed:.0f} mm/min)"))
    ax.set_xlim(rpm_lo_display, rpm_hi_display)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Spindle speed (rpm)")
    tool_mat_label = _result_tool_material_label(result)
    title_parts = [f"Natural freq. {result.natural_freq_hz:.1f} Hz"]
    title_parts.append(f"Tool: {tool_mat_label}" + (f" Ø{result.tool_diameter_mm} mm" if result.tool_diameter_mm is not None else ""))
    title_parts.append(f"{result.n_teeth_used} flutes, max {result.max_rpm:.0f} rpm")
    title_parts.append(default_material_label(material_name))
    title_parts.append("Metric")
    ax.set_title("  ·  ".join(title_parts))
    from matplotlib.lines import Line2D
    suggested_line = Line2D([0], [0], color="#1e8449", linewidth=2, linestyle="--", label=f"Suggested: {result.suggested_rpm_min:.0f}–{result.suggested_rpm_max:.0f} rpm")
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (chatter risk)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal")
    ax.legend(handles=[avoid_patch, optimal_patch, suggested_line] + handles_legend, loc="upper right", fontsize=8)
    ax.set_ylabel(" ")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# Stepover percentages to visualize for feeds & speeds (chip thinning, TEA, feed at ref RPM).
DEFAULT_STEPOVER_PERCENTAGES = (10, 25, 35, 50, 75, 100)


def plot_feeds_speeds_stepover_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    chip_load_mm: float = 0.05,
    reference_rpm: float | None = None,
    depth_of_cut_mm: float | None = None,
    stepover_percentages: tuple[float, ...] = DEFAULT_STEPOVER_PERCENTAGES,
    avoid_width_fraction: float = 0.05,
    figsize: tuple[float, float] = (10, 9),
    material_name: str | None = None,
    reference_chip_load: float | None = None,
    spindle_power_w: float | None = None,
    min_chipload_mm: float | None = None,
):
    """
    Feeds & speeds by stepover: RPM band + stepover % vs adjusted chipload, feed, TEA (and optional MRR).

    Uses result (tool diameter, flutes, suggested RPM, avoid RPM, lobes) and for each stepover
    (10, 25, 35, 50, 75, 100% by default) shows chip-thinning adjusted chipload, feed at reference
    RPM, tool engagement angle, and optionally MRR if depth_of_cut_mm is set.

    Good/bad zones: chip evacuation (target chipload >= min_chipload_mm) and spindle load
    (cutting power <= spindle_power_w). Default spindle 1.5 kW. Bars and table show Zone (OK, Low fz, Overload).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing import feeds_speeds as fs

    if material_name is None:
        material_name = get_config().material_name
    mat = get_material_or_default(material_name)
    mat_label = mat.name

    spindle_w = spindle_power_w if spindle_power_w is not None and spindle_power_w > 0 else fs.DEFAULT_SPINDLE_POWER_W
    min_fz = min_chipload_mm if min_chipload_mm is not None else fs.MIN_CHIPLOAD_MM_PER_TOOTH

    diam_mm = result.tool_diameter_mm if result.tool_diameter_mm is not None else 6.35
    ref_rpm = reference_rpm
    if ref_rpm is None or ref_rpm <= 0:
        ref_rpm = (result.suggested_rpm_min + result.suggested_rpm_max) / 2.0
    ref_cl = reference_chip_load if reference_chip_load is not None and reference_chip_load > 0 else chip_load_mm

    # Build data for each stepover %
    pcts = list(stepover_percentages)
    stepover_mm_list = [diam_mm * (p / 100.0) for p in pcts]
    adjusted_cl = [fs.chipload_adjusted_for_stepover_mm(diam_mm, woc, chip_load_mm) for woc in stepover_mm_list]
    feed_list = [feed_rate_mm_min(acl, result.n_teeth_used, ref_rpm) for acl in adjusted_cl]
    tea_list = [fs.tool_engagement_angle_deg(diam_mm, woc) for woc in stepover_mm_list]
    mrr_list = None
    power_list = None
    if depth_of_cut_mm is not None and depth_of_cut_mm > 0:
        mrr_list = [fs.mrr_mm3_per_min(woc, depth_of_cut_mm, f) for woc, f in zip(stepover_mm_list, feed_list)]
        if mat_label in fs.UNIT_POWER_W_MIN_PER_MM3:
            unit_power = fs.UNIT_POWER_W_MIN_PER_MM3[mat_label]
            power_list = [fs.cutting_power_w(mrr, unit_power) for mrr in mrr_list]

    # Zone status: chip evacuation (target chipload >= min) and spindle load (power <= spindle_w)
    zone_list = []
    for i in range(len(pcts)):
        power_w = power_list[i] if power_list is not None else None
        z = fs.cutting_zone_status(chip_load_mm, power_w, spindle_power_w=spindle_w, min_chipload_mm=min_fz)
        zone_list.append(z)

    rpm_lo_display, rpm_hi_display = chart_rpm_range(result)
    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction, rpm_min=rpm_lo_display)

    fig, (ax_rpm, ax_stepover, ax_table) = plt.subplots(
        3, 1, figsize=figsize, height_ratios=[0.9, 1.2, 1.0], sharex=False
    )

    # Top: RPM band (compact) — overall tooling RPM (suggested range), avoid, ref RPM
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"
        alpha = 0.45 if kind == "avoid" else 0.35
        ax_rpm.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    ax_rpm.axvline(result.suggested_rpm_min, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    ax_rpm.axvline(result.suggested_rpm_max, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None and rpm_lo_display <= prac_hi and prac_lo <= rpm_hi_display:
            ax_rpm.axvspan(
                max(prac_lo, rpm_lo_display), min(prac_hi, rpm_hi_display),
                color="#1e8449", alpha=0.35, zorder=0,
            )
    ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, ref_rpm)
    if rpm_lo_display <= ref_rpm <= rpm_hi_display:
        ax_rpm.axvline(ref_rpm, color="#8e44ad", linewidth=2, linestyle="-", alpha=0.95)
        ax_rpm.scatter([ref_rpm], [0.5], color="#8e44ad", s=100, zorder=6, edgecolors="white", linewidths=2)
        ax_rpm.text(ref_rpm, 0.08, f"Ref\n{ref_rpm:.0f}\n{ref_feed:.0f}", fontsize=7, ha="center", va="bottom", color="#8e44ad", fontweight="bold")
    ax_rpm.set_xlim(rpm_lo_display, rpm_hi_display)
    ax_rpm.set_ylim(0, 1)
    ax_rpm.set_yticks([])
    ax_rpm.set_xlabel("Spindle speed (rpm)")
    tool_mat_label = _result_tool_material_label(result)
    ax_rpm.set_title(
        f"Stability · fn={result.natural_freq_hz:.1f} Hz · {tool_mat_label} Ø{diam_mm} mm, {result.n_teeth_used} fl · "
        f"Ref {ref_rpm:.0f} rpm ({ref_feed:.0f} mm/min) · {default_material_label(material_name)}"
    )
    from matplotlib.lines import Line2D
    ax_rpm.legend(
        handles=[
            mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid"),
            mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal"),
            Line2D([0], [0], color="#1e8449", linewidth=2, linestyle="--", label=f"Suggested {result.suggested_rpm_min:.0f}–{result.suggested_rpm_max:.0f}"),
            Line2D([0], [0], color="#8e44ad", linewidth=2, label="Ref RPM"),
        ],
        loc="upper right", fontsize=8,
    )
    ax_rpm.grid(True, alpha=0.3, axis="x")

    # Middle: Stepover % vs Adjusted chipload, Feed, TEA (twin axis for TEA); bar color = zone
    n_bars = len(pcts)
    x_pos = np.arange(n_bars)
    width = 0.35
    # Zone colors: OK=green, Low fz=amber, Overload or both=red
    bar_colors_cl = []
    bar_colors_feed = []
    for z in zone_list:
        if z["label"] == "OK":
            bar_colors_cl.append("#27ae60")
            bar_colors_feed.append("#2ecc71")
        elif z["label"].startswith("Low fz") and "Overload" not in z["label"]:
            bar_colors_cl.append("#f39c12")
            bar_colors_feed.append("#f1c40f")
        else:
            bar_colors_cl.append("#c0392b")
            bar_colors_feed.append("#e74c3c")
    ax2 = ax_stepover.twinx()
    bars1 = ax_stepover.bar(x_pos - width / 2, adjusted_cl, width=width, color=bar_colors_cl, alpha=0.8, label="Adj. chipload (mm/tooth)")
    bars2 = ax_stepover.bar(x_pos + width / 2, [f / 100.0 for f in feed_list], width=width, color=bar_colors_feed, alpha=0.8, label="Feed / 100 (mm/min)")
    ax_stepover.set_ylabel("Adjusted chipload (mm/tooth)  ·  Feed/100 (mm/min)", fontsize=9)
    ax_stepover.set_ylim(bottom=0)
    line_tea, = ax2.plot(x_pos, tea_list, color="#e67e22", linewidth=2.5, marker="o", markersize=8, label="TEA (°)")
    ax2.set_ylabel("Tool engagement angle TEA (°)", fontsize=9, color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax2.set_ylim(0, 185)
    ax_stepover.set_xlabel("Stepover (% of tool diameter)")
    ax_stepover.set_xticks(x_pos)
    ax_stepover.set_xticklabels([f"{p:.0f}%" for p in pcts])
    ax_stepover.set_title(
        f"Feeds & speeds by stepover · target chipload {chip_load_mm} mm/tooth → adjusted chipload & feed at {ref_rpm:.0f} rpm  ·  Zone: green=OK, amber=Low fz, red=Overload"
    )
    from matplotlib.lines import Line2D
    zone_handles = [
        mpatches.Patch(facecolor="#27ae60", alpha=0.8, label="OK (chip evac + spindle)"),
        mpatches.Patch(facecolor="#f39c12", alpha=0.8, label="Low fz (chip evac)"),
        mpatches.Patch(facecolor="#c0392b", alpha=0.8, label="Overload / Low fz+Overload"),
    ]
    ax_stepover.legend(handles=zone_handles + [Line2D([0], [0], color="#e67e22", linewidth=2, label="TEA (°)")], loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=8)
    ax_stepover.grid(True, alpha=0.3, axis="y")

    # Bottom: table Stepover %, WOC (mm), Adj chipload, Feed, TEA, [MRR], [Power W], Zone
    ax_table.set_axis_off()
    headers = ["Stepover %", "WOC (mm)", "Adj. fz (mm/tooth)", "Feed (mm/min)", "TEA (°)"]
    if mrr_list is not None:
        headers.append("MRR (mm³/min)")
    if power_list is not None:
        headers.append("Power (W)")
    headers.append("Zone")
    cell_text = []
    for i, p in enumerate(pcts):
        row = [f"{p:.0f}", f"{stepover_mm_list[i]:.2f}", f"{adjusted_cl[i]:.3f}", f"{feed_list[i]:.0f}", f"{tea_list[i]:.0f}"]
        if mrr_list is not None:
            row.append(f"{mrr_list[i]:.0f}")
        if power_list is not None:
            row.append(f"{power_list[i]:.0f}")
        row.append(zone_list[i]["label"])
        cell_text.append(row)
    table = ax_table.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8f4f8"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    # Color Zone column cells by zone
    zone_col_idx = len(headers) - 1
    for i in range(len(cell_text)):
        z = zone_list[i]
        if z["label"] == "OK":
            table[(i + 1, zone_col_idx)].set_facecolor("#d5f4e6")
        elif "Overload" in z["label"]:
            table[(i + 1, zone_col_idx)].set_facecolor("#fadbd8")
        else:
            table[(i + 1, zone_col_idx)].set_facecolor("#fdebd0")
    doc_note = f"  DOC={depth_of_cut_mm} mm" if depth_of_cut_mm else ""
    ax_table.set_title(
        f"Stepover summary (tool Ø{diam_mm} mm, {result.n_teeth_used} fl, ref {ref_rpm:.0f} rpm, spindle ≤{spindle_w:.0f} W){doc_note}",
        fontsize=10,
    )
    ax_table.text(
        0.5, 0.02,
        "Chip thinning: adjusted chipload compensates for stepover < 50%. TEA = tool engagement angle (slotting = 180°). "
        f"Zone: chip evacuation (fz ≥ {min_fz:.2f} mm) and spindle load (power ≤ {spindle_w:.0f} W).",
        transform=ax_table.transAxes, fontsize=8, ha="center", va="bottom", wrap=True,
    )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_optimal_loads_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    chip_load_mm: float = 0.05,
    n_lobes: int = 6,
    avoid_width_fraction: float = 0.05,
    figsize: tuple[float, float] = (10, 8),
    material_name: str | None = None,
    reference_rpm: float | None = None,
    reference_chip_load: float | None = None,
    chip_load_bands_mm: tuple[float, ...] | None = None,
    min_feed_mm_min: float = 1000.0,
    max_feed_mm_min: float | None = 2000.0,
):
    """
    Chart of all optimal-load data points for a given tool: RPM bands, feed vs RPM
    (chip-load bands), and a table of (RPM, feed, f_tooth) for each optimal point.

    Lobe speeds are filtered to those where feed (at chip_load_mm) lies in
    [min_feed_mm_min, max_feed_mm_min]; default 1000–2000 mm/min. Set max_feed_mm_min
    to None for no upper limit (e.g. allow higher top-end feed).

    reference_rpm: If set, draw a vertical line at this RPM and add a table row for it.
    chip_load_bands_mm: Chip loads (mm/tooth) to plot as feed vs RPM curves; default (0.03, 0.05, 0.08).
    min_feed_mm_min: Minimum feed (mm/min) for lobe points to be shown; default 1000.
    max_feed_mm_min: Maximum feed (mm/min) for lobe points; default 2000. None = no upper limit.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing.milling_dynamics import (
        spindle_aligned_rpms,
        stability_lobe_best_spindle_speed_rpm,
        tooth_passing_frequency_hz,
    )

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)

    rpm_min, rpm_max = chart_rpm_range(result)
    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction, rpm_min=rpm_min)

    # When spindle operating frequency is set, use spindle-aligned speeds (multiples of spindle freq)
    # so optimal points line up with what the spindle can actually run. Otherwise use stability lobe formula.
    spindle_hz = result.spindle_operating_frequency_hz
    if spindle_hz is None and reference_rpm is not None and reference_rpm > 0:
        spindle_hz = reference_rpm / 60.0
    use_spindle_aligned = spindle_hz is not None and spindle_hz > 0

    if use_spindle_aligned:
        # Use spindle min/max range only (not extended chart range) so we don't show
        # spindle-aligned speeds below the spindle's usable range (e.g. avoid < 8000 rpm).
        all_lobe_points = spindle_aligned_rpms(spindle_hz, result.min_rpm, result.max_rpm)
        _optimal_point_label = "S"  # S1, S2, ... (spindle subharmonic)
        _theory_lobe_rpms = [
            stability_lobe_best_spindle_speed_rpm(result.natural_freq_hz, result.n_teeth_used, lobe_index_n=n)
            for n in range(min(n_lobes, 6))
            if rpm_min <= stability_lobe_best_spindle_speed_rpm(result.natural_freq_hz, result.n_teeth_used, lobe_index_n=n) <= rpm_max
        ]
    else:
        all_lobe_points = []
        for n in range(n_lobes):
            rpm = stability_lobe_best_spindle_speed_rpm(
                result.natural_freq_hz, result.n_teeth_used, lobe_index_n=n
            )
            if rpm_min <= rpm <= rpm_max:
                all_lobe_points.append((n, rpm))
        _optimal_point_label = "N"  # N0, N1, ... (lobe index)
        _theory_lobe_rpms = []
    # Filter to lobe speeds that yield feed in [min_feed_mm_min, max_feed_mm_min] at chip_load_mm
    max_feed_cap = max_feed_mm_min if max_feed_mm_min is not None else float("inf")
    optimal_rpm_points = [
        (n, rpm) for n, rpm in all_lobe_points
        if min_feed_mm_min <= feed_rate_mm_min(chip_load_mm, result.n_teeth_used, rpm) <= max_feed_cap
    ]
    _feed_band_fallback_note = False
    if not optimal_rpm_points and all_lobe_points:
        # No lobes in feed band; show all lobes and note it in the table
        optimal_rpm_points = list(all_lobe_points)
        _feed_band_fallback_note = True

    feed_band_str = f"{min_feed_mm_min:.0f}–{max_feed_mm_min:.0f}" if max_feed_mm_min is not None else f"≥{min_feed_mm_min:.0f}"

    if chip_load_bands_mm is None:
        chip_load_bands_mm = (0.03, 0.05, 0.08)

    fig, (ax_bands, ax_feed, ax_table) = plt.subplots(
        3, 1, figsize=(figsize[0], figsize[1] + 2.2), height_ratios=[1.0, 1.0, 1.0], sharex=False
    )

    # Top: RPM bands + optimal lobe speeds as points
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"
        alpha = 0.45 if kind == "avoid" else 0.35
        ax_bands.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    # Practical range at spindle (intersection of suggested with ±10% of reference RPM)
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None and rpm_min <= prac_hi and prac_lo <= rpm_max:
            ax_bands.axvspan(prac_lo, prac_hi, color="#1e8449", alpha=0.35, zorder=0)
    ax_bands.set_xlim(rpm_min, rpm_max)
    ax_bands.set_ylim(0, 1)
    ax_bands.set_yticks([])
    if _theory_lobe_rpms:
        for trpm in _theory_lobe_rpms:
            ax_bands.axvline(trpm, color="#2980b9", linewidth=0.8, linestyle=":", alpha=0.4)
    if optimal_rpm_points:
        rpms = [r for _, r in optimal_rpm_points]
        lobe_label = "Spindle-aligned (optimal)" if use_spindle_aligned else "Best stability lobe (optimal)"
        ax_bands.scatter(
            rpms, [0.5] * len(rpms), color="#2980b9", s=80, zorder=5,
            edgecolors="white", linewidths=1.5, label=lobe_label,
        )
        for idx, rpm in optimal_rpm_points:
            feed = feed_rate_mm_min(chip_load_mm, result.n_teeth_used, rpm)
            ax_bands.annotate(
                f"{_optimal_point_label}{idx}", (rpm, 0.5), xytext=(0, 10), textcoords="offset points",
                fontsize=9, ha="center", va="bottom", fontweight="bold",
            )
            ax_bands.annotate(
                f"{rpm:.0f} rpm · {feed:.0f} mm/min",
                (rpm, 0.5), xytext=(0, -22), textcoords="offset points", fontsize=7,
                ha="center", va="top", color="#2980b9",
            )
    if reference_rpm is not None and rpm_min <= reference_rpm <= rpm_max:
        ref_cl = reference_chip_load if reference_chip_load and reference_chip_load > 0 else chip_load_mm
        ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
        ax_bands.axvline(reference_rpm, color="#8e44ad", linewidth=2, linestyle="-", alpha=0.95)
        ax_bands.scatter([reference_rpm], [0.5], color="#8e44ad", s=120, zorder=6, edgecolors="white", linewidths=2)
        ax_bands.annotate(f"Ref {reference_rpm:.0f}\n{ref_feed:.0f} mm/min", (reference_rpm, 0.5), xytext=(0, -18), textcoords="offset points", fontsize=8, ha="center", va="top", color="#8e44ad", fontweight="bold")
    ax_bands.set_xlabel("Spindle speed (rpm)")
    ax_bands.set_ylabel(" ")
    tool_mat_label = _result_tool_material_label(result)
    tool_desc = f"{tool_mat_label} Ø{result.tool_diameter_mm} mm" if result.tool_diameter_mm is not None else tool_mat_label
    mode_note = f"spindle {spindle_hz:.0f} Hz aligned" if use_spindle_aligned else ""
    title = (
        f"Optimal loads · Tool {tool_desc}, {result.n_teeth_used} fl · "
        f"fn={result.natural_freq_hz:.1f} Hz · feed {feed_band_str} mm/min"
    )
    if mode_note:
        title += f" · {mode_note}"
    title += f" · {default_material_label(material_name)}"
    ax_bands.set_title(title)
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (chatter risk)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal band")
    handles = [avoid_patch, optimal_patch]
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None:
            spindle_hz = result.spindle_operating_frequency_hz or (reference_rpm / 60.0)
            handles.append(mpatches.Patch(facecolor="#1e8449", alpha=0.35, label=f"Practical at spindle ({spindle_hz:.0f} Hz)"))
    if _theory_lobe_rpms:
        from matplotlib.lines import Line2D
        handles.append(
            Line2D([0], [0], color="#2980b9", linewidth=1.5, linestyle=":", alpha=0.5, label="Lobe (theory)")
        )
    if optimal_rpm_points:
        from matplotlib.lines import Line2D
        pt_label = "Spindle-aligned speeds" if use_spindle_aligned else "Best lobe speeds (data points)"
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9", markeredgecolor="white",
                   markersize=10, label=pt_label)
        )
    if reference_rpm is not None and rpm_min <= reference_rpm <= rpm_max:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#8e44ad", linewidth=2, label="Your reference RPM"))
    ax_bands.legend(handles=handles, loc="upper right", fontsize=8)
    ax_bands.grid(True, alpha=0.3, axis="x")

    # Middle: Feed (mm/min) vs RPM for different chip loads (feeds band by tooling)
    rpm_axis = np.linspace(rpm_min, rpm_max, 200)
    colors_fz = ("#16a085", "#2980b9", "#8e44ad")  # teal, blue, purple
    for i, fz in enumerate(chip_load_bands_mm):
        feed_curve = feed_rate_mm_min(fz, result.n_teeth_used, rpm_axis)
        ax_feed.plot(rpm_axis, feed_curve, color=colors_fz[i % len(colors_fz)], linewidth=1.5, label=f"fz={fz:.2f} mm/tooth")
    if reference_rpm is not None and rpm_min <= reference_rpm <= rpm_max:
        ref_cl = reference_chip_load if reference_chip_load and reference_chip_load > 0 else chip_load_mm
        ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
        ax_feed.axvline(reference_rpm, color="#8e44ad", linewidth=1.5, linestyle="--", alpha=0.9)
        ax_feed.scatter([reference_rpm], [ref_feed], color="#8e44ad", s=80, zorder=5, edgecolors="white", linewidths=1.5)
        ax_feed.annotate(f"Ref {reference_rpm:.0f}\n{ref_feed:.0f} mm/min", (reference_rpm, ref_feed), xytext=(8, 0), textcoords="offset points", fontsize=8, va="center", color="#8e44ad")
    for idx, rpm in optimal_rpm_points:
        feed_at_rpm = feed_rate_mm_min(chip_load_mm, result.n_teeth_used, rpm)
        ax_feed.axvline(rpm, color="#2980b9", linewidth=0.9, linestyle=":", alpha=0.7)
        ax_feed.scatter([rpm], [feed_at_rpm], color="#2980b9", s=50, zorder=4, edgecolors="white", linewidths=1)
        ax_feed.annotate(
            f"{_optimal_point_label}{idx}: {rpm:.0f} rpm, {feed_at_rpm:.0f} mm/min",
            (rpm, feed_at_rpm), xytext=(6, 0), textcoords="offset points", fontsize=7,
            va="center", color="#2980b9",
        )
    ax_feed.set_xlim(rpm_min, rpm_max)
    ax_feed.set_ylim(bottom=0)
    ax_feed.set_xlabel("Spindle speed (rpm)")
    ax_feed.set_ylabel("Feed (mm/min)")
    ax_feed.set_title(f"Feed vs RPM · {result.n_teeth_used} flutes  (Feed = fz × Nt × RPM)")
    ax_feed.legend(loc="upper left", fontsize=8)
    ax_feed.grid(True, alpha=0.3)

    # Bottom: optimal cutting parameters as a data table (granular chart data)
    ax_table.set_axis_off()
    if optimal_rpm_points:
        # Build table data: Label, RPM, Feed (mm/min), fz (mm/tooth), f_tooth (Hz)
        headers = ["Label", "RPM", "Feed (mm/min)", "fz (mm/tooth)", "f_tooth (Hz)"]
        cell_text = []
        for idx, rpm in optimal_rpm_points:
            feed = feed_rate_mm_min(chip_load_mm, result.n_teeth_used, rpm)
            f_tooth = tooth_passing_frequency_hz(rpm, result.n_teeth_used)
            cell_text.append([f"{_optimal_point_label}{idx}", f"{rpm:.0f}", f"{feed:.1f}", f"{chip_load_mm:.2f}", f"{f_tooth:.1f}"])
        if reference_rpm is not None and rpm_min <= reference_rpm <= rpm_max:
            ref_cl = reference_chip_load if reference_chip_load and reference_chip_load > 0 else chip_load_mm
            ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
            ref_ft = tooth_passing_frequency_hz(reference_rpm, result.n_teeth_used)
            cell_text.append(["Ref", f"{reference_rpm:.0f}", f"{ref_feed:.1f}", f"{ref_cl:.2f}", f"{ref_ft:.1f}"])
        table = ax_table.table(
            cellText=cell_text,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colColours=["#e8f4f8"] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 2.0)
        ax_table.set_title(
            "Optimal cutting parameters (tooling)" + (
                f"  ·  Spindle {spindle_hz:.0f} Hz, feed band {feed_band_str} mm/min" if use_spindle_aligned else f"  ·  Feed band {feed_band_str} mm/min"
            ) + ("  (all points in RPM range; none in feed band)" if _feed_band_fallback_note else ""),
            fontsize=10,
        )
        # Summary line below table
        summary = (
            f"Suggested RPM: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f}  ·  "
            f"Avoid (resonance): {', '.join(f'{r:.0f}' for r in result.avoid_rpm[:6])}{'…' if len(result.avoid_rpm) > 6 else ''}"
        )
        ax_table.text(0.5, 0.02, summary, transform=ax_table.transAxes, fontsize=8, ha="center", va="bottom", wrap=True)
    else:
        no_pt_msg = "No spindle-aligned speeds in range with feed in band." if use_spindle_aligned else "No best-stability lobe speeds in spindle range with feed in band."
        ax_table.text(
            0.5, 0.7, no_pt_msg,
            transform=ax_table.transAxes, fontsize=11, ha="center", va="center",
        )
        ax_table.text(
            0.5, 0.4, f"Feed band: {feed_band_str} mm/min  at  fz = {chip_load_mm} mm/tooth\nMin RPM = {rpm_min:.0f}, Max RPM = {rpm_max:.0f}.",
            transform=ax_table.transAxes, fontsize=9, ha="center", va="center",
        )
        ax_table.text(
            0.5, 0.15, "Try higher chip load, higher max RPM, or relax min/max feed.",
            transform=ax_table.transAxes, fontsize=9, ha="center", va="center",
        )
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spectrum_figure(
    result: TapTestResult,
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 3),
    freq_max_hz: float | None = None,
    tap_series_list: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    """
    Plot FFT magnitude spectrum for impact-test analysis.

    Shows frequency (Hz) vs magnitude, with the detected natural frequency
    and optional uncertainty band marked. If tap_series_list is provided,
    plots each tap's spectrum (lighter) and the average (bold).

    Args:
        result: TapTestResult (defines natural_freq_hz and magnitude_axis).
        t: Time vector (s), unused if sample_rate_hz is set; used to infer sr if needed.
        data: (3, N) array [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        output_path: If set, save figure.
        figsize: Figure size.
        freq_max_hz: Upper frequency limit for x-axis (default: Nyquist or 2× fn).
        tap_series_list: Optional list of (t, data) per tap; when set, draw per-tap + average.
    """
    import matplotlib.pyplot as plt

    colors = ["#3498db", "#9b59b6", "#1abc9c"]

    def _signal_from_data(data: np.ndarray) -> np.ndarray:
        ax_g, ay_g, az_g = data[0], data[1], data[2]
        if result.magnitude_axis == "magnitude":
            return np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
        if result.magnitude_axis == "x":
            return ax_g
        if result.magnitude_axis == "y":
            return ay_g
        return az_g

    if tap_series_list and len(tap_series_list) > 0:
        n_min = min(d.shape[1] for _, d in tap_series_list)
        signals = [_signal_from_data(d)[:n_min] for _, d in tap_series_list]
        avg_signal = np.mean(signals, axis=0)
        freqs, magnitude = fft_magnitude_spectrum(avg_signal, sample_rate_hz)
    else:
        signal = _signal_from_data(data)
        freqs, magnitude = fft_magnitude_spectrum(signal, sample_rate_hz)

    if len(freqs) == 0:
        fig, ax_plot = plt.subplots(figsize=figsize)
        ax_plot.text(0.5, 0.5, "Insufficient data for spectrum", ha="center", va="center", transform=ax_plot.transAxes)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    fn = result.natural_freq_hz
    u_hz = result.natural_freq_hz_uncertainty or 0.0
    if freq_max_hz is None:
        freq_max_hz = min(sample_rate_hz / 2.0, max(freqs[-1], 2.0 * fn))

    fig, ax_plot = plt.subplots(figsize=figsize)
    if tap_series_list and len(tap_series_list) > 0:
        n_min = min(d.shape[1] for _, d in tap_series_list)
        for i, (_, d) in enumerate(tap_series_list):
            sig = _signal_from_data(d)[:n_min]
            freqs_i, mag_i = fft_magnitude_spectrum(sig, sample_rate_hz)
            if len(freqs_i) > 0:
                mask_i = freqs_i <= freq_max_hz
                ax_plot.plot(freqs_i[mask_i], mag_i[mask_i], color=colors[i % len(colors)], alpha=0.35, linewidth=0.8, label=f"Tap {i + 1}" if i < 3 else None)
        mask = freqs <= freq_max_hz
        ax_plot.plot(freqs[mask], magnitude[mask], color="#2980b9", linewidth=1.5, label="Average")
    else:
        mask = freqs <= freq_max_hz
        ax_plot.plot(freqs[mask], magnitude[mask], color="#2980b9", linewidth=1.0, label="FFT magnitude")
    ax_plot.axvline(fn, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Natural freq. {fn:.1f} Hz")
    if u_hz > 0:
        ax_plot.axvspan(fn - u_hz, fn + u_hz, color="#c0392b", alpha=0.15)
    ax_plot.set_xlim(0, freq_max_hz)
    ax_plot.set_xlabel("Frequency (Hz)")
    ax_plot.set_ylabel("Magnitude")
    ax_plot.set_title("Impact test — frequency spectrum (per-tap + average)" if tap_series_list else "Impact test — frequency spectrum (verify dominant peak)")
    ax_plot.legend(loc="upper right", fontsize=9)
    ax_plot.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_signal_figure(
    result: TapTestResult,
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    show_envelope: bool = True,
    show_axes_xyz: bool = True,
    tap_series_list: list[tuple[np.ndarray, np.ndarray]] | None = None,
    live_spindle: bool = False,
) -> "matplotlib.figure.Figure":
    """
    Plot time-domain signal: magnitude (and optional envelope), then x/y/z axes.

    Supports both tap-test data (decay envelope, Tn markers) and live spindle data
    (continuous; no envelope/Tn). If tap_series_list is provided (tap test only),
    plots each tap (lighter) and the average (bold).
    """
    import matplotlib.pyplot as plt

    if live_spindle:
        show_envelope = False

    colors = ["#3498db", "#9b59b6", "#1abc9c"]
    fn = result.natural_freq_hz
    Tn = 1.0 / fn if (fn > 0 and not live_spindle) else None
    n_rows = 2 if show_axes_xyz else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]
    ax_mag = axes[0]

    if tap_series_list and len(tap_series_list) > 0 and not live_spindle:
        n_min = min(d.shape[1] for _, d in tap_series_list)
        magnitudes: list[np.ndarray] = []
        t_plot = None
        for i, (t_i, d) in enumerate(tap_series_list):
            ax_g, ay_g, az_g = d[0], d[1], d[2]
            mag = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)[:n_min]
            if t_plot is None:
                t_plot = t_i[:n_min] if len(t_i) >= n_min else t_i
            magnitudes.append(mag)
            ax_mag.plot(t_i[:n_min], mag, color=colors[i % len(colors)], alpha=0.5, linewidth=0.9, label=f"Tap {i + 1}")
        if t_plot is None:
            t_plot = t[:n_min] if len(t) >= n_min else t
        avg_mag = np.mean(magnitudes, axis=0)
        ax_mag.plot(t_plot, avg_mag, color="black", linewidth=1.8, label="Average")
        mag = avg_mag
        t_use = t_plot
    else:
        ax_g, ay_g, az_g = data[0], data[1], data[2]
        mag = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
        ax_mag.plot(t, mag, color="#2980b9", linewidth=1.0, label="Magnitude (g)")
        t_use = t

    if show_envelope and fn > 0 and sample_rate_hz > 0:
        _, envelope, zeta_est = decay_envelope_and_damping(t_use, mag, fn, sample_rate_hz)
        if zeta_est is not None and np.any(np.isfinite(envelope)):
            ax_mag.plot(t_use, envelope, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.9, label=f"Decay envelope (ζ≈{zeta_est:.3f})")
    if Tn is not None and Tn > 0 and len(t_use) > 0:
        t0 = float(t_use[0])
        n_periods = max(1, int((float(t_use[-1]) - t0) / Tn))
        ax_mag.axvline(t0 + Tn, color="#7f8c8d", linestyle="--", linewidth=1.0, label=f"T_n = {Tn*1000:.1f} ms")
        for k in range(2, min(n_periods + 1, 8)):
            ax_mag.axvline(t0 + k * Tn, color="#95a5a6", linestyle=":", linewidth=0.8, alpha=0.7)
    ax_mag.set_ylabel("Magnitude (g)")
    if live_spindle:
        ax_mag.set_title("Time signal — magnitude (live spindle)")
    else:
        ax_mag.set_title("Time signal — magnitude (per-tap + average)" if tap_series_list else "Time signal — magnitude (tap response)")
    ax_mag.legend(loc="upper right", fontsize=8)
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_ylim(bottom=0)

    if show_axes_xyz:
        ax_xyz = axes[1]
        if tap_series_list and len(tap_series_list) > 0 and not live_spindle:
            n_min = min(d.shape[1] for _, d in tap_series_list)
            for i, (t_i, d) in enumerate(tap_series_list):
                ax_xyz.plot(t_i[:n_min], d[0][:n_min], color="#e74c3c", alpha=0.35, linewidth=0.7)
                ax_xyz.plot(t_i[:n_min], d[1][:n_min], color="#27ae60", alpha=0.35, linewidth=0.7)
                ax_xyz.plot(t_i[:n_min], d[2][:n_min], color="#3498db", alpha=0.35, linewidth=0.7)
            avg_ax = np.mean([d[0][:n_min] for _, d in tap_series_list], axis=0)
            avg_ay = np.mean([d[1][:n_min] for _, d in tap_series_list], axis=0)
            avg_az = np.mean([d[2][:n_min] for _, d in tap_series_list], axis=0)
            t_xyz = tap_series_list[0][0][:n_min]
            ax_xyz.plot(t_xyz, avg_ax, color="#e74c3c", linewidth=1.2, alpha=0.95, label="ax")
            ax_xyz.plot(t_xyz, avg_ay, color="#27ae60", linewidth=1.2, alpha=0.95, label="ay")
            ax_xyz.plot(t_xyz, avg_az, color="#3498db", linewidth=1.2, alpha=0.95, label="az")
        else:
            ax_xyz.plot(t, data[0], color="#e74c3c", linewidth=0.8, alpha=0.9, label="ax")
            ax_xyz.plot(t, data[1], color="#27ae60", linewidth=0.8, alpha=0.9, label="ay")
            ax_xyz.plot(t, data[2], color="#3498db", linewidth=0.8, alpha=0.9, label="az")
        ax_xyz.set_ylabel("Accel (g)")
        ax_xyz.set_title("Time signal — x, y, z axes" + (" (live spindle)" if live_spindle else ""))
        ax_xyz.legend(loc="upper right", fontsize=8)
        ax_xyz.grid(True, alpha=0.3)
        ax_xyz.axhline(0, color="#bdc3c7", linewidth=0.5)

    axes[-1].set_xlabel("Time (s)")
    duration_s = float(t_use[-1] - t_use[0]) if len(t_use) > 1 else 0.0
    if live_spindle:
        fig.suptitle(f"Time signal map  ·  Live spindle dataset  ·  {duration_s*1000:.0f} ms  ·  {sample_rate_hz:.0f} Hz" + (f"  ·  f_n = {fn:.1f} Hz" if fn > 0 else ""), fontsize=10)
    else:
        fig.suptitle(f"Time signal map  ·  f_n = {fn:.1f} Hz  ·  {duration_s*1000:.0f} ms  ·  {sample_rate_hz:.0f} Hz", fontsize=10)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def verify_frf_data(
    t: np.ndarray,
    data: np.ndarray,
    force_x: np.ndarray | None,
    force_y: np.ndarray | None,
    sample_rate_hz: float,
) -> tuple[bool, str, dict]:
    """
    Verify that we have the data needed to compute and plot FRF.

    Returns:
        (ok, message, details): ok True if force has usable energy and lengths match;
        details has keys like force_columns, n_samples, force_peak_x, force_peak_y.
    """
    details: dict = {}
    if force_x is None:
        return False, "No force column (need Fx_N, Fy_N, or F_N in CSV header).", details
    n = len(t)
    if data.shape[1] != n:
        return False, f"Length mismatch: t has {n}, data has {data.shape[1]} samples.", details
    n_force = len(force_x)
    if n_force != n:
        return False, f"Length mismatch: t has {n}, force has {n_force} samples.", details
    if n < 4:
        return False, f"Too few samples ({n}); need at least 4 for FRF.", details
    peak_x = float(np.max(np.abs(force_x)))
    details["n_samples"] = n
    details["force_peak_x_N"] = peak_x
    details["force_columns"] = "Fx_N" if force_y is None else "Fx_N, Fy_N"
    if force_y is not None:
        details["force_peak_y_N"] = float(np.max(np.abs(force_y)))
    if peak_x < 1e-9:
        return False, "Force signal is effectively zero (peak < 1e-9 N); check force sensor and units.", details
    if sample_rate_hz <= 0:
        return False, "Sample rate unknown or zero; need # sample_rate_hz in CSV or regular time steps.", details
    return True, "FRF data OK.", details


# Bounds for estimated impact duration when no force sensor (seconds).
_ESTIMATED_IMPACT_DURATION_MIN_S = 0.0005
_ESTIMATED_IMPACT_DURATION_MAX_S = 0.02
_IMPACT_THRESHOLD_FRACTION = 0.1


def estimate_impact_duration_from_response(
    response_magnitude: np.ndarray,
    sample_rate_hz: float,
) -> float:
    """
    Estimate impact duration from the response magnitude (e.g. sqrt(ax²+ay²+az²)).
    Uses first/last time magnitude exceeds a fraction of its maximum; clamps to sensible bounds.
    Returns duration in seconds.
    """
    if sample_rate_hz <= 0 or len(response_magnitude) < 4:
        return _ESTIMATED_IMPACT_DURATION_MAX_S
    mag = np.abs(response_magnitude)
    peak = float(np.max(mag))
    if peak <= 0:
        return (_ESTIMATED_IMPACT_DURATION_MIN_S + _ESTIMATED_IMPACT_DURATION_MAX_S) / 2
    thresh = peak * _IMPACT_THRESHOLD_FRACTION
    above = np.where(mag >= thresh)[0]
    if len(above) < 2:
        return (_ESTIMATED_IMPACT_DURATION_MIN_S + _ESTIMATED_IMPACT_DURATION_MAX_S) / 2
    first, last = int(above[0]), int(above[-1])
    duration_s = (last - first + 1) / sample_rate_hz
    return float(np.clip(duration_s, _ESTIMATED_IMPACT_DURATION_MIN_S, _ESTIMATED_IMPACT_DURATION_MAX_S))


def synthetic_force_from_hammer_mass(
    n_samples: int,
    sample_rate_hz: float,
    hammer_mass_kg: float,
    impact_velocity_m_s: float,
    tap_center_index: int,
    duration_s: float,
) -> np.ndarray:
    """
    Build a synthetic force signal (N) for rough FRF when no force sensor is used.
    Uses a half-sine pulse with impulse I = 2 * m * v (full rebound), so F_peak = m*v*π/T.

    The pulse is placed starting at tap_center_index and lasts duration_s seconds.
    """
    force = np.zeros(n_samples, dtype=float)
    if sample_rate_hz <= 0 or duration_s <= 0 or hammer_mass_kg <= 0:
        return force
    n_impact = max(2, int(round(duration_s * sample_rate_hz)))
    n_impact = min(n_impact, n_samples - max(0, tap_center_index))
    if n_impact < 2:
        return force
    # Impulse I = 2*m*v; half-sine F(t)=F_peak*sin(π*t/T) gives I = F_peak * 2*T/π => F_peak = π*m*v/T
    T = duration_s
    F_peak = hammer_mass_kg * impact_velocity_m_s * np.pi / T
    t = np.arange(n_impact, dtype=float) / sample_rate_hz
    pulse = F_peak * np.sin(np.pi * t / T)
    start = max(0, tap_center_index)
    end = min(start + n_impact, n_samples)
    force[start:end] = pulse[: end - start]
    return force


def compute_frf_from_impact(
    response_signal: np.ndarray,
    force_signal: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FRF (H1: response/force) in frequency domain from impact test signals.

    Both signals are demeaned and Hanning-windowed. H = FFT(response) / FFT(force);
    division is protected (zero force bins yield NaN). Used when force is measured
    (e.g. impact hammer with transducer).

    Returns:
        freqs: Frequency in Hz (one-sided).
        H: Complex FRF, same length as freqs (inertance A/F if response is acceleration).
    """
    n = min(len(response_signal), len(force_signal))
    if n < 4:
        return np.array([]), np.array([])
    r = response_signal[:n] - np.mean(response_signal[:n])
    f = force_signal[:n] - np.mean(force_signal[:n])
    window = np.hanning(n)
    r = r * window
    f = f * window
    R = np.fft.rfft(r)
    F = np.fft.rfft(f)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(np.abs(F) > 1e-12 * np.max(np.abs(F)), R / F, np.nan + 1j * np.nan)
    return freqs, H


def plot_frf_figure(
    freqs: np.ndarray,
    H: np.ndarray,
    output_path: str | Path | None = None,
    form: str = "inertance",
    figsize: tuple[float, float] = (8, 5),
    title_note: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot FRF magnitude and phase (when force is measured in impact test).

    form: 'inertance' (A/F) or 'receptance' (X/F = (A/F)/ω²). Requires tap_testing.transducers
    for receptance conversion. title_note: optional suffix (e.g. "rough, estimated from hammer mass").
    """
    import matplotlib.pyplot as plt

    valid = np.isfinite(H) & (freqs > 0)
    if not np.any(valid):
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize)
        ax_mag.text(0.5, 0.5, "No valid FRF data", ha="center", va="center", transform=ax_mag.transAxes)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig
    f_plot = freqs[valid]
    H_plot = H[valid].copy()
    if form == "receptance":
        from tap_testing.transducers import inertance_to_receptance
        omega = 2.0 * np.pi * f_plot
        H_plot = inertance_to_receptance(H_plot, omega)
    mag = np.abs(H_plot)
    phase_deg = np.degrees(np.angle(H_plot))
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_mag.semilogy(f_plot, mag, color="#2980b9", linewidth=1.0)
    ax_mag.set_ylabel(f"|H| ({'m/N' if form == 'receptance' else 'g/N'})")
    title = f"FRF ({form}) — impact testing with force measurement"
    if title_note:
        title += f" — {title_note}"
    ax_mag.set_title(title)
    ax_mag.grid(True, alpha=0.3)
    ax_phase.plot(f_plot, phase_deg, color="#2980b9", linewidth=1.0)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (°)")
    ax_phase.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_engagement_vs_lobe_figure(
    result: TapTestResult,
    axial_depth_mm: float,
    helix_angle_deg: float,
    output_path: str | Path | None = None,
    n_lobes: int = 6,
    reference_rpm: float | None = None,
    figsize: tuple[float, float] = (8, 4),
    material_name: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Lobe engagement on a linear RPM scale at a given axial depth (helix-based).

    X-axis: spindle speed (rpm), linear scale. Y-axis: engagement % (0–100).
    Plots engagement at each lobe's RPM so you can see variation across RPM and
    compare to a set RPM (reference_rpm). As you get more data (depth, helix,
    tool), the same linear scale lets you compare variations.
    """
    import matplotlib.pyplot as plt

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing.milling_dynamics import engagement_vs_lobe_index_data

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)

    diameter_mm = result.tool_diameter_mm or 10.0
    n_teeth = result.n_teeth_used
    data = engagement_vs_lobe_index_data(
        result.natural_freq_hz,
        n_teeth,
        diameter_mm,
        axial_depth_mm,
        helix_angle_deg,
        n_lobes=n_lobes,
    )
    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Lobe engagement vs RPM (no data)")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    rpms = [d[1] for d in data]
    engagement_pcts = [d[2] for d in data]
    lobe_labels = [f"N{d[0]}" for d in data]

    # Linear RPM scale: use lobe RPM range, optionally extend to include reference_rpm
    rpm_min = min(rpms)
    rpm_max = max(rpms)
    if reference_rpm is not None and reference_rpm > 0:
        rpm_min = min(rpm_min, reference_rpm)
        rpm_max = max(rpm_max, reference_rpm)
    # Add a little margin
    span = max(rpm_max - rpm_min, 500)
    x_min = max(0, rpm_min - span * 0.05)
    x_max = rpm_max + span * 0.05

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(rpms, engagement_pcts, color="#2980b9", s=80, zorder=3, edgecolors="white", linewidths=1.5, label="Engagement % at lobe RPM")
    for (n, rpm, pct), lbl in zip(data, lobe_labels):
        ax.annotate(lbl, (rpm, pct), fontsize=8, ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("Spindle speed (rpm) — linear scale")
    ax.set_ylabel("Engagement % (at depth, helix-based)")
    ax.set_ylim(0, 105)
    ax.set_xlim(x_min, x_max)
    ax.axhline(100, color="#27ae60", linewidth=1, linestyle="--", alpha=0.7)
    if reference_rpm is not None and reference_rpm > 0 and x_min <= reference_rpm <= x_max:
        ax.axvline(reference_rpm, color="#8e44ad", linewidth=2, linestyle="-", alpha=0.9, label=f"Set RPM {reference_rpm:.0f}")
        # Engagement at set RPM is the same as at this depth (same depth/helix)
        eng_at_ref = engagement_pcts[0] if engagement_pcts else 0
        ax.scatter([reference_rpm], [eng_at_ref], color="#8e44ad", s=120, zorder=4, edgecolors="white", linewidths=2)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Lobe engagement at depth b = {axial_depth_mm:.1f} mm (helix {helix_angle_deg}°)  ·  "
        f"fn = {result.natural_freq_hz:.1f} Hz, {n_teeth} fl, Ø{diameter_mm} mm"
    )
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Lobe engagement vs RPM (linear scale)  ·  {default_material_label(material_name)}", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_resonance_map_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    avoid_width_fraction: float = 0.05,
    max_axial_depth_mm: float = 25.0,
    helix_angle_deg: float | None = None,
    radial_immersion_percent: float | None = None,
    figsize: tuple[float, float] = (10, 5),
    material_name: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Map tooth-pass resonance and stable RPM vs axial depth of cut (and optional width/helix).

    Red vertical bands = avoid RPM (tooth-pass resonance at f_n); risk at any depth/width.
    Green = suggested stable RPM range. Optional: horizontal line at constant-force axial
    depth (helix); table of radial immersion → Nt*, φs, φe. Full blim(Ω) boundary
    requires FRF from impact testing; this map shows resonance zones only.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing.milling_dynamics import (
        average_teeth_in_cut,
        constant_force_axial_depth_mm,
        engagement_vs_lobe_index_data,
        exit_angle_up_milling_deg,
        stability_lobe_best_spindle_speed_rpm,
    )

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)

    rpm_min, rpm_max = chart_rpm_range(result)
    diameter_mm = result.tool_diameter_mm or 10.0
    radius_mm = diameter_mm / 2.0
    n_teeth = result.n_teeth_used
    tooth_pitch_deg = 360.0 / n_teeth if n_teeth else 0

    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction, rpm_min=rpm_min)
    depth_max = max(0.1, max_axial_depth_mm)

    fig, (ax_map, ax_tbl) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]})
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"
        alpha = 0.45 if kind == "avoid" else 0.35
        ax_map.axvspan(rpm_lo, rpm_hi, 0, 1, color=color, alpha=alpha)
    ax_map.set_xlim(rpm_min, rpm_max)
    ax_map.set_ylim(0, depth_max)
    ax_map.set_xlabel("Spindle speed (rpm)")
    ax_map.set_ylabel("Axial depth of cut (mm)")
    ax_map.set_title("Tooth-pass resonance map (red = avoid at any depth; full blim requires FRF)")
    if helix_angle_deg is not None and helix_angle_deg > 0:
        b_const = constant_force_axial_depth_mm(diameter_mm, tooth_pitch_deg, helix_angle_deg)
        if 0 < b_const <= depth_max * 1.2:
            ax_map.axhline(b_const, color="#8e44ad", linewidth=1.5, linestyle="--", alpha=0.9)
            ax_map.text(rpm_max * 0.98, b_const, f"  b_const={b_const:.1f} mm (helix {helix_angle_deg}°)", fontsize=8, va="center", ha="right", color="#8e44ad")
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (tooth-pass resonance)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Suggested RPM range")
    handles = [avoid_patch, optimal_patch]
    if helix_angle_deg is not None and helix_angle_deg > 0:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#8e44ad", linewidth=2, linestyle="--", label=f"Constant-force depth (helix {helix_angle_deg}°)"))
    ax_map.legend(handles=handles, loc="upper right", fontsize=8)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_ylim(0, depth_max)

    # Right panel: cutting geometry table (radial immersion → Nt*, φs, φe; lobe speeds)
    ax_tbl.set_axis_off()
    fn_str = f"{result.natural_freq_hz:.1f}"
    if result.natural_freq_hz_uncertainty and result.natural_freq_hz_uncertainty > 0:
        fn_str += f" ± {result.natural_freq_hz_uncertainty:.2f}"
    lines = [
        "Resonance map (tap-test only)",
        f"  fn = {fn_str} Hz  ·  {n_teeth} fl  ·  Ø{diameter_mm} mm",
        f"  Suggested RPM: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f}",
        "",
        "Radial immersion (width of cut)",
        "  imm%   Nt*   φs   φe (up milling)",
    ]
    for pct in [10, 25, 50, 100]:
        if pct == 100:
            radial_depth = diameter_mm
        else:
            radial_depth = diameter_mm * (pct / 100.0)
        phi_s = 0.0
        phi_e = exit_angle_up_milling_deg(radius_mm, radial_depth)
        Nt_star = average_teeth_in_cut(phi_s, phi_e, n_teeth)
        lines.append(f"  {pct:3.0f}%   {Nt_star:.2f}   {phi_s:.0f}°  {phi_e:.0f}°")
    lines.extend([
        "",
        "Best lobe speeds (rpm)",
    ])
    for n in range(4):
        rpm = stability_lobe_best_spindle_speed_rpm(result.natural_freq_hz, n_teeth, lobe_index_n=n)
        if rpm_min <= rpm <= rpm_max:
            lines.append(f"  N{n}: {rpm:.0f}")
    # Per-tooth engagement % vs lobe index at constant-force depth (when helix set)
    if helix_angle_deg is not None and helix_angle_deg > 0:
        b_const = constant_force_axial_depth_mm(diameter_mm, tooth_pitch_deg, helix_angle_deg)
        if b_const > 0:
            engagement_data = engagement_vs_lobe_index_data(
                result.natural_freq_hz, n_teeth, diameter_mm, b_const, helix_angle_deg, n_lobes=4
            )
            lines.extend(["", "Engagement % vs lobe (at depth b_const, helix)"])
            for n, rpm, eng_pct in engagement_data:
                if rpm_min <= rpm <= rpm_max:
                    lines.append(f"  N{n}: {rpm:.0f} rpm  {eng_pct:.0f}%")
    lines.extend([
        "",
        "Red bands: tooth-pass resonance (avoid any depth/width).",
        "Full stability boundary (blim vs Ω) needs FRF + cutting coeffs.",
    ])
    ax_tbl.text(0.02, 0.98, "\n".join(lines), transform=ax_tbl.transAxes, fontsize=9, va="top", ha="left", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.4))
    fig.suptitle(f"Lobe resonance map  ·  {default_material_label(material_name)}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def compute_stability_lobe_boundary(
    freqs_hz: np.ndarray,
    Hxx: np.ndarray,
    Hyy: np.ndarray,
    Ks_n_per_mm2: float,
    force_angle_beta_deg: float,
    n_teeth: int,
    phi_s_deg: float,
    phi_e_deg: float,
    rpm_min: float,
    rpm_max: float,
    fn_nominal_hz: float,
    n_points: int = 200,
    *,
    up_milling: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute stability lobe boundary blim (mm) vs spindle speed (rpm).

    Uses oriented FRF_orient = μx·Hxx + μy·Hyy at the nominal natural frequency
    (response-only tap test gives fn; full FRF from impact with force gives Hxx, Hyy).
    Per Eq. 4.23/4.109 (ref): blim = −1 / (2·Ks·Re(FRF_orient)·Nt*). In the valid
    chatter frequency range Re(FRF_orient) < 0, so the minus sign yields positive blim.
    Requires x,y FRFs from impact testing with force.

    Directional factors μx, μy depend on cut type: up milling (ref Fig. 4.74, Example 4.5)
    or down milling (ref Fig. 4.24, Example 4.4). If up_milling is None, it is inferred
    from phi_s, phi_e (down milling when φe == 180° and φs > 0).

    Returns:
        rpm_axis: Spindle speed (rpm), shape (n_points,).
        blim_mm: Limiting chip width (mm), shape (n_points,). NaN where Re(FRF_orient) >= 0 (invalid).
    """
    from tap_testing.milling_dynamics import (
        average_teeth_in_cut,
        average_tooth_angle_deg,
        directional_factors_down_milling,
        directional_factors_up_milling,
    )
    phi_ave = average_tooth_angle_deg(phi_s_deg, phi_e_deg)
    if up_milling is None:
        # Down milling: φe = 180° and φs > 0 (ref Sect. 4.1); else up milling
        up_milling = not (phi_e_deg >= 179.0 and phi_s_deg > 0)
    if up_milling:
        mu_x, mu_y = directional_factors_up_milling(force_angle_beta_deg, phi_ave)
    else:
        mu_x, mu_y = directional_factors_down_milling(force_angle_beta_deg, phi_ave)
    Nt_star = average_teeth_in_cut(phi_s_deg, phi_e_deg, n_teeth)
    if Nt_star <= 0 or Ks_n_per_mm2 <= 0:
        rpm_axis = np.linspace(rpm_min, rpm_max, n_points)
        return rpm_axis, np.full(n_points, np.nan)
    Hxx_interp = np.interp(fn_nominal_hz, freqs_hz, np.real(Hxx)) + 1j * np.interp(fn_nominal_hz, freqs_hz, np.imag(Hxx))
    Hyy_interp = np.interp(fn_nominal_hz, freqs_hz, np.real(Hyy)) + 1j * np.interp(fn_nominal_hz, freqs_hz, np.imag(Hyy))
    FRF_orient = mu_x * Hxx_interp + mu_y * Hyy_interp
    Re_orient = np.real(FRF_orient)
    rpm_axis = np.linspace(rpm_min, rpm_max, n_points)
    # Valid chatter range: Re(FRF_orient) < 0 (ref Eq. 4.23, 4.109)
    if Re_orient >= 0:
        return rpm_axis, np.full(n_points, np.nan)
    blim_mm = -1.0 / (2.0 * Ks_n_per_mm2 * Re_orient * Nt_star)
    return rpm_axis, np.full(n_points, blim_mm)


def plot_stability_lobe_figure(
    rpm_axis: np.ndarray,
    blim_mm: np.ndarray,
    output_path: str | Path | None = None,
    title: str | None = "Stability lobe (blim vs Ω) — requires x,y FRFs + cutting coefficients",
    figsize: tuple[float, float] = (8, 4),
) -> "matplotlib.figure.Figure":
    """Plot stability boundary: spindle speed (rpm) vs limiting chip width (mm)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    valid = np.isfinite(blim_mm) & (blim_mm > 0)
    if np.any(valid):
        ax.fill_between(rpm_axis, 0, blim_mm, where=valid, color="#27ae60", alpha=0.4, label="Stable (below boundary)")
        ax.plot(rpm_axis, blim_mm, color="#2980b9", linewidth=1.5, label="blim (mm)")
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Limiting chip width (mm)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
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
    material_name: str | None = None,
    reference_rpm: float | None = None,
    reference_chip_load: float | None = None,
):
    """
    Build a figure with (1) RPM band chart, (2) tap cycles + average magnitude vs time,
    (3) FFT magnitude spectrum (natural freq. marked), and (4) milling dynamics summary.

    tap_series_list: list of (t, data) with data shape (3, N) in g. Typically 3 taps.
    material_name: Workpiece material for label. If None, uses config.
    reference_rpm: If set, draw vertical line at this RPM (e.g. from spindle_operating_frequency_hz * 60).
    reference_chip_load: Chip load (mm/tooth) for reference RPM feed label; default 0.05.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from tap_testing.config import get_config
    from tap_testing.material import default_material_label, get_material_or_default

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)  # validate

    fig, (ax_rpm, ax_traces, ax_spectrum, ax_milling) = plt.subplots(
        4, 1, figsize=(figsize[0], figsize[1] + 2.4), height_ratios=[1, 1.2, 0.9, 0.85],
    )

    # Top: RPM band chart (overall tooling RPM; extend range so red avoid zones visible when below min_rpm)
    rpm_lo_display, rpm_hi_display = chart_rpm_range(result)
    zones = get_rpm_zones(result, avoid_width_fraction=avoid_width_fraction, rpm_min=rpm_lo_display)
    for rpm_lo, rpm_hi, kind in zones:
        color = "#c0392b" if kind == "avoid" else "#27ae60"
        alpha = 0.45 if kind == "avoid" else 0.35
        ax_rpm.axvspan(rpm_lo, rpm_hi, color=color, alpha=alpha)
    # Highlight suggested RPM range (calculated optimal band)
    ax_rpm.axvline(result.suggested_rpm_min, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    ax_rpm.axvline(result.suggested_rpm_max, color="#1e8449", linewidth=2, linestyle="--", alpha=0.9)
    # Practical range at spindle (±10% of reference RPM)
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None and rpm_lo_display <= prac_hi and prac_lo <= rpm_hi_display:
            ax_rpm.axvspan(
                max(prac_lo, rpm_lo_display), min(prac_hi, rpm_hi_display),
                color="#1e8449", alpha=0.35, zorder=0,
            )
    if reference_rpm is not None and reference_rpm > 0 and rpm_lo_display <= reference_rpm <= rpm_hi_display:
        ref_cl = reference_chip_load if reference_chip_load is not None and reference_chip_load > 0 else 0.05
        ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
        ax_rpm.axvline(reference_rpm, color="#8e44ad", linewidth=2, linestyle="-", alpha=0.95)
        ax_rpm.scatter([reference_rpm], [0.5], color="#8e44ad", s=120, zorder=6, edgecolors="white", linewidths=2)
        ax_rpm.text(reference_rpm, 0.08, f"Ref\n{reference_rpm:.0f}\n{ref_feed:.0f} mm/min", fontsize=7, ha="center", va="bottom", color="#8e44ad", fontweight="bold")
    ax_rpm.set_xlim(rpm_lo_display, rpm_hi_display)
    ax_rpm.set_ylim(0, 1)
    ax_rpm.set_yticks([])
    ax_rpm.set_ylabel(" ")
    ax_rpm.set_xlabel("Spindle speed (rpm)")
    tool_mat_label = _result_tool_material_label(result)
    title_parts = [f"Natural freq. {result.natural_freq_hz:.1f} Hz"]
    title_parts.append(f"Tool: {tool_mat_label}" + (f" Ø{result.tool_diameter_mm} mm" if result.tool_diameter_mm is not None else ""))
    title_parts.append(f"{result.n_teeth_used} flutes, max {result.max_rpm:.0f} rpm")
    title_parts.append(default_material_label(material_name))
    title_parts.append("Metric")
    ax_rpm.set_title("  ·  ".join(title_parts))
    from matplotlib.lines import Line2D
    suggested_line = Line2D([0], [0], color="#1e8449", linewidth=2, linestyle="--", label=f"Suggested: {result.suggested_rpm_min:.0f}–{result.suggested_rpm_max:.0f} rpm")
    avoid_patch = mpatches.Patch(color="#c0392b", alpha=0.45, label="Avoid (chatter risk)")
    optimal_patch = mpatches.Patch(color="#27ae60", alpha=0.35, label="Optimal")
    legend_handles = [avoid_patch, optimal_patch, suggested_line]
    if reference_rpm is not None and reference_rpm > 0:
        prac_lo, prac_hi = practical_suggested_range_at_spindle(
            result.suggested_rpm_min, result.suggested_rpm_max, reference_rpm, tolerance=0.10
        )
        if prac_lo is not None and prac_hi is not None:
            spindle_hz = result.spindle_operating_frequency_hz or (reference_rpm / 60.0)
            legend_handles.append(mpatches.Patch(facecolor="#1e8449", alpha=0.35, label=f"Practical at spindle ({spindle_hz:.0f} Hz)"))
        if rpm_lo_display <= reference_rpm <= rpm_hi_display:
            ref_cl = reference_chip_load if reference_chip_load is not None and reference_chip_load > 0 else 0.05
            ref_feed = feed_rate_mm_min(ref_cl, result.n_teeth_used, reference_rpm)
            legend_handles.append(Line2D([0], [0], color="#8e44ad", linewidth=2, linestyle="-", label=f"Spindle ref: {reference_rpm:.0f} rpm ({ref_feed:.0f} mm/min)"))
    ax_rpm.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # Tap cycles + average magnitude
    avg_signal_for_spectrum: np.ndarray | None = None
    sample_rate_for_spectrum: float = 0.0
    signals_for_spectrum: list[np.ndarray] | None = None
    t_plot_cycle: np.ndarray | None = None
    avg_mag_cycle: np.ndarray | None = None
    if not tap_series_list:
        ax_traces.text(0.5, 0.5, "No tap series", ha="center", va="center", transform=ax_traces.transAxes)
    else:
        n_min = min(d.shape[1] for _, d in tap_series_list)
        magnitudes: list[np.ndarray] = []
        signals_for_axis: list[np.ndarray] = []  # same axis as result.magnitude_axis
        t_plot = None
        colors = ["#3498db", "#9b59b6", "#1abc9c"]  # blue, purple, teal
        for i, (t, data) in enumerate(tap_series_list):
            ax_g, ay_g, az_g = data[0], data[1], data[2]
            mag = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
            if result.magnitude_axis == "magnitude":
                sig = mag
            elif result.magnitude_axis == "x":
                sig = ax_g
            elif result.magnitude_axis == "y":
                sig = ay_g
            else:
                sig = az_g
            if mag.shape[0] > n_min:
                mag = mag[:n_min]
                sig = sig[:n_min]
                t = t[:n_min]
            if t_plot is None:
                t_plot = t
            elif len(t) > n_min:
                t = t[:n_min]
            magnitudes.append(mag[:n_min])
            signals_for_axis.append(sig[:n_min])
            label = f"Tap {i + 1}"
            ax_traces.plot(t[:n_min], mag[:n_min], color=colors[i % len(colors)], alpha=0.6, label=label)

        if t_plot is not None and magnitudes:
            t_plot = t_plot[:n_min]
            avg_mag = np.mean(magnitudes, axis=0)
            ax_traces.plot(t_plot, avg_mag, color="black", linewidth=2, label="Average")
            avg_signal_for_spectrum = np.mean(signals_for_axis, axis=0)
            dt = float(np.median(np.diff(t_plot)))
            sample_rate_for_spectrum = 1.0 / dt if dt > 0 else 0.0
            t_plot_cycle = t_plot
            avg_mag_cycle = avg_mag
            signals_for_spectrum = list(signals_for_axis)
            # Decay envelope overlay (optional)
            _, envelope, zeta_est = decay_envelope_and_damping(
                t_plot, avg_mag, result.natural_freq_hz, sample_rate_for_spectrum
            )
            if zeta_est is not None and np.any(np.isfinite(envelope)):
                ax_traces.plot(t_plot, envelope, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.9, label=f"Decay envelope (ζ≈{zeta_est:.3f})")
        ax_traces.set_xlabel("Time (s)")
        ax_traces.set_ylabel("Magnitude (g)")
        ax_traces.set_title("Tap cycles (independent) and average")
        ax_traces.legend(loc="upper right", fontsize=8)
        ax_traces.grid(True, alpha=0.3)

    # FFT magnitude spectrum: per-tap (light) + average (bold), natural freq. marked
    zeta_est_spectrum: float | None = None
    if avg_signal_for_spectrum is not None and sample_rate_for_spectrum > 0:
        fn = result.natural_freq_hz
        u_hz = result.natural_freq_hz_uncertainty or 0.0
        freqs, magnitude = fft_magnitude_spectrum(avg_signal_for_spectrum, sample_rate_for_spectrum)
        if len(freqs) > 0:
            freq_max = min(sample_rate_for_spectrum / 2.0, max(freqs[-1], 2.0 * fn))
            if signals_for_spectrum:
                for i, sig in enumerate(signals_for_spectrum):
                    freqs_i, mag_i = fft_magnitude_spectrum(sig, sample_rate_for_spectrum)
                    if len(freqs_i) > 0:
                        mask_i = freqs_i <= freq_max
                        ax_spectrum.plot(freqs_i[mask_i], mag_i[mask_i], color=colors[i % len(colors)], alpha=0.25, linewidth=0.8)
            mask = freqs <= freq_max
            ax_spectrum.plot(freqs[mask], magnitude[mask], color="#2980b9", linewidth=1.5, label="Average")
            ax_spectrum.axvline(fn, color="#c0392b", linestyle="--", linewidth=1.5, label=f"f_n = {fn:.1f} Hz")
            if u_hz > 0:
                ax_spectrum.axvspan(fn - u_hz, fn + u_hz, color="#c0392b", alpha=0.15)
            ax_spectrum.set_xlim(0, freq_max)
            if t_plot_cycle is not None and avg_mag_cycle is not None:
                _, _, zeta_est_spectrum = decay_envelope_and_damping(
                    t_plot_cycle, avg_mag_cycle, result.natural_freq_hz, sample_rate_for_spectrum
                )
            ax_spectrum.set_xlabel("Frequency (Hz)")
            ax_spectrum.set_ylabel("Magnitude")
            spectrum_title = "FFT spectrum (per-tap + average)"
            if zeta_est_spectrum is not None:
                spectrum_title += f"  ·  ζ≈{zeta_est_spectrum:.3f}"
            ax_spectrum.set_title(spectrum_title)
            ax_spectrum.legend(loc="upper right", fontsize=8)
            ax_spectrum.grid(True, alpha=0.3)
    else:
        ax_spectrum.set_axis_off()
        ax_spectrum.text(0.5, 0.5, "No spectrum (no tap data)", ha="center", va="center", transform=ax_spectrum.transAxes)

    # Milling dynamics summary (guidance text for near real-time use)
    milling_text = format_milling_guidance_for_cycle(result, material_name, n_lobes=3)
    ax_milling.set_axis_off()
    ax_milling.text(
        0.02, 0.98, milling_text,
        transform=ax_milling.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.5),
        family="monospace",
    )

    fig.subplots_adjust(left=0.10, right=0.96, top=0.94, bottom=0.06, hspace=0.50)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_milling_dynamics_figure(
    result: TapTestResult,
    output_path: str | Path | None = None,
    n_lobes: int = 3,
    figsize: tuple[float, float] = (8, 5),
    material_name: str | None = None,
):
    """
    Build a figure that maps tap-test results to milling dynamics for the user.

    Shows: natural frequency (and uncertainty), avoid RPM, suggested range,
    tooth-passing formula, and best stability lobe speeds (N=0, 1, ...) for the
    tool (flute count, diameter). material_name: workpiece material for label; if None, uses config.
    """
    import matplotlib.pyplot as plt

    from tap_testing.config import get_config
    from tap_testing.milling_dynamics import (
        angle_increment_deg,
        simulation_steps_rev,
        simulation_time_step_s,
        stability_lobe_best_spindle_speed_rpm,
        tooth_period_s,
        tooth_passing_frequency_hz,
    )
    from tap_testing.material import default_material_label, get_material_or_default, list_material_names, list_tool_material_names, metric_note, normalize_tool_material

    if material_name is None:
        material_name = get_config().material_name
    get_material_or_default(material_name)  # validate

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    tool_mat_label = _result_tool_material_label(result)
    tool_desc = f"{tool_mat_label} Ø{result.tool_diameter_mm} mm" if result.tool_diameter_mm is not None else tool_mat_label
    title = f"Milling dynamics · Tool: {tool_desc}, {result.n_teeth_used} flutes"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Build text block: tap result, uncertainty, tool/workpiece material, tooth passing, stability lobes
    fn_str = f"{result.natural_freq_hz:.1f}"
    if result.natural_freq_hz_uncertainty is not None and result.natural_freq_hz_uncertainty > 0:
        fn_str += f" ± {result.natural_freq_hz_uncertainty:.2f}"
    fn_str += " Hz"

    mat_label = default_material_label(material_name)
    other_mats = [m for m in list_material_names() if m != mat_label]
    _current_tool = normalize_tool_material(result.tool_material or get_config().tool_material)
    other_tool_mats = [m for m in list_tool_material_names() if m != _current_tool]

    lines = [
        "Tap-test result (metric: Hz, rpm, mm)",
        f"  Natural frequency: {fn_str}",
        f"  Measurement uncertainty: FFT resolution + tap-to-tap spread" if (result.natural_freq_hz_uncertainty and result.natural_freq_hz_uncertainty > 0) else None,
        f"  Avoid RPM (tooth-pass resonance): {result.avoid_rpm}",
        f"  Suggested RPM range: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f} rpm",
        "",
        "Tool material (cutter)",
        f"  {tool_mat_label}",
        f"  Other tool materials: --tool-material " + " | ".join(other_tool_mats) if other_tool_mats else None,
        "",
        "Workpiece material",
        f"  {mat_label}",
        f"  Other materials: --material '" + "' | '".join(other_mats) + "'" if other_mats else None,
        "",
        "Tooth passing",
        "  f_tooth = Ω × Nt / 60  (Hz),  Ω in rpm",
        f"  Example at 10,000 rpm: f_tooth = {tooth_passing_frequency_hz(10000.0, result.n_teeth_used):.0f} Hz",
        "",
        "Best stability lobe speeds (rpm)",
        "  Ω_best = fn × 60 / ((N+1) × Nt),  N = lobe index",
        "  Ideal parameters: RPM (tap test), depth (from lobe when FRF), feed (chip load).",
        "  Full stability lobe (blim vs Ω) requires x,y FRFs (impact with force) and cutting coefficients.",
    ]
    lines = [x for x in lines if x is not None]
    for n in range(n_lobes):
        rpm = stability_lobe_best_spindle_speed_rpm(
            result.natural_freq_hz,
            result.n_teeth_used,
            lobe_index_n=n,
        )
        if result.min_rpm <= rpm <= result.max_rpm:
            lines.append(f"  N = {n}: {rpm:.0f} rpm (within range)")
        else:
            lines.append(f"  N = {n}: {rpm:.0f} rpm")
    # Time-domain simulation (Sect. 4.4): regenerative chip thickness, surf, dφ, dt
    steps_rev = simulation_steps_rev(650, result.n_teeth_used)
    example_rpm = 10000.0
    tau_s = tooth_period_s(example_rpm, result.n_teeth_used)
    dt_s = simulation_time_step_s(steps_rev, example_rpm)
    dphi_deg = angle_increment_deg(steps_rev)
    lines.extend([
        "",
        "Time-domain simulation (Sect. 4.4)",
        "  h = ft·sin(φ) + n(t−τ) − n(t); n = x·sin(φ) − y·cos(φ); τ = 60/(Ω·Nt) s",
        "  Steps: chip thickness → force → displacements → increment angle; surf stores n(t−τ).",
        f"  Example Ω={example_rpm:.0f} rpm, Nt={result.n_teeth_used}: τ={tau_s:.4f} s, steps_rev={steps_rev}, dt={dt_s:.2e} s, dφ={dphi_deg:.2f}°",
    ])
    lines.append("")
    lines.append(metric_note(material_name))

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _tooling_config_to_summary_lines(config: ToolingConfiguration, n_lobes: int = 3) -> tuple[str, list[str]]:
    """Build title and body lines for one tooling configuration (for visualization)."""
    from tap_testing.material import default_material_label, get_material_or_default
    from tap_testing.milling_dynamics import stability_lobe_best_spindle_speed_rpm

    get_material_or_default(config.material_name)
    tool_desc = f"Ø{config.tool_diameter_mm} mm" if config.tool_diameter_mm is not None else "—"
    mat_label = default_material_label(config.material_name)
    title = f"{tool_desc}  ·  {config.n_teeth} fl  ·  {mat_label}"
    if config.label:
        title = f"{config.label}  —  {title}"

    avoid_rpm = rpm_to_avoid(
        config.natural_freq_hz,
        config.n_teeth,
        harmonic_orders=config.harmonic_orders,
        rpm_min=config.min_rpm,
        rpm_max=config.max_rpm,
    )
    sug_min, sug_max = suggested_rpm_range(
        avoid_rpm, rpm_min=config.min_rpm, rpm_max=config.max_rpm
    )
    fn_str = f"{config.natural_freq_hz:.1f}"
    if config.natural_freq_hz_uncertainty and config.natural_freq_hz_uncertainty > 0:
        fn_str += f" ± {config.natural_freq_hz_uncertainty:.2f}"
    fn_str += " Hz"

    avoid_str = ", ".join(f"{r:.0f}" for r in avoid_rpm[:4])
    if len(avoid_rpm) > 4:
        avoid_str += " …"

    best_speeds = [
        stability_lobe_best_spindle_speed_rpm(config.natural_freq_hz, config.n_teeth, n)
        for n in range(n_lobes)
    ]
    best_str = ", ".join(f"N{n}={r:.0f}" for n, r in enumerate(best_speeds))

    lines = [
        f"fn = {fn_str}",
        f"Avoid RPM: {avoid_str}",
        f"Suggested: {sug_min:.0f} – {sug_max:.0f} rpm",
        f"Best stability: {best_str}",
        "Process damping helps at low rpm. Chatter → set Ω so f_tooth = fc.",
    ]
    return title, lines


def plot_tooling_configurations_figure(
    configurations: list[ToolingConfiguration],
    output_path: str | Path | None = None,
    n_lobes: int = 3,
    figsize: tuple[float, float] | None = None,
    max_cols: int = 2,
):
    """
    Visualize multiple tooling configurations in one figure for comparison.

    Each configuration is shown in a card with: tool (diameter, flutes), material,
    natural frequency, avoid RPM, suggested range, best stability lobe speeds (N=0,1,2),
    and short hints (process damping, chatter strategy). Use ToolingConfiguration.from_tap_result
    to build configs from tap-test results, or construct ToolingConfiguration directly
    for assumed/synthetic fn.

    Args:
        configurations: List of ToolingConfiguration (at least one).
        output_path: If set, save figure to this path.
        n_lobes: Number of best-stability speeds per config (N=0, 1, ...).
        figsize: (width, height). If None, chosen from number of configs.
        max_cols: Maximum number of cards per row (grid layout).
    """
    import matplotlib.pyplot as plt

    if not configurations:
        raise ValueError("At least one ToolingConfiguration is required")

    n = len(configurations)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5.0 * ncols + 1.0, 2.8 * nrows + 0.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    facecolors = ["#e8f4f8", "#fff4e6", "#f0e8f0", "#e6ffe6"][:4]
    for i, config in enumerate(configurations):
        ax = axes[i]
        ax.set_axis_off()
        title, lines = _tooling_config_to_summary_lines(config, n_lobes=n_lobes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        text = "\n".join(lines)
        fc = facecolors[i % len(facecolors)]
        ax.text(
            0.03, 0.97, text,
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, alpha=0.6),
            family="monospace",
        )
    for j in range(len(configurations), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        "Tooling configurations · fn, avoid RPM, suggested range, best stability lobes",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="Analyze tap-test CSV and suggest spindle speeds and feed. Or use --calc-* to compute missing feed/RPM/chip-load."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="*",
        default=None,
        help="Tap-test CSV file(s), or a single cycle directory. Directory: use tap_*.csv and combined.csv inside it (e.g. data/cycle/RUN_ID). Files: one = analyze that CSV; multiple = combine and show per-tap + average.",
    )
    parser.add_argument(
        "-n", "--flutes",
        type=int,
        default=4,
        dest="flute_count",
        help="Flute (tooth) count (default 4)",
    )
    parser.add_argument(
        "--chatter",
        action="store_true",
        help="Run chatter identification on CSV (cutting recording). Requires --natural-freq from tap test.",
    )
    parser.add_argument(
        "--natural-freq",
        type=float,
        default=None,
        metavar="HZ",
        dest="natural_freq_hz_chatter",
        help="Natural frequency in Hz (from tap test); required for --chatter. Used as reference band for chatter detection.",
    )
    parser.add_argument(
        "--natural-freq-uncertainty",
        type=float,
        default=None,
        metavar="HZ",
        dest="natural_freq_uncertainty_hz_chatter",
        help="± band around natural freq. for chatter (Hz). Default 3%% of fn.",
    )
    parser.add_argument(
        "--chatter-plot-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save chatter spectrum figure (cutting FFT + fn band + peaks) to file",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        metavar="RPM",
        dest="rpm_chatter",
        help="Current spindle RPM during cut (optional; used with --chatter for tooth-passing context)",
    )
    parser.add_argument(
        "--calc-rpm",
        type=float,
        default=None,
        metavar="RPM",
        help="RPM for feed calculator (give two of rpm, chip-load, feed to get the third)",
    )
    parser.add_argument(
        "--calc-chip-load",
        type=float,
        default=None,
        metavar="MM",
        dest="calc_chip_load",
        help="Chip load (mm/tooth) for calculator",
    )
    parser.add_argument(
        "--calc-feed",
        type=float,
        default=None,
        metavar="MM_MIN",
        help="Feed (mm/min) for calculator",
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
        "--no-subtract-background",
        action="store_true",
        dest="no_subtract_background",
        help="Do not extract tap cycle or subtract background (analyze full recording as-is)",
    )
    parser.add_argument(
        "--live-spindle",
        action="store_true",
        dest="live_spindle",
        help="Data is from live spindle (continuous) rather than tap test; time signal shows magnitude/axes without decay/Tn; full recording analyzed.",
    )
    parser.add_argument(
        "--chip-load",
        type=float,
        default=None,
        metavar="MM",
        help="Chip load in mm/tooth to print example feed at suggested RPM",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        metavar="KG",
        dest="mass_kg",
        help="Effective mass (kg) of tool–holder–spindle; print effective stiffness k from f_n",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=None,
        metavar="N_PER_M",
        dest="stiffness_n_per_m",
        help="Effective stiffness (N/m); print effective mass m from f_n",
    )
    parser.add_argument(
        "--vibration-theory",
        action="store_true",
        dest="vibration_theory",
        help="Print nutshell summaries for free, forced, and self-excited (chatter) vibration",
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
    parser.add_argument(
        "--plot-spectrum",
        action="store_true",
        help="Show FFT magnitude spectrum (frequency vs magnitude, natural freq. marked)",
    )
    parser.add_argument(
        "--plot-spectrum-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save spectrum image to file (for impact-test analysis)",
    )
    parser.add_argument(
        "--plot-time-signal",
        action="store_true",
        help="Generate time-signal map (magnitude + x/y/z vs time; tap test: decay envelope, Tn; use --live-spindle for spindle data)",
    )
    parser.add_argument(
        "--plot-time-signal-out",
        type=Path,
        default=None,
        metavar="FILE",
        dest="plot_time_signal_out",
        help="Save time-signal figure to file",
    )
    parser.add_argument(
        "--plot-frf",
        action="store_true",
        help="Plot FRF (magnitude and phase); requires force columns in CSV or --hammer-mass for rough FRF",
    )
    parser.add_argument(
        "--plot-frf-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save FRF figure to file (requires force columns in CSV or --hammer-mass for rough FRF)",
    )
    parser.add_argument(
        "--hammer-mass",
        type=float,
        default=None,
        metavar="KG",
        dest="hammer_mass_kg",
        help="Hammer/tapper mass (kg) for rough FRF when CSV has no force columns; force estimated from F≈2*m*v impulse",
    )
    parser.add_argument(
        "--impact-velocity",
        type=float,
        default=None,
        metavar="M_S",
        dest="impact_velocity_m_s",
        help="Typical impact velocity (m/s) for rough FRF with --hammer-mass (default 0.5)",
    )
    parser.add_argument(
        "--plot-optimal-loads",
        action="store_true",
        help="Generate RPM + feed-rate chart (best lobe speeds and feed table); same as example optimal_loads chart",
    )
    parser.add_argument(
        "--plot-optimal-loads-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save optimal-loads chart to file (RPM bands + feed mm/min table at lobe speeds)",
    )
    parser.add_argument(
        "--plot-milling-dynamics",
        action="store_true",
        help="Generate milling dynamics summary figure (fn, avoid RPM, best lobe speeds, material); same as example module",
    )
    parser.add_argument(
        "--plot-milling-dynamics-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save milling dynamics figure to file",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate all visuals: time signal, spectrum, FRF (if force), RPM chart, optimal-loads, feeds & speeds by stepover, resonance map, milling dynamics. Saves each to <stem>_<name>.png.",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Recommended workflow order: time signal → spectrum → FRF (if force) → RPM/optimal/resonance → milling dynamics. Same outputs as --plot-all in optimal order (see docs/ANALYSIS_WORKFLOW.md).",
    )
    parser.add_argument(
        "--plot-feeds-speeds",
        action="store_true",
        dest="plot_feeds_speeds",
        help="Show feeds & speeds by stepover chart (10%%–100%% stepover, chip thinning, TEA, feed at ref RPM).",
    )
    parser.add_argument(
        "--plot-feeds-speeds-out",
        type=Path,
        default=None,
        metavar="FILE",
        dest="plot_feeds_speeds_out",
        help="Save feeds & speeds by stepover chart to file (default: <stem>_feeds_speeds_stepover.png).",
    )
    parser.add_argument(
        "--plot-resonance-map",
        action="store_true",
        help="Generate lobe resonance map (RPM vs axial depth, avoid bands, radial immersion table, optional helix)",
    )
    parser.add_argument(
        "--plot-resonance-map-out",
        type=Path,
        default=None,
        metavar="FILE",
        dest="plot_resonance_map_out",
        help="Save resonance map figure to file (RPM vs depth + cutting geometry)",
    )
    parser.add_argument(
        "--helix",
        type=float,
        default=None,
        metavar="DEG",
        help="Helix angle (deg) for constant-force axial depth line on resonance map",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=25.0,
        metavar="MM",
        dest="max_axial_depth_mm",
        help="Max axial depth (mm) for resonance map Y-axis (default 25)",
    )
    from tap_testing.material import list_material_names, list_tool_material_names
    parser.add_argument(
        "--material",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Workpiece material for chart label (default: 6061 aluminum). Choices: {', '.join(list_material_names())}",
    )
    parser.add_argument(
        "--tool-material",
        type=str,
        default=None,
        metavar="NAME",
        dest="tool_material",
        help=f"Tool (cutter) material for chart label (default: carbide). Choices: {', '.join(list_tool_material_names())}",
    )
    from tap_testing.config import get_config as _get_config
    _cfg = _get_config()
    parser.add_argument(
        "--reference-rpm",
        type=float,
        default=None,
        metavar="RPM",
        help="Mark this RPM on charts and in output (e.g. 18000); show feed at this speed and whether it is in suggested range",
    )
    parser.add_argument(
        "--spindle-frequency",
        type=float,
        default=None,
        metavar="HZ",
        dest="spindle_frequency_hz",
        help=f"Spindle operating frequency in Hz (rev/s); used as default reference operating point when --reference-rpm is not set. rpm = HZ × 60 (default {_cfg.spindle_operating_frequency_hz}). Set TAP_SPINDLE_OPERATING_FREQUENCY_HZ for config default.",
    )
    parser.add_argument(
        "--reference-chip-load",
        type=float,
        default=None,
        metavar="MM",
        dest="reference_chip_load",
        help="Chip load (mm/tooth) for --reference-rpm feed (default: same as --chip-load or 0.05)",
    )
    parser.add_argument(
        "--min-feed",
        type=float,
        default=1000.0,
        metavar="MM_MIN",
        dest="min_feed_mm_min",
        help="Minimum feed (mm/min) for optimal-loads lobe filter; show only lobes with feed >= this (default 1000)",
    )
    parser.add_argument(
        "--max-feed",
        type=float,
        default=2000.0,
        metavar="MM_MIN",
        dest="max_feed_mm_min",
        help="Maximum feed (mm/min) for optimal-loads lobe filter; 0 = no upper limit (default 2000)",
    )
    parser.add_argument(
        "--stepover",
        type=float,
        default=None,
        metavar="PCT",
        dest="stepover_percent",
        help="Stepover as percent of tool diameter (e.g. 25 for 25%%). Enables chip-thinning and TEA/MRR summary.",
    )
    parser.add_argument(
        "--depth-of-cut",
        type=float,
        default=None,
        metavar="MM",
        dest="depth_of_cut_mm",
        help="Axial depth of cut per pass (mm). With --stepover, prints MRR and optional power/force.",
    )
    parser.add_argument(
        "--spindle-power",
        type=float,
        default=None,
        metavar="W",
        dest="spindle_power_w",
        help="Spindle power rating in watts for overload zone (default 1500 W = 1.5 kW).",
    )
    parser.add_argument(
        "--min-chipload",
        type=float,
        default=None,
        metavar="MM",
        dest="min_chipload_mm",
        help="Minimum chipload (mm/tooth) for chip evacuation zone; below = Low fz (default 0.025).",
    )
    args = parser.parse_args()

    # Chatter identification: analyze cutting recording against tap-test natural frequency.
    if getattr(args, "chatter", False):
        fn = getattr(args, "natural_freq_hz_chatter", None)
        if fn is None or fn <= 0:
            parser.error("--chatter requires --natural-freq HZ (from tap test)")
        _ch_csv_list = list(args.csv) if args.csv else []
        if not _ch_csv_list:
            parser.error("--chatter requires a CSV file or cycle directory (positional csv)")
        if len(_ch_csv_list) > 1:
            parser.error("--chatter expects a single file or one directory, not multiple CSV paths")
        _ch_p = _ch_csv_list[0]
        if _ch_p.is_dir():
            try:
                _ch_combined, _ch_taps, _ch_homing, _ = discover_cycle_directory(_ch_p)
            except ValueError as e:
                parser.error(str(e))
            if _ch_combined is None and not _ch_taps and _ch_homing is None:
                parser.error(
                    f"No tap_*.csv, combined.csv, or homing.csv found in directory: {_ch_p}"
                )
            csv_path = (
                _ch_combined
                if _ch_combined is not None
                else (_ch_taps[0] if _ch_taps else _ch_homing)
            )
        else:
            csv_path = _ch_p
        if not csv_path.exists():
            parser.error(f"CSV not found: {csv_path}")
        t_ch, data_ch, sr_ch = load_tap_csv(csv_path)
        if sr_ch <= 0 and len(t_ch) > 1:
            sr_ch = 1.0 / float(np.median(np.diff(t_ch)))
        if sr_ch <= 0:
            parser.error("Could not determine sample rate from CSV")
        signal = np.sqrt(data_ch[0] ** 2 + data_ch[1] ** 2 + data_ch[2] ** 2)
        chatter_result = assess_chatter(
            signal,
            sr_ch,
            fn,
            natural_freq_uncertainty_hz=getattr(args, "natural_freq_uncertainty_hz_chatter", None),
            n_teeth=args.flute_count if args.flute_count > 0 else None,
            rpm=getattr(args, "rpm_chatter", None),
        )
        print("Chatter assessment (cutting recording):")
        print(f"  Natural frequency (tap test): {chatter_result.natural_freq_hz:.1f} Hz  band: [{chatter_result.natural_freq_band_hz[0]:.1f}, {chatter_result.natural_freq_band_hz[1]:.1f}] Hz")
        print(f"  Dominant frequency in signal: {chatter_result.dominant_freq_hz:.1f} Hz")
        if chatter_result.peak_near_natural_hz is not None:
            print(f"  Peak in natural-freq band: {chatter_result.peak_near_natural_hz:.1f} Hz")
        print(f"  Chatter likely: {'Yes' if chatter_result.is_chatter_likely else 'No'}")
        if chatter_result.peaks:
            print("  Top peaks (Hz): " + ", ".join(f"{f:.0f}" for f, _ in chatter_result.peaks[:5]))
        if chatter_result.suggested_rpm:
            print("  Suggested spindle speeds (stability lobes): " + ", ".join(f"{r:.0f}" for r in chatter_result.suggested_rpm[:4]) + " rpm")
        chatter_plot = getattr(args, "chatter_plot_out", None)
        if chatter_plot is not None or chatter_result.is_chatter_likely:
            freqs_ch, mag_ch = fft_magnitude_spectrum(signal, sr_ch)
            out_path = chatter_plot or (csv_path.parent / f"{csv_path.stem}_chatter_spectrum.png")
            plot_chatter_spectrum_figure(freqs_ch, mag_ch, chatter_result, output_path=out_path)
            print(f"  Chatter spectrum saved to {out_path}")
        return

    # Feed/RPM/chip-load calculator: give two of the three to get the third (F = fz × Nt × RPM).
    calc_rpm = getattr(args, "calc_rpm", None)
    calc_chip_load = getattr(args, "calc_chip_load", None)
    calc_feed = getattr(args, "calc_feed", None)
    n_calc = sum(x is not None for x in (calc_rpm, calc_chip_load, calc_feed))
    if n_calc >= 2:
        n_teeth = args.flute_count
        if n_teeth <= 0:
            parser.error("--flutes must be positive for calculator")
        if calc_rpm is not None and calc_chip_load is not None and calc_feed is not None:
            # Verify: feed should equal chip_load * n_teeth * rpm
            expected = calc_chip_load * n_teeth * calc_rpm
            if abs(calc_feed - expected) > 0.01:
                print(f"Warning: feed {calc_feed} does not match fz×Nt×RPM = {expected:.2f} mm/min")
            else:
                print(f"OK: feed = chip_load × flutes × RPM  ({calc_feed:.2f} = {calc_chip_load} × {n_teeth} × {calc_rpm})")
        elif calc_rpm is not None and calc_chip_load is not None:
            feed = feed_rate_mm_min(calc_chip_load, n_teeth, calc_rpm)
            print(f"Feed = chip_load × flutes × RPM  →  feed = {feed:.2f} mm/min")
            print(f"  ({calc_chip_load} mm/tooth × {n_teeth} fl × {calc_rpm:.0f} rpm)")
        elif calc_rpm is not None and calc_feed is not None:
            if calc_rpm <= 0:
                parser.error("--calc-rpm must be positive to solve for chip load")
            chip_load = calc_feed / (n_teeth * calc_rpm)
            print(f"Chip load = feed / (flutes × RPM)  →  chip load = {chip_load:.4f} mm/tooth")
            print(f"  ({calc_feed} mm/min ÷ ({n_teeth} × {calc_rpm:.0f}))")
        else:
            # chip_load and feed given → RPM
            if calc_chip_load <= 0:
                parser.error("--calc-chip-load must be positive to solve for RPM")
            rpm = calc_feed / (calc_chip_load * n_teeth)
            print(f"RPM = feed / (chip_load × flutes)  →  RPM = {rpm:.0f}")
            print(f"  ({calc_feed} mm/min ÷ ({calc_chip_load} × {n_teeth}))")
        sys.exit(0)

    csv_list = list(args.csv) if args.csv else []
    if not csv_list:
        parser.error("the following arguments are required: csv (file(s), cycle directory, or use --calc-* for feed calculator)")

    # Default reference RPM from spindle operating frequency (400 Hz → 24000 rpm) when --reference-rpm not set
    from tap_testing.config import rpm_from_spindle_frequency_hz
    _ref_rpm = getattr(args, "reference_rpm", None)
    if _ref_rpm is None:
        _spindle_hz = getattr(args, "spindle_frequency_hz", None)
        if _spindle_hz is None:
            _spindle_hz = _get_config().spindle_operating_frequency_hz
        _ref_rpm = rpm_from_spindle_frequency_hz(_spindle_hz) if _spindle_hz > 0 else None
    args.reference_rpm = _ref_rpm

    _spindle_hz = getattr(args, "spindle_frequency_hz", None)
    if _spindle_hz is None:
        _spindle_hz = _get_config().spindle_operating_frequency_hz

    tap_series_list: list[tuple[np.ndarray, np.ndarray]] | None = None
    _full_recording_used = False  # True when homing/live: full recording analyzed (no tap-cycle subset)

    # Single argument that is a directory → cycle run: discover tap_*.csv and combined.csv
    if len(csv_list) == 1 and csv_list[0].is_dir():
        cycle_dir = csv_list[0]
        try:
            combined_path, tap_paths, homing_path, output_base = discover_cycle_directory(cycle_dir)
        except ValueError as e:
            parser.error(str(e))
        if combined_path is None and not tap_paths and homing_path is None:
            parser.error(f"No tap_*.csv, combined.csv, or homing.csv found in directory: {cycle_dir}")
        if combined_path is not None:
            _no_tap_cycle = getattr(args, "no_subtract_background", False) or getattr(args, "live_spindle", False)
            r = analyze_tap(
                combined_path,
                flute_count=args.flute_count,
                use_axis=args.axis,
                max_rpm=args.max_rpm,
                tool_diameter_mm=args.tool_diameter_mm,
                tool_material=getattr(args, "tool_material", None),
                subtract_background=False if _no_tap_cycle else None,
                spindle_operating_frequency_hz=_spindle_hz,
            )
            t_loaded, data_loaded, sr_loaded = load_tap_csv(combined_path)
            if tap_paths:
                tap_series_list = [(load_tap_csv(p)[0], load_tap_csv(p)[1]) for p in tap_paths]
        elif tap_paths:
            t_combined, data_combined, sr_combined = combine_tap_csvs(tap_paths)
            r = analyze_tap_data(
                t_combined, data_combined, sr_combined,
                flute_count=args.flute_count,
                max_rpm=args.max_rpm,
                tool_diameter_mm=args.tool_diameter_mm,
                tool_material=getattr(args, "tool_material", None),
                use_axis=args.axis,
                spindle_operating_frequency_hz=_spindle_hz,
            )
            t_loaded, data_loaded, sr_loaded = t_combined, data_combined, sr_combined
            tap_series_list = [(load_tap_csv(p)[0], load_tap_csv(p)[1]) for p in tap_paths]
        else:
            # homing_path is the only CSV (e.g. homing calibration run): always use full
            # recording so live data is not analyzed as a small tap-cycle subsection.
            _no_tap_cycle = True
            _full_recording_used = True
            r = analyze_tap(
                homing_path,
                flute_count=args.flute_count,
                use_axis=args.axis,
                max_rpm=args.max_rpm,
                tool_diameter_mm=args.tool_diameter_mm,
                tool_material=getattr(args, "tool_material", None),
                subtract_background=False if _no_tap_cycle else None,
                spindle_operating_frequency_hz=_spindle_hz,
            )
            t_loaded, data_loaded, sr_loaded = load_tap_csv(homing_path)
        csv_path_for_outputs = output_base
        csv_file_for_loading = (
            combined_path if combined_path is not None
            else (tap_paths[0] if tap_paths else homing_path if homing_path is not None else output_base)
        )
    elif len(csv_list) == 1:
        # Single file: use full recording for homing.csv (live spindle) so we don't
        # analyze only a small tap-cycle subsection; otherwise respect flags/config.
        _is_homing_file = csv_list[0].name == "homing.csv"
        _no_tap_cycle = (
            getattr(args, "no_subtract_background", False)
            or getattr(args, "live_spindle", False)
            or _is_homing_file
        )
        if _is_homing_file:
            _full_recording_used = True
        r = analyze_tap(
            csv_list[0],
            flute_count=args.flute_count,
            use_axis=args.axis,
            max_rpm=args.max_rpm,
            tool_diameter_mm=args.tool_diameter_mm,
            tool_material=getattr(args, "tool_material", None),
            subtract_background=False if _no_tap_cycle else None,
            spindle_operating_frequency_hz=_spindle_hz,
        )
        t_loaded, data_loaded, sr_loaded = load_tap_csv(csv_list[0])
        csv_path_for_outputs = csv_list[0]
        csv_file_for_loading = csv_list[0]
    else:
        t_combined, data_combined, sr_combined = combine_tap_csvs(csv_list)
        r = analyze_tap_data(
            t_combined, data_combined, sr_combined,
            flute_count=args.flute_count,
            max_rpm=args.max_rpm,
            tool_diameter_mm=args.tool_diameter_mm,
            tool_material=getattr(args, "tool_material", None),
            use_axis=args.axis,
            spindle_operating_frequency_hz=_spindle_hz,
        )
        t_loaded, data_loaded, sr_loaded = t_combined, data_combined, sr_combined
        tap_series_list = [(load_tap_csv(p)[0], load_tap_csv(p)[1]) for p in csv_list]
        csv_path_for_outputs = csv_list[0]
        csv_file_for_loading = csv_list[0]

    # csv_path_for_outputs: base for output filenames (.parent / f"{.stem}_..."); csv_file_for_loading: actual CSV for FRF/fallback load
    if getattr(args, "live_spindle", False) or _full_recording_used:
        print("Data source: live spindle (full recording analyzed; time signal without decay/Tn)")
    unc_str = f" ± {r.natural_freq_hz_uncertainty:.2f}" if r.natural_freq_hz_uncertainty else ""
    print(f"Natural frequency (dominant): {r.natural_freq_hz:.1f}{unc_str} Hz")
    tool_mat_label = _result_tool_material_label(r)
    tool_desc = f"Tool: {tool_mat_label}"
    if r.tool_diameter_mm is not None:
        tool_desc += f" Ø{r.tool_diameter_mm} mm"
    tool_desc += f", {r.n_teeth_used} flutes, max {r.max_rpm:.0f} RPM"
    print(tool_desc)
    print(f"Avoid spindle RPM (tooth-pass resonance): {r.avoid_rpm}")
    print(f"Suggested stable RPM range: {r.suggested_rpm_min:.0f} – {r.suggested_rpm_max:.0f} RPM")

    # M593 input shaping (RRF): reduce ringing at tap-test natural frequency
    m593_line = format_m593_input_shaping(r.natural_freq_hz, damping_ratio=None)
    print(f"RRF input shaping (config.g): {m593_line}")

    # Feed rates: use --chip-load if set, else default 0.05 mm/tooth for guidance
    chip_load_mm = args.chip_load if args.chip_load is not None else 0.05
    rpm_mid = (r.suggested_rpm_min + r.suggested_rpm_max) / 2
    feed_mid = feed_rate_mm_min(chip_load_mm, r.n_teeth_used, rpm_mid)
    chip_note = "" if args.chip_load is not None else " (use --chip-load to change; default 0.05 mm/tooth)"
    print(f"Feed at suggested mid RPM: at {rpm_mid:.0f} RPM, chip load {chip_load_mm} mm/tooth{chip_note} → feed ≈ {feed_mid:.1f} mm/min")
    from tap_testing.milling_dynamics import stability_lobe_best_spindle_speed_rpm
    for n in range(3):
        lobe_rpm = stability_lobe_best_spindle_speed_rpm(r.natural_freq_hz, r.n_teeth_used, lobe_index_n=n)
        if r.min_rpm <= lobe_rpm <= r.max_rpm:
            lobe_feed = feed_rate_mm_min(chip_load_mm, r.n_teeth_used, lobe_rpm)
            print(f"  Best lobe N{n}: {lobe_rpm:.0f} RPM → feed ≈ {lobe_feed:.1f} mm/min")

    # Tool optimization (chip thinning, TEA, MRR, power/force) when --stepover given
    stepover_pct = getattr(args, "stepover_percent", None)
    if stepover_pct is not None and stepover_pct > 0:
        from tap_testing import feeds_speeds as fs
        diam_mm = r.tool_diameter_mm or 6.35  # default 1/4" if not set
        stepover_mm = diam_mm * (stepover_pct / 100.0)
        adj_cl = fs.chipload_adjusted_for_stepover_mm(diam_mm, stepover_mm, chip_load_mm)
        feed_adj = feed_rate_mm_min(adj_cl, r.n_teeth_used, rpm_mid)
        tea_deg = fs.tool_engagement_angle_deg(diam_mm, stepover_mm)
        print("Tool optimization (stepover / chip thinning):")
        print(f"  Stepover: {stepover_pct:.0f}% of Ø{diam_mm} mm = {stepover_mm:.2f} mm WOC")
        print(f"  Chip thinning: target chipload {chip_load_mm:.3f} → adjusted {adj_cl:.3f} mm/tooth → feed ≈ {feed_adj:.1f} mm/min at {rpm_mid:.0f} RPM")
        print(f"  Tool engagement angle (TEA): {tea_deg:.0f}°")
        doc_mm = getattr(args, "depth_of_cut_mm", None)
        if doc_mm is not None and doc_mm > 0:
            mrr = fs.mrr_mm3_per_min(stepover_mm, doc_mm, feed_adj)
            print(f"  MRR (WOC×DOC×feed): {mrr:.0f} mm³/min")
            mat_name = getattr(args, "material", None) or ""
            from tap_testing.material import get_material_or_default
            mat_label = get_material_or_default(mat_name or None).name
            if mat_label in fs.UNIT_POWER_W_MIN_PER_MM3:
                up = fs.UNIT_POWER_W_MIN_PER_MM3[mat_label]
                power_w = fs.cutting_power_w(mrr, up)
                torque = fs.cutting_torque_nm(power_w, rpm_mid)
                force_n = fs.cutting_force_n(torque, diam_mm / 2.0)
                print(f"  Cutting power (material={mat_label}): ≈ {power_w:.1f} W  torque ≈ {torque:.3f} N·m  force ≈ {force_n:.0f} N")

    # Reference operating point (e.g. 18k RPM) — show feed and how it relates to suggested range / avoid / lobes
    reference_rpm = getattr(args, "reference_rpm", None)
    if reference_rpm is not None and reference_rpm > 0:
        ref_cl = getattr(args, "reference_chip_load", None) or chip_load_mm
        ref_feed = feed_rate_mm_min(ref_cl, r.n_teeth_used, reference_rpm)
        in_suggested = r.suggested_rpm_min <= reference_rpm <= r.suggested_rpm_max
        zones = get_rpm_zones(r, avoid_width_fraction=0.05)
        in_avoid = any(kind == "avoid" and rpm_lo <= reference_rpm <= rpm_hi for rpm_lo, rpm_hi, kind in zones)
        nearest_lobe_n = None
        nearest_lobe_rpm = None
        best_dist = float("inf")
        for n in range(6):
            lr = stability_lobe_best_spindle_speed_rpm(r.natural_freq_hz, r.n_teeth_used, lobe_index_n=n)
            if r.min_rpm <= lr <= r.max_rpm:
                d = abs(lr - reference_rpm)
                if d < best_dist:
                    best_dist = d
                    nearest_lobe_n = n
                    nearest_lobe_rpm = lr
        print(f"Reference operating point: {reference_rpm:.0f} RPM, chip load {ref_cl} mm/tooth → feed ≈ {ref_feed:.1f} mm/min")
        if in_suggested:
            print(f"  → This RPM is within the suggested stable range.")
        else:
            if reference_rpm < r.suggested_rpm_min:
                print(f"  → Below suggested range (min {r.suggested_rpm_min:.0f} RPM).")
            else:
                print(f"  → Above suggested range (max {r.suggested_rpm_max:.0f} RPM).")
        if in_avoid:
            print(f"  → In an avoid band (tooth-pass resonance risk).")
        if nearest_lobe_n is not None:
            pct = 100 * (reference_rpm - nearest_lobe_rpm) / nearest_lobe_rpm if nearest_lobe_rpm else 0
            print(f"  → Nearest stability lobe: N{nearest_lobe_n} at {nearest_lobe_rpm:.0f} RPM ({pct:+.0f}% from your ref).")

    # Resonance map summary: radial immersion (width) → Nt*, φs, φe; constant-force depth if helix given
    helix_deg = getattr(args, "helix", None)
    if helix_deg is not None or getattr(args, "plot_resonance_map", False) or getattr(args, "plot_resonance_map_out", None) is not None:
        from tap_testing.milling_dynamics import (
            average_teeth_in_cut,
            constant_force_axial_depth_mm,
            exit_angle_up_milling_deg,
            stability_lobe_best_spindle_speed_rpm,
        )
        diam = r.tool_diameter_mm or 10.0
        radius_mm = diam / 2.0
        tooth_pitch_deg = 360.0 / r.n_teeth_used if r.n_teeth_used else 0
        print("Resonance map (cutting geometry):")
        print("  Radial immersion (width of cut)  Nt*   φs   φe (up milling)")
        for pct in [10, 25, 50, 100]:
            radial_depth = diam * (pct / 100.0) if pct < 100 else diam
            phi_s, phi_e = 0.0, exit_angle_up_milling_deg(radius_mm, radial_depth)
            Nt_star = average_teeth_in_cut(phi_s, phi_e, r.n_teeth_used)
            print(f"    {pct:3.0f}%   (radial depth {radial_depth:.1f} mm)     {Nt_star:.2f}   {phi_s:.0f}°  {phi_e:.0f}°")
        if helix_deg is not None and helix_deg > 0 and tooth_pitch_deg > 0:
            b_const = constant_force_axial_depth_mm(diam, tooth_pitch_deg, helix_deg)
            print(f"  Constant-force axial depth (helix {helix_deg}°): b = {b_const:.1f} mm")
        print("  Red bands (avoid RPM) = tooth-pass resonance at any depth/width. Full blim(Ω) requires FRF.")

    # SDOF interpretation: f_n = (1/(2π))√(k/m)
    if args.mass_kg is not None or args.stiffness_n_per_m is not None:
        interp = compute_sdof_interpretation(
            r,
            mass_kg=args.mass_kg,
            stiffness_n_per_m=args.stiffness_n_per_m,
        )
        if "effective_stiffness_n_per_m" in interp:
            print(f"SDOF (given mass {args.mass_kg} kg): effective stiffness ≈ {interp['effective_stiffness_n_per_m']:.0f} N/m")
        if "effective_mass_kg" in interp:
            print(f"SDOF (given stiffness {args.stiffness_n_per_m:.0f} N/m): effective mass ≈ {interp['effective_mass_kg']:.4f} kg")
        print(f"  Free: {interp['nutshell']}")
        if getattr(args, "vibration_theory", False):
            print(f"  Forced (resonance): {sdof.forced_vibration_summary_nutshell()}")
            print(f"  FRF mag/phase: {sdof.frequency_response_magnitude_phase_nutshell()}")
            print(f"  Self-excited (chatter): {sdof.self_excited_vibration_summary_nutshell()}")

    # Vibration theory summary (free, forced, self-excited)
    if getattr(args, "vibration_theory", False) and not (args.mass_kg is not None or args.stiffness_n_per_m is not None):
        theory = sdof.vibration_theory_summary()
        print("Vibration theory (nutshell):")
        print(f"  Free:      {theory['free']}")
        print(f"  Forced:    {theory['forced']}")
        print(f"  FRF mag/φ: {theory['frf_magnitude_phase']}")
        print(f"  Self-exc.: {theory['self_excited']}")
        print(f"  Display:   {theory['display_scale']}")

    # --plot-all / --workflow: enable all outputs; --workflow also uses recommended order below
    if getattr(args, "plot_all", False) or getattr(args, "workflow", False):
        args.plot = True
        args.plot_spectrum = True
        args.plot_optimal_loads = True
        args.plot_feeds_speeds = True
        args.plot_milling_dynamics = True
        args.plot_resonance_map = True
        args.plot_time_signal = True
        args.plot_frf = True  # will run only if force columns present
        if args.plot_spectrum_out is None:
            args.plot_spectrum_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_spectrum.png"
        if getattr(args, "plot_optimal_loads_out", None) is None:
            args.plot_optimal_loads_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_optimal_loads.png"
        if getattr(args, "plot_feeds_speeds_out", None) is None:
            args.plot_feeds_speeds_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_feeds_speeds_stepover.png"
        if getattr(args, "plot_milling_dynamics_out", None) is None:
            args.plot_milling_dynamics_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_milling_dynamics.png"
        if getattr(args, "plot_resonance_map_out", None) is None:
            args.plot_resonance_map_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_resonance_map.png"
        if getattr(args, "plot_time_signal_out", None) is None:
            args.plot_time_signal_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_time_signal.png"
        if getattr(args, "plot_frf_out", None) is None:
            args.plot_frf_out = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_frf.png"

    # Load tap data once for workflow order (time signal, spectrum, and optionally FRF)
    t_loaded, data_loaded, sr_loaded = None, None, None
    if (
        args.plot_spectrum or args.plot_spectrum_out is not None
        or getattr(args, "plot_time_signal", False)
        or getattr(args, "plot_time_signal_out", None) is not None
    ):
        t_loaded, data_loaded, sr_loaded = load_tap_csv(csv_file_for_loading)
        if sr_loaded <= 0 and len(t_loaded) > 1:
            sr_loaded = 1.0 / float(np.median(np.diff(t_loaded)))

    # --- Recommended workflow order (see docs/ANALYSIS_WORKFLOW.md) ---
    # 1. Time signal map (data quality)
    if getattr(args, "plot_time_signal", False) or getattr(args, "plot_time_signal_out", None) is not None:
        if t_loaded is None:
            t_loaded, data_loaded, sr_loaded = load_tap_csv(csv_file_for_loading)
            if sr_loaded <= 0 and len(t_loaded) > 1:
                sr_loaded = 1.0 / float(np.median(np.diff(t_loaded)))
        out_path = getattr(args, "plot_time_signal_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_time_signal.png")
        live_spindle = getattr(args, "live_spindle", False)
        plot_time_signal_figure(r, t_loaded, data_loaded, sr_loaded, output_path=out_path, tap_series_list=tap_series_list, live_spindle=live_spindle)
        print(f"Time signal map saved to {out_path} (view anytime)" + (" [live spindle]" if live_spindle else ""))
        if getattr(args, "plot_time_signal", False):
            import matplotlib.pyplot as plt
            plt.show()

    # 2. Spectrum (validate f_n)
    if args.plot_spectrum or args.plot_spectrum_out is not None:
        if t_loaded is None:
            t_loaded, data_loaded, sr_loaded = load_tap_csv(csv_file_for_loading)
            if sr_loaded <= 0 and len(t_loaded) > 1:
                sr_loaded = 1.0 / float(np.median(np.diff(t_loaded)))
        spectrum_path = args.plot_spectrum_out or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_spectrum.png")
        plot_spectrum_figure(r, t_loaded, data_loaded, sr_loaded, output_path=spectrum_path, tap_series_list=tap_series_list)
        print(f"Spectrum saved to {spectrum_path} (view anytime)")
        if args.plot_spectrum:
            import matplotlib.pyplot as plt
            plt.show()

    # 3. FRF (if force measured — foundation for blim(Ω) later; or rough FRF from --hammer-mass)
    t_f = data_f = None
    sr_f = 0.0
    force_x = force_y = None
    if getattr(args, "plot_frf", False) or getattr(args, "plot_frf_out", None) is not None:
        try:
            t_f, data_f, sr_f, force_x, force_y = load_tap_csv_with_force(csv_file_for_loading)
        except Exception as e:
            force_x = force_y = None
            if getattr(args, "plot_frf_out", None) is not None or getattr(args, "plot_frf", False):
                print(f"FRF load failed: {e}")
        if force_x is not None:
            ok, msg, details = verify_frf_data(t_f, data_f, force_x, force_y, sr_f)
            if ok:
                print(f"FRF data: {details.get('force_columns', '?')}, {details.get('n_samples', 0)} samples, force peak ≈ {details.get('force_peak_x_N', 0):.2f} N")
                if sr_f > 0:
                    if force_y is not None and not np.array_equal(force_x, force_y):
                        freqs_f, Hxx = compute_frf_from_impact(data_f[0], force_x, sr_f)
                        freqs_f, Hyy = compute_frf_from_impact(data_f[1], force_y, sr_f)
                        H_plot = Hxx
                    else:
                        mag_resp = np.sqrt(data_f[0] ** 2 + data_f[1] ** 2 + data_f[2] ** 2)
                        freqs_f, H_plot = compute_frf_from_impact(mag_resp, force_x, sr_f)
                    frf_path = getattr(args, "plot_frf_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_frf.png")
                    plot_frf_figure(freqs_f, H_plot, output_path=frf_path, form="inertance")
                    print(f"FRF saved to {frf_path} (impact testing with force measurement)")
                    if getattr(args, "plot_frf", False):
                        import matplotlib.pyplot as plt
                        plt.show()
            else:
                print(f"FRF skipped: {msg}")
                if details:
                    for k, v in details.items():
                        print(f"  {k}: {v}")
        else:
            # No force columns: try rough FRF from hammer mass if requested
            want_frf = getattr(args, "plot_frf_out", None) is not None or getattr(args, "plot_frf", False)
            if want_frf:
                from tap_testing.config import get_config
                cfg = get_config()
                hammer_kg = getattr(args, "hammer_mass_kg", None)
                if hammer_kg is None:
                    hammer_kg = cfg.hammer_mass_kg
                if (t_f is not None and data_f is not None and hammer_kg is not None
                        and hammer_kg > 0 and sr_f > 0 and data_f.size >= 4):
                    impact_vel = getattr(args, "impact_velocity_m_s", None)
                    if impact_vel is None:
                        impact_vel = cfg.impact_velocity_m_s
                    if impact_vel is None or impact_vel <= 0:
                        impact_vel = 0.5
                    mag_resp = np.sqrt(data_f[0] ** 2 + data_f[1] ** 2 + data_f[2] ** 2)
                    tap_center = int(np.argmax(mag_resp))
                    duration_s = estimate_impact_duration_from_response(mag_resp, sr_f)
                    n_samples = len(t_f)
                    force_synth = synthetic_force_from_hammer_mass(
                        n_samples, sr_f, hammer_kg, impact_vel, tap_center, duration_s
                    )
                    freqs_f, H_plot = compute_frf_from_impact(mag_resp, force_synth, sr_f)
                    if len(freqs_f) > 0:
                        frf_path = getattr(args, "plot_frf_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_frf.png")
                        plot_frf_figure(
                            freqs_f, H_plot,
                            output_path=frf_path,
                            form="inertance",
                            title_note="rough (force estimated from hammer mass)",
                        )
                        print(f"FRF (rough) saved to {frf_path} (hammer mass {hammer_kg} kg, v≈{impact_vel} m/s)")
                        if getattr(args, "plot_frf", False):
                            import matplotlib.pyplot as plt
                            plt.show()
                    else:
                        print("FRF (rough) skipped: could not compute FRF from synthetic force.")
                else:
                    print("FRF plot skipped: CSV has no force columns (Fx_N, Fy_N or F_N). Use --hammer-mass KG for rough FRF.")

    # Save chart image by default so it can be viewed later
    save_chart = not args.no_save_chart
    chart_path = args.plot_out
    if save_chart and chart_path is None:
        chart_path = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_rpm_chart.png"
    # With --plot-all/--workflow, always set RPM chart path so it is saved even with --no-save-chart
    if (getattr(args, "plot_all", False) or getattr(args, "workflow", False)) and chart_path is None:
        chart_path = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_rpm_chart.png"
    ref_rpm = getattr(args, "reference_rpm", None)
    ref_chip = getattr(args, "reference_chip_load", None) or (chip_load_mm if ref_rpm else None)
    save_rpm_chart = (save_chart or getattr(args, "plot_all", False) or getattr(args, "workflow", False)) and chart_path is not None
    if save_rpm_chart:
        plot_result_figure(r, output_path=chart_path, material_name=args.material, reference_rpm=ref_rpm, reference_chip_load=ref_chip)
        print(f"Chart saved to {chart_path} (view anytime)")
    # Also save optimal-loads chart (RPM + feed table) by default so feed rates are visualized
    optimal_loads_path = None
    if save_chart and chart_path is not None:
        optimal_loads_path = getattr(args, "plot_optimal_loads_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_optimal_loads.png")
        _max_feed = getattr(args, "max_feed_mm_min", 2000.0)
        plot_optimal_loads_figure(
            r,
            output_path=optimal_loads_path,
            chip_load_mm=chip_load_mm,
            n_lobes=6,
            material_name=args.material,
            reference_rpm=ref_rpm,
            reference_chip_load=ref_chip,
            min_feed_mm_min=getattr(args, "min_feed_mm_min", 1000.0),
            max_feed_mm_min=_max_feed if _max_feed and _max_feed > 0 else None,
        )
        print(f"Optimal loads (feed table) saved to {optimal_loads_path} (view anytime)")

    # Save feeds & speeds by stepover chart by default (10%, 25%, 35%, 50%, 75%, 100%)
    if save_chart and chart_path is not None:
        stepover_path = getattr(args, "plot_feeds_speeds_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_feeds_speeds_stepover.png")
        plot_feeds_speeds_stepover_figure(
            r,
            output_path=stepover_path,
            chip_load_mm=chip_load_mm,
            reference_rpm=ref_rpm,
            depth_of_cut_mm=getattr(args, "depth_of_cut_mm", None),
            material_name=args.material,
            reference_chip_load=ref_chip,
            spindle_power_w=getattr(args, "spindle_power_w", None),
            min_chipload_mm=getattr(args, "min_chipload_mm", None),
        )
        print(f"Feeds & speeds by stepover saved to {stepover_path} (view anytime)")

    do_plot = args.plot and not args.no_plot
    if do_plot:
        plot_result(r, output_path=None, show=True, material_name=args.material, reference_rpm=ref_rpm, reference_chip_load=ref_chip)

    # 4. Optimal-loads chart (explicit path or show; may duplicate default save)
    if getattr(args, "plot_optimal_loads", False) or getattr(args, "plot_optimal_loads_out", None) is not None:
        out_path = getattr(args, "plot_optimal_loads_out", None)
        if out_path is None:
            out_path = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_optimal_loads.png"
        _max_feed = getattr(args, "max_feed_mm_min", 2000.0)
        plot_optimal_loads_figure(
            r,
            output_path=out_path,
            chip_load_mm=chip_load_mm,
            n_lobes=6,
            material_name=args.material,
            reference_rpm=ref_rpm,
            reference_chip_load=ref_chip,
            min_feed_mm_min=getattr(args, "min_feed_mm_min", 1000.0),
            max_feed_mm_min=_max_feed if _max_feed and _max_feed > 0 else None,
        )
        if out_path != optimal_loads_path:
            print(f"Optimal loads (feed table) saved to {out_path} (view anytime)")
        if getattr(args, "plot_optimal_loads", False):
            import matplotlib.pyplot as plt
            plt.show()

    # 4b. Feeds & speeds by stepover (explicit show or custom path)
    if getattr(args, "plot_feeds_speeds", False) or getattr(args, "plot_feeds_speeds_out", None) is not None:
        out_path = getattr(args, "plot_feeds_speeds_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_feeds_speeds_stepover.png")
        plot_feeds_speeds_stepover_figure(
            r,
            output_path=out_path,
            chip_load_mm=chip_load_mm,
            reference_rpm=ref_rpm,
            depth_of_cut_mm=getattr(args, "depth_of_cut_mm", None),
            material_name=args.material,
            reference_chip_load=ref_chip,
            spindle_power_w=getattr(args, "spindle_power_w", None),
            min_chipload_mm=getattr(args, "min_chipload_mm", None),
        )
        if getattr(args, "plot_feeds_speeds_out", None) is not None:
            print(f"Feeds & speeds by stepover saved to {out_path}")
        if getattr(args, "plot_feeds_speeds", False):
            import matplotlib.pyplot as plt
            plt.show()

    # 5. Resonance map (RPM vs depth, helix, radial immersion table)
    if getattr(args, "plot_resonance_map", False) or getattr(args, "plot_resonance_map_out", None) is not None:
        out_path = getattr(args, "plot_resonance_map_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_resonance_map.png")
        helix_deg = getattr(args, "helix", None)
        plot_resonance_map_figure(
            r,
            output_path=out_path,
            max_axial_depth_mm=getattr(args, "max_axial_depth_mm", 25.0),
            helix_angle_deg=helix_deg,
            figsize=(10, 5),
            material_name=args.material,
        )
        print(f"Resonance map saved to {out_path} (view anytime)")
        # Per-tooth engagement % vs lobe index at depth (when helix set)
        if helix_deg is not None and helix_deg > 0:
            from tap_testing.milling_dynamics import constant_force_axial_depth_mm
            diam = r.tool_diameter_mm or 10.0
            tooth_pitch = 360.0 / r.n_teeth_used if r.n_teeth_used else 0
            depth_at_helix = constant_force_axial_depth_mm(diam, tooth_pitch, helix_deg)
            eng_path = csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_engagement_vs_lobe.png"
            set_rpm = getattr(args, "reference_rpm", None)
            if set_rpm is None and r.spindle_operating_frequency_hz and r.spindle_operating_frequency_hz > 0:
                from tap_testing.config import rpm_from_spindle_frequency_hz
                set_rpm = rpm_from_spindle_frequency_hz(r.spindle_operating_frequency_hz)
            plot_engagement_vs_lobe_figure(
                r,
                axial_depth_mm=depth_at_helix,
                helix_angle_deg=helix_deg,
                output_path=eng_path,
                n_lobes=6,
                reference_rpm=set_rpm,
                material_name=args.material,
            )
            print(f"Engagement vs lobe (at depth {depth_at_helix:.1f} mm, linear RPM scale) saved to {eng_path}")
        if getattr(args, "plot_resonance_map", False):
            import matplotlib.pyplot as plt
            plt.show()

    # 6. Milling dynamics summary (text panel)
    if getattr(args, "plot_milling_dynamics", False) or getattr(args, "plot_milling_dynamics_out", None) is not None:
        out_path = getattr(args, "plot_milling_dynamics_out", None) or (csv_path_for_outputs.parent / f"{csv_path_for_outputs.stem}_milling_dynamics.png")
        plot_milling_dynamics_figure(
            r,
            output_path=out_path,
            n_lobes=4,
            material_name=args.material,
        )
        print(f"Milling dynamics summary saved to {out_path} (view anytime)")
        if getattr(args, "plot_milling_dynamics", False):
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
