"""
Tap cycle extraction and background vibration subtraction.

We treat the recording as containing:
- A pre-tap segment (background): vibration/noise before the impact.
- An impact event: detected by magnitude or derivative threshold.
- A tap phase: ~1 s of data starting at impact (the "tap cycle").

Background is estimated from the pre-tap segment and subtracted from the tap-cycle
data to produce a cleaner signal for FFT/natural frequency analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TapCycleResult:
    """Result of extracting tap cycle and subtracting background."""

    t: np.ndarray  # time for tap cycle (s)
    data: np.ndarray  # (3, N) cleaned [ax, ay, az] in g
    sample_rate_hz: float
    impact_index: int  # index in original array where impact was detected (-1 if not found)
    background_start_index: int
    background_end_index: int
    tap_start_index: int
    tap_end_index: int
    baseline_per_axis: np.ndarray  # (3,) mean of background per axis, in g


def _magnitude_g(data: np.ndarray) -> np.ndarray:
    """(3, N) -> (N,) magnitude in g."""
    return np.sqrt(np.sum(data ** 2, axis=0))


def detect_impact(
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    *,
    threshold_g: float = 0.5,
    min_background_samples: Optional[int] = None,
    use_derivative: bool = True,
    derivative_threshold_g_per_s: float = 50.0,
) -> int:
    """
    Find the sample index where the tap impact occurs.

    Impact is detected when either:
    - Magnitude exceeds threshold_g (above baseline), or
    - Rate of change of magnitude exceeds derivative_threshold_g_per_s (optional).

    We require at least min_background_samples (or ~5% of length) before the
    detected impact so we have a valid background window.

    Args:
        t: Time vector (s), shape (N,).
        data: (3, N) [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        threshold_g: Magnitude (g) above baseline to consider impact. Baseline is
            taken as mean of first 10% of samples.
        min_background_samples: Minimum number of samples before impact. If None,
            uses max(10, 5% of N).
        use_derivative: If True, also check for sharp rise (jerk) as impact.
        derivative_threshold_g_per_s: Threshold for d(magnitude)/dt (g/s) to mark impact.

    Returns:
        Index of first sample where impact is detected, or 0 if no clear impact
        (caller can then use full recording with background = initial segment).
    """
    n = data.shape[1]
    if n < 10:
        return 0

    mag = _magnitude_g(data)
    # Baseline from first 10% of recording (pre-tap quiet period)
    n_bg = max(10, n // 10)
    baseline_mag = float(np.mean(mag[:n_bg]))
    mag_above = mag - baseline_mag

    if min_background_samples is None:
        min_background_samples = max(10, n // 20)

    # Magnitude-based: first index where mag exceeds baseline + threshold
    dt_s = 1.0 / sample_rate_hz if sample_rate_hz > 0 else (float(t[-1] - t[0]) / (n - 1)) if n > 1 else 1.0
    impact_idx = -1
    for i in range(min_background_samples, n - 1):
        if mag_above[i] >= threshold_g:
            impact_idx = i
            break

    # Optional: derivative (jerk) - sharp rise indicates impact
    if use_derivative and impact_idx < 0:
        dmag = np.abs(np.diff(mag)) / dt_s
        for i in range(min_background_samples, len(dmag)):
            if dmag[i] >= derivative_threshold_g_per_s:
                impact_idx = i
                break

    if impact_idx < 0:
        return 0
    return impact_idx


def extract_tap_cycle_and_subtract_background(
    t: np.ndarray,
    data: np.ndarray,
    sample_rate_hz: float,
    *,
    tap_cycle_duration_s: float = 1.0,
    background_duration_s: Optional[float] = None,
    background_fraction: float = 0.2,
    impact_threshold_g: float = 0.5,
    use_impact_detection: bool = True,
) -> TapCycleResult:
    """
    Extract the tap cycle (~1 s from impact) and subtract background from it.

    Background is estimated from the segment *before* the tap (start of recording
    up to impact, or a fixed initial window). The mean of each axis over that
    segment is subtracted from the tap-cycle data to reduce baseline drift and
    ambient vibration.

    Args:
        t: Time vector (s), shape (N,).
        data: (3, N) [ax, ay, az] in g.
        sample_rate_hz: Sample rate in Hz.
        tap_cycle_duration_s: Duration of the tap phase in seconds (default 1.0).
        background_duration_s: If set, use this many seconds at the start as
            background (ignores impact for background window). If None, background
            is from start to impact (or background_fraction of length if no impact).
        background_fraction: If background_duration_s is None and no impact found,
            use this fraction of the recording length for background (default 0.2).
        impact_threshold_g: Threshold (g) for impact detection (see detect_impact).
        use_impact_detection: If True, run impact detection to define tap start; otherwise
            use start of recording as tap start.

    Returns:
        TapCycleResult with cleaned (t, data) for the tap cycle and metadata.
    """
    n = data.shape[1]
    if n < 10:
        raise ValueError("Tap data too short for cycle extraction")

    dt_s = 1.0 / sample_rate_hz if sample_rate_hz > 0 else (float(t[-1] - t[0]) / (n - 1)) if n > 1 else 1.0 / 800.0
    n_tap_samples = int(round(tap_cycle_duration_s * sample_rate_hz))
    n_tap_samples = min(n_tap_samples, n)

    if use_impact_detection:
        impact_idx = detect_impact(
            t, data, sample_rate_hz,
            threshold_g=impact_threshold_g,
        )
    else:
        impact_idx = 0

    # Background window: either fixed duration at start, or from start to impact
    if background_duration_s is not None:
        n_bg = min(int(round(background_duration_s * sample_rate_hz)), impact_idx if impact_idx > 0 else n // 5)
        n_bg = max(10, n_bg)
        background_end = min(n_bg, n)
    else:
        if impact_idx > 10:
            background_end = impact_idx
        else:
            background_end = max(10, int(n * background_fraction))
    background_start = 0
    tap_start = impact_idx
    tap_end = min(tap_start + n_tap_samples, n)
    if tap_end <= tap_start:
        tap_end = min(tap_start + 1, n)

    # Baseline = mean of each axis over background window
    baseline = np.mean(data[:, background_start:background_end], axis=1)  # (3,)
    # Extract tap segment and subtract baseline
    t_tap = t[tap_start:tap_end].copy()
    data_tap = data[:, tap_start:tap_end].copy()
    data_cleaned = data_tap - baseline[:, np.newaxis]

    return TapCycleResult(
        t=t_tap,
        data=data_cleaned,
        sample_rate_hz=sample_rate_hz,
        impact_index=impact_idx,
        background_start_index=background_start,
        background_end_index=background_end,
        tap_start_index=tap_start,
        tap_end_index=tap_end,
        baseline_per_axis=baseline,
    )
