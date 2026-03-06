"""
Measurement uncertainties for tap-test and FRF (frequency response function) results.

A complete description of an FRF should include both the frequency-dependent mean
values (e.g. real and imaginary parts) and the frequency-dependent uncertainty
in those values. Because the FRF is complex-valued, a defensible uncertainty
statement requires a bivariate uncertainty analysis: the real and imaginary parts
can be correlated, and confidence regions at each frequency are ellipsoids in the
complex plane. The total scalar uncertainty can be obtained from an eigenanalysis
of the FRF covariance matrix.

Important uncertainty sources in tool-point impact testing include:
  • Statistical variations (repeatability)
  • Uncertain calibration coefficients (hammer and vibration transducer) — in
    one study these comprised >80% of the total uncertainty, ~2% of the mean
    tool point direct receptance
  • Misalignment between intended and actual force/hammer direction vs.
    accelerometer axis (cosine-type bias)
  • Accelerometer mass loading of the structure (measurement bias)

This module provides:
  • FFT-based frequency resolution uncertainty for dominant natural frequency
  • Optional correction formulas for FRF bias (misalignment, mass loading) when
    complex-valued FRF data are available
  • Natural frequency uncertainty from the spread across multiple tap cycles
"""

from __future__ import annotations

import math
from typing import Sequence


def fft_frequency_resolution_uncertainty_hz(
    sample_rate_hz: float,
    n_samples: int,
    half_bin: bool = True,
) -> float:
    """
    Approximate uncertainty in frequency (Hz) due to FFT bin width.

    The frequency resolution is df = sample_rate_hz / n_samples. The true peak
    can lie anywhere in the bin; a common choice for uncertainty is half the
    bin width (half_bin=True) or the full bin width (half_bin=False).

    Args:
        sample_rate_hz: Sample rate in Hz.
        n_samples: Number of samples in the FFT (length of time record).
        half_bin: If True, return 0.5 * df; else return df.

    Returns:
        Frequency uncertainty in Hz (≥ 0).
    """
    if sample_rate_hz <= 0 or n_samples < 2:
        return 0.0
    df = sample_rate_hz / n_samples
    return (0.5 if half_bin else 1.0) * df


def correct_frf_misalignment_bias(
    frf_measured: complex,
    u_beta_rad: float,
) -> complex:
    """
    Correct FRF for bias due to misalignment between hammer and accelerometer axes.

    A small bias is introduced when the force input direction is misaligned with
    the accelerometer axis. The first-order correction (cosine-type) is:

        FRF_true = FRF_measured * (1 - 0.5 * u(β)²)

    where u(β) is the uncertainty in the misalignment angle β in radians.
    Typical misalignment uncertainty is a few degrees.

    Args:
        frf_measured: Measured FRF value (e.g. receptance in m/N).
        u_beta_rad: Uncertainty in misalignment angle (radians).

    Returns:
        Corrected FRF (same units as frf_measured).
    """
    factor = 1.0 - 0.5 * (u_beta_rad ** 2)
    return frf_measured * factor


def correct_frf_mass_loading(
    frf_measured: complex,
    omega_rad_s: float,
    accelerometer_mass_kg: float,
) -> complex:
    """
    Correct FRF for bias due to accelerometer mass loading.

    Mass loading of the structure by the accelerometer introduces a measurement
    bias. The correction is:

        FRF_true = FRF_measured / (1 + m_a * ω² * FRF_measured)

    with m_a = accelerometer mass (kg), ω in rad/s, FRF in m/N (receptance).
    For compatibility with other units, ensure omega is in rad/s and FRF has
    units such that m_a * ω² * FRF is dimensionless.

    Args:
        frf_measured: Measured FRF (receptance, m/N).
        omega_rad_s: Angular frequency in rad/s (2π * f_Hz).
        accelerometer_mass_kg: Accelerometer mass in kg.

    Returns:
        Corrected FRF (same units as frf_measured).
    """
    denom = 1.0 + accelerometer_mass_kg * (omega_rad_s ** 2) * frf_measured
    if abs(denom) < 1e-20:
        return frf_measured  # avoid division by zero
    return frf_measured / denom


def natural_freq_uncertainty_from_tap_spread(
    natural_freqs_hz: Sequence[float],
) -> float:
    """
    Standard uncertainty (one standard deviation) of natural frequency from
    multiple tap cycles.

    Use when several impact tests are run and each yields a natural frequency;
    the spread across taps provides a repeatability uncertainty. For a small
    number of taps, consider using the sample standard deviation as the
    uncertainty; for reporting, an expanded uncertainty (e.g. k=2) can be applied
    by the caller.

    Args:
        natural_freqs_hz: List of natural frequencies (Hz) from each tap.

    Returns:
        Sample standard deviation in Hz, or 0.0 if fewer than two values.
    """
    if not natural_freqs_hz or len(natural_freqs_hz) < 2:
        return 0.0
    n = len(natural_freqs_hz)
    mean_f = sum(natural_freqs_hz) / n
    variance = sum((f - mean_f) ** 2 for f in natural_freqs_hz) / (n - 1)
    return math.sqrt(max(0.0, variance))


def combined_natural_freq_uncertainty_hz(
    fft_uncertainty_hz: float,
    tap_spread_std_hz: float | None = None,
) -> float:
    """
    Combine FFT resolution uncertainty and (optional) tap-to-tap spread.

    Assumes the two contributions are independent; combined uncertainty
    u = sqrt(u_fft² + u_spread²). If tap_spread_std_hz is None or 0, returns
    fft_uncertainty_hz.

    Args:
        fft_uncertainty_hz: Uncertainty from FFT bin resolution (e.g. half-bin).
        tap_spread_std_hz: Standard deviation of natural frequency across taps, or None.

    Returns:
        Combined standard uncertainty in Hz.
    """
    u_fft = max(0.0, fft_uncertainty_hz)
    u_spread = max(0.0, tap_spread_std_hz) if tap_spread_std_hz is not None else 0.0
    if u_spread <= 0:
        return u_fft
    return math.sqrt(u_fft ** 2 + u_spread ** 2)
