"""
Vibration measurement via impact excitation: transducers, inertance vs receptance, phase correction.

We use the ADXL345 for measurement: a low-mass accelerometer suitable for tool-point
tap testing with impulse excitation.

Transducer types:
  Noncontact (capacitance probes, laser vibrometers) do not influence system dynamics
  but are less convenient. Contacting types (accelerometers) are easier to implement.
  As a compromise, low-mass accelerometers are often used for tool-point FRF testing:
  for most tools, a few grams or less does not appreciably alter the response, and
  the accelerometer can be attached with wax and removed without damaging the tool.

Accelerometer output and FRF form:
  Accelerometers produce a signal proportional to acceleration, so the measured FRF
  is inertance (A/F). For harmonic motion x = X·e^(iωt), ẍ = −ω²·X·e^(iωt), so
  (1/ω²)·(X/F) = (A/F), i.e. receptance = inertance / ω² (Eq. 2.70). This is
  equivalent to double integration in the frequency domain.

Time delay and phase error:
  In impact testing, the transducer’s amplifying electronics (e.g. analog low-pass
  filters) can introduce a time delay between the physical motion and the output
  voltage. The DAQ may also introduce synchronization errors between force and
  response. A constant time delay yields a phase error that increases linearly with
  frequency: Δφ(f) = S·f (e.g. S in deg/Hz). The measured phase is corrected as
  φ_c(f) = φ_m − Δφ = φ_m − S·f (Eq. 2.72). Given the corrected phase, the real and
  imaginary parts of the FRF are then updated from the magnitude and φ_c.
"""
from __future__ import annotations

from typing import Union

import numpy as np

# Default vibration sensor used in this project (low-mass accelerometer for tap testing).
DEFAULT_ACCELEROMETER = "ADXL345"


def inertance_to_receptance(
    omega_rad_s: Union[float, np.ndarray],
    H_inertance: Union[complex, np.ndarray],
) -> Union[complex, np.ndarray]:
    """
    Convert inertance (A/F) to receptance (X/F) at given frequency (or frequencies).

    For harmonic motion, ẍ = −ω²·x, so X/F = (A/F) / ω² (Eq. 2.70). At ω = 0 the
    formula is singular; callers should avoid ω = 0 or handle separately.

    Args:
        omega_rad_s: Angular frequency (rad/s), scalar or array.
        H_inertance: Complex inertance (acceleration/force), same shape as omega.

    Returns:
        Complex receptance (displacement/force) = H_inertance / ω².
    """
    omega = np.asarray(omega_rad_s, dtype=float)
    H = np.asarray(H_inertance, dtype=complex)
    if np.any(np.abs(omega) < 1e-20):
        raise ValueError("omega must be non-zero for inertance-to-receptance conversion")
    return H / (omega * omega)


def receptance_to_inertance(
    omega_rad_s: Union[float, np.ndarray],
    H_receptance: Union[complex, np.ndarray],
) -> Union[complex, np.ndarray]:
    """
    Convert receptance (X/F) to inertance (A/F): A/F = ω²·(X/F).
    """
    omega = np.asarray(omega_rad_s, dtype=float)
    H = np.asarray(H_receptance, dtype=complex)
    return H * (omega * omega)


def phase_error_from_time_delay(
    time_delay_s: float,
    freq_hz: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Phase error (deg) due to a constant time delay: Δφ = −360·f·Δt (linear in f).

    A delay Δt shifts the measured signal by Δφ = −2π·f·Δt radians = −360·f·Δt degrees
    (phase lag). So the slope S in deg/Hz is S = −360·Δt (with Δt in seconds).

    Args:
        time_delay_s: Time delay in seconds (e.g. 50e-6 for 50 μs).
        freq_hz: Frequency in Hz, scalar or array.

    Returns:
        Phase error in degrees (negative = lag).
    """
    f = np.asarray(freq_hz, dtype=float)
    return -360.0 * f * time_delay_s


def slope_deg_per_hz_from_time_delay(time_delay_s: float) -> float:
    """
    Slope S (deg/Hz) of the linear phase error vs frequency for a constant time delay.

    Δφ(f) = S·f with S = −360·Δt when Δt is in seconds. E.g. 50 μs → −18 deg/kHz
    = −0.018 deg/Hz.

    Returns:
        S in deg/Hz.
    """
    return -360.0 * time_delay_s


def correct_phase_linear(
    freq_hz: Union[float, np.ndarray],
    phase_measured_deg: Union[float, np.ndarray],
    slope_deg_per_hz: float,
) -> Union[float, np.ndarray]:
    """
    Correct measured phase for linear time-delay error: φ_c(f) = φ_m − S·f (Eq. 2.72).

    Args:
        freq_hz: Frequency in Hz.
        phase_measured_deg: Measured phase in degrees.
        slope_deg_per_hz: Slope S of phase error vs frequency (deg/Hz), e.g. from
            slope_deg_per_hz_from_time_delay(delay_s) or from calibration (phase
            difference between transducers vs frequency).

    Returns:
        Corrected phase in degrees.
    """
    f = np.asarray(freq_hz, dtype=float)
    phi_m = np.asarray(phase_measured_deg, dtype=float)
    return phi_m - slope_deg_per_hz * f


def correct_frf_real_imag_from_phase_slope(
    magnitude: Union[float, np.ndarray],
    phase_measured_deg: Union[float, np.ndarray],
    freq_hz: Union[float, np.ndarray],
    slope_deg_per_hz: float,
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Correct FRF real and imaginary parts for linear phase error (time delay).

    Corrected phase φ_c = φ_m − S·f. Then Re_c = |H|·cos(φ_c), Im_c = |H|·sin(φ_c)
    (magnitude unchanged; phase corrected). Returns (Re_corrected, Im_corrected).

    Args:
        magnitude: |H(f)| (unchanged by phase correction).
        phase_measured_deg: Measured phase in degrees.
        freq_hz: Frequency in Hz.
        slope_deg_per_hz: S (deg/Hz) from time delay or calibration.

    Returns:
        (Re_corrected, Im_corrected) in same units as magnitude.
    """
    mag = np.asarray(magnitude, dtype=float)
    phi_m = np.asarray(phase_measured_deg, dtype=float)
    f = np.asarray(freq_hz, dtype=float)
    phi_c_deg = correct_phase_linear(f, phi_m, slope_deg_per_hz)
    phi_c_rad = np.deg2rad(phi_c_deg)
    Re_c = mag * np.cos(phi_c_rad)
    Im_c = mag * np.sin(phi_c_rad)
    return (Re_c, Im_c)


# Nutshell text for docs or UI (re-exported from tap_testing.docs.transducers)
from tap_testing.docs.transducers import (
    inertance_receptance_nutshell,
    phase_error_time_delay_nutshell,
    transducers_nutshell,
)
