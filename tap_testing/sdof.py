"""
SDOF vibration theory: free, forced, and self-excited vibration.

─── Free vibration ─────────────────────────────────────────────────────────
  Natural frequency f_n = (1/(2π))√(k/m). High stiffness or low mass → high f_n.
  The tap test excites the tool–holder–spindle structure; the dominant frequency
  in the response is (approximately) the first natural frequency.

─── Forced vibration ───────────────────────────────────────────────────────
  A continuous external periodic excitation produces a response at the same
  frequency (after initial transients decay). Analyzed in the frequency domain:
  magnitude and phase depend on frequency, which makes natural frequencies easy
  to identify. Large vibrations occur when the forcing frequency ω is near the
  system natural frequency ω_n — this is resonance, and is generally avoided.
  In machining, tooth-passing is a periodic forcing; we avoid spindle RPMs where
  tooth-passing frequency (or a harmonic) equals the natural frequency.

─── Frequency response (magnitude and phase) ────────────────────────────────
  The magnitude (motion size vs force) and phase (time delay between force
  peak and displacement peak) depend on excitation frequency and natural
  frequency. Low excitation freq: motion small, nearly in phase (0°). As
  frequency increases: motion grows; displacement lags force. Near ω_n: motion
  largest; displacement is zero when force is maximum (90° out of phase). Above
  ω_n: motion shrinks. Very high excitation: motion very small; displacement
  max when force is min (180° out of phase).

─── Self-excited vibration ────────────────────────────────────────────────
  A steady input force is modulated into vibration at or near a natural frequency.
  Examples: violin, aeroelastic flutter, acoustic feedback, and chatter in machining.
  Chatter is self-excited: the cutting force is modulated so that the tool vibrates
  at a natural frequency. Avoiding forced resonance (tooth-pass = f_n) reduces the
  risk of exciting chatter as well.

─── Display scale (linear vs logarithmic) ───────────────────────────────────
  Log scale is used when data spans a broad range in one graph (e.g. acoustics:
  whisper to jet engine). The math of acoustics and mechanical vibrations is
  similar, so many texts use log scales from acoustics for vibrations too. For
  mechanical vibrations the signal range is often not so large, and linear
  scale is more intuitive for most people; we use linear scale almost exclusively.

Formulas (SI):
  ω_n = sqrt(k/m) [rad/s],  f_n = ω_n/(2π) [Hz],  k = m*(2π*f_n)²,  m = k/(2π*f_n)².
"""
from __future__ import annotations


def natural_frequency_hz(stiffness_n_per_m: float, mass_kg: float) -> float:
    """
    Natural frequency in Hz for a SDOF system: f_n = (1/(2π)) * sqrt(k/m).

    Args:
        stiffness_n_per_m: Stiffness k in N/m.
        mass_kg: Mass m in kg.

    Returns:
        Natural frequency in Hz.
    """
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if stiffness_n_per_m <= 0:
        raise ValueError("stiffness_n_per_m must be positive")
    import math
    return (1.0 / (2.0 * math.pi)) * math.sqrt(stiffness_n_per_m / mass_kg)


def stiffness_from_natural_frequency(
    natural_freq_hz: float,
    mass_kg: float,
) -> float:
    """
    Effective stiffness k (N/m) given measured natural frequency and assumed mass.

    k = m * (2π * f_n)². Use when you know (or estimate) the effective mass of the
    tool–holder–spindle system and have measured f_n from a tap test.

    Args:
        natural_freq_hz: Measured natural frequency in Hz (e.g. from FFT peak).
        mass_kg: Effective mass in kg.

    Returns:
        Stiffness in N/m.
    """
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if natural_freq_hz <= 0:
        raise ValueError("natural_freq_hz must be positive")
    import math
    omega = 2.0 * math.pi * natural_freq_hz
    return mass_kg * (omega * omega)


def mass_from_natural_frequency(
    natural_freq_hz: float,
    stiffness_n_per_m: float,
) -> float:
    """
    Effective mass m (kg) given measured natural frequency and assumed stiffness.

    m = k / (2π * f_n)². Use when you have a stiffness estimate and measured f_n.

    Args:
        natural_freq_hz: Measured natural frequency in Hz.
        stiffness_n_per_m: Stiffness in N/m.

    Returns:
        Effective mass in kg.
    """
    if stiffness_n_per_m <= 0:
        raise ValueError("stiffness_n_per_m must be positive")
    if natural_freq_hz <= 0:
        raise ValueError("natural_freq_hz must be positive")
    import math
    omega = 2.0 * math.pi * natural_freq_hz
    return stiffness_n_per_m / (omega * omega)


# Nutshell/summary text for docs or UI (re-exported from tap_testing.docs.sdof)
from tap_testing.docs.sdof import (
    forced_vibration_summary_nutshell,
    frequency_response_magnitude_phase_nutshell,
    linear_scale_display_nutshell,
    sdof_summary_nutshell,
    self_excited_vibration_summary_nutshell,
    vibration_theory_summary,
)


def is_near_resonance(
    forcing_freq_hz: float,
    natural_freq_hz: float,
    tolerance_hz: float = 5.0,
) -> bool:
    """
    True if forcing frequency is near the natural frequency (resonance condition).

    Resonance: ω ≈ ω_n. In Hz, |f_forcing - f_n| within tolerance indicates
    risk of large forced response.

    Args:
        forcing_freq_hz: Excitation frequency in Hz (e.g. tooth-passing frequency).
        natural_freq_hz: System natural frequency in Hz (e.g. from tap test).
        tolerance_hz: Band (Hz) around f_n to count as "near resonance".

    Returns:
        True if |forcing_freq_hz - natural_freq_hz| <= tolerance_hz.
    """
    return abs(forcing_freq_hz - natural_freq_hz) <= tolerance_hz
