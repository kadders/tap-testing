"""
SDOF vibration theory: nutshell summaries for documentation or UI.

Free, forced, and self-excited vibration; FRF magnitude/phase; display scale.
See tap_testing.sdof for computation (natural_frequency_hz, stiffness_from_natural_frequency, etc.).
"""
from __future__ import annotations

__all__ = [
    "sdof_summary_nutshell",
    "forced_vibration_summary_nutshell",
    "frequency_response_magnitude_phase_nutshell",
    "self_excited_vibration_summary_nutshell",
    "linear_scale_display_nutshell",
    "vibration_theory_summary",
]


def sdof_summary_nutshell() -> str:
    """Short high-level summary of SDOF free vibration for documentation or UI."""
    return (
        "SDOF free vibration: natural frequency f_n = (1/(2π))·√(k/m). "
        "High stiffness or low mass → high f_n; low stiffness or high mass → low f_n. "
        "The tap test excites the structure; the dominant frequency in the response "
        "is the first natural frequency."
    )


def forced_vibration_summary_nutshell() -> str:
    """
    Short summary of forced vibration and resonance for documentation or UI.

    Forced vibration: external periodic excitation → response at same frequency
    (after transients). Resonance when forcing frequency ω ≈ natural frequency ω_n;
    large vibrations then, generally avoided. Tooth-passing in milling is a periodic
    forcing; we avoid RPMs where tooth-pass frequency (or harmonics) equals f_n.
    """
    return (
        "Forced vibration: periodic excitation produces response at the same frequency. "
        "Resonance when forcing frequency ≈ natural frequency (ω ≈ ω_n); large vibrations, "
        "generally avoided. In milling, tooth-passing is the forcing; we avoid spindle RPMs "
        "where tooth-pass frequency or its harmonics match the measured natural frequency."
    )


def frequency_response_magnitude_phase_nutshell() -> str:
    """
    Magnitude and phase of the frequency response function (FRF) for forced vibration.

    (1) Response occurs at the frequency of the exciting force. (2) Magnitude (motion
    size vs force) and phase (delay between force peak and displacement peak) depend
    on excitation frequency and natural frequency. Low excitation: motion small, nearly
    in phase (0°). Increasing frequency: motion larger, displacement lags force. Near
    natural frequency: displacement as large as possible; displacement zero when force
    is maximum (90° out of phase). Further increase: motion reduced. Very high
    excitation: motion very small; displacement maximum when force is minimum (180°
    out of phase).
    """
    return (
        "FRF magnitude and phase: (1) response at the excitation frequency; (2) magnitude "
        "(motion/force) and phase (delay force peak → displacement peak) depend on excitation "
        "and natural frequency. Low ω: small motion, ~0° phase. Near ω_n: motion largest, 90° "
        "out of phase (displacement zero when force max). High ω: small motion, 180° out of phase."
    )


def self_excited_vibration_summary_nutshell() -> str:
    """
    Short summary of self-excited vibration and chatter for documentation or UI.

    Self-excited: steady input is modulated into vibration at/near a natural frequency.
    Chatter in machining is self-excited vibration; avoiding forced resonance helps
    reduce the risk of exciting chatter.
    """
    return (
        "Self-excited vibration: steady input is modulated into vibration at or near "
        "a natural frequency (e.g. violin, flutter, chatter). Chatter in machining is "
        "self-excited; avoiding spindle speeds where tooth-passing matches the natural "
        "frequency reduces the risk of exciting chatter."
    )


def linear_scale_display_nutshell() -> str:
    """
    Why we use linear scale for displaying vibration signals (not log).

    Log scale suits data spanning a huge range (e.g. sound pressure: whisper to
    jet engine). Acoustics and mechanical vibrations share similar math, so
    textbooks often use log scales from acoustics for vibrations. For mechanical
    vibrations the signal range is typically not so large; linear scale is more
    intuitive, so we use it almost exclusively.
    """
    return (
        "Display scale: log scale suits a broad range in one graph (e.g. acoustics). "
        "Mechanical vibrations use similar math but often a smaller signal range; "
        "linear scale is more intuitive, so we use linear scale almost exclusively."
    )


def vibration_theory_summary() -> dict[str, str]:
    """
    Nutshell summaries for free, forced, self-excited, FRF magnitude/phase, and display scale.

    Returns:
        Dict with keys: "free", "forced", "self_excited", "frf_magnitude_phase", "display_scale".
    """
    return {
        "free": sdof_summary_nutshell(),
        "forced": forced_vibration_summary_nutshell(),
        "self_excited": self_excited_vibration_summary_nutshell(),
        "frf_magnitude_phase": frequency_response_magnitude_phase_nutshell(),
        "display_scale": linear_scale_display_nutshell(),
    }
