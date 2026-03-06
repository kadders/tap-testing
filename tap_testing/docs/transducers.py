"""
Vibration transducers and FRF form: nutshell summaries for documentation or UI.

Inertance vs receptance, time-delay phase error. See tap_testing.transducers for
computation (inertance_to_receptance, phase_error_from_time_delay, correct_phase_linear, etc.).
"""
from __future__ import annotations

__all__ = [
    "transducers_nutshell",
    "inertance_receptance_nutshell",
    "phase_error_time_delay_nutshell",
]


def transducers_nutshell() -> str:
    """Short summary of transducer choice for vibration measurement (impact testing)."""
    return (
        "Vibration transducers: noncontact (e.g. capacitance, laser) do not influence dynamics; "
        "contacting (e.g. accelerometers) are more convenient. We use the ADXL345 (low-mass "
        "accelerometer) for tool-point tap testing—a few grams or less often does not appreciably "
        "alter the response; attach with wax and remove without damaging the tool."
    )


def inertance_receptance_nutshell() -> str:
    """Short summary of inertance vs receptance and conversion (Eq. 2.70)."""
    return (
        "Accelerometers give inertance (A/F). Receptance (X/F) = inertance / ω² "
        "(double integration in the frequency domain); for harmonic motion ẍ = −ω²·x."
    )


def phase_error_time_delay_nutshell() -> str:
    """Short summary of time-delay phase error and correction (Eqs. 2.71–2.72)."""
    return (
        "Amplifier and DAQ can introduce a constant time delay → phase error linear in frequency. "
        "Corrected phase: φ_c(f) = φ_m − S·f, where S (deg/Hz) is from the time delay or from "
        "calibration (e.g. phase difference between transducers). Then correct real and imaginary "
        "parts from magnitude and φ_c."
    )
