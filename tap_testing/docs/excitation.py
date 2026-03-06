"""
Force excitation types for FRF measurement and how they apply to tap testing.

Common force inputs (Section 2.6.1):
  • Fixed-frequency sine (sine-sweep): complex response one frequency at a time,
    with averaging at each frequency over a short time interval.
  • Random: broadband (white noise) or truncated (pink noise); averaging over a fixed
    period.
  • Impulse (impact): short-duration impact excites a broad range of frequencies in
    one short test. Multiple impacts are typically averaged in the frequency domain
    to improve coherence between force and response.

In a Nutshell: hitting the structure with a hammer excites many frequencies at almost
the same level at the same time.

Hardware:
  • Shaker: harmonically driven armature (electrodynamic or hydraulic), stinger, load
    cell (adds mass—can alter FRF on low-mass structures), isolation from structure.
  • Impact hammer: force transducer in the tip; short setup, popular for tool-holder
    testing (impact testing). Input energy ∝ hammer mass; bandwidth depends on mass
    and tip stiffness. Stiffer tips → wider bandwidth, energy spread; softer tips →
    lower range, concentrated. Plastic tips common for tool testing (no damage to
    cutting edge, sufficient bandwidth).

Our tap test: we default to impulse excitation (hammer tap). We record the acceleration
response only (no force transducer in the basic setup), so we obtain the free decay /
response spectrum and identify the dominant natural frequency from the FFT. With a
force sensor at the tap point, we could compute the FRF (response/force) and apply
full modal fitting. Multiple taps are averaged (e.g. run_cycle) to improve the
estimate of the natural frequency.
"""
from __future__ import annotations

from tap_testing.config import DEFAULT_EXCITATION_TYPE

__all__ = [
    "DEFAULT_EXCITATION_TYPE",
    "excitation_types_nutshell",
    "impact_excitation_nutshell",
    "impact_hammer_hardware_nutshell",
    "tap_test_excitation_nutshell",
    "force_input_summary",
]


def excitation_types_nutshell() -> str:
    """
    Short summary of the three common force excitation types for FRF measurement.
    """
    return (
        "Force excitation types: (1) Sine-sweep—one frequency at a time, averaging at each. "
        "(2) Random—broadband or band-limited noise, averaging over time. "
        "(3) Impulse (impact)—hammer tap excites many frequencies at once; multiple "
        "impacts averaged in frequency domain for coherence."
    )


def impact_excitation_nutshell() -> str:
    """
    Short summary of impact excitation: hammer excites many frequencies at once.
    """
    return (
        "Impact excitation: hitting the structure with a hammer excites many frequencies "
        "at almost the same level at the same time. Multiple tests are averaged in the "
        "frequency domain to improve coherence between force and response."
    )


def impact_hammer_hardware_nutshell() -> str:
    """
    Short summary of impact hammer hardware and tip choice for tool-holder testing.
    """
    return (
        "Impact hammer: force transducer in tip; short setup, popular for tool-holder testing. "
        "Energy ∝ hammer mass; bandwidth depends on mass and tip stiffness. Stiffer tips → "
        "wider bandwidth (energy spread); softer tips → lower range, concentrated. "
        "Plastic tips common for tool testing: no damage to cutting edge, sufficient bandwidth."
    )


def tap_test_excitation_nutshell() -> str:
    """
    How the tap-test workflow relates to impact excitation and what we measure.
    We default to impulse excitation.
    """
    return (
        "Tap test defaults to impulse excitation (hammer tap). We record acceleration response only "
        "(no force sensor in basic setup), so we get free decay / response spectrum and identify "
        "dominant natural frequency from FFT. Multiple taps (e.g. run_cycle) are averaged to improve "
        "the estimate. With a force sensor at the tap point, we could compute the FRF and apply "
        "full modal fitting."
    )


def force_input_summary() -> dict[str, str]:
    """
    All excitation-related nutshells for docs or UI.

    Returns:
        Dict with keys: "types", "impact", "impact_hammer", "tap_test".
    """
    return {
        "types": excitation_types_nutshell(),
        "impact": impact_excitation_nutshell(),
        "impact_hammer": impact_hammer_hardware_nutshell(),
        "tap_test": tap_test_excitation_nutshell(),
    }
