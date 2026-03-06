"""
Feeds and speeds tool optimization (practical CNC).

Formulas follow practical feeds & speeds guides (e.g. Shapeoko CNC A–Z:
https://shapeokoenthusiasts.gitbook.io/shapeoko-cnc-a-to-z/feeds-and-speeds-basics).
All units are metric (mm, mm/min, mm/tooth, mm³/min, m/s, W, N, N·m).

Use with tap-test stability results: pick a stable RPM and chipload, then
adjust for stepover (chip thinning), and check MRR / power / force if needed.
"""
from __future__ import annotations

import math

# Default spindle power (W) for overload checks. 1.5 kW is a common small CNC spindle.
DEFAULT_SPINDLE_POWER_W = 1500.0

# Minimum effective chipload (mm/tooth) for acceptable chip evacuation; below this, risk of rubbing
# and heat (poor chip formation). ~0.001" ≈ 0.0254 mm per Shapeoko/CNC guidelines; use 0.025.
MIN_CHIPLOAD_MM_PER_TOOTH = 0.025

# Unit power (W·min/mm³) = 1 / K where K is mm³/(min·W). From US K (in³/min/HP):
# K_mm3_per_W = K_in3_per_HP * (16387/745.7) ≈ K_in3_per_HP * 22; unit_power = 1/K_mm3_per_W.
# 6061-T6: K ≈ 3.34 in³/(min·HP) → ~73.5 mm³/(min·W); hard woods/plastics ~10; soft woods ~30.
UNIT_POWER_W_MIN_PER_MM3: dict[str, float] = {
    "6061 aluminum": 1.0 / 73.5,
    "7075 aluminum": 1.0 / 65.0,
    "A36 steel": 1.0 / 25.0,
    "hard wood": 1.0 / 100.0,
    "soft wood": 1.0 / 300.0,
}

__all__ = [
    "chipload_adjusted_for_stepover_mm",
    "chipload_from_feed_mm_per_tooth",
    "cutting_force_n",
    "cutting_power_w",
    "cutting_torque_nm",
    "cutting_zone_status",
    "DEFAULT_SPINDLE_POWER_W",
    "feed_rate_for_chipload_mm_min",
    "MIN_CHIPLOAD_MM_PER_TOOTH",
    "mrr_mm3_per_min",
    "rpm_from_sfm",
    "sfm_from_rpm",
    "tool_engagement_angle_deg",
    "UNIT_POWER_W_MIN_PER_MM3",
]


def cutting_zone_status(
    chipload_target_mm: float,
    power_w: float | None,
    spindle_power_w: float = DEFAULT_SPINDLE_POWER_W,
    min_chipload_mm: float = MIN_CHIPLOAD_MM_PER_TOOTH,
) -> dict:
    """
    Classify cutting conditions into good/bad zones for chip evacuation and spindle load.

    Chip evacuation: target effective chipload should be >= min_chipload_mm to avoid rubbing
    and poor chip formation. Spindle load: cutting power should not exceed spindle_power_w.

    Args:
        chipload_target_mm: Target effective chipload (mm/tooth) for the cut.
        power_w: Cutting power at tool (W), or None if MRR/material unknown.
        spindle_power_w: Spindle power rating (W); default 1.5 kW.
        min_chipload_mm: Minimum chipload for acceptable chip evacuation; default 0.025 mm (~0.001").

    Returns:
        Dict with: chip_evac_ok (bool), spindle_ok (bool), power_w (float | None),
        label (str) e.g. "OK", "Low fz", "Overload", "Low fz + Overload", "Low fz (power n/a)".
    """
    chip_evac_ok = chipload_target_mm >= min_chipload_mm
    if power_w is None or spindle_power_w <= 0:
        spindle_ok = True  # unknown power → assume OK for display
        if chip_evac_ok:
            label = "OK"
        else:
            label = "Low fz (power n/a)"
    else:
        spindle_ok = power_w <= spindle_power_w
        if chip_evac_ok and spindle_ok:
            label = "OK"
        elif not chip_evac_ok and spindle_ok:
            label = "Low fz"
        elif chip_evac_ok and not spindle_ok:
            label = "Overload"
        else:
            label = "Low fz + Overload"
    return {
        "chip_evac_ok": chip_evac_ok,
        "spindle_ok": spindle_ok,
        "power_w": power_w,
        "label": label,
    }


def chipload_adjusted_for_stepover_mm(
    diameter_mm: float,
    stepover_mm: float,
    chipload_nominal_mm: float,
) -> float:
    """
    Target chipload to use when stepover < 50% so effective chip thickness is correct (chip thinning).

    When radial engagement (stepover) is less than half the diameter, the actual
    chip thickness is smaller than the nominal chipload. To get a desired
    effective chip thickness, increase the target (programmed) chipload by the
    chip-thinning factor:

        chipload_adjusted = (D / (2 * sqrt(D×S - S²))) × chipload_nominal

    with D = diameter, S = stepover (WOC), all in mm. For stepover ≥ 50% the
    factor is 1 (no adjustment). Reference: Shapeoko feeds & speeds (chip thinning).

    Args:
        diameter_mm: Tool diameter (mm).
        stepover_mm: Width of cut / radial depth (mm). Can be computed as
            diameter_mm * (stepover_percent / 100).
        chipload_nominal_mm: Desired effective chipload (mm/tooth).

    Returns:
        Adjusted chipload (mm/tooth) to use in feed = chipload × n_teeth × RPM.
    """
    if diameter_mm <= 0 or chipload_nominal_mm <= 0:
        return chipload_nominal_mm
    stepover_mm = max(0.0, min(stepover_mm, diameter_mm))
    # At or above 50% engagement, no chip thinning
    if stepover_mm >= 0.5 * diameter_mm:
        return chipload_nominal_mm
    inner = diameter_mm * stepover_mm - stepover_mm * stepover_mm
    if inner <= 0:
        return chipload_nominal_mm
    factor = diameter_mm / (2.0 * math.sqrt(inner))
    return factor * chipload_nominal_mm


def feed_rate_for_chipload_mm_min(
    chipload_mm: float,
    n_teeth: int,
    rpm: float,
) -> float:
    """
    Feed rate (mm/min) to achieve a given chipload: F = fz × Nt × RPM.
    """
    if n_teeth <= 0 or rpm <= 0:
        return 0.0
    return chipload_mm * n_teeth * rpm


def chipload_from_feed_mm_per_tooth(
    feed_mm_min: float,
    n_teeth: int,
    rpm: float,
) -> float:
    """
    Chipload (mm/tooth) from feed rate: fz = F / (Nt × RPM).
    """
    if n_teeth <= 0 or rpm <= 0:
        return 0.0
    return feed_mm_min / (n_teeth * rpm)


def tool_engagement_angle_deg(diameter_mm: float, stepover_mm: float) -> float:
    """
    Tool engagement angle TEA (degrees) from stepover (width of cut).

    TEA = cos⁻¹(1 - stepover / (0.5 × diameter)). For 50% stepover (0.5×D),
    TEA = 90°. For slotting (stepover = diameter), TEA = 180°. Higher TEA
    means higher cutting forces; slotting is hardest.

    Args:
        diameter_mm: Tool diameter (mm).
        stepover_mm: Width of cut / radial depth (mm).

    Returns:
        TEA in degrees [0, 180].
    """
    if diameter_mm <= 0:
        return 0.0
    half_d = 0.5 * diameter_mm
    stepover_mm = max(0.0, min(stepover_mm, diameter_mm))
    x = 1.0 - stepover_mm / half_d
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))


def mrr_mm3_per_min(
    width_of_cut_mm: float,
    depth_of_cut_mm: float,
    feed_mm_min: float,
) -> float:
    """
    Material removal rate in mm³/min: MRR = WOC × DOC × Feed.

    Used to compare cutting scenarios and to estimate required power.

    Args:
        width_of_cut_mm: Radial depth / stepover (mm).
        depth_of_cut_mm: Axial depth per pass (mm).
        feed_mm_min: Feed rate (mm/min).

    Returns:
        MRR in mm³/min.
    """
    return width_of_cut_mm * depth_of_cut_mm * feed_mm_min


def sfm_from_rpm(rpm: float, diameter_mm: float) -> float:
    """
    Surface feet per minute from RPM and tool diameter (metric inputs).

    SFM = π × D_in × RPM / 12. With D_mm: D_in = D_mm / 25.4, so
    SFM = π × (D_mm / 25.4) × RPM / 12 = π × D_mm × RPM / (12 × 25.4).

    Args:
        rpm: Spindle speed (rpm).
        diameter_mm: Tool diameter (mm).

    Returns:
        Cutting speed at tool periphery in ft/min (SFM).
    """
    if rpm <= 0 or diameter_mm <= 0:
        return 0.0
    diameter_in = diameter_mm / 25.4
    return math.pi * diameter_in * rpm / 12.0


def rpm_from_sfm(sfm_ft_per_min: float, diameter_mm: float) -> float:
    """
    RPM to achieve a given surface speed (SFM): RPM = SFM / (0.262 × D_in).

    Args:
        sfm_ft_per_min: Surface feet per minute (cutting speed at periphery).
        diameter_mm: Tool diameter (mm).

    Returns:
        Spindle speed (rpm).
    """
    if sfm_ft_per_min <= 0 or diameter_mm <= 0:
        return 0.0
    diameter_in = diameter_mm / 25.4
    return sfm_ft_per_min / (0.262 * diameter_in)


def cutting_power_w(
    mrr_mm3_per_min: float,
    unit_power_w_min_per_mm3: float,
) -> float:
    """
    Cutting power at the tool (W) from MRR and material unit power.

    Power = MRR × UnitPower. Unit power is material-dependent (in³/min per HP
    in US sources; convert: 1 HP ≈ 745.7 W, 1 in³ ≈ 16387 mm³, so
    K_mm3_per_W ≈ K_in3_per_HP × 16387 / 745.7). Then UnitPower (W·min/mm³)
    = 1 / K_mm3_per_W. So Power_W = MRR_mm3_min × UnitPower.

    Alternatively pass unit_power as (1/K) where K is mm³/(min·W) capacity:
    Power_W = MRR / K.

    Args:
        mrr_mm3_per_min: Material removal rate (mm³/min).
        unit_power_w_min_per_mm3: Unit power in W·min/mm³ (power per volume rate).
            Example: 6061-T6 ≈ 1/73.5 ≈ 0.0136 W·min/mm³ (K ≈ 73.5 mm³/(min·W)).

    Returns:
        Power at cutter in watts.
    """
    if unit_power_w_min_per_mm3 <= 0:
        return 0.0
    return mrr_mm3_per_min * unit_power_w_min_per_mm3


def cutting_torque_nm(power_w: float, rpm: float) -> float:
    """
    Torque at spindle (N·m) from cutting power and RPM.

    Power = Torque × ω, ω = 2π × RPM/60 rad/s, so Torque = Power / ω.

    Args:
        power_w: Cutting power (W).
        rpm: Spindle speed (rpm).

    Returns:
        Torque in N·m.
    """
    if rpm <= 0:
        return 0.0
    omega_rad_s = 2.0 * math.pi * rpm / 60.0
    return power_w / omega_rad_s


def cutting_force_n(torque_nm: float, radius_mm: float) -> float:
    """
    Tangential cutting force (N) from torque and tool radius.

    Force = Torque / radius (radius in m for SI: F = T/r_m).

    Args:
        torque_nm: Torque at spindle (N·m).
        radius_mm: Tool radius (mm); converted to m internally.

    Returns:
        Force in N.
    """
    if radius_mm <= 0:
        return 0.0
    return torque_nm / (radius_mm / 1000.0)
