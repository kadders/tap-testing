"""
Milling dynamics: how cutting conditions differ from turning and why engagement is complex.

Units: All formulas in this module use metric (mm, mm/min, mm/tooth, N/mm², m/s). Textbook
references (Sect. 4.x, Eq. 4.14, etc.) may appear in imperial in the original source (e.g.
Machining Dynamics.pdf); here they are implemented in SI/metric. See
docs/METRIC_FORMULAS_REFERENCE.md for equation-by-equation mapping and conversions.

In turning, chip thickness and chip width are fixed. In milling they are not:
  • Straight slotting: chip thickness varies continuously as each tooth enters and
    exits the cut.
  • Pocket milling: radial depth of cut may also change.
  • Sculptured surface milling: axial depth of cut may vary as well.

Start and exit angles (up vs down milling):
  • Up milling: start angle φs = 0; exit angle φe = cos⁻¹((r - a) / r) where r is
    cutter radius and a is radial depth of cut. A 10% radial immersion cut (e.g.
    a = 1.9 mm, diameter 19 mm) gives φe ≈ 37°.
  • Down milling: φs = 180° - cos⁻¹((r - a) / r), φe = 180°. For 25% radial
    immersion (a = r/2), φs = 120° and φe = 180°.

Example 4.2 (25% radial immersion): a = r/2 gives φe = 60° (up) or φs = 120°
(down). With four teeth at 90° intervals, each tooth is engaged for 60°; force is
zero for the remaining 30° between teeth. Up milling: force grows as φ increases
(chip thickness increases); down milling: maximum force at entry then decreases
as the final surface is created (often preferred for finishing). Resultant
F = √(Fx² + Fy²). Typical parameters (e.g. aluminum): kt = 750 N/mm², kn = 250 N/mm²,
b = 5 mm, ft = 0.1 mm/tooth.

Cutting force (rigid tool/workpiece, geometry only): Unlike turning, the cutting
force is not constant; it is a function of cutter angle φ. Total force F = Ks·A =
Ks·b·h (chip area × specific force). Normal and tangential components in the
rotating frame: Fn = kn·b·h, Ft = kt·b·h. For measurement in a fixed frame (e.g.
dynamometer x, y), project using φ: Fx = Ft·cos(φ) + Fn·sin(φ), Fy = Ft·sin(φ) -
Fn·cos(φ). With instantaneous chip thickness h = ft·sin(φ), the forces are periodic
in φ and depend on how many teeth are engaged and on the cut geometry.

Tooth passing frequency (Sect. 4.1.1): Convert cutter angle φ (deg) to time (s) by
t = φ · 60 / (Ω · 360) with Ω in rpm. The cutting force spectrum has content at
the fundamental tooth passing frequency f_tooth = Ω · Nt / 60 Hz and at integer
multiples (harmonics). Sharper force impulses produce stronger higher harmonics.
Our tap-test analysis uses the measured natural frequency to avoid spindle RPM at
which f_tooth or one of its harmonics equals that natural frequency (resonance).

Multiple teeth engaged: With more teeth or higher radial immersion, more than one
tooth can be in the cut at once, and the number can vary with angle (e.g. 75%
immersion, 4 teeth: up 0–90° and down 90–120°; two teeth (4 and 1) then one (1)
then two (1 and 2); force never drops to zero). The total force is the sum over
all teeth (Eqs. 4.15–4.16), with φj = angle of tooth j (tooth pitch = 360°/Nt);
only teeth with φs ≤ φj ≤ φe contribute. Slotting with an even number of teeth
Nt > 2 yields constant cutting force.

Stability lobe diagrams (Sect. 4.3): Tlusty’s average-tooth-angle approach makes
the system time-invariant; the limiting chip width is blim = 1/(2·Ks·Re[FRF_orient]·Nt*),
with Nt* = (φe − φs)/360 · Nt the average number of teeth in cut. The oriented FRF
is FRF_orient = μx·FRFx + μy·FRFy (directional factors μx, μy depend on radial immersion
and force angle β). Best spindle speeds (rpm) are Ω_best = fn·60/((N+1)·Nt) for
lobe N; the tap-test natural frequency fn can be used to target these speeds. Points
above the (Ω, blim) boundary are unstable; below, stable.

Time-domain simulation (Sect. 4.4): The “Regenerative Force, Dynamic Deflection”
model (Smith & Tlusty) gives local force and vibration for chosen conditions;
analytical lobes give a global stability picture. Chip thickness (regenerative):
h(t) = ft·sin(φ) + n(t−τ) − n(t), with tooth period τ = 60/(Ω·Nt) s (Ω in rpm).
Normal-direction vibration n = x·sin(φ) − y·cos(φ). The simulation discretizes
angle (dφ = 360/steps_rev, dt = 60/(steps_rev·Ω)); steps_rev/Nt must be an integer.
A surf vector stores the previous tooth’s normal at each angle; h = ft·sin(φ) + surf − n.
Experimental cutting force coefficients (Fig. 4.8): Tests use a known feed per tooth
and axial depth and measure the x (feed), y, and z (axial) cutting force components
in the dynamometer's fixed frame. Square endmill geometry is presumed. The force
model and corresponding coefficients (kt, kn; for ball endmills also ka) are revisited
before applying numerical and experimental techniques to obtain their values.
"""
from __future__ import annotations

import math

# Nutshell/summary text for docs or UI (re-exported from tap_testing.docs.milling_dynamics)
from tap_testing.docs.milling_dynamics import (
    milling_ball_endmill_simulation_nutshell,
    milling_chatter_to_stable_strategy_nutshell,
    milling_cutting_force_geometry_nutshell,
    milling_cutting_forces_example_4_2_nutshell,
    milling_dynamics_nutshell,
    milling_dynamics_summary,
    milling_example_6_2_competing_lobes_nutshell,
    milling_experimental_cutting_coefficients_nutshell,
    milling_experimental_techniques_nutshell,
    milling_helical_teeth_nutshell,
    milling_linear_regression_coefficients_nutshell,
    milling_multiple_teeth_engagement_nutshell,
    milling_nonlinear_optimization_coefficients_nutshell,
    milling_process_damping_nutshell,
    milling_process_damping_stability_nutshell,
    milling_simulation_loop_and_examples_nutshell,
    milling_stability_lobes_nutshell,
    milling_straight_vs_helical_forces_nutshell,
    milling_start_exit_angles_nutshell,
    milling_surf_update_rule_nutshell,
    milling_time_domain_simulation_nutshell,
    milling_updated_force_model_nutshell,
    tooth_passing_frequency_nutshell,
)

__all__ = [
    "angle_increment_deg",
    "angle_to_time_sec",
    "average_teeth_in_cut",
    "average_tooth_angle_deg",
    "ball_chip_width_db_mm",
    "ball_forces_xyz_N",
    "ball_kappa_prime_rad",
    "ball_slice_width_dz_mm",
    "ball_theta0_rad",
    "chip_thickness_ball_mm",
    "chip_thickness_regenerative_mm",
    "cutting_forces_xy_N",
    "axial_force_with_edge_N",
    "coefficient_of_determination_r2",
    "constant_force_axial_depth_mm",
    "cutting_coefficients_from_slotting_regression",
    "cutting_speed_m_per_s",
    "directional_factors_up_milling",
    "linear_regression_slope_intercept",
    "mean_force_per_rev_slotting_N",
    "exit_angle_up_milling_deg",
    "helical_lag_angle_deg",
    "milling_cutting_force_geometry_nutshell",
    "milling_cutting_forces_example_4_2_nutshell",
    "milling_dynamics_nutshell",
    "milling_dynamics_summary",
    "milling_ball_endmill_simulation_nutshell",
    "milling_chatter_to_stable_strategy_nutshell",
    "milling_example_6_2_competing_lobes_nutshell",
    "milling_experimental_cutting_coefficients_nutshell",
    "milling_updated_force_model_nutshell",
    "milling_linear_regression_coefficients_nutshell",
    "milling_nonlinear_optimization_coefficients_nutshell",
    "milling_experimental_techniques_nutshell",
    "milling_process_damping_nutshell",
    "milling_process_damping_stability_nutshell",
    "milling_helical_teeth_nutshell",
    "milling_straight_vs_helical_forces_nutshell",
    "milling_start_exit_angles_nutshell",
    "milling_multiple_teeth_engagement_nutshell",
    "milling_simulation_loop_and_examples_nutshell",
    "milling_stability_lobes_nutshell",
    "milling_time_domain_simulation_nutshell",
    "milling_surf_update_rule_nutshell",
    "normal_force_N",
    "normal_force_with_edge_N",
    "normal_velocity_from_xy_mm_s",
    "normal_surface_vibration_mm",
    "process_damping_cnew_x_up_milling",
    "process_damping_cnew_y_up_milling",
    "process_damping_force_N",
    "normal_force_with_process_damping_N",
    "radial_immersion_percent",
    "simulation_steps_rev",
    "simulation_time_step_s",
    "spindle_aligned_rpms",
    "stability_lobe_best_spindle_speed_rpm",
    "stability_lobe_phase_epsilon_rad",
    "start_angle_down_milling_deg",
    "tangential_force_N",
    "tangential_force_with_edge_N",
    "tooth_angle_deg",
    "tooth_period_s",
    "tooth_passing_frequency_hz",
    "tooth_passing_frequency_nutshell",
]


def radial_immersion_percent(radial_depth_mm: float, diameter_mm: float) -> float:
    """
    Radial immersion as a percentage of cutter diameter.

    Immersion = (radial depth of cut / diameter) × 100. Example: 1.9 mm radial
    depth with 19 mm diameter → 10% radial immersion. A full slot (radial depth
    = diameter) is 100% radial immersion.

    Args:
        radial_depth_mm: Radial depth of cut (mm).
        diameter_mm: Cutter diameter (mm).

    Returns:
        Percentage in [0, 100] (clamped if out of range).
    """
    if diameter_mm <= 0:
        return 0.0
    pct = 100.0 * radial_depth_mm / diameter_mm
    return max(0.0, min(100.0, pct))


def exit_angle_up_milling_deg(radius_mm: float, radial_depth_mm: float) -> float:
    """
    Exit angle φe (degrees) for up milling from geometry.

    φe = cos⁻¹((r - a) / r), where r = radius, a = radial depth of cut. For up
    milling the start angle is φs = 0. Example: r = 9.5 mm, a = 1.9 mm →
    (9.5 - 1.9) / 9.5 = 0.8 → φe = cos⁻¹(0.8) ≈ 37°.

    Args:
        radius_mm: Cutter radius (mm).
        radial_depth_mm: Radial depth of cut (mm).

    Returns:
        Exit angle in degrees [0, 180]. Returns 0 if r <= 0 or a >= r (no cut).
    """
    if radius_mm <= 0:
        return 0.0
    x = (radius_mm - radial_depth_mm) / radius_mm
    x = max(-1.0, min(1.0, x))  # clamp for acos
    return math.degrees(math.acos(x))


def start_angle_down_milling_deg(radius_mm: float, radial_depth_mm: float) -> float:
    """
    Start angle φs (degrees) for down milling from geometry.

    φs = 180° - cos⁻¹((r - a) / r), with φe = 180°. Example: 25% radial immersion
    (a = r/2) gives (r - a)/r = 0.5, so φs = 180° - 60° = 120°.

    Args:
        radius_mm: Cutter radius (mm).
        radial_depth_mm: Radial depth of cut (mm).

    Returns:
        Start angle in degrees [0, 180]. Returns 180 if r <= 0 or a >= r.
    """
    if radius_mm <= 0:
        return 180.0
    x = (radius_mm - radial_depth_mm) / radius_mm
    x = max(-1.0, min(1.0, x))
    return 180.0 - math.degrees(math.acos(x))


def tooth_angle_deg(cutter_angle_deg: float, tooth_index_1based: int, n_teeth: int) -> float:
    """
    Angle of the j-th tooth (deg) when the cutter (tooth 1) is at the given angle.

    Tooth pitch is 360° / Nt. With φ = angle of tooth 1, tooth j is at
    φ - (j - 1) * (360 / Nt). Example: 4 teeth, tooth 1 at 40° → tooth 2 at 310°,
    tooth 3 at 220°, tooth 4 at 130°. Result is normalized to [0, 360).

    Args:
        cutter_angle_deg: Angle of tooth 1 (cutter reference angle), deg.
        tooth_index_1based: Tooth index j = 1, 2, ..., Nt.
        n_teeth: Number of teeth.

    Returns:
        Angle of tooth j in [0, 360) deg, or 0 if n_teeth <= 0.
    """
    if n_teeth <= 0:
        return 0.0
    pitch = 360.0 / n_teeth
    phi_j = cutter_angle_deg - (tooth_index_1based - 1) * pitch
    while phi_j < 0:
        phi_j += 360.0
    while phi_j >= 360.0:
        phi_j -= 360.0
    return phi_j


def angle_to_time_sec(angle_deg: float, rpm: float) -> float:
    """
    Convert cutter angle (degrees) to time (seconds) at a given spindle speed.

    t = φ · 60 / (Ω · 360) = φ / (6·Ω), with φ in deg and Ω in rpm. Example:
    90° at 7500 rpm → 90 / (6 · 7500) = 0.002 s (tooth period for a four-tooth
    cutter, since 90° is the angular spacing between teeth).

    Args:
        angle_deg: Cutter angle in degrees (e.g. 90 for one tooth interval).
        rpm: Spindle speed in rpm.

    Returns:
        Time in seconds. Returns 0 if rpm <= 0.
    """
    if rpm <= 0:
        return 0.0
    return angle_deg * 60.0 / (rpm * 360.0)


def tooth_passing_frequency_hz(rpm: float, n_teeth: int) -> float:
    """
    Fundamental tooth passing frequency in Hz (Eq. 4.14).

    f_tooth = Ω · Nt / 60, with Ω in rpm. Example: 7500 rpm, 4 teeth →
    f_tooth = 7500 · 4 / 60 = 500 Hz. The cutting force spectrum has content
    at f_tooth and at integer multiples (2·f_tooth, 3·f_tooth, ...). Our tap-test
    analysis avoids spindle RPM where f_tooth or a harmonic equals the measured
    natural frequency (resonance).

    Args:
        rpm: Spindle speed in rpm.
        n_teeth: Number of teeth (flutes).

    Returns:
        Tooth passing frequency in Hz (≥ 0).
    """
    if rpm <= 0 or n_teeth <= 0:
        return 0.0
    return rpm * n_teeth / 60.0


def average_teeth_in_cut(phi_s_deg: float, phi_e_deg: float, n_teeth: int) -> float:
    """
    Average number of teeth in the cut, Nt* (Eq. 4.26).

    Nt* = (φe - φs) / 360 · Nt, with start and exit angles in degrees. Example:
    slotting (φs = 0, φe = 180°) with 4 teeth gives Nt* = (180/360)·4 = 2.

    Args:
        phi_s_deg: Start angle (deg), e.g. from up/down milling geometry.
        phi_e_deg: Exit angle (deg).
        n_teeth: Number of teeth.

    Returns:
        Nt* ≥ 0.
    """
    if n_teeth <= 0:
        return 0.0
    span = max(0.0, phi_e_deg - phi_s_deg)
    return (span / 360.0) * n_teeth


def average_tooth_angle_deg(phi_s_deg: float, phi_e_deg: float) -> float:
    """
    Average angle of a tooth in the cut (deg): φ_ave = (φs + φe) / 2.

    Used in the stability lobe approach for directional orientation factors and
    surface normal. Example: slotting φs = 0, φe = 180° → φ_ave = 90°.

    Args:
        phi_s_deg: Start angle (deg).
        phi_e_deg: Exit angle (deg).

    Returns:
        Average angle in degrees.
    """
    return 0.5 * (phi_s_deg + phi_e_deg)


def stability_lobe_best_spindle_speed_rpm(
    fn_hz: float,
    n_teeth: int,
    lobe_index_n: int = 0,
) -> float:
    """
    Best spindle speed (rpm) for a given stability lobe N (Eq. 4.29).

    Formula: Ω = fn · 60 / ((N + 1) · Nt).

    At this speed, tooth-passing frequency f_tooth = Ω·Nt/60 = fn/(N+1), so the
    (N+1)-th harmonic of f_tooth equals fn. The phase between successive teeth
    is favorable for stability (high limiting chip width). Cross-check:
    tooth_passing_frequency_hz(Ω, Nt) == fn/(N+1).

    For N = 0: Ω = fn·60/Nt (e.g. fn = 500 Hz, 4 teeth → 7500 rpm). Note: this
    is the same RPM as "avoid" for the first harmonic (k=1); at the lobe peak
    the regenerative phase allows stable cutting despite excitation at fn.

    When regulating out of chatter (Ex. 6.2): use the *detected chatter frequency*
    fc as fn_hz; the result is the new spindle speed to try.

    Args:
        fn_hz: Natural frequency in Hz (e.g. from tap test), or detected chatter frequency fc.
        n_teeth: Number of teeth.
        lobe_index_n: Lobe index N = 0, 1, 2, ... (integer waves between teeth).

    Returns:
        Spindle speed in rpm (≥ 0). Returns 0 if n_teeth <= 0 or N+1 <= 0.
    """
    if n_teeth <= 0 or lobe_index_n < 0:
        return 0.0
    denom = (lobe_index_n + 1) * n_teeth
    if denom <= 0:
        return 0.0
    return fn_hz * 60.0 / denom


def spindle_aligned_rpms(
    spindle_freq_hz: float,
    rpm_min: float,
    rpm_max: float,
    max_subharmonics: int = 12,
) -> list[tuple[int, float]]:
    """
    Spindle speeds (rpm) that are submultiples of the spindle operating frequency.

    Many spindles run at a fixed frequency (e.g. 400 Hz). Speed control often
    gives speeds that are submultiples: rpm = (spindle_freq_hz * 60) / k for
    k = 1, 2, 3, ... (i.e. 24000, 12000, 8000, 6000, ... for 400 Hz). These
    are the "spindle-aligned" operating points. Use them as optimal speed
    candidates so the visual shows how they line up with avoid/optimal bands.

    Returns:
        List of (k, rpm) with rpm in [rpm_min, rpm_max], ordered by k ascending
        (rpm descending). k=1 is the base speed (e.g. 24000 rpm at 400 Hz).
    """
    if spindle_freq_hz <= 0 or rpm_min < 0 or rpm_max <= 0:
        return []
    base_rpm = spindle_freq_hz * 60.0
    out: list[tuple[int, float]] = []
    for k in range(1, max_subharmonics + 1):
        rpm = base_rpm / k
        if rpm_min <= rpm <= rpm_max:
            out.append((k, rpm))
        if rpm < rpm_min:
            break
    return out


def stability_lobe_phase_epsilon_rad(
    Re_FRF_orient: float,
    Im_FRF_orient: float,
) -> float:
    """
    Phase ε (rad) between current and previous tooth vibration for stability (Eq. 4.110).

    ε = 2π − 2·atan(Im[FRF_orient] / Re[FRF_orient]). Used to relate chatter frequency
    fc to spindle speed: fc/(Ω·Nt) = N + ε/(2π). When Re is zero, atan2(Im, Re) gives
    ±π/2 so ε is well-defined.

    Args:
        Re_FRF_orient: Real part of the oriented FRF at the chatter frequency.
        Im_FRF_orient: Imaginary part of the oriented FRF.

    Returns:
        ε in radians, in [0, 2π) typically.
    """
    return 2.0 * math.pi - 2.0 * math.atan2(Im_FRF_orient, Re_FRF_orient)


def process_damping_force_N(
    process_damping_constant_C: float,
    chip_width_mm: float,
    cutting_speed_mm_s: float,
    velocity_normal_mm_s: float,
) -> float:
    """
    Process damping force (N) in the surface normal direction (Eq. 4.108).

    Fd = C·b·ṅ/v. The force is proportional to velocity ṅ (viscous damping): when the
    tool moves on a sinusoidal path, the clearance angle varies and the thrust force
    change is 90° out of phase with displacement and opposite to velocity. C is a
    constant; b is chip width, v is cutting speed. Use consistent units so Fd is in N
    (e.g. b and velocities in mm and mm/s, C in N·s/mm²).

    Args:
        process_damping_constant_C: Constant C (e.g. N·s/mm²).
        chip_width_mm: Chip width b (mm).
        cutting_speed_mm_s: Cutting speed v (mm/s).
        velocity_normal_mm_s: Velocity in surface normal direction ṅ (mm/s).

    Returns:
        Fd in N. Returns 0 if cutting_speed_mm_s <= 0 to avoid division by zero.
    """
    if cutting_speed_mm_s <= 0:
        return 0.0
    return (
        process_damping_constant_C
        * chip_width_mm
        * velocity_normal_mm_s
        / cutting_speed_mm_s
    )


def cutting_speed_m_per_s(diameter_m: float, rpm: float) -> float:
    """
    Cutting speed v (m/s) from tool diameter and spindle speed: v = π·d·Ω/60.

    Used in process damping (cnew depends on v) and in the converging stability
    algorithm. Returns 0 if rpm <= 0 or diameter_m <= 0.

    Metric: diameter in m, result in m/s. If the textbook gives diameter in
    inches, convert: diameter_m = diameter_in * 0.0254. SFM (ft/min) to m/s:
    v_m_s = SFM * 0.00508.

    Args:
        diameter_m: Tool diameter d (m).
        rpm: Spindle speed Ω (rpm).

    Returns:
        v in m/s.
    """
    if diameter_m <= 0 or rpm <= 0:
        return 0.0
    return math.pi * diameter_m * rpm / 60.0


def process_damping_cnew_x_up_milling(
    damping_x: float,
    process_damping_constant_C_N_per_m: float,
    axial_depth_m: float,
    cutting_speed_m_per_s: float,
    phi_ave_deg: float,
) -> float:
    """
    New x-direction damping including process damping for up milling (Eq. 4.113).

    cnew,x = cx + (C·b/v)·cos²(90° − φave). Use SI: C in N/m, b in m, v in m/s.
    For down milling the same formula gives identical result for same φave (cos²(90°−φave) = cos²(φave−90°)).

    Args:
        damping_x: Original structural damping cx (N·s/m).
        process_damping_constant_C_N_per_m: Process damping constant C (N/m).
        axial_depth_m: Axial depth b (m).
        cutting_speed_m_per_s: Cutting speed v (m/s).
        phi_ave_deg: Average tooth angle φave (deg).

    Returns:
        cnew,x in N·s/m. Returns damping_x if v <= 0.
    """
    if cutting_speed_m_per_s <= 0:
        return damping_x
    angle_deg = 90.0 - phi_ave_deg
    cos_sq = (math.cos(math.radians(angle_deg))) ** 2
    return damping_x + (process_damping_constant_C_N_per_m * axial_depth_m / cutting_speed_m_per_s) * cos_sq


def process_damping_cnew_y_up_milling(
    damping_y: float,
    process_damping_constant_C_N_per_m: float,
    axial_depth_m: float,
    cutting_speed_m_per_s: float,
    phi_ave_deg: float,
) -> float:
    """
    New y-direction damping including process damping for up milling (Eq. 4.114).

    cnew,y = cy + (C·b/v)·cos²(180° − φave). Same for down milling (Eq. 4.116).

    Args:
        damping_y: Original structural damping cy (N·s/m).
        process_damping_constant_C_N_per_m: C (N/m).
        axial_depth_m: b (m).
        cutting_speed_m_per_s: v (m/s).
        phi_ave_deg: φave (deg).

    Returns:
        cnew,y in N·s/m.
    """
    if cutting_speed_m_per_s <= 0:
        return damping_y
    angle_deg = 180.0 - phi_ave_deg
    cos_sq = (math.cos(math.radians(angle_deg))) ** 2
    return damping_y + (process_damping_constant_C_N_per_m * axial_depth_m / cutting_speed_m_per_s) * cos_sq


def directional_factors_up_milling(
    force_angle_beta_deg: float,
    phi_ave_deg: float,
) -> tuple[float, float]:
    """
    Directional factors μx, μy for oriented FRF in up milling (Fig. 4.74).

    FRF_orient = μx·FRFx + μy·FRFy. μx = cos(β − (90°−φave))·cos(90°−φave),
    μy = cos((180°−φave) − β)·cos(180°−φave). β is the force angle.

    Args:
        force_angle_beta_deg: Force angle β (deg).
        phi_ave_deg: Average tooth angle φave (deg).

    Returns:
        (μx, μy).
    """
    a = 90.0 - phi_ave_deg
    b_ang = 180.0 - phi_ave_deg
    beta_rad = math.radians(force_angle_beta_deg)
    mu_x = math.cos(math.radians(a) - beta_rad) * math.cos(math.radians(a))
    mu_y = math.cos(math.radians(b_ang) - beta_rad) * math.cos(math.radians(b_ang))
    return (mu_x, mu_y)


def normal_velocity_from_xy_mm_s(
    x_dot_mm_s: float,
    y_dot_mm_s: float,
    phi_rad: float,
) -> float:
    """
    Velocity in surface normal direction (mm/s): ṅ = ẋ·sin(φ) − ẏ·cos(φ).

    Same projection as normal displacement; used for process damping in time-domain
    simulation (e.g. n_dot = x_dot*sin(phia) − y_dot*cos(phia)).

    Args:
        x_dot_mm_s: Velocity in x (mm/s).
        y_dot_mm_s: Velocity in y (mm/s).
        phi_rad: Tooth angle φ (rad).

    Returns:
        ṅ in mm/s.
    """
    return x_dot_mm_s * math.sin(phi_rad) - y_dot_mm_s * math.cos(phi_rad)


def normal_force_with_process_damping_N(
    kn_n_per_mm2: float,
    chip_width_mm: float,
    chip_thickness_mm: float,
    process_damping_constant_C: float,
    cutting_speed_mm_s: float,
    velocity_normal_mm_s: float,
) -> float:
    """
    Normal force (N) including process damping for time-domain simulation (Sect. 4.8, Ex. 4.18).

    Fn = kn·b·h − C·b/v·ṅ. The process damping term subtracts because it opposes velocity.
    Use consistent units (e.g. C in N·s/mm², b and h in mm, v and ṅ in mm/s) so Fn is in N.

    Args:
        kn_n_per_mm2: kn (N/mm²).
        chip_width_mm: Chip width b (mm).
        chip_thickness_mm: h (mm).
        process_damping_constant_C: C (e.g. N·s/mm²).
        cutting_speed_mm_s: v (mm/s).
        velocity_normal_mm_s: ṅ (mm/s).

    Returns:
        Fn in N (can be negative if process damping dominates).
    """
    cutting = kn_n_per_mm2 * chip_width_mm * chip_thickness_mm if chip_thickness_mm > 0 else 0.0
    if cutting_speed_mm_s <= 0:
        return cutting
    damping_term = process_damping_constant_C * chip_width_mm * velocity_normal_mm_s / cutting_speed_mm_s
    return cutting - damping_term


def tooth_period_s(rpm: float, n_teeth: int) -> float:
    """
    Tooth period τ in seconds: τ = 60 / (Ω · Nt), with Ω in rpm.

    Time between successive teeth passing the same angular position. Used in the
    regenerative chip thickness equation h(t) = ft·sin(φ) + n(t−τ) − n(t).

    Args:
        rpm: Spindle speed in rpm.
        n_teeth: Number of teeth.

    Returns:
        Tooth period in seconds (≥ 0). Returns 0 if rpm or n_teeth <= 0.
    """
    if rpm <= 0 or n_teeth <= 0:
        return 0.0
    return 60.0 / (rpm * n_teeth)


def normal_surface_vibration_mm(x_mm: float, y_mm: float, phi_rad: float) -> float:
    """
    Normal-direction vibration (mm) at tooth angle φ: n = x·sin(φ) − y·cos(φ).

    Used in the regenerative chip thickness model. Positive x opposes feed; φ in
    radians. Same units as x_mm, y_mm (typically mm).

    Args:
        x_mm: X-direction vibration (mm).
        y_mm: Y-direction vibration (mm).
        phi_rad: Tooth angle in radians.

    Returns:
        Normal-direction displacement (mm).
    """
    return x_mm * math.sin(phi_rad) - y_mm * math.cos(phi_rad)


def chip_thickness_regenerative_mm(
    feed_per_tooth_mm: float,
    phi_rad: float,
    n_current_mm: float,
    n_previous_mm: float,
) -> float:
    """
    Regenerative instantaneous chip thickness (mm): h = ft·sin(φ) + n(t−τ) − n(t).

    n_previous_mm is the normal-direction vibration of the previous tooth at the
    same angle (stored in surf in the simulation); n_current_mm is the current
    normal vibration. If the tooth leaves the cut, the surf update uses the
    nominal feed not removed. Result can be clamped to non-negative in the
    simulation if needed.

    Args:
        feed_per_tooth_mm: Feed per tooth ft (mm).
        phi_rad: Tooth angle in radians.
        n_current_mm: Current normal vibration n(t) (mm).
        n_previous_mm: Previous tooth normal at same angle n(t−τ) (mm).

    Returns:
        Chip thickness h in mm.
    """
    return feed_per_tooth_mm * math.sin(phi_rad) + n_previous_mm - n_current_mm


def tangential_force_N(
    kt_n_per_mm2: float,
    axial_depth_mm: float,
    chip_thickness_mm: float,
) -> float:
    """
    Tangential cutting force (N) from Eq. 4.8: Ft = kt·b·h.

    Chip width b is the axial depth of cut for straight teeth. If chip thickness
    is negative (tool vibrated out of the cut), no cutting occurs and Ft = 0.

    All units metric: kt in N/mm², b and h in mm. If the textbook gives kt in
    lb/in² (psi), convert: N/mm² = psi × 0.006895.

    Args:
        kt_n_per_mm2: Tangential specific force kt (N/mm²).
        axial_depth_mm: Axial depth of cut b (mm).
        chip_thickness_mm: Instantaneous chip thickness h (mm).

    Returns:
        Ft in N (≥ 0).
    """
    if chip_thickness_mm <= 0.0:
        return 0.0
    return kt_n_per_mm2 * axial_depth_mm * chip_thickness_mm


def normal_force_N(
    kn_n_per_mm2: float,
    axial_depth_mm: float,
    chip_thickness_mm: float,
) -> float:
    """
    Normal cutting force (N) in the rotating frame: Fn = kn·b·h.

    Set to zero when h ≤ 0 (tooth out of cut). Used with Ft for projection to x, y.

    All units metric: kn in N/mm², b and h in mm. If the textbook gives kn in
    lb/in² (psi), convert: N/mm² = psi × 0.006895.

    Args:
        kn_n_per_mm2: Normal specific force kn (N/mm²).
        axial_depth_mm: Axial depth of cut b (mm).
        chip_thickness_mm: Instantaneous chip thickness h (mm).

    Returns:
        Fn in N (≥ 0).
    """
    if chip_thickness_mm <= 0.0:
        return 0.0
    return kn_n_per_mm2 * axial_depth_mm * chip_thickness_mm


# ----- Updated force model with edge effect (Sect. 4.7, Eqs. 4.79–4.81) -----

def tangential_force_with_edge_N(
    kt_n_per_mm2: float,
    kte_n_per_mm: float,
    axial_depth_mm: float,
    chip_thickness_mm: float,
) -> float:
    """
    Tangential force (N) with edge effect (Eq. 4.80): Ft = kt·b·h + kte·b.

    The rubbing (edge) term kte·b is independent of chip thickness; force is
    nonzero as h → 0. kte has units N/mm. Used for regression from mean forces.

    Args:
        kt_n_per_mm2: Cutting coefficient kt (N/mm²).
        kte_n_per_mm: Edge coefficient kte (N/mm).
        axial_depth_mm: Axial depth b (mm).
        chip_thickness_mm: Chip thickness h (mm).

    Returns:
        Ft in N.
    """
    return kt_n_per_mm2 * axial_depth_mm * chip_thickness_mm + kte_n_per_mm * axial_depth_mm


def normal_force_with_edge_N(
    kn_n_per_mm2: float,
    kne_n_per_mm: float,
    axial_depth_mm: float,
    chip_thickness_mm: float,
) -> float:
    """
    Normal force (N) with edge effect (Eq. 4.79): Fn = kn·b·h + kne·b.

    Args:
        kn_n_per_mm2: Cutting coefficient kn (N/mm²).
        kne_n_per_mm: Edge coefficient kne (N/mm).
        axial_depth_mm: Axial depth b (mm).
        chip_thickness_mm: Chip thickness h (mm).

    Returns:
        Fn in N.
    """
    return kn_n_per_mm2 * axial_depth_mm * chip_thickness_mm + kne_n_per_mm * axial_depth_mm


def axial_force_with_edge_N(
    ka_n_per_mm2: float,
    kae_n_per_mm: float,
    axial_depth_mm: float,
    chip_thickness_mm: float,
) -> float:
    """
    Axial force (N) with edge effect (Eq. 4.81): Fa = ka·b·h + kae·b.

    Args:
        ka_n_per_mm2: Cutting coefficient ka (N/mm²).
        kae_n_per_mm: Edge coefficient kae (N/mm).
        axial_depth_mm: Axial depth b (mm).
        chip_thickness_mm: Chip thickness h (mm).

    Returns:
        Fa in N.
    """
    return ka_n_per_mm2 * axial_depth_mm * chip_thickness_mm + kae_n_per_mm * axial_depth_mm


def mean_force_per_rev_slotting_N(
    n_teeth: int,
    axial_depth_mm: float,
    feed_per_tooth_mm: float,
    kt_n_per_mm2: float,
    kn_n_per_mm2: float,
    ka_n_per_mm2: float,
    kte_n_per_mm: float,
    kne_n_per_mm: float,
    kae_n_per_mm: float,
) -> tuple[float, float, float]:
    """
    Mean cutting force per revolution (N) for 100% radial immersion slotting (Eqs. 4.96–4.98).

    φs = 0°, φe = 180°. Used with linear regression over a range of feed-per-tooth
    values to determine the six coefficients from measured mean Fx, Fy, Fz.

    Fx_mean = Nt·b·kn/4 · ft + Nt·b·kne/π
    Fy_mean = Nt·b·kt/4 · ft + Nt·b·kte/π
    Fz_mean = −Nt·b·ka/π · ft − Nt·b·kae/2

    Args:
        n_teeth: Number of teeth Nt.
        axial_depth_mm: Axial depth b (mm).
        feed_per_tooth_mm: Feed per tooth ft (mm).
        kt_n_per_mm2, kn_n_per_mm2, ka_n_per_mm2: Cutting coefficients (N/mm²).
        kte_n_per_mm, kne_n_per_mm, kae_n_per_mm: Edge coefficients (N/mm).

    Returns:
        (Fx_mean, Fy_mean, Fz_mean) in N.
    """
    if n_teeth <= 0 or axial_depth_mm <= 0:
        return (0.0, 0.0, 0.0)
    b = axial_depth_mm
    ft = feed_per_tooth_mm
    Nt = n_teeth
    Fx_mean = Nt * b * (kn_n_per_mm2 / 4.0 * ft + kne_n_per_mm / math.pi)
    Fy_mean = Nt * b * (kt_n_per_mm2 / 4.0 * ft + kte_n_per_mm / math.pi)
    Fz_mean = -Nt * b * (ka_n_per_mm2 / math.pi * ft + kae_n_per_mm / 2.0)
    return (Fx_mean, Fy_mean, Fz_mean)


# ----- Linear regression for cutting force coefficients (Sect. 4.7, Eqs. 4.99–4.104) -----

def linear_regression_slope_intercept(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float]:
    """
    Least-squares slope and intercept for y = a0 + a1*x (Eqs. 4.99 and 4.100).

    For mean force vs feed per tooth: x = ft, y = F_mean. Slope a1 and intercept a0
    are determined by minimizing the sum of squared errors. Requires len(x) == len(y)
    and n ≥ 2; if variance of x is zero, returns (0.0, mean(y)).

    Args:
        x_values: Independent variable (e.g. feed per tooth ft in mm).
        y_values: Dependent variable (e.g. mean force in N).

    Returns:
        (slope_a1, intercept_a0).
    """
    n = len(x_values)
    if n != len(y_values) or n < 2:
        if n == 1 and len(y_values) == 1:
            return (0.0, y_values[0])
        return (0.0, sum(y_values) / len(y_values) if y_values else 0.0)
    sx = sum(x_values)
    sy = sum(y_values)
    sxx = sum(xi * xi for xi in x_values)
    sxy = sum(xi * yi for xi, yi in zip(x_values, y_values))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-20:
        return (0.0, sy / n)
    a1 = (n * sxy - sx * sy) / denom
    a0 = sy / n - a1 * (sx / n)
    return (a1, a0)


def coefficient_of_determination_r2(
    x_values: list[float],
    y_values: list[float],
    slope: float,
    intercept: float,
) -> float:
    """
    Coefficient of determination r² for the linear fit (Eq. 4.101).

    r² describes how well the linear model explains the data (e.g. r² = 0.95 → 95%
    of the variance explained). r² = 1 − Σ(Ei²) / Σ(yi − y_mean)², with Ei = yi − (a0 + a1*xi).

    Args:
        x_values: Independent variable.
        y_values: Dependent variable (measured).
        slope: Fitted slope a1.
        intercept: Fitted intercept a0.

    Returns:
        r² in [0, 1] (can exceed 1 or be negative with numerical noise or bad fit).
    """
    if len(x_values) != len(y_values) or len(y_values) < 2:
        return 0.0
    n = len(y_values)
    y_mean = sum(y_values) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_values)
    if ss_tot <= 0.0:
        return 1.0
    errors = [yi - (intercept + slope * xi) for xi, yi in zip(x_values, y_values)]
    ss_res = sum(e * e for e in errors)
    return 1.0 - ss_res / ss_tot


def cutting_coefficients_from_slotting_regression(
    n_teeth: int,
    axial_depth_mm: float,
    slope_x: float,
    intercept_x: float,
    slope_y: float,
    intercept_y: float,
    slope_z: float,
    intercept_z: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Six cutting force coefficients from slotting linear regression (Eqs. 4.102–4.104).

    From the slopes and intercepts of Fx_mean, Fy_mean, Fz_mean vs feed per tooth:
    kn = 4*a1x/(Nt*b), kne = π*a0x/(Nt*b); kt = 4*a1y/(Nt*b), kte = π*a0y/(Nt*b);
    ka = −π*a1z/(Nt*b), kae = −2*a0z/(Nt*b). (Fz_mean = −Nt*b*(ka/π*ft + kae/2).)

    Args:
        n_teeth: Number of teeth Nt.
        axial_depth_mm: Axial depth b (mm).
        slope_x, intercept_x: From Fx_mean vs ft (a1x, a0x).
        slope_y, intercept_y: From Fy_mean vs ft (a1y, a0y).
        slope_z, intercept_z: From Fz_mean vs ft (a1z, a0z).

    Returns:
        (kn, kne, kt, kte, ka, kae): kn, kt, ka in N/mm²; kne, kte, kae in N/mm.
    """
    if n_teeth <= 0 or axial_depth_mm <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Nt = n_teeth
    b = axial_depth_mm
    kn = 4.0 * slope_x / (Nt * b)
    kne = math.pi * intercept_x / (Nt * b)
    kt = 4.0 * slope_y / (Nt * b)
    kte = math.pi * intercept_y / (Nt * b)
    ka = -math.pi * slope_z / (Nt * b)
    kae = -2.0 * intercept_z / (Nt * b)
    return (kn, kne, kt, kte, ka, kae)


def cutting_forces_xy_N(Ft_N: float, Fn_N: float, phi_rad: float) -> tuple[float, float]:
    """
    Project tangential and normal forces into fixed x and y (Eqs. 4.11 and 4.12).

    Fx = Ft·cos(φ) + Fn·sin(φ),  Fy = Ft·sin(φ) − Fn·cos(φ).
    Positive x opposes the feed direction; φ is the tooth angle in radians.

    Args:
        Ft_N: Tangential force (N).
        Fn_N: Normal force (N).
        phi_rad: Tooth angle in radians.

    Returns:
        (Fx, Fy) in N.
    """
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    Fx = Ft_N * c + Fn_N * s
    Fy = Ft_N * s - Fn_N * c
    return (Fx, Fy)


def helical_lag_angle_deg(
    helix_angle_deg: float,
    axial_depth_mm: float,
    diameter_mm: float,
) -> float:
    """
    Lag angle (deg) between free end and helical edge at axial depth b (Eq. 4.68).

    χ = 2·b·tan(γ) / d (rad), converted to degrees. The cutter rotates χ degrees
    between the time the free end of the helical tooth enters the cut and the time
    the helical edge at axial depth b is engaged. Used to explain saw-tooth force
    and constant-force axial depth for helical endmills.

    Args:
        helix_angle_deg: Helix angle γ in degrees.
        axial_depth_mm: Axial depth of cut b (mm).
        diameter_mm: Cutter diameter d (mm).

    Returns:
        Lag angle in degrees. Returns 0 if diameter_mm <= 0.
    """
    if diameter_mm <= 0:
        return 0.0
    gamma_rad = math.radians(helix_angle_deg)
    chi_rad = 2.0 * axial_depth_mm * math.tan(gamma_rad) / diameter_mm
    return math.degrees(chi_rad)


def constant_force_axial_depth_mm(
    diameter_mm: float,
    tooth_pitch_deg: float,
    helix_angle_deg: float,
) -> float:
    """
    Axial depth b (mm) at which lag angle equals tooth pitch → constant cutting force (Eq. 4.69).

    When χ = φp, the same cutting edge length is engaged regardless of cutter angle;
    b = d·φp / (2·tan(γ)), with φp in radians. Example: 19 mm diameter, 4 teeth (pitch 90°),
    45° helix → b ≈ 14.9 mm. Multiples of this b can give the same constant-force
    behavior if stable and flute length allows.

    Args:
        diameter_mm: Cutter diameter d (mm).
        tooth_pitch_deg: Tooth pitch φp in degrees (360 / Nt).
        helix_angle_deg: Helix angle γ in degrees.

    Returns:
        Axial depth b in mm. Returns 0 if tan(γ) <= 0 (e.g. 0° helix).
    """
    gamma_rad = math.radians(helix_angle_deg)
    t = math.tan(gamma_rad)
    if t <= 0:
        return 0.0
    phi_p_rad = math.radians(tooth_pitch_deg)
    return diameter_mm * phi_p_rad / (2.0 * t)


# ----- Ball endmill (Sect. 4.6): slice geometry and force projection -----

def ball_slice_width_dz_mm(
    diameter_mm: float,
    angle_increment_rad: float,
    helix_angle_deg: float,
) -> float:
    """
    Axial slice width dz (mm) for ball endmill so cutter angle aligns with phi grid (Eq. 4.70).

    dz = d·dφ / (2·tan(γ)), with d = diameter, dφ in rad, γ = helix angle. Ensures the
    instantaneous cutter angle at each slice coincides with a predefined φ angle.

    Args:
        diameter_mm: Cutter diameter d (mm).
        angle_increment_rad: Angular step dφ in radians (e.g. 2π / steps_per_rev).
        helix_angle_deg: Global helix angle γ in degrees.

    Returns:
        dz in mm. Returns 0 if tan(γ) <= 0 or diameter_mm <= 0.
    """
    if diameter_mm <= 0:
        return 0.0
    gamma_rad = math.radians(helix_angle_deg)
    t = math.tan(gamma_rad)
    if t <= 0:
        return 0.0
    return diameter_mm * angle_increment_rad / (2.0 * t)


def ball_kappa_prime_rad(
    slice_index_1based_j: int,
    slice_width_dz_mm: float,
    diameter_mm: float,
) -> float:
    """
    Angle κ′ (rad) between tool axis and ball surface normal for axial slice j (Eq. 4.71).

    κ′ = cos⁻¹(1 − 2·j·dz/d). Used to project chip thickness from the x-y plane onto the
    ball surface normal and to project Ft, Fn, Fa onto x-y-z. When the slice is at or
    past the ball apex (2·j·dz/d ≥ 1), returns π/2 (90°) so Eq. 4.78 applies.

    Args:
        slice_index_1based_j: Axial slice index j = 1, 2, ... (cnt4 in p_4_15_1.m).
        slice_width_dz_mm: Slice width dz (mm) from ball_slice_width_dz_mm.
        diameter_mm: Cutter diameter d (mm).

    Returns:
        κ′ in radians, in [0, π/2].
    """
    if diameter_mm <= 0 or slice_index_1based_j <= 0:
        return 0.0
    x = 1.0 - 2.0 * slice_index_1based_j * slice_width_dz_mm / diameter_mm
    if x <= 0.0:
        return math.pi / 2.0  # at or past ball apex: κ′ = 90° (Eq. 4.78)
    x = min(1.0, x)
    return math.acos(x)


def ball_theta0_rad(
    slice_index_1based_j: int,
    slice_width_dz_mm: float,
    diameter_mm: float,
) -> float:
    """
    Angle θ0 (rad) at the lower edge of slice j for chip width on the ball (Eq. 4.73).

    θ0 = cos⁻¹(1 − 2·dz·(j−1)/d). For j = 1, θ0 = 0. Used with κ′ to get angular span
    θ = κ′ − θ0 for chip width db = r·θ.

    Args:
        slice_index_1based_j: Axial slice index j (1-based).
        slice_width_dz_mm: Slice width dz (mm).
        diameter_mm: Cutter diameter d (mm).

    Returns:
        θ0 in radians, in [0, π].
    """
    if diameter_mm <= 0 or slice_index_1based_j <= 0:
        return 0.0
    if slice_index_1based_j == 1:
        return 0.0
    x = 1.0 - 2.0 * slice_width_dz_mm * (slice_index_1based_j - 1) / diameter_mm
    x = max(-1.0, min(1.0, x))
    return math.acos(x)


def ball_chip_width_db_mm(
    radius_mm: float,
    kappa_prime_rad: float,
    theta0_rad: float,
) -> float:
    """
    Chip width db (mm) for a ball endmill slice: arc length on the ball (Eq. 4.72).

    db = r·θ, with θ = κ′ − θ0. When θ ≤ 0 (slice not on ball), returns 0.

    Args:
        radius_mm: Ball radius r (mm), half of diameter.
        kappa_prime_rad: κ′ from ball_kappa_prime_rad (rad).
        theta0_rad: θ0 from ball_theta0_rad (rad).

    Returns:
        db in mm (≥ 0).
    """
    if radius_mm <= 0:
        return 0.0
    theta = kappa_prime_rad - theta0_rad
    if theta <= 0.0:
        return 0.0
    return radius_mm * theta


def chip_thickness_ball_mm(h_xy_plane_mm: float, kappa_prime_rad: float) -> float:
    """
    Chip thickness (mm) on the ball surface from x-y plane thickness (Sect. 4.6).

    h = (h_xy)·sin(κ′). The x-y plane thickness is ft·sin(φ) + surf − n; projecting
    onto the ball surface normal gives h used in Ft = kt·h·db, Fn = kn·h·db, Fa = ka·h·db.

    Args:
        h_xy_plane_mm: Chip thickness in the x-y plane (mm).
        kappa_prime_rad: κ′ from ball_kappa_prime_rad (rad).

    Returns:
        Chip thickness in mm (≥ 0).
    """
    if h_xy_plane_mm <= 0.0:
        return 0.0
    return h_xy_plane_mm * math.sin(kappa_prime_rad)


def ball_forces_xyz_N(
    Ft_N: float,
    Fn_N: float,
    Fa_N: float,
    phi_rad: float,
    kappa_prime_rad: float,
) -> tuple[float, float, float]:
    """
    Project tangential, normal, and axial forces into fixed x-y-z (Eq. 4.77).

    [ Fx ]   [ cos(φ)    sin(φ)sin(κ′)   sin(φ)cos(κ′) ] [ Ft ]
    [ Fy ] = [ sin(φ)   −cos(φ)sin(κ′)  −cos(φ)cos(κ′) ] [ Fn ]
    [ Fz ]   [ 0         cos(κ′)        −sin(κ′)      ] [ Fa ]

    When commanded axial depth exceeds the ball radius, κ′ is set to 90° and the
    matrix collapses to Eq. 4.78: Fx, Fy as for square endmill; Fz = Fa.

    Args:
        Ft_N: Tangential force (N), from kt·h·db.
        Fn_N: Normal force (N), from kn·h·db.
        Fa_N: Axial force (N), from ka·h·db.
        phi_rad: Tooth angle φ (rad).
        kappa_prime_rad: κ′ in rad; use π/2 when depth > ball radius.

    Returns:
        (Fx, Fy, Fz) in N.
    """
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    sk = math.sin(kappa_prime_rad)
    ck = math.cos(kappa_prime_rad)
    Fx = c * Ft_N + s * ck * Fa_N + s * sk * Fn_N
    Fy = s * Ft_N - c * ck * Fa_N - c * sk * Fn_N
    Fz = ck * Fn_N - sk * Fa_N
    return (Fx, Fy, Fz)


def angle_increment_deg(steps_per_rev: int) -> float:
    """
    Angular increment per simulation step (deg): dφ = 360 / steps_rev.

    Args:
        steps_per_rev: Number of steps per revolution (steps_rev).

    Returns:
        dφ in degrees. Returns 0 if steps_per_rev <= 0.
    """
    if steps_per_rev <= 0:
        return 0.0
    return 360.0 / steps_per_rev


def simulation_time_step_s(steps_per_rev: int, rpm: float) -> float:
    """
    Time step (s) for time-domain simulation: dt = 60 / (steps_rev · Ω).

    Ω in rpm. Corresponds to one angular increment dφ = 360/steps_rev.

    Args:
        steps_per_rev: Number of steps per revolution (must satisfy steps_rev/Nt integer).
        rpm: Spindle speed in rpm.

    Returns:
        dt in seconds. Returns 0 if steps_per_rev <= 0 or rpm <= 0.
    """
    if steps_per_rev <= 0 or rpm <= 0:
        return 0.0
    return 60.0 / (steps_per_rev * rpm)


def simulation_steps_rev(steps_per_rev: int, n_teeth: int) -> int:
    """
    Adjust steps_per_rev so that steps_rev / Nt is an integer (required for simulation).

    Example: 650 steps, 4 teeth → 650/4 = 162.5; round to 163 → return 163*4 = 652.

    Args:
        steps_per_rev: Requested steps per revolution.
        n_teeth: Number of teeth.

    Returns:
        Adjusted steps_per_rev such that (return value) / n_teeth is an integer.
        Returns 0 if n_teeth <= 0.
    """
    if n_teeth <= 0:
        return 0
    # Round half up (162.5 → 163) so Example 4.10 gives 652; Python's round() is round-half-to-even
    steps_per_tooth = max(1, int(steps_per_rev / n_teeth + 0.5))
    return steps_per_tooth * n_teeth


