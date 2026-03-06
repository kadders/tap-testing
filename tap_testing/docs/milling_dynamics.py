"""
Milling dynamics: nutshell summaries for documentation or UI.

Start/exit angles, cutting force geometry, tooth passing, stability lobes,
time-domain simulation, helical teeth, coefficients, process damping, etc.
See tap_testing.milling_dynamics for computation (radial_immersion_percent,
stability_lobe_best_spindle_speed_rpm, etc.).
"""
from __future__ import annotations

__all__ = [
    "milling_dynamics_nutshell",
    "milling_start_exit_angles_nutshell",
    "milling_cutting_force_geometry_nutshell",
    "milling_cutting_forces_example_4_2_nutshell",
    "milling_multiple_teeth_engagement_nutshell",
    "tooth_passing_frequency_nutshell",
    "milling_stability_lobes_nutshell",
    "milling_time_domain_simulation_nutshell",
    "milling_surf_update_rule_nutshell",
    "milling_helical_teeth_nutshell",
    "milling_ball_endmill_simulation_nutshell",
    "milling_experimental_cutting_coefficients_nutshell",
    "milling_updated_force_model_nutshell",
    "milling_linear_regression_coefficients_nutshell",
    "milling_nonlinear_optimization_coefficients_nutshell",
    "milling_chatter_to_stable_strategy_nutshell",
    "milling_example_6_2_competing_lobes_nutshell",
    "milling_experimental_techniques_nutshell",
    "milling_process_damping_nutshell",
    "milling_process_damping_stability_nutshell",
    "milling_straight_vs_helical_forces_nutshell",
    "milling_simulation_loop_and_examples_nutshell",
    "milling_dynamics_summary",
]


def milling_dynamics_nutshell() -> str:
    """
    In-a-nutshell: milling vs turning, and why cutting conditions in milling are complex.

    Whereas in turning operations the chip thickness and chip width are fixed,
    this is not the case in milling. In a straight slotting cut, the chip thickness
    encountered by each tooth varies continuously as that tooth enters and exits
    the cut. In pocket milling, the radial depth of cut may also change. Finally,
    in sculptured surface milling, the axial depth of cut may vary as well. Even
    though the motion of the rotating tool with respect to the workpiece may be
    relatively simple to visualize, the exact conditions of the cutting-edge
    engagement with the workpiece can be surprisingly complicated.
    """
    return (
        "Whereas in turning operations the chip thickness and chip width are fixed, "
        "this is not the case in milling. In a straight slotting cut, the chip thickness "
        "encountered by each tooth varies continuously as that tooth enters and exits "
        "the cut. In pocket milling, the radial depth of cut may also change. Finally, "
        "in sculptured surface milling, the axial depth of cut may vary as well. Even "
        "though the motion of the rotating tool with respect to the workpiece may be "
        "relatively simple to visualize, the exact conditions of the cutting-edge "
        "engagement with the workpiece can be surprisingly complicated."
    )


def milling_start_exit_angles_nutshell() -> str:
    """
    Start and exit angles for up milling; radial immersion and Example 4.1.
    """
    return (
        "For peripheral up milling, the start angle is φs = 0. The exit angle is "
        "φe = cos⁻¹((r - a) / r), where r is the cutter radius and a is the radial "
        "depth of cut. A 10% radial immersion cut (e.g. 1.9 mm radial depth with "
        "19 mm diameter, so r = 9.5 mm) gives φe = cos⁻¹(0.8) ≈ 37°. A slotting cut "
        "with radial depth equal to the diameter is 100% radial immersion. "
        "The instantaneous chip thickness between φs and φe is defined by the "
        "feed per tooth and the cutter angle. Down milling: φs = 180° - cos⁻¹((r - a)/r) "
        "and φe = 180° (e.g. 25% immersion with a = r/2 gives φs = 120°)."
    )


def milling_cutting_force_geometry_nutshell() -> str:
    """
    Cutting force in milling: not constant, function of angle; projection to x, y.
    """
    return (
        "Unlike turning, the cutting force in milling is not constant under rigid "
        "tool and workpiece assumptions; it is a function of the cutter angle φ. "
        "The total force on a tooth is F = Ks·A = Ks·b·h (specific force × chip area). "
        "Normal and tangential components in the rotating frame are Fn = kn·b·h and "
        "Ft = kt·b·h. For measurement in a fixed frame (e.g. dynamometer), project "
        "into x and y: Fx = Ft·cos(φ) + Fn·sin(φ), Fy = Ft·sin(φ) - Fn·cos(φ). "
        "Replacing the instantaneous chip thickness h with the expression in terms of "
        "φ (e.g. h = ft·sin(φ)) gives forces that are periodic in φ; the final force "
        "also depends on how many teeth are simultaneously engaged and on the "
        "start/exit angles of the cut."
    )


def milling_cutting_forces_example_4_2_nutshell() -> str:
    """
    Example 4.2: 25% radial immersion up and down milling, force profiles, typical parameters.
    """
    return (
        "Example 4.2 considers 25% radial immersion peripheral milling (a = r/2). "
        "Up milling: φs = 0, φe = cos⁻¹(0.5) = 60°. Down milling: φs = 180° - 60° = 120°, "
        "φe = 180°. With four teeth at 90° intervals, each tooth is engaged for 60°; "
        "for the remaining 30° between teeth the forces are zero. Force components "
        "in the fixed x-y frame use Eqs. 4.11 and 4.12; the resultant is F = √(Fx² + Fy²). "
        "In up milling the chip thickness increases with angle so force levels grow until "
        "exit at 60°. In down milling the maximum force occurs near entry and then "
        "decreases as the final surface is created, which is why down milling is often "
        "selected for finishing. The x direction force is positive in up milling and "
        "negative (acting to the right) in down milling. Typical parameters used for "
        "illustration (e.g. aluminum): kt = 750 N/mm², kn = 250 N/mm², b = 5 mm, "
        "ft = 0.1 mm/tooth (Ks ≈ 791 N/mm², β ≈ 71.6°)."
    )


def milling_multiple_teeth_engagement_nutshell() -> str:
    """
    Multiple teeth in cut; 75% radial immersion example; force summation (Eqs 4.15–4.16); slotting.
    """
    return (
        "With more teeth or higher radial immersion, more than one tooth can be engaged "
        "at a given time, and the number can alternate (e.g. one tooth over some interval, "
        "two in another). At 75% radial immersion with four teeth, the cut includes both "
        "up (0 ≤ φ ≤ 90°) and down (90° < φ ≤ 120°) milling. When 0 ≤ φ ≤ 30°, teeth 4 and 1 "
        "are cutting; when 30° < φ ≤ 90°, only tooth 1; when 90° < φ ≤ 120°, teeth 1 and 2. "
        "The cutting force never drops to zero. During intervals when two teeth are engaged, "
        "the force can be constant. The total force is the sum over all teeth: Fx and Fy "
        "use Eqs. 4.15 and 4.16, where φj is the angle of each tooth (tooth pitch = 360°/Nt; "
        "e.g. tooth 1 at 40° gives φ2 = 310°, φ3 = 220°, φ4 = 130° for Nt = 4). Only teeth "
        "with φs ≤ φj ≤ φe contribute; otherwise add zero. Slotting with an even number of "
        "teeth Nt > 2 yields constant cutting force at all times. When evaluating tooling "
        "in tap-test analysis, flute count Nt determines tooth-passing frequency and avoid "
        "RPM; higher Nt or higher immersion changes how many teeth are in cut and the "
        "force profile, but the same natural frequency from the test still dictates which "
        "spindle speeds to avoid."
    )


def tooth_passing_frequency_nutshell() -> str:
    """
    Tooth passing frequency and its link to tap-test analysis and our test-cycle data.
    """
    return (
        "Cutting force signals in partial immersion milling resemble trains of periodic "
        "impulses. To get the frequency spectrum, convert the abscissa from tooth angle "
        "(deg) to time (s): t = φ · 60 / (Ω · 360), with Ω in rpm. Each tooth passage "
        "(e.g. 90° for a four-tooth cutter) then corresponds to a fixed time interval "
        "(e.g. 0.002 s at 7500 rpm). The Fourier transform of the force vs. time signal "
        "shows content at the fundamental tooth passing frequency f_tooth = Ω · Nt / 60 Hz "
        "and at integer multiples (second harmonic 2·f_tooth, third 3·f_tooth, etc.). "
        "Sharper force impulses lead to larger higher-order harmonics; DC content is "
        "also common. Our tap-test cycles measure the tool-holder natural frequency from "
        "the response spectrum. We use that natural frequency to avoid spindle RPM at "
        "which f_tooth or one of its harmonics equals the natural frequency (resonance), "
        "so the avoid-RPM list is exactly the speeds where excitation from tooth passing "
        "would coincide with the mode we identified in the test data."
    )


def milling_stability_lobes_nutshell() -> str:
    """
    Stability lobe diagrams (Tlusty average-tooth-angle approach) and link to tap-test.
    """
    return (
        "Stability lobe diagrams map spindle speed Ω vs. limiting chip width blim: "
        "above the boundary cutting is unstable (chatter); below, stable. Tlusty's "
        "average-tooth-angle approach assumes a fixed average force direction so the "
        "system is time-invariant. The limiting chip width is blim = 1/(2·Ks·Re[FRF_orient]·Nt*), "
        "where the oriented FRF is FRF_orient = μx·FRFx + μy·FRFy (directional factors μx, μy "
        "depend on radial immersion and force angle β). Nt* = (φe − φs)/360 · Nt is the "
        "average number of teeth in the cut. The phase ε between current and previous "
        "tooth vibrations and the lobe index N (integer waves between teeth) determine "
        "spindle speed. Best spindle speeds (rpm) are Ω_best = fn·60/((N+1)·Nt) for lobe N; "
        "the natural frequency fn from our tap-test cycles can be used to evaluate these "
        "speeds. In milling, stable zones appear at lower spindle speeds than in turning, "
        "making the stability lobe diagram especially useful for selecting high-performance "
        "stable conditions. Our analysis provides fn and avoid-RPM; full stability lobes "
        "additionally require x and y direction FRFs (e.g. from impact testing with force "
        "measurement) and cutting force parameters (Ks, β) to compute blim and the full "
        "(Ω, blim) boundary."
    )


def milling_time_domain_simulation_nutshell() -> str:
    """
    Time-domain milling simulation (Smith & Tlusty regenerative model) and analytics link.
    """
    return (
        "Time-domain simulation for milling is based on the regenerative force, dynamic "
        "deflection model (Smith & Tlusty). Analytical stability lobes give a global "
        "picture of stability; simulation gives local cutting force and tool vibration "
        "for selected conditions. The simulation uses numerical integration of "
        "time-delayed equations of motion and includes the nonlinearity when a tooth "
        "leaves the cut. Assumptions include straight teeth, circular tool path, and "
        "square endmill geometry. Steps: (1) Instantaneous chip thickness from current "
        "and previous tooth vibration at the tooth angle: h = ft·sin(φ) + n(t−τ) − n(t), "
        "with τ = 60/(Ω·Nt) s (tooth period) and n = x·sin(φ) − y·cos(φ). (2) Cutting "
        "force from h. (3) Force → new displacements (modal x, y). (4) Increment angle "
        "by dφ, repeat. The angle is discretized: dφ = 360/steps_rev (deg), dt = 60/"
        "(steps_rev·Ω) s; steps_rev/Nt must be an integer so tooth positions index "
        "correctly into the angle and surf vectors. A surf vector stores the previous "
        "tooth's normal at each angle; if the tooth vibrates out of the cut, surf is "
        "updated with the nominal feed not removed. Nothing represents reality better "
        "than actual cutting; simulation removes many analytical simplifications and "
        "gives detailed local information but loses the global view—producing a full "
        "stability lobe diagram requires many simulation runs. Helical teeth (varying "
        "engagement along the axis, deflection imprinted on the surface) are straightforward "
        "in time-domain simulation but difficult in analytical formulations. Our tap-test natural "
        "frequency and modal parameters feed into such simulations as the dynamics "
        "in the x and y directions."
    )


def milling_surf_update_rule_nutshell() -> str:
    """
    Surf vector update rule for time-domain milling (Sect. 4.4).

    surf(angle) stores the previous tooth's normal-direction vibration at that angle;
    it supplies n(t−τ) in h = ft·sin(φ) + n(t−τ) − n(t). After computing chip thickness
    and force at the current angle: (1) If the tooth is in the cut (between φs and φe
    and tangential force ≥ 0), set surf(angle) = n (current normal vibration).
    (2) If the tooth is in the angular range but has left the cut (tangential force < 0),
    set surf(angle) = surf(angle) + ft·sin(φ) so the nominal feed not removed is
    carried forward. Otherwise do not update surf.
    """
    return (
        "Surf stores previous-tooth normal at each angle for h = ft·sin(φ) + surf − n. "
        "Update: in cut (φs ≤ φ ≤ φe, Ft ≥ 0) → surf = n; out of cut (Ft < 0) → surf += ft·sin(φ)."
    )


def milling_helical_teeth_nutshell() -> str:
    """
    Helical teeth: varying engagement along the axis, surface imprint, simulation vs analytical.

    With helical teeth the engaged portion of each tooth varies as the tool rotates;
    the changing deflection is imprinted on the machined surface parallel to the
    tool axis. This is difficult to include in analytical formulations but relatively
    straightforward in time-domain simulation, which is therefore a better
    representation of the cutting process than the analytical approaches—at the
    cost of increased computational effort.
    """
    return (
        "With helical teeth, the portion of each tooth in the cut varies as the tool "
        "rotates, and the changing tool deflection is imprinted on the surface "
        "parallel to the tool axis. This is difficult to include in analytical "
        "stability formulations but relatively straightforward in time-domain simulation, "
        "so time-domain simulation is a better representation of the process than either "
        "analytical formulation. The trade-off is that increased accuracy is "
        "computationally more intensive."
    )


def milling_ball_endmill_simulation_nutshell() -> str:
    """
    Ball endmill time-domain simulation (Sect. 4.6): slice geometry, κ′, chip width, Fx,Fy,Fz.
    """
    return (
        "Ball endmill simulation extends helical square endmill simulation with spherical "
        "geometry. The tool is sectioned into axial slices of width dz = d·dφ/(2·tan(γ)) "
        "(Eq. 4.70) so the cutter angle at each slice matches the φ grid. For each slice j, "
        "κ′ = cos⁻¹(1 − 2·j·dz/d) (Eq. 4.71) is the angle between the tool axis and the ball "
        "surface normal. The x-y plane chip thickness (ft·sin(φ) + surf − n) is projected "
        "onto the ball normal: h = (h_xy)·sin(κ′). Chip width is the arc length db = r·θ "
        "with θ = κ′ − θ0, θ0 = cos⁻¹(1 − 2·dz·(j−1)/d) (Eqs. 4.72–4.73). Forces per slice "
        "are Ft = kt·h·db, Fn = kn·h·db, Fa = ka·h·db (Eqs. 4.74–4.76). These are projected "
        "into fixed x-y-z by Eq. 4.77 (using φ and κ′). When axial depth exceeds the ball "
        "radius, κ′ = 90° and Eq. 4.78 gives Fx, Fy as for the square endmill and Fz = Fa. "
        "Example 4.15 compares ball and square helical endmills (30% radial immersion, 5 mm "
        "axial depth, Ks = 600 N/mm², β = 60°, ka = kn); differences in Fx, Fy, Fz arise "
        "from the ball surface normal and projections; the resultant force magnitude is "
        "the same for both tools."
    )


def milling_experimental_cutting_coefficients_nutshell() -> str:
    """
    Experimental determination of cutting force coefficients (Fig. 4.8); force model revisited.
    """
    return (
        "Experimental determination of the cutting force coefficients for the milling force "
        "model is discussed with a square endmill geometry again presumed. As shown in Fig. 4.8, "
        "these tests are carried out by prescribing a known feed per tooth and axial depth and "
        "measuring the x (feed), y, and z (axial) direction cutting force components in the "
        "dynamometer's fixed coordinate frame. Prior to detailing the numerical and experimental "
        "techniques used to obtain the coefficient values, the force model and corresponding "
        "coefficients are revisited: in the rotating frame Ft = kt·b·h and Fn = kn·b·h (tangential "
        "and normal specific forces kt, kn in N/mm²; chip width b, thickness h); these project to "
        "Fx, Fy in the fixed frame via the cutter angle φ; for helical or ball endmills an axial "
        "component Fa = ka·h·db and Fz are also considered. The coefficients kt, kn (and ka where "
        "applicable), or the equivalent Ks and force angle β, are then obtained from the measured "
        "forces using the techniques that follow."
    )


def milling_updated_force_model_nutshell() -> str:
    """
    Updated force model (Sect. 4.7): edge effect, six coefficients, mean force slotting.
    """
    return (
        "Equations 4.6–4.10 and 4.34 assume the resultant cutting force is proportional to "
        "chip thickness and axial depth and independent of other parameters; this is reasonable "
        "for stability lobe diagrams but can disagree with measured forces. A nonzero cutting "
        "edge radius causes increased rubbing (or plowing) when chip thickness is near the "
        "edge radius, so force exceeds the F = Ks·b·h model; cutting speed can also affect "
        "force (strain rate, temperature). Coefficients are therefore not material properties "
        "but process- and tool-workpiece-dependent. The model is augmented with edge terms "
        "(Eqs. 4.79–4.81): Fn = kn·b·h + kne·b, Ft = kt·b·h + kte·b, Fa = ka·b·h + kae·b. "
        "The six coefficients (kt, kn, ka, kte, kne, kae) are determined by linear regression "
        "using the average cutting forces over a range of feed per tooth; the DC rubbing terms "
        "are neglected in linear stability analysis so previous lobe derivations remain valid. "
        "For 100% radial immersion (slotting), the mean force per revolution simplifies to "
        "Eqs. 4.96–4.98 (Fx_mean, Fy_mean, Fz_mean in terms of ft and the six coefficients)."
    )


def milling_linear_regression_coefficients_nutshell() -> str:
    """
    Linear regression over feed per tooth to determine six coefficients (Eqs. 4.99–4.104).
    """
    return (
        "Given the mean-force expressions for slotting (Eqs. 4.96–4.98), linear regression "
        "over feed per tooth ft is used to determine the six cutting force coefficients. "
        "The first term in each equation depends on ft (slope), the second does not (intercept); "
        "so F_mean = a0 + a1*ft matches slope-intercept form with ft as the independent variable. "
        "For each direction (x, y, z), the slope and intercept are obtained by minimizing the "
        "sum of squared errors (Eqs. 4.99–4.100). The coefficient of determination r² (Eq. 4.101) "
        "describes how well the linear model explains the data (e.g. r² = 0.95 → 95% of variance "
        "explained). From the fitted slopes and intercepts, the coefficients are recovered "
        "(Eqs. 4.102–4.104): kn = 4*a1x/(Nt*b), kne = π*a0x/(Nt*b); kt = 4*a1y/(Nt*b), "
        "kte = π*a0y/(Nt*b); ka = −π*a1z/(Nt*b), kae = −2*a0z/(Nt*b). Example 4.16 demonstrates "
        "the procedure with simulated slotting data (e.g. kn = 190, kne = 5, kt = 710 N/mm² or N/mm, "
        "ft = 0.05–0.25 mm/tooth, Nt = 2, b = 5 mm) and linear regression to recover the coefficients."
    )


def milling_nonlinear_optimization_coefficients_nutshell() -> str:
    """
    Instantaneous force, nonlinear least-squares identification (Sect. 4.7.3).
    """
    return (
        "Alternatively, the six cutting force coefficients (and optionally flute-to-flute runout) "
        "may be determined by instantaneous-force nonlinear optimization. The method solves a "
        "nonlinear least-squares curve-fitting problem: at each time step, cutting forces are "
        "simulated in the time domain (chip thickness, force from Eqs. 4.79–4.81, coordinate "
        "transformation from rotating Ft, Fn, Fa to fixed Fx, Fy, Fz via the cutter angle φ), "
        "and the residual is the difference between simulated and measured Fx, Fy, Fz. The "
        "objective is to minimize the sum of squared residuals over all time steps (Eq. 4.107). "
        "A trust-region reflective algorithm is typically used, with bounds on the decision variables. "
        "One advantage over linear regression is that only a single cut is required, which is "
        "important when tool wear is high. The mechanistic model may also be extended to "
        "nonlinear dependence on chip thickness."
    )


def milling_chatter_to_stable_strategy_nutshell() -> str:
    """
    Special topics: simple 4-step strategy to move from unstable to stable milling.
    """
    return (
        "Given the complexity of milling (tooth passing content and system dynamics), a simple "
        "strategy exists for directing unstable cuts into stable zones. (1) Record the frequency "
        "content of the cutting signal—a microphone works well; accelerometers, dynamometers, "
        "or even remote smartphone recording are also possible. (2) Ignore frequency components "
        "caused by teeth passing (filter or ignore them). Know the actual spindle speed accurately "
        "(it often differs from commanded due to load or digitization); measure it or infer from "
        "runout frequency and harmonics. (3) If a significant frequency peak remains, that is the "
        "chatter frequency (chatter detector). Define 'significant' e.g. by recording without "
        "cutting and setting the threshold just above that level. (4) Choose a new spindle speed "
        "so that the tooth passing frequency or a multiple of it equals the detected chatter "
        "frequency. This approach identifies a stable speed if one is available at the selected "
        "axial and radial depths; it may take a few iterations but not many."
    )


def milling_example_6_2_competing_lobes_nutshell() -> str:
    """
    Example 6.2: Selecting new spindle speeds with competing lobes (two-mode system).
    """
    return (
        "Example 6.2 considers a two-mode symmetric system (fn1=800 Hz, fn2=900 Hz) and 25% "
        "radial immersion up milling with a four-tooth cutter (Ks=700 N/mm², β=66°). Exit angle "
        "φe=60° (Eq. 4.4), φs=0, φave=30°. Directional factors μx = cos(β−60°)cos(60°), "
        "μy = cos(150°−β)cos(150°) define the oriented FRF. The two modes give two valid chatter "
        "frequency ranges and competing lobes: the overall stability limit is the minimum of the "
        "two at each spindle speed where they overlap. Initial cut at 20,000 rpm, 2 mm axial depth "
        "is unstable; chatter frequency fc=919 Hz, f_tooth=1333 Hz, undulation phase ε=248°. "
        "First regulation: Ω = 60·fc/((N+1)·Nt) = 60·919/4 = 13,785 rpm (N=0). Cut still unstable; "
        "chatter moves to 804 Hz (first competing lobe), ε=315°. Second regulation: Ω = 60·804/4 = "
        "12,060 rpm gives stable cutting. The new speed lies to the left of the first competing "
        "lobe for N=0, where stable axial depth is larger; f_tooth=804 Hz, ε=358°."
    )


def milling_experimental_techniques_nutshell() -> str:
    """
    Dynamometer setup, bandwidth, and stable cutting tests (Sect. 4.7.4).
    """
    return (
        "Cutting forces are typically measured with a table-top dynamometer mounted to the "
        "machine table and aligned with the feed direction. Data acquisition range should "
        "be chosen for adequate resolution; simulation (e.g. with assumed coefficients) can "
        "guide maximum and minimum signal levels. The dynamometer is a dynamic system with "
        "its own frequency response; if the cutting force has content near the dynamometer "
        "natural frequency, the data can be corrupted. The tooth passing frequency provides "
        "a lower bound on the desired dynamometer bandwidth. Impact testing with the mounted "
        "workpiece (which mass-loads the dynamometer) can determine the frequency response; "
        "tooth passing can be kept below the first natural frequency, or inverse filtering "
        "of the measured force can be applied. Cutting tests should be stable: use the tool "
        "point FRF and stability lobe algorithms with assumed coefficients to select a stable "
        "axial depth for the chosen radial depth, number of teeth, and spindle speed."
    )


def milling_process_damping_nutshell() -> str:
    """
    Process damping (Sect. 4.8): physical mechanism, clearance angle, viscous damping (Eq. 4.108).
    """
    return (
        "Process damping is described by a tool moving on a sinusoidal path while shearing the chip. "
        "The clearance angle γ between the flank face and the work surface tangent varies: at one "
        "point it equals the nominal relief angle; where the surface slopes into the tool it can "
        "become small or negative (interference, increased thrust in the surface normal direction n); "
        "where it opens up, thrust is decreased. The force change from the sinusoidal path is 90° "
        "out of phase with displacement and has the opposite sign from velocity, so it behaves as "
        "viscous damping (force proportional to velocity). The process damping force in the surface "
        "normal direction is Fd = C·b·ṅ/v (Eq. 4.108), where ṅ is velocity in the normal direction, "
        "b is chip width, v is cutting speed, and C is a constant."
    )


def milling_process_damping_stability_nutshell() -> str:
    """
    Process damping in the stability algorithm (Sect. 4.8.2): cnew, blim, converging, time domain.
    """
    return (
        "Process damping is incorporated in the average-tooth-angle approach. The process damping "
        "force in the surface normal n is projected onto x and y using the average tooth angle φave: "
        "cnew,x = cx + (C·b/v)·cos²(90°−φave), cnew,y = cy + (C·b/v)·cos²(180°−φave) (Eqs. 4.113–4.116; "
        "up and down milling give identical results for the same φave). The oriented FRF is "
        "FRF_orient = μx·FRFx + μy·FRFy with μx, μy from the force angle β and φave (Fig. 4.74). "
        "Limiting chip width blim = 1/(2·Ks·Re[FRF_orient]·Nt*) (Eq. 4.109); fc/(Ω·Nt) = N + ε/(2π) "
        "(Eq. 4.110). Cutting speed v = π·d·Ω/60 (m/s). The analysis is converging: (1) compute "
        "stability boundary with no process damping for initial b and Ω; (2) compute cnew from b and v; "
        "(3) repeat with new damping until blim and Ω converge. In time-domain simulation, velocity "
        "in the normal direction is ṅ = ẋ·sin(φ) − ẏ·cos(φ), and Fn = kn·b·h − C·b/v·ṅ (Ex. 4.18)."
    )


def milling_straight_vs_helical_forces_nutshell() -> str:
    """
    Example 4.14: straight vs helical tooth force comparison; lag angle; constant-force b.

    Same conditions (30% radial immersion up, φs=0°, φe=66.4°; 4-tooth 19 mm, ft=0.15;
    Ks=600 N/mm², β=60° → kt=520, kn=300) at Ω=15,000 rpm, b=5 mm: straight teeth
    give higher max force and force drops abruptly to zero at exit; helical (45°)
    gives lower max force and a saw-tooth decay that does not reach zero (lag/wrapping).
    Lag angle χ = 2·b·tan(γ)/d (e.g. b=5 mm, γ=45°, d=19 mm → χ≈30.2°). When χ equals
    tooth pitch, same cutting edge length is engaged at all angles → constant force;
    b = d·φp/(2·tan(γ)) (e.g. 14.9 mm for 4 teeth, 45° helix, 19 mm). At that b, straight
    tooth can border instability (regeneration); helical gives nearly constant force with
    lower maximum.
    """
    return (
        "Example 4.14 compares straight and helical teeth (30% radial immersion up, "
        "4-tooth 19 mm, ft = 0.15 mm/tooth, Ks = 600 N/mm², β = 60°, Ω = 15,000 rpm). "
        "At b = 5 mm: straight teeth give higher max resultant force and force drops "
        "abruptly to zero at cut exit; 45° helix gives lower max force and a saw-tooth "
        "decay that does not reach zero (lag: χ = 2·b·tan(γ)/d; at φ = φe the free end "
        "exits but the helical portion near the spindle remains engaged). When lag "
        "angle equals tooth pitch, the same cutting edge length is engaged regardless "
        "of angle → constant force; b = d·φp/(2·tan(γ)) (e.g. 14.9 mm for 4 teeth, 45° "
        "helix, 19 mm). At that axial depth, straight-tooth cut can border instability; "
        "helical gives nearly constant force with much lower maximum."
    )


def milling_simulation_loop_and_examples_nutshell() -> str:
    """
    Simulation loop (three activities per step) and Examples 4.11–4.12 (slotting vs partial immersion).
    """
    return (
        "At each time step the simulation does three things: (1) Rotate the cutter by dφ "
        "(e.g. add one to each entry in the teeth vector). (2) For each tooth: if φs ≤ φ ≤ φe, "
        "compute chip thickness and cutting force (including the nonlinearity when a tooth "
        "leaves the cut); otherwise set force to zero; sum contributions over teeth. "
        "(3) Update displacement by numerical integration. Example 4.11 (slotting): symmetric "
        "x/y dynamics fn = 500 Hz, 4-tooth endmill, kt = 695 and kn = 281 N/mm² (Ks = 750, β = 68°), "
        "φs = 0°, φe = 180°, 652 steps/rev, ft = 0.15 mm/tooth, 20 rev. Four (Ω, b) cases: "
        "(1) 7500 rpm, b = 3 mm and (3) 5000 rpm, b = 0.1 mm are stable (constant force after "
        "transients); (2) 7500 rpm, b = 5 mm and (4) 5000 rpm, b = 0.5 mm are unstable (chatter, "
        "force/displacement grow; resultant force shows nonlinearity when tooth leaves cut). "
        "Time domain agrees with Fourier-series stability; the average-tooth-angle boundary "
        "disagrees for cases 2 and 4 because in slotting μy = 0 so y-direction dynamics are "
        "neglected. Example 4.12 (20% radial immersion down): φs = 126.9°, φe = 180°, fnx = 900 Hz, "
        "fny = 950 Hz, 3-tooth 19 mm, ft = 0.2 mm/tooth, Ks = 2250 N/mm², β = 75° (kt ≈ 2173, "
        "kn ≈ 582), 801 steps/rev, 40 rev. Case 1: Ω = 17,000 rpm, b = 4 mm stable (forced vibration "
        "only, force profile as expected for down milling). Case 2: Ω = 13,000 rpm, b = 4 mm "
        "unstable (chatter near natural frequency plus tooth-passing harmonics). In partial "
        "immersion only one tooth is engaged at a time so force drops to zero between teeth "
        "(interrupted cutting); down milling force is largest at entry where chip thickness is maximum."
    )


def milling_dynamics_summary() -> dict[str, str]:
    """
    All milling-dynamics nutshells for docs or UI.

    Returns:
        Dict with keys: "nutshell", "start_exit_angles", "cutting_force_geometry",
        "cutting_forces_example_4_2", "multiple_teeth_engagement", "tooth_passing_frequency",
        "stability_lobes", "time_domain_simulation", "surf_update_rule", "helical_teeth", "straight_vs_helical_forces". Can be extended with "process_damping", etc.
    """
    return {
        "nutshell": milling_dynamics_nutshell(),
        "start_exit_angles": milling_start_exit_angles_nutshell(),
        "cutting_force_geometry": milling_cutting_force_geometry_nutshell(),
        "cutting_forces_example_4_2": milling_cutting_forces_example_4_2_nutshell(),
        "multiple_teeth_engagement": milling_multiple_teeth_engagement_nutshell(),
        "tooth_passing_frequency": tooth_passing_frequency_nutshell(),
        "stability_lobes": milling_stability_lobes_nutshell(),
        "time_domain_simulation": milling_time_domain_simulation_nutshell(),
        "surf_update_rule": milling_surf_update_rule_nutshell(),
        "helical_teeth": milling_helical_teeth_nutshell(),
        "straight_vs_helical_forces": milling_straight_vs_helical_forces_nutshell(),
        "ball_endmill_simulation": milling_ball_endmill_simulation_nutshell(),
        "experimental_cutting_coefficients": milling_experimental_cutting_coefficients_nutshell(),
        "updated_force_model": milling_updated_force_model_nutshell(),
        "linear_regression_coefficients": milling_linear_regression_coefficients_nutshell(),
        "nonlinear_optimization_coefficients": milling_nonlinear_optimization_coefficients_nutshell(),
        "experimental_techniques": milling_experimental_techniques_nutshell(),
        "chatter_to_stable_strategy": milling_chatter_to_stable_strategy_nutshell(),
        "example_6_2_competing_lobes": milling_example_6_2_competing_lobes_nutshell(),
        "process_damping": milling_process_damping_nutshell(),
        "process_damping_stability": milling_process_damping_stability_nutshell(),
    }
