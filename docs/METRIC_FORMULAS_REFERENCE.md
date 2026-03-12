# Machining Dynamics: Metric Formula Reference

This document maps **textbook equations** (e.g. from *Machining Dynamics* / Sect. 4.x) to our **metric implementation**. The textbook may use imperial units (IPM, in/tooth, SFM, lb/in²); we use **metric only** (mm/min, mm/tooth, m/s, N/mm²). Use the conversions below when reading the PDF or other US references.

**Reference:** Use `reference/machining_dynamics_ref.txt` (text export of Schmitz & Smith, *Machining Dynamics*, 2nd ed.) to cross-check equation numbers (Sect. 4.1.1, Eq. 4.14, etc.). If verification finds mismatches, see **[VERIFICATION_CHANGE_OUTLINE.md](VERIFICATION_CHANGE_OUTLINE.md)** for what to change.

---

## 1. Feed rate and chip load

| Textbook (often imperial) | Our implementation | Conversion |
|---------------------------|--------------------|------------|
| Feed F = fz × Nt × n (F in **in/min**, fz in **in/tooth**, n in rpm) | `feed_rate_mm_min(chip_load_mm, n_teeth, rpm)` in **mm/min** | fz: **in/tooth × 25.4 = mm/tooth**. F: **IPM × 25.4 = mm/min**. |
| Chip load fz (in/tooth) from tables | Chip load in **mm/tooth** (e.g. default 0.05) | 0.002 in/tooth → 0.0508 mm/tooth |

**Formula (unchanged):** F = fz × N_teeth × RPM. Use consistent units: fz in mm/tooth ⇒ F in mm/min.

---

## 2. Tooth passing and time (angle, frequency, period)

These use **RPM** (revolutions per minute) and **Hz**; the textbook typically uses the same. No unit conversion.

| Textbook | Our function | Notes |
|----------|--------------|--------|
| f_tooth = Ω·Nt/60 (Hz), Ω in rpm | `tooth_passing_frequency_hz(rpm, n_teeth)` | Eq. 4.14. Returns Hz. |
| t = φ·60/(Ω·360) (s), φ in deg, Ω in rpm | `angle_to_time_sec(angle_deg, rpm)` | Angle (deg) to time (s). |
| τ = 60/(Ω·Nt) (s), tooth period | `tooth_period_s(rpm, n_teeth)` | Ω in rpm. |
| dt = 60/(steps_rev·Ω) (s) | `simulation_time_step_s(steps_per_rev, rpm)` | Simulation time step. |

### RPS vs RPM (ref vs code)

The textbook uses **Ω in rev/s (rps)** in Eqs. 3.6, 4.24, 4.110 (e.g. fc/Ω = N + ε/(2π)). We use **rpm** everywhere in the API and in saved values.

| Conversion | Formula | Use in code |
|------------|---------|-------------|
| **rpm → rev/s (Hz)** | Ω_rps = Ω_rpm / 60 | `spindle_hz = reference_rpm / 60.0`; compare to `spindle_operating_frequency_hz` |
| **rev/s (Hz) → rpm** | Ω_rpm = Ω_rps × 60 | `config.rpm_from_spindle_frequency_hz(freq_hz)`; `base_rpm = spindle_freq_hz * 60.0` |

**Verification (end values):**

- **Tooth passing:** f_tooth (Hz) = rpm × Nt / 60 → e.g. 7500 rpm, 4 teeth → 500 Hz ✓  
- **Avoid RPM:** rpm = 60×fn/(Nt×k) → e.g. fn=500 Hz, Nt=4, k=1 → 7500 rpm ✓  
- **Best lobe speed:** Ω_rpm = fn×60/((N+1)×Nt) → e.g. fn=500 Hz, N=0, Nt=4 → 7500 rpm ✓ (ref Eq. 4.29; ref uses Ω_rps = fn/((N+1)×Nt) then ×60 for rpm)  
- **Spindle reference:** 400 Hz (rev/s) → 400×60 = 24000 rpm ✓

---

## 3. Stability lobe speeds (avoid RPM and best speeds)

Same formula in both systems (RPM and Hz). No conversion.

| Textbook | Our function | Notes |
|----------|--------------|--------|
| RPM_avoid: k·f_tooth = fn ⇒ RPM = 60·fn/(Nt·k) | `rpm_to_avoid(natural_freq_hz, n_teeth, ...)` in `analyze.py` | k = 1, 2, 3, … |
| Ω_best = fn·60/((N+1)·Nt) for lobe N | `stability_lobe_best_spindle_speed_rpm(fn_hz, n_teeth, lobe_index_n)` | Eq. 4.29. |

---

## 4. Cutting speed (surface speed)

| Textbook (often imperial) | Our implementation | Conversion |
|---------------------------|--------------------|------------|
| v = π·D·n (D in **in**, n in rpm → v in **in/min**) or SFM = π·D·n/12 (ft/min) | `cutting_speed_m_per_s(diameter_m, rpm)` → **m/s** | **Diameter:** in → m: × 0.0254 (or ÷ 39.37). **SFM → m/s:** SFM × 0.00508 = m/s. |
| Formula: v = π·D·Ω/60 | Same: v = π·diameter_m·rpm/60 | With D in **metres**, result is m/s. |

---

## 5. Geometry (angles, immersion, depth)

Angles are in **degrees**; lengths in the textbook may be in inches. We use **mm** everywhere.

| Textbook | Our function | Conversion |
|----------|--------------|------------|
| φe = cos⁻¹((r−a)/r), up milling | `exit_angle_up_milling_deg(radius_mm, radial_depth_mm)` | r, a in **mm**. |
| φs = 180° − cos⁻¹((r−a)/r), down milling | `start_angle_down_milling_deg(radius_mm, radial_depth_mm)` | r, a in **mm**. |
| Radial immersion % = 100·(radial depth)/diameter | `radial_immersion_percent(radial_depth_mm, diameter_mm)` | Use **mm**. in → mm: × 25.4. |
| Nt* = (φe−φs)/360·Nt | `average_teeth_in_cut(phi_s_deg, phi_e_deg, n_teeth)` | Angles in deg. |
| Lag angle χ = 2·b·tan(γ)/d (rad → deg) | `helical_lag_angle_deg(helix_angle_deg, axial_depth_mm, diameter_mm)` | b, d in **mm**. |
| Constant-force depth b = d·φp/(2·tan(γ)) | `constant_force_axial_depth_mm(diameter_mm, tooth_pitch_deg, helix_angle_deg)` | **mm**. |

---

## 6. Cutting force coefficients and forces

Textbooks often give **kt, kn** in **lb/in²** (psi) or N/mm². We use **N/mm²** and **N/mm** for edge terms.

| Textbook | Our implementation | Conversion |
|----------|--------------------|------------|
| Ft = kt·b·h, Fn = kn·b·h (Eq. 4.8) | `tangential_force_N`, `normal_force_N` | kt, kn in **N/mm²**; b, h in **mm**. **psi → N/mm²:** × 0.006895. |
| Ft = kt·b·h + kte·b, Fn = kn·b·h + kne·b (Eqs. 4.79–4.81) | `tangential_force_with_edge_N`, `normal_force_with_edge_N` | kte, kne in **N/mm**. lb/in → N/mm: × 0.1751. |
| Mean force slotting (Eqs. 4.96–4.98) | `mean_force_per_rev_slotting_N(...)` | All lengths in **mm**, forces in **N**. |
| Regression slopes/intercepts → kn, kt, kte, kne (Eqs. 4.102–4.104) | `cutting_coefficients_from_slotting_regression(...)` | Input slopes/intercepts from F vs **ft in mm**; outputs N/mm², N/mm. |

---

## 7. Process damping

| Textbook | Our implementation | Conversion |
|----------|--------------------|------------|
| Fd = C·b·ṅ/v (Eq. 4.108) | `process_damping_force_N(..., cutting_speed_mm_s, velocity_normal_mm_s)` | Use **mm/s** for v and ṅ; C in N·s/mm². |
| cnew,x = cx + (C·b/v)·cos²(90°−φave) (Eqs. 4.113–4.116) | `process_damping_cnew_x_up_milling`, `process_damping_cnew_y_up_milling` | **SI:** C in N/m, b in **m**, v in **m/s**. So convert b: mm → m (÷ 1000); v: use `cutting_speed_m_per_s`. |

---

## 8. Stability lobe boundary (blim)

| Textbook | Our implementation | Conversion |
|----------|--------------------|------------|
| blim = −1/(2·Ks·Re[FRF_orient]·Nt*) (valid when Re[FRF_orient] < 0) | `compute_stability_lobe_boundary` in `analyze.py` | Ks in **N/mm²**; FRF in consistent length/force units (e.g. mm/N); blim in **mm**. |
| Ks, β from material/tests | Cutting coefficients | If Ks given in lb/in², × 0.006895 → N/mm². |
| Oriented FRF: μx, μy from β and φave (up milling Fig. 4.74; down milling Fig. 4.24, Ex. 4.4) | `directional_factors_up_milling`, `directional_factors_down_milling` in `milling_dynamics.py` | Use `up_milling=True/False` or infer from φs, φe in `compute_stability_lobe_boundary`. |

**Conventions (ref vs code):**

- **Spindle speed in ref:** Eqs. 3.6, 4.24, 4.110 use **Ω in rev/s (rps)**. We use **rpm** everywhere; Ω_rpm = 60·Ω_rps (e.g. best speed fn·60/((N+1)·Nt) in rpm).
- **Phase ε (Eq. 4.110):** We use ε = 2π − 2·atan2(Im, Re) (argument of FRF). The textbook may write tan⁻¹(Re/Im) or tan⁻¹(Im/Re); the relationship is equivalent modulo quadrant. See `stability_lobe_phase_epsilon_rad` in `milling_dynamics.py`.

---

## 9. Quick conversion table (imperial → metric)

| Quantity | Imperial | To metric |
|----------|----------|-----------|
| Length / depth / diameter | in | × 25.4 → mm |
| Length (for cutting speed in m/s) | in | × 0.0254 → m |
| Feed rate | IPM (in/min) | × 25.4 → mm/min |
| Chip load (feed per tooth) | in/tooth | × 25.4 → mm/tooth |
| Cutting speed | SFM (ft/min) | × 0.00508 → m/s |
| Force | lb | × 4.448 → N |
| Pressure / specific force | psi (lb/in²) | × 0.006895 → N/mm² |
| Force per width | lb/in | × 0.1751 → N/mm |
| Power | HP | × 745.7 → W |
| Volume rate (for K factor) | in³/(min·HP) | × 21.97 ≈ mm³/(min·W); unit_power = 1/K |

All formulas in `tap_testing.milling_dynamics` and `tap_testing.analyze` use the **metric** units above. When the textbook (e.g. *Machining Dynamics.pdf*) shows an equation in imperial, apply these conversions to the **inputs** before using our functions, or use the metric form listed in this doc.

---

## 10. Tool optimization (feeds & speeds, practical)

These formulas support **practical** feeds and speeds selection (chipload, stepover, MRR, power). Source: [Shapeoko CNC A–Z: Feeds & speeds](https://shapeokoenthusiasts.gitbook.io/shapeoko-cnc-a-to-z/feeds-and-speeds-basics). Implemented in `tap_testing.feeds_speeds`.

| Concept | Formula / reference | Our function | Notes |
|--------|----------------------|--------------|--------|
| **Chip thinning** | Chipload_adjusted = (D / (2√(D·S − S²))) × Chipload | `chipload_adjusted_for_stepover_mm(diameter_mm, stepover_mm, chipload_nominal_mm)` | Use when stepover &lt; 50% of diameter so effective chip thickness is correct. |
| **Feed from chipload** | F = fz × Nt × RPM | `feed_rate_for_chipload_mm_min(chipload_mm, n_teeth, rpm)` | Same as `analyze.feed_rate_mm_min`. |
| **Chipload from feed** | fz = F / (Nt × RPM) | `chipload_from_feed_mm_per_tooth(feed_mm_min, n_teeth, rpm)` | |
| **Tool engagement angle (TEA)** | TEA = cos⁻¹(1 − stepover/(0.5×D)) | `tool_engagement_angle_deg(diameter_mm, stepover_mm)` | 50% stepover → 90°; slotting → 180°. |
| **Material removal rate** | MRR = WOC × DOC × Feed | `mrr_mm3_per_min(width_of_cut_mm, depth_of_cut_mm, feed_mm_min)` | Result in **mm³/min**. |
| **SFM from RPM** | SFM = π·D_in·RPM/12 | `sfm_from_rpm(rpm, diameter_mm)` | Input D in **mm**; output **ft/min**. |
| **RPM from SFM** | RPM = SFM/(0.262×D_in) | `rpm_from_sfm(sfm_ft_per_min, diameter_mm)` | |
| **Cutting power** | Power = MRR × UnitPower | `cutting_power_w(mrr_mm3_per_min, unit_power_w_min_per_mm3)` | UnitPower in **W·min/mm³**; use `UNIT_POWER_W_MIN_PER_MM3[material]` for common materials. |
| **Cutting torque** | T = Power/ω, ω = 2π·RPM/60 | `cutting_torque_nm(power_w, rpm)` | Result in **N·m**. |
| **Cutting force** | F = T / radius | `cutting_force_n(torque_nm, radius_mm)` | Tangential force in **N**. |
| **Good/bad zones** | Chip evac: fz ≥ min; Spindle: power ≤ limit | `cutting_zone_status(chipload_target_mm, power_w, spindle_power_w, min_chipload_mm)` | Returns chip_evac_ok, spindle_ok, label. Default spindle **1500 W** (1.5 kW); default min chipload **0.025 mm** (~0.001") for chip evacuation. |

Suggested workflow: choose target chipload and RPM (e.g. from stability lobes); if stepover &lt; 50%, compute adjusted chipload and feed; compute MRR and optionally power/force to check machine limits. Use **cutting_zone_status** (and the stepover chart) to see OK / Low fz (chip evacuation) / Overload (spindle) zones.
