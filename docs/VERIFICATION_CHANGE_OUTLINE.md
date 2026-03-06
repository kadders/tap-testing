# Verification Against Machining Dynamics Reference — Change Outline

This document outlines **what to change** in the tap-testing codebase if verification against `reference/machining_dynamics_ref.txt` (Schmitz & Smith, *Machining Dynamics*, 2nd ed.) finds formulas, units, or conventions that do not match. Use it as a checklist when running verification and when applying fixes.

**Reference text file:** `reference/machining_dynamics_ref.txt`  
**Code:** `tap_testing/milling_dynamics.py`, `tap_testing/analyze.py`  
**Metric reference:** `docs/METRIC_FORMULAS_REFERENCE.md`

---

## 1. Feed per tooth and linear feed (Eq. 4.3)

| Item | Reference (ref text) | Our implementation | If wrong, change |
|------|----------------------|--------------------|------------------|
| **Formula** | ft = f / (Ω·Nt); f = ft·Ω·Nt. Units: mm/tooth, rpm, mm/min, teeth/rev. | `feed_rate_mm_min(chip_load_mm, n_teeth, rpm)` → F = chip_load_mm × n_teeth × rpm (mm/min) | In `analyze.py`: adjust `feed_rate_mm_min` body or docstring. Ensure chip_load is **per-tooth** and RPM is **per minute**. |
| **Convention** | f = linear feed, ft = feed per tooth, Ω = spindle speed (rpm), Nt = number of teeth. | Same: F = fz×Nt×RPM. | If ref uses rev/s instead of rpm: multiply by 60 in the formula or document conversion. |
| **Units** | Ref states: mm/tooth, rpm, mm/min. | We use mm/tooth, rpm, mm/min. | If ref ever uses in/min or in/tooth in a later chapter, keep our inputs in metric and document conversion in METRIC_FORMULAS_REFERENCE.md. |

**Files:** `tap_testing/analyze.py` (`feed_rate_mm_min`), `docs/METRIC_FORMULAS_REFERENCE.md`.

---

## 2. Tooth passing frequency (Eq. 4.14)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Formula** | f_tooth = Ω·Nt/60 (Hz), Ω in rpm. | `tooth_passing_frequency_hz(rpm, n_teeth)` → rpm * n_teeth / 60.0 | In `milling_dynamics.py`: fix `tooth_passing_frequency_hz`. If ref used rev/s: f = Ω_rps·Nt (no 60); then we’d need to accept rpm and divide by 60. |
| **RPM vs RPS** | Ref: “spindle speed is again expressed in rpm”. Example: 7500·4/60 = 500 Hz. | We use rpm and divide by 60. | If you find Ω in rev/s elsewhere: add a clear comment and/or a branch (e.g. parameter `rpm=True`). |

**Files:** `tap_testing/milling_dynamics.py` (`tooth_passing_frequency_hz`), `tap_testing/analyze.py` (uses it for avoid RPM).

---

## 3. Avoid RPM (resonance: k·f_tooth = fn)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Formula** | Avoid when tooth-pass harmonic equals natural frequency: f_tooth·k = fn ⇒ RPM = 60·fn/(Nt·k). | `rpm_to_avoid(natural_freq_hz, n_teeth, harmonic_orders, rpm_min, rpm_max)` → rpm = 60.0 * natural_freq_hz / (n_teeth * k) | In `analyze.py`: fix `rpm_to_avoid` formula or bounds. |
| **Indexing** | k = 1, 2, 3, … (harmonic order). | We use k = 1, 2, …, harmonic_orders. | If ref uses 0-based or different harmonic numbering, adjust loop and/or docstring. |

**Files:** `tap_testing/analyze.py` (`rpm_to_avoid`).

---

## 4. Best spindle speed for stability lobe N (Eq. 4.29)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Formula** | Ωbest = fn·60 / ((N+1)·Nt) (rpm). Example: fn=1000, N=0, Nt=4 → 15000 rpm. | `stability_lobe_best_spindle_speed_rpm(fn_hz, n_teeth, lobe_index_n)` → fn_hz * 60.0 / ((lobe_index_n + 1) * n_teeth) | In `milling_dynamics.py`: fix `stability_lobe_best_spindle_speed_rpm`. |
| **N convention** | N = 0, 1, 2, … (lobe index). Ref also uses j in Eq. 4.59 for same role. | lobe_index_n = 0, 1, 2, … | If ref defines N differently (e.g. 1-based), adjust so our lobe_index_n matches ref’s N. |

**Files:** `tap_testing/milling_dynamics.py` (`stability_lobe_best_spindle_speed_rpm`), `tap_testing/analyze.py` (calls it for “best lobe speeds”).

---

## 5. Angle and time (Sect. 4.1; Eq. 4.14 context)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Angle to time** | t = φ·60/(Ω·360) with φ in deg, Ω in rpm; tooth period 0.002 s for 7500 rpm, 4 teeth → 500 Hz. | `angle_to_time_sec(angle_deg, rpm)` → angle_deg * 60.0 / (rpm * 360.0) | In `milling_dynamics.py`: fix `angle_to_time_sec`. |
| **Tooth period** | τ = 60/(Ω·Nt) (s). | `tooth_period_s(rpm, n_teeth)` → 60.0 / (rpm * n_teeth) | In `milling_dynamics.py`: fix `tooth_period_s`. |
| **Simulation dt** | dt = 60/(steps_rev·Ω). | `simulation_time_step_s(steps_per_rev, rpm)` → 60.0 / (steps_per_rev * rpm) | In `milling_dynamics.py`: fix `simulation_time_step_s`. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 6. Start/exit angles (Eqs. 4.4, 4.5)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Up milling** | φe = cos⁻¹((r−a)/r); φs = 0. | `exit_angle_up_milling_deg(radius_mm, radial_depth_mm)` | In `milling_dynamics.py`: fix formula or argument order (r, a). |
| **Down milling** | φs = 180° − cos⁻¹((r−a)/r); φe = 180°. | `start_angle_down_milling_deg(radius_mm, radial_depth_mm)` | Same file: fix formula. |
| **Units** | Ref examples in mm (e.g. 1.9 mm, 9.5 mm). | We use radius_mm, radial_depth_mm. | If ref ever uses inches, keep our API in mm; document conversion. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 7. Cutting force (Eqs. 4.7, 4.8) and coefficients

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Ft, Fn** | Ft = kt·b·h, Fn = kn·b·h. Ref examples: kt = 700 N/mm², kn = 210 N/mm², b in mm, ft in mm/tooth. | `tangential_force_N`, `normal_force_N` (kt, kn in N/mm²; b, h in mm) | In `milling_dynamics.py`: fix formulas or units in docstrings/implementation. |
| **Edge terms (4.79–4.81)** | Ft = kt·b·h + kte·b, etc. | `tangential_force_with_edge_N`, etc. | Same file: align coefficients and formula. |
| **Slotting mean force (4.96–4.98)** | Fx_mean, Fy_mean, Fz_mean vs ft. | `mean_force_per_rev_slotting_N` | Same file: fix coefficients or regression extraction. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 8. Cutting speed (for process damping, etc.)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Formula** | v = π·D·n; ref uses metric (mm, then m/s where needed). | `cutting_speed_m_per_s(diameter_m, rpm)` → π·diameter_m·rpm/60 (m/s) | In `milling_dynamics.py`: fix `cutting_speed_m_per_s`. If ref gives D in mm, we expect diameter_m in **metres**; document. |

**Files:** `tap_testing/milling_dynamics.py`, `docs/METRIC_FORMULAS_REFERENCE.md`.

---

## 9. Process damping (Sect. 4.8; Eqs. 4.108, 4.113–4.116)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Fd = C·b·ṅ/v** | Process damping force; C, b, v, ṅ in consistent units. | `process_damping_force_N(..., cutting_speed_mm_s, velocity_normal_mm_s)` (mm/s) | In `milling_dynamics.py`: fix units or formula. |
| **cnew,x, cnew,y** | cnew = c + (C·b/v)·cos²(…); ref may use SI (m, m/s). | `process_damping_cnew_x_up_milling`, `process_damping_cnew_y_up_milling` (axial_depth_m, cutting_speed_m_per_s) | Same file: align SI (m, m/s) with ref. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 10. Stability boundary blim (Sect. 4.3)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **blim formula** | blim = 1/(2·Ks·Re[FRF_orient]·Nt*); blim in mm. | `compute_stability_lobe_boundary` in `analyze.py`; blim_mm. | In `analyze.py`: fix formula, sign of Re[FRF], or Nt* definition. |
| **Nt*** | Nt* = (φe−φs)/360·Nt (average teeth in cut). | `average_teeth_in_cut(phi_s_deg, phi_e_deg, n_teeth)` | In `milling_dynamics.py`: fix if ref defines Nt* differently. |

**Files:** `tap_testing/analyze.py`, `tap_testing/milling_dynamics.py`.

---

## 11. Regenerative chip thickness and simulation (Sect. 4.4)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **h(t)** | h = ft·sin(φ) + n(t−τ) − n(t); τ = 60/(Ω·Nt). | `chip_thickness_regenerative_mm(feed_per_tooth_mm, phi_rad, n_current_mm, n_previous_mm)` | In `milling_dynamics.py`: fix regenerative formula or surf update. |
| **Steps per rev** | Integer steps per tooth; steps_rev/Nt integer. | `simulation_steps_rev`, `simulation_time_step_s` | Same file: ensure dt and angle increment match ref. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 12. Helical lag and constant-force depth (Sect. 4.5; Eqs. 4.68, 4.69)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Lag angle** | χ = 2·b·tan(γ)/d (rad → deg). | `helical_lag_angle_deg(helix_angle_deg, axial_depth_mm, diameter_mm)` | In `milling_dynamics.py`: fix formula or argument order. |
| **Constant-force b** | b = d·φp/(2·tan(γ)). | `constant_force_axial_depth_mm(diameter_mm, tooth_pitch_deg, helix_angle_deg)` | Same file: fix formula. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 13. Ball endmill (Sect. 4.6; Eqs. 4.70–4.78)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **dz, κ′, θ0, db** | Eqs. 4.70–4.73. | `ball_slice_width_dz_mm`, `ball_kappa_prime_rad`, `ball_theta0_rad`, `ball_chip_width_db_mm` | In `milling_dynamics.py`: align each with ref equation. |
| **Forces Fx,Fy,Fz** | Eq. 4.77, 4.78. | `ball_forces_xyz_N` | Same file: fix projection matrix or inputs. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 14. Linear regression for coefficients (Sect. 4.7; Eqs. 4.99–4.104)

| Item | Reference | Our implementation | If wrong, change |
|------|-----------|--------------------|------------------|
| **Slope/intercept** | F_mean = a0 + a1·ft; regression for kn, kne, kt, kte, ka, kae. | `linear_regression_slope_intercept`, `cutting_coefficients_from_slotting_regression` | In `milling_dynamics.py`: fix mapping from slopes/intercepts to coefficients. |

**Files:** `tap_testing/milling_dynamics.py`.

---

## 15. Tests to update after any formula change

| Change in | Update tests in |
|-----------|------------------|
| `feed_rate_mm_min` | `tests/test_analyze.py` (TestFeedRateMmMin) |
| `rpm_to_avoid` | `tests/test_analyze.py` (TestRpmToAvoid), `tests/test_milling_dynamics.py` |
| `tooth_passing_frequency_hz`, `stability_lobe_best_spindle_speed_rpm` | `tests/test_milling_dynamics.py` (TestStabilityLobeBestSpindleSpeedRpm, tooth period tests) |
| `angle_to_time_sec`, `tooth_period_s`, `simulation_time_step_s` | `tests/test_milling_dynamics.py` |
| Exit/start angles | `tests/test_milling_dynamics.py` (if such tests exist) |
| Cutting force, ball, process damping, blim | `tests/test_milling_dynamics.py`, `tests/test_analyze.py` (as applicable) |

After editing any formula, run:

```bash
pytest tests/ -v
```

---

## 16. Documentation to update after changes

- **METRIC_FORMULAS_REFERENCE.md** — Update the equation mapping and conversion table if you change units or formula form.
- **README.md** — “Units and reference formulas” and “From tap test to spindle speed and feed” if feed/RPM/avoid logic changes.
- **Module docstrings** in `milling_dynamics.py` and `analyze.py` — Keep equation numbers and units in sync with the ref and METRIC_FORMULAS_REFERENCE.md.

---

## Summary

- The reference text uses **metric** (mm, mm/tooth, mm/min, N/mm²) in Chapter 4; our code is intended to match.
- If verification finds a **formula mismatch**: locate the row above, apply the change in the listed file(s), then update the tests and docs as in §§15–16.
- If the ref uses **different indexing** (e.g. N vs j, or 1-based lobe index): adjust our parameters and docstrings so our convention is clearly documented and consistent with the ref.
- If the ref introduces **imperial** in a section we use: do not switch our API to imperial; add conversion in METRIC_FORMULAS_REFERENCE.md and docstrings.
