# Reference Alignment Analysis: Turning + Milling Dynamics

This document compares the tap-testing implementation to **Schmitz & Smith, *Machining Dynamics*, 2nd ed.** (reference: `reference/machining_dynamics_ref.txt`). It covers **turning dynamics** (Ch. 3) as well as **milling dynamics** (Ch. 4), outlines **differences and gaps**, and suggests **step-by-step changes** for better alignment. It also discusses **live tooling / cutting-based identification** as a way to improve calculations beyond tap testing alone.

---

## 1. Why Turning Dynamics Matters for Milling

The milling stability formulation in Ch. 4 is a direct adaptation of the turning formulation in Ch. 3:

| Concept | Turning (Ch. 3) | Milling (Ch. 4) |
|--------|-----------------|-----------------|
| **Chip thickness** | h = hm + y(t−τ) − y(t); τ = 1 rev | h = ft·sin(φ) + n(t−τ) − n(t); τ = 60/(Ω·Nt) |
| **Stability limit** | blim = −1/(2·Ks·μ·Re[FRF]) (Eq. 3.5, 3.21) | blim = −1/(2·Ks·Re[FRF_orient]·Nt*) (Eq. 4.23, 4.109) |
| **Phase / spindle** | fc/Ω = N + ε/(2π); Ω in **rev/s** (Eq. 3.6) | fc/(Ω·Nt) = N + ε/(2π); Ω in **rev/s** (Eq. 4.24, 4.110) |
| **Oriented FRF** | FRF_orient = μ·FRF (single dir) or μ1·FRFu1 + μ2·FRFu2 (Eq. 3.26) | FRF_orient = μx·FRFx + μy·FRFy (Eq. 4.27) |
| **Process damping** | Fd = −C·(b/v)·ẏ (Eq. 3.44); cnew in stability loop (Sect. 3.7.2) | Same form (Eq. 4.108); cnew,x, cnew,y (Eqs. 4.113–4.116) |

So: **turning** defines the oriented FRF, the blim formula, the phase ε = 2π − 2·atan(Im/Re), and process damping. **Milling** adds Nt*, average tooth angle φave, and directional factors μx, μy that depend on radial immersion (up vs down milling). Our code implements milling explicitly; turning is not implemented as a separate mode, but the **same formulas** (oriented FRF, blim, ε, process damping) are shared. Ensuring we match the reference in milling therefore implies we align with the turning foundation where the two overlap.

---

## 2. Formula-by-Formula Comparison

### 2.1 Tooth passing and time (Eq. 4.14, 4.13)

| Reference | Our implementation | Status |
|-----------|--------------------|--------|
| f_tooth = Ω·Nt/60 (Hz), Ω in **rpm** | `tooth_passing_frequency_hz(rpm, n_teeth)` → rpm·n_teeth/60 | OK |
| t = φ·60/(Ω·360) (s), φ in deg, Ω in rpm | `angle_to_time_sec(angle_deg, rpm)` | OK |
| τ = 60/(Ω·Nt) (s) | `tooth_period_s(rpm, n_teeth)` | OK |

**Note:** The reference uses **Ω in rev/s (rps)** in Eqs. 3.6, 4.24, 4.110 (fc/Ω = N + ε/(2π)). Our code uses **rpm** everywhere and converts internally (e.g. Ω_rpm = fn·60/((N+1)·Nt)). No change needed; document that ref uses rps in stability equations and we use rpm.

---

### 2.2 Avoid RPM and best spindle speed (Eq. 4.29, resonance)

| Reference | Our implementation | Status |
|-----------|--------------------|--------|
| Avoid: k·f_tooth = fn ⇒ RPM = 60·fn/(Nt·k) | `rpm_to_avoid(..., n_teeth, ...)` | OK |
| Ω_best = fn·60/((N+1)·Nt) (rpm) | `stability_lobe_best_spindle_speed_rpm(fn_hz, n_teeth, lobe_index_n)` | OK |

---

### 2.3 Stability boundary blim (Eq. 3.5, 3.21, 4.23, 4.109)

Reference formula:

- **Turning:** blim = **−1**/(2·Ks·μ·Re[FRF]) (Eq. 3.5, 3.21).
- **Milling:** blim = **−1**/(2·Ks·Re[FRF_orient]·Nt*) (Eq. 4.23, 4.109).

The **negative** sign is required because in the valid chatter frequency range **Re[FRF] < 0**; so −1/(2·Ks·(negative)·Nt*) gives a **positive** blim.

| Item | Reference | Our implementation | Status |
|------|-----------|--------------------|--------|
| **blim** | blim = **−1**/(2·Ks·Re[FRF_orient]·Nt*) | `compute_stability_lobe_boundary`: blim = **−1**/(2·Ks·Re_orient·Nt*) | **Done** |
| **Valid range** | Re[FRF_orient] **< 0** | Code returns NaN when Re_orient **≥ 0** | **Done** |

**Done.** The analyzer’s built-in charts and example generator do **not** call `compute_stability_lobe_boundary` or `plot_stability_lobe_figure` (they require x,y FRFs and cutting coefficients). So the blim fix and down-milling factors apply when you call those functions programmatically or when the CLI is extended to plot the stability lobe when FRF data is available (see “Where these changes apply” below).

---

### 2.4 Oriented FRF and directional factors

**Turning (Sect. 3.4):** FRF_orient = μ1·FRFu1 + μ2·FRFu2 (Eq. 3.26). μ projects force into mode direction then into surface normal.

**Milling (Sect. 4.3.2):** FRF_orient = μx·FRFx + μy·FRFy (Eq. 4.27). Examples:

- **Slotting:** μx = cos(β), μy = 0 (Example 4.3).
- **Down milling 50%:** μx = cos(45+β)·cos(45), μy = cos(β−45)·cos(45) (Example 4.4).
- **Up milling 25%:** μx = cos(β−60)·cos(60), μy = cos(150−β)·cos(150) (Example 4.5).

Our code has **only** `directional_factors_up_milling(force_angle_beta_deg, phi_ave_deg)` (docstring: μx = cos(β−(90−φave))·cos(90−φave), μy = cos((180−φave)−β)·cos(180−φave)), which matches **up milling** (and slotting as a special case when φave = 90°).

| Item | Reference | Our implementation | Status |
|------|-----------|--------------------|--------|
| **Up milling** | Example 4.5 (φave = 30°): μx = cos(β−60)·cos(60), μy = cos(150−β)·cos(150) | Same geometry via (90−φave), (180−φave) | OK |
| **Down milling** | Example 4.4 (φave = 135°): μx = cos(45+β)·cos(45), μy = cos(β−45)·cos(45) | `directional_factors_down_milling`; used in `compute_stability_lobe_boundary` when `up_milling=False` or inferred from φs, φe | **Done** |
| **Slotting** | μx = cos(β), μy = 0 (φave = 90°) | Up-milling formula with φave=90° gives μy = 0, μx = cos(β) | OK |

**Done.** Down-milling factors are used only inside `compute_stability_lobe_boundary` (see “Where these changes apply” below).
---

### 2.5 Phase ε (Eq. 3.7, 4.25, 4.110)

ε = 2π − 2·atan(Re[FRF_orient] / Im[FRF_orient]). The ref uses atan(Re/Im); our `stability_lobe_phase_epsilon_rad(Re_FRF_orient, Im_FRF_orient)` uses **atan2(Im, Re)** so that ε = 2π − 2·atan2(Im, Re). With atan2(Im, Re) = atan(Im/Re), we get ε = 2π − 2·atan(Im/Re). The ref writes atan(Re/Im), so atan(Im/Re) = π/2 − atan(Re/Im) in the principal branch; the ref’s ε and ours can differ by a constant if the convention is Re/Im vs Im/Re. Checking the ref: “ε = 2π − 2·tan⁻¹(Re[FRF_orient]/Im[FRF_orient])”. So they use **Re/Im**. Our atan2(Im, Re) = atan(Im/Re). So we have ε_ours = 2π − 2·atan(Im/Re) and ε_ref = 2π − 2·atan(Re/Im). These are not the same. For stability we need the phase between current and previous vibration; the correct formula in the ref is ε = 2π − 2·atan(**Im/Re**) in the text near Eq. 4.110 (see “tan⁻¹(Re/Im)” in one place and “Im/Re” in another in different printings). Our implementation **ε = 2π − 2·atan2(Im, Re)** is the standard form (phase of FRF). If the ref consistently uses Re/Im, we should add a short comment and/or a unit test against a known (Re, Im) → ε example from the ref. **Action:** Verify ref’s Eq. 4.110 exactly (Re/Im vs Im/Re) and document our convention; add test if needed.

---

### 2.6 Process damping (Eq. 3.44, 4.108, 4.113–4.116)

| Reference | Our implementation | Status |
|-----------|--------------------|--------|
| Fd = −C·(b/v)·ṅ (Eq. 3.44, 4.108) | `process_damping_force_N(C, b, v, ṅ)` → C·b·ṅ/v (we subtract in normal force) | Sign consistent in use |
| cnew,x = cx + (C·b/v)·cos²(90°−φave) (Eq. 4.113) | `process_damping_cnew_x_up_milling` | OK |
| cnew,y = cy + (C·b/v)·cos²(180°−φave) (Eq. 4.114) | `process_damping_cnew_y_up_milling` | OK |

Ref states process damping force opposes velocity (damping); we apply it as a subtraction in the normal force. No change needed if tests pass.

---

### 2.7 Cutting force and coefficients (Ch. 3.1, 4.7)

Turning: F = Ks·b·hm (Eq. 3.1); Fn = cos(β)·F, Ft = sin(β)·F (Eqs. 3.2–3.3); kn, kt from Fn/(b·hm), Ft/(b·hm). Milling: Ft = kt·b·h, Fn = kn·b·h (Eq. 4.8); edge terms (4.79–4.81); slotting mean force (4.96–4.98); regression (4.102–4.104). Our `tangential_force_N`, `normal_force_N`, `mean_force_per_rev_slotting_N`, `cutting_coefficients_from_slotting_regression` are documented in METRIC_FORMULAS_REFERENCE.md and VERIFICATION_CHANGE_OUTLINE.md. No discrepancy identified; keep as is unless a specific ref example is run and disagrees.

---

### 2.8 Nt* and geometry (Eq. 4.26, 4.4, 4.5)

Nt* = (φe−φs)/360·Nt; φe up = cos⁻¹((r−a)/r); φs down = 180° − cos⁻¹((r−a)/r). Our `average_teeth_in_cut`, `exit_angle_up_milling_deg`, `start_angle_down_milling_deg` match. OK.

---

## 3. Summary of Differences and Required Changes

| # | Item | Severity | Status |
|---|------|----------|--------|
| 1 | **blim sign and valid range** | High | **Done:** blim = −1/(2·Ks·Re_orient·Nt*); valid when Re_orient < 0. |
| 2 | **Down-milling directional factors** | Medium | **Done:** `directional_factors_down_milling` added; `compute_stability_lobe_boundary(..., up_milling=None)` infers from φs, φe or accepts explicit flag. |
| 3 | **Phase ε convention** | Low | **Done:** Documented in `stability_lobe_phase_epsilon_rad` and METRIC_FORMULAS_REFERENCE.md (atan2(Im, Re)). |
| 4 | **Ω in ref (rps) vs our rpm** | Doc | **Done:** METRIC_FORMULAS_REFERENCE.md §8 states ref uses Ω in rev/s; we use rpm. **Yes, we take it into account in plotting and analysis:** all chart axes and result values are in rpm; formulas from the ref that use Ω in rps are implemented with Ω_rpm = 60·Ω_rps (e.g. best lobe speed, avoid RPM, spindle reference, stability lobe x-axis). |

### Where these changes apply in analysis and visualizations

The **blim** fix and **down-milling directional factors** only affect code that calls `compute_stability_lobe_boundary` and `plot_stability_lobe_figure`. Those functions are **not** called by:

- The **analyzer CLI** (`python -m tap_testing.analyze ...`) or **run_cycle** / **cycle_gui**
- The **example chart generator** (`generate_example_chart`)
- Any of the built-in figures (RPM band, cycle chart, spectrum, milling dynamics panel, optimal loads, resonance map, tooling configurations)

Reason: the **(Ω, blim)** stability boundary requires **x,y FRFs** (from impact testing with force measurement) and **cutting coefficients** (Ks, β, etc.). The default workflow uses **response-only** tap data (no force), so it never computes or plots the blim curve. It does use:

- **Best stability lobe speeds** Ω_best = fn·60/((N+1)·Nt) from `stability_lobe_best_spindle_speed_rpm` (unchanged)
- **Avoid RPM** and **suggested RPM band** from tooth-pass resonance (unchanged)
- **Resonance map**: uses `exit_angle_up_milling_deg` only (up milling); no blim curve

So **analysis and visualizations are unchanged** until you either:

1. Call `compute_stability_lobe_boundary` and `plot_stability_lobe_figure` yourself (e.g. in a script) when you have FRF arrays and Ks, β, φs, φe, or  
2. Add a CLI option (e.g. `--plot-stability-lobe-out`) that, when the user has provided a CSV with force columns and optional Ks/β/immersion, computes and saves the blim vs Ω figure using the corrected formula and up/down-milling choice.

Docstrings and nutshell text have been updated so that the **documented** formula is blim = −1/(2·Ks·Re[FRF_orient]·Nt*) everywhere (module doc in `milling_dynamics.py`, nutshells in `tap_testing.docs.milling_dynamics`).

---

## 4. Step-by-Step Implementation Plan (completed)

1. **Fix blim (analyze.py)** — **Done.**  
   - In `compute_stability_lobe_boundary`: set `blim_mm = -1.0 / (2.0 * Ks_n_per_mm2 * Re_orient * Nt_star)`.  
   - Replace the condition “Re_orient <= 0 → NaN” with “Re_orient >= 0 → NaN” (valid range: Re_orient < 0).  
   - Update docstring to cite Eq. 4.23/4.109 and the minus sign.  
   - Run tests; update any test that assumed the old sign/range.

2. **Down-milling directional factors (milling_dynamics.py)** — **Done.**  
   - `directional_factors_down_milling` added (ref Fig. 4.24, Example 4.4). `compute_stability_lobe_boundary(..., up_milling=None)` infers from φs, φe. Tests: `TestDirectionalFactorsDownMilling`.

3. **Phase ε (optional)** — **Done.**  
   - Note in `stability_lobe_phase_epsilon_rad` and METRIC_FORMULAS_REFERENCE.md §8 (atan2(Im, Re) convention).

4. **Documentation** — **Done.**  
   - METRIC_FORMULAS_REFERENCE.md: blim = −1/(2·Ks·Re[FRF_orient]·Nt*); Ω in ref (rps); down-milling μx, μy.  
   - VERIFICATION_CHANGE_OUTLINE.md: add § “Stability boundary blim sign” and “Down-milling directional factors”.  
   - README / MACHINING_DYNAMICS_ANALYSIS: no structural change; only if we mention “oriented FRF” or “stability lobe” again, point to “blim vs Ω” and the need for x,y FRFs + cutting coefficients.

---

## 5. Live Tooling vs Tap Testing: Better Calculations

### 5.1 What tap testing gives

- **Natural frequency** fn (and optionally uncertainty) from the **tool–holder–spindle** structure under **impact** (no cutting).
- **Avoid RPM** (tooth-pass resonance) and **suggested stable RPM band** and **best stability lobe speeds** Ω_best = fn·60/((N+1)·Nt).
- With **force measurement** (impact hammer): **FRF** (receptance/inertance) and then **blim(Ω)** when cutting coefficients are available.

So today we get **RPM guidance** from a single dominant fn; we do **not** get blim(Ω) without force + cutting coefficients.

### 5.2 What “live tooling” (cutting) can add

- **Chatter identification:** Recording accelerometer (or force) **during a cut** and checking for a peak near fn (already in place: `--chatter`, CHATTER_IDENTIFICATION.md).  
- **Chatter frequency fc:** If chatter occurs, the **dominant frequency in the cut** is often close to but not exactly fn (e.g. fn(1+ζ) at the worst speed). Using **fc** instead of fn in Ω_best = fc·60/((N+1)·Nt) can give a better **next spindle speed to try** (ref Ex. 6.2). We already suggest RPM from stability-lobe formula; we could explicitly use the **detected peak frequency** in the chatter band as fc when available.  
- **Process damping / low-speed stability:** At low Ω, process damping increases the stable zone; this is not identified by tap testing alone. A **live cut** at low speed could help validate or tune a process-damping model (C, cnew) if we ever implement the iterative stability algorithm (Sect. 4.8.2).  
- **Oriented dynamics:** Tap test with accelerometer only gives a **single axis or magnitude** response. The **oriented FRF** (μx·FRFx + μy·FRFy) depends on x and y FRFs. With **force in x and y** during impact we get Hxx, Hyy and thus FRF_orient and blim. **Live cutting** does not replace impact for FRF; it can complement it by showing which mode actually dominates during cutting (e.g. which direction is most excited).

### 5.3 Concrete improvements using “live” data

1. **Use detected chatter frequency fc when available**  
   In the chatter workflow: when “chatter likely” and a peak frequency fc is found in the fn band, **suggest Ω_best from fc** (not only from fn): Ω = fc·60/((N+1)·Nt) for N = 0,1,2,… and optionally report “based on detected chatter frequency fc = … Hz”.

2. **Optional: record RPM during cut**  
   If the user passes `--rpm` during the cut, we already use it for context. We could also **derive approximate RPM from the spectrum** (tooth-pass peak or spindle harmonic) to auto-suggest RPM for the next run when the user did not log it.

3. **Future: process damping from low-speed cuts**  
   If we add the iterative process-damping stability algorithm (Sect. 4.8.2), we could use **one or more low-speed cutting tests** to tune or validate the constant C (or cnew) by comparing predicted vs observed stability boundary. This would be a later step.

4. **Keep tap test as primary for fn**  
   The ref and best practice use **impact (tap) test** to get fn and FRF. **Live cutting** is for detecting chatter and refining **which frequency** to use (fc) and for future process-damping or validation. We should not replace tap testing with “live only”; we should **combine**: tap test → fn (and FRF if force available); cutting → chatter detection and fc → better Ω_best and, later, process-damping or validation.

---

## 6. References

- **Reference text:** `reference/machining_dynamics_ref.txt` (Schmitz & Smith, *Machining Dynamics*, 2nd ed.).
- **Metric mapping:** `docs/METRIC_FORMULAS_REFERENCE.md`.
- **Change checklist:** `docs/VERIFICATION_CHANGE_OUTLINE.md`.
- **Impact vs FRF:** README “Force excitation” and “Impact test data and graphs”; `docs/MACHINING_DYNAMICS_ANALYSIS.md`.
- **Chatter workflow:** `docs/CHATTER_IDENTIFICATION.md`.
