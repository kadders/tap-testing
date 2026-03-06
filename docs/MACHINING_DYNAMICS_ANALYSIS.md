# Machining Dynamics: Impact Test & Milling — Gaps and Corrections

This document analyzes what is needed to generate **proper graphs for impact-test data analysis** and to apply **machining-dynamics principles** to milling and **ideal tooling parameters**. It aligns with typical “Machining Dynamics” textbook content (FRF measurement, stability lobes, tooth passing, process damping) and the current tap-testing codebase.

**Reference:** For mapping textbook equations (Sect. 4.x) to our **metric** implementation and imperial→metric conversions, see **[docs/METRIC_FORMULAS_REFERENCE.md](METRIC_FORMULAS_REFERENCE.md)**. You can place `Machining Dynamics.pdf` in `docs/` to cross-check equation numbers.

---

## 1. Current State: What the Documentation and Code Provide

### 1.1 Impact test workflow (README, `run_cycle`, `cycle_gui`)

- **Excitation:** Impulse (hammer tap); accelerometer records response only (no force measurement).
- **Data:** CSV with `t_s`, `ax_g`, `ay_g`, `az_g`; optional `# sample_rate_hz`.
- **Analysis:** FFT → **dominant natural frequency** → avoid RPM (tooth-pass resonance) and suggested stable RPM band; optional uncertainty from FFT resolution and tap-to-tap spread.
- **Outputs:** RPM band chart (red = avoid, green = optimal); cycle chart with 3 tap traces + average magnitude vs time + milling guidance panel; standalone milling dynamics text figure; multi-configuration comparison.

### 1.2 Graphs currently produced

| Graph | Purpose | Status |
|-------|---------|--------|
| **RPM band** (avoid / optimal) | Choose spindle speed from f_n and N_teeth | ✅ Implemented |
| **Tap cycles + average** (magnitude vs time) | Visualize repeatability and decay | ✅ Implemented |
| **Milling dynamics panel** | fn, uncertainty, avoid RPM, best lobe speeds, material | ✅ Text panel |
| **Tooling configurations** | Compare multiple tool setups | ✅ Implemented |

### 1.3 What is *not* graphed (gaps for impact-test analysis)

- **Frequency spectrum (FFT magnitude vs frequency)**  
  The dominant natural frequency is *computed* from the FFT but the spectrum is **never plotted**. For proper impact-test analysis we need:
  - A **magnitude spectrum** (one-sided FFT magnitude vs Hz) to verify the peak, spot secondary modes, and check for leakage or noise.
  - Optionally: spectrum from **each tap** and an **average spectrum** (or averaged FFT) to show coherence and reduce noise.

- **Decay / damping visualization**  
  Time traces are plotted, but there is no:
  - Exponential envelope overlay (e.g. decay rate from log-decrement or half-power method).
  - Damping ratio or quality factor in the figure (modal_fit and theory support it, but only when FRF is available).

- **FRF (Frequency Response Function)**  
  - **Receptance (X/F) or inertance (A/F)** is not available in the basic setup because **force is not measured**. The README correctly states: *“We do not measure force in the basic setup, so we get the response spectrum (FFT) and identify the dominant natural frequency; with a force sensor at the tap point we could compute the full FRF.”*
  - **Correction:** Documentation should explicitly list “response-only” vs “full FRF” so users know what graphs are possible with and without a force sensor. The code already has `transducers.inertance_to_receptance`, `measurement_uncertainties` FRF bias corrections, and `modal_fit` for when FRF *is* available.

- **Stability lobe diagram (blim vs Ω)**  
  - **Limiting chip width vs spindle speed** is the central graph in machining dynamics for choosing “ideal” milling parameters. The code implements **best spindle speeds** Ω_best = fn·60/((N+1)·Nt) and **avoid RPM**, but does **not** plot the (Ω, blim) stability boundary.
  - **Reason:** That boundary requires **oriented FRF** (μx·FRFx + μy·FRFy), which in turn needs **x and y FRFs** at the tool point—i.e. impact testing **with force measurement** in two directions (and optionally cross FRFs). `milling_dynamics.py` states this explicitly: *“full stability lobes additionally require x and y direction FRFs (e.g. from impact testing with force measurement).”*
  - So: no error in the doc; the **limitation** (response-only tap test → no blim curve) should be stated clearly where we discuss “ideal parameters” and “stability lobe diagram.”

---

## 2. What Needs to Be Done for Proper Impact-Test Graphs

### 2.1 High priority (needed to analyze impact-test data properly)

1. **Add FFT magnitude spectrum plot**
   - **What:** Plot frequency (Hz) vs magnitude (e.g. one-sided FFT magnitude of the analyzed signal: magnitude of ax, ay, az or combined).
   - **Where:** Option in `analyze` (e.g. `--plot-spectrum`) and/or a fourth subplot or second figure in the cycle result (e.g. “Spectrum” next to “Tap cycles”).
   - **Detail:** Mark the detected dominant natural frequency (and optional uncertainty band) on the spectrum; optionally show spectra for each tap and an average.
   - **Why:** Standard practice in impact-test analysis; validates the chosen f_n and reveals other modes or bad data.

2. **Document “response-only” vs “full FRF”**
   - In README and/or a short “Impact test data” section: clearly state that with the **current hardware** (accelerometer only) we produce:
     - Time-domain response and FFT magnitude spectrum (once implemented),
     - Dominant natural frequency and avoid/suggested RPM,
     but **not** receptance/inertance FRF or stability lobe (blim vs Ω). Add that with an **impact hammer with force transducer** (x and y at tool point) one could compute FRFs and then stability lobes and blim.

### 2.2 Medium priority (improve interpretation)

3. **Optional decay envelope / damping on time plot**
   - Fit exponential envelope to the decay (e.g. magnitude after peak) and optionally report damping ratio or decay rate; display envelope on the tap-traces figure. Useful when moving toward modal identification (e.g. for future FRF-based modal_fit).

4. **Per-tap vs combined spectrum**
   - When multiple taps are used: either plot each tap’s spectrum (lighter) and the average (bold), or document that the combined time series is what is FFT’d. This clarifies how “one” natural frequency is obtained from multiple impacts.

### 2.3 Lower priority (when force is available)

5. **FRF plot (when force is measured)**
   - If force is ever recorded (e.g. same CSV with Fx, Fy or Fz): compute H = response/force in frequency domain and plot magnitude and phase (receptance or inertance). Then `modal_fit` and `transducers` helpers become directly applicable.

6. **Stability lobe diagram (blim vs Ω)**
   - Requires x and y FRFs and cutting parameters (e.g. Ks, β, Nt*). Implement as a separate function/figure when FRF data and material/cutting coefficients are available; document dependency on “impact testing with force measurement.”

---

## 3. Milling and Ideal Tooling Parameters (Applying the PDF Principles)

### 3.1 What the code already applies

- **Tooth passing:** f_tooth = Ω·Nt/60; avoid RPM where f_tooth or harmonics equal f_n (resonance).
- **Best stability lobe speeds:** Ω_best = fn·60/((N+1)·Nt) for N = 0, 1, 2, … (displayed in milling panel and milling dynamics figure).
- **Suggested stable RPM band:** Largest gap between avoid RPMs in [min_rpm, max_rpm].
- **Material:** Workpiece material (e.g. 6061 aluminum) for labels and future use (e.g. cutting coefficients).
- **Uncertainty:** FFT resolution and tap-to-tap spread for f_n; shown in milling guidance and figures.
- **Process damping / cutting coefficients:** Mentioned in guidance text (Sect. 4.8, 4.7); not computed (need cutting tests or coefficients).

### 3.2 What is incomplete for “ideal parameters” (and ties to the PDF)

- **Stability lobe diagram (blim vs Ω)**  
  The “ideal” chip width (or axial depth) vs spindle speed is given by the **stability boundary** blim(Ω). That needs:
  - Oriented FRF (μx·FRFx + μy·FRFy) from **x and y tool-point FRFs** (impact test with force).
  - Cutting force model (e.g. Ks, β) and Nt* from cut geometry.
  - The code has the formulas (`stability_lobe_phase_epsilon_rad`, `directional_factors_*`, `milling_stability_lobes_nutshell`); what’s missing is **measured FRF data** and a plotting path.

- **Radial/axial depth and feed**  
  “Ideal” milling parameters also include:
  - **Radial and axial depth** (below blim for stability),
  - **Feed** from chip load: F = fz·Nt·RPM (already in README and `feed_rate_mm_min`).
  Chip load (fz) comes from material/tool guidelines or cutting tests, not from the tap test; the doc already separates “tap test → RPM” from “feed from chip load.”

- **Process damping**  
  At low spindle speed, the stable zone can increase (Sect. 4.8). The code mentions it in the milling guidance; implementing it in a stability boundary would require a process-damping model and possibly FRF.

- **Cutting coefficients (kt, kn, Ks, β)**  
  Needed for blim and for force prediction; obtained from slotting tests or inverse identification (Sect. 4.7). The code references this; no correction needed, only clarity that “ideal parameters” for **depth vs speed** depend on these when moving beyond “avoid RPM” and “best lobe speeds.”

### 3.3 Recommended documentation corrections/additions

- **README (“From tap test to spindle speed and feed”):**
  - Add one sentence: “With the current setup we do not measure force, so we cannot plot the FRF or the stability lobe diagram (blim vs Ω); we provide natural frequency, avoid RPM, suggested RPM band, and best stability lobe speeds.”
  - When describing “ideal” or “optimal” parameters, specify: (1) **RPM** from tap test (avoid + suggested + best lobe speeds); (2) **depth** from stability lobe when FRF and cutting coefficients are available; (3) **feed** from chip load and RPM.

- **Milling dynamics figure / panel:**
  - Add a short line: “Full stability lobe (blim vs Ω) requires x,y FRFs from impact testing with force measurement and cutting coefficients.”

- **New or updated section “Impact test data and graphs”:**
  - List required graphs for impact-test analysis: (1) Time-domain response (done), (2) **Frequency spectrum (FFT magnitude)** (to be added), (3) Optional: decay envelope; (4) FRF magnitude/phase when force is measured; (5) Stability lobe when FRF + cutting data exist.

---

## 4. Summary: Priority Actions (implementation status)

| Priority | Action | Status |
|----------|--------|--------|
| **P1** | Add **FFT magnitude spectrum** plot (frequency vs magnitude, with f_n marked); expose in analyze and/or cycle figure. | Done: `plot_spectrum_figure`, `--plot-spectrum` / `--plot-spectrum-out`, spectrum subplot in cycle figure. |
| **P1** | Document **response-only vs full FRF** and that **stability lobe (blim vs Ω)** requires force measurement and x,y FRFs. | Done: README "Impact test data and graphs", milling guidance and figure. |
| **P2** | Optionally add **decay envelope** and/or **per-tap vs average spectrum** to improve impact-test interpretation. | Done: `decay_envelope_and_damping`, envelope on tap traces, ζ in title; per-tap + average in spectrum. |
| **P2** | In README and milling panel, spell out “ideal parameters”: RPM (from tap test), depth (from lobes when FRF available), feed (from chip load). | Done: README subsection, `format_milling_guidance_for_cycle`, `plot_milling_dynamics_figure`. |
| **P3** | When force is available: FRF plot; then stability lobe plot (blim vs Ω) using existing formulas and directional factors. | Done: `load_tap_csv_with_force`, `compute_frf_from_impact`, `plot_frf_figure`, `--plot-frf` / `--plot-frf-out`; `compute_stability_lobe_boundary`, `plot_stability_lobe_figure`. |

This keeps the current tap-test workflow correct and complete for **response-only** impact testing, adds the **missing graph** (spectrum) needed for proper impact-test analysis, and aligns documentation with machining-dynamics principles for **milling and ideal tooling parameters** (RPM, depth, feed, and the conditions under which full stability lobes can be generated).
