# Analysis Workflow: Optimal Feeds and Speeds from Tap Test

This document defines the **recommended order** to work through tap-test data so you get accurate visualizations and optimal feeds/speeds for the tool under test. The order is chosen for **efficiency** (minimal redundant work, clear dependencies) and **accuracy** (validate data quality first, then add FRF when available, then cutting geometry).

---

## 1. Data and Dependency Overview

| Output | Depends on | Purpose |
|--------|------------|--------|
| **Time signal map** | Tap CSV (t, ax, ay, az) | Verify clean impact, decay, no clipping; qualitative ζ |
| **FFT spectrum** | Tap CSV + analyzed fn | Confirm dominant peak at f_n; spot other modes |
| **FRF** (optional) | Tap CSV **with force** (Fx_N, Fy_N) | Full frequency response → enables blim(Ω) |
| **RPM chart** | f_n, flutes, min/max RPM | Avoid bands + suggested range + best lobe markers |
| **Optimal loads** | f_n, flutes, chip load | Feed (mm/min) at lobe speeds and suggested range |
| **Resonance map** | f_n, flutes, diameter, optional helix | RPM vs axial depth; radial immersion table; constant-force depth |
| **Milling dynamics** | f_n, flutes, material | Text summary: fn, avoid, lobes, tooth-pass formula |
| **Stability lobe blim(Ω)** (optional) | **FRF** + Ks, β, immersion | Limiting depth vs speed (when force was measured) |

**Key point:** Start with **FRF when available** (force measured); then all other outputs use the same tap data and tool parameters. When force is *not* measured, we still get f_n from the response FFT and can produce RPM guidance, feeds, and resonance map—but not the full blim(Ω) curve.

---

## 2. Recommended Work Order (Most Efficient)

Use this order so each step builds on the previous and you don’t repeat work.

### Step 0: Tool and test parameters (before or with analysis)

- **Flutes** (e.g. 3)
- **Tool diameter** (mm) — for resonance map and labels
- **Helix angle** (deg) — for constant-force axial depth and cutting behavior
- **Chip load** (mm/tooth) — for feed calculations
- **Spindle operating frequency** (Hz) — rev/s of your spindle; default 400 Hz → 24000 rpm. Used as the **reference operating point** on charts when you don’t set `--reference-rpm`.
- **Reference RPM** (optional) — e.g. 18k to mark a specific speed on charts and in text. If omitted, the default reference is **spindle frequency × 60** (e.g. 400 Hz → 24000 rpm).
- **Material** — for labels and future cutting coefficients

### Step 1: Data quality (time domain)

1. **Time signal map** — magnitude and x/y/z vs time, decay envelope, Tn markers.
   - **Why first:** Confirms the tap is clean, decay is visible, and axes look reasonable before trusting f_n.
   - **CLI:** `--plot-time-signal` / `--plot-time-signal-out`

### Step 2: Identify dynamics (frequency domain)

2. **FFT spectrum** — frequency vs magnitude with f_n (and uncertainty) marked.
   - **Why next:** Validates the dominant natural frequency and reveals other modes or noise.
   - **CLI:** `--plot-spectrum` / `--plot-spectrum-out`

### Step 3: FRF (only when force is measured)

3. **FRF** — magnitude (and phase) of response/force.
   - **Why here:** If your CSV has force columns (Fx_N, Fy_N or F_N), compute FRF right after spectrum so you have the full dynamics. This is the foundation for a **stability lobe (blim vs Ω)** if you later add cutting coefficients (Ks, β) and immersion.
   - **CLI:** `--plot-frf` / `--plot-frf-out`
   - **Note:** Without force, skip this step; RPM and feed guidance still come from f_n.

### Step 4: RPM and feed guidance (no FRF required)

4. **RPM chart** — avoid bands (red), suggested range (green), **practical at spindle** band (dark green, when spindle frequency is set), best lobe speeds (blue), reference RPM (purple).
5. **Optimal loads** — RPM bands (with practical-at-spindle band), **feed vs RPM** (chip-load bands by tooling), and feed table. When spindle operating frequency is set (e.g. 400 Hz), optimal speeds are **spindle-aligned** (rpm = spindle_freq × 60 / k: 24000, 12000, 8000, … rpm) so they line up with what the spindle can run; thin dashed lines show theoretical lobe (N) positions for comparison.
6. **Resonance map** — RPM vs axial depth (avoid/suggested), radial immersion table (Nt*, φs, φe), constant-force depth line if `--helix` set.

   - **Why together:** They all use the same inputs (f_n, flutes, diameter, helix, chip load). Viewing them in sequence gives: which RPMs to avoid → which speeds are best → how depth and width (immersion) interact with resonance.

   - **CLI:** Default save for RPM + optimal loads; add `--plot-resonance-map`, `--helix`, `--max-depth` for the resonance map.

### Step 5: Summary and (optional) full stability lobe

7. **Milling dynamics** — one-panel text: fn, avoid RPM, suggested range, best lobe speeds, tooth-pass formula, material.
8. **Stability lobe blim(Ω)** — only when **FRF + cutting coefficients (Ks, β) + radial immersion** are available; plot limiting chip width vs spindle speed.

   - **CLI:** `--plot-milling-dynamics`; blim(Ω) is not yet wired to a single CSV run in the CLI (use `compute_stability_lobe_boundary` + `plot_stability_lobe_figure` with your FRF and cutting data).

---

## 3. Single-Command Workflow

To run the **recommended order** in one go (all visualizations, FRF when force exists):

```bash
python -m tap_testing.analyze data/cycle/<run_id>/combined.csv \
  --flutes 3 \
  --tool-diameter 6 \
  --helix 45 \
  --chip-load 0.05 \
  --spindle-frequency 400 \
  --material "6061 aluminum" \
  --workflow
```

Use `--reference-rpm 18000` to force a specific reference RPM; otherwise the reference is taken from `--spindle-frequency` (default 400 Hz → 24000 rpm).

`--workflow` turns on all plot outputs and generates them in this order:

1. Time signal map  
2. Spectrum  
3. FRF (if force columns present)  
4. RPM chart + optimal loads + resonance map  
5. Milling dynamics summary  

Files are written next to the CSV (e.g. `combined_time_signal.png`, `combined_spectrum.png`, …). Use `--no-save-chart` to skip saving the default RPM/optimal-loads only; with `--workflow` you still get the workflow figures.

---

## 4. Spindle Operating Frequency and Reference RPM (Visualizations)

The **spindle operating frequency** (default 400 Hz = 24000 rpm) is stored in config and can be overridden per run. It is used as the **reference operating point** on RPM-related charts when you do not set `--reference-rpm` explicitly.

### Where the reference appears (analytic side)

| Visualization | Reference RPM effect |
|---------------|----------------------|
| **RPM band chart** | Purple vertical line at reference RPM with label and feed (mm/min) at that speed; legend entry “Your ref” / “Spindle ref”. |
| **Optimal loads** (feed table) | Same purple line on the bands subplot; table row “ref” with RPM, feed, tooth-pass frequency. |
| **Cycle result figure** (run_cycle / cycle_gui) | Same purple “Spindle ref” line on the RPM band panel. |
| **Text output** (analyze CLI) | “Reference operating point: … RPM … feed …” and whether it lies in suggested range or avoid band. |
| **Resonance map** | No reference line (RPM vs depth only; ref could be added later). |
| **Milling dynamics panel** | Text only (fn, avoid, lobes); no reference line. |
| **Stability lobe blim(Ω)** | Optional reference line when `reference_rpm` is passed to the plot function. |

So: **RPM chart**, **optimal loads**, and **cycle result** all show your default (or chosen) operating point; resonance map and milling dynamics text do not draw the ref line.

### CLI options (all three entry points)

- **analyze:** `--spindle-frequency HZ` (default from config), `--reference-rpm RPM` (overrides default ref), `--reference-chip-load MM`.
- **run_cycle:** `--spindle-frequency HZ`; reference for the saved chart is spindle_frequency × 60 unless you add analysis with `--reference-rpm` later.
- **cycle_gui:** `--spindle-frequency HZ`; same as run_cycle for the embedded chart.

Config default: `spindle_operating_frequency_hz` (env: `TAP_SPINDLE_OPERATING_FREQUENCY_HZ`). The **dataset** (e.g. `TapTestResult`) stores `spindle_operating_frequency_hz` when provided so re-plots and reports can use it without re-passing.

---

## 5. Factors That Affect Cutting (Included in This Workflow)

- **Natural frequency f_n** — from tap test; drives avoid RPM, best lobe speeds, and (with FRF) blim(Ω).
- **Flutes (Nt)** — tooth-pass frequency and lobe formula; feed = fz × Nt × RPM.
- **Helix angle** — constant-force axial depth b = d·φp/(2·tan(γ)); shown on resonance map when `--helix` is set.
- **Tool diameter** — for radial immersion (width) and constant-force depth.
- **Radial immersion (width of cut)** — Nt*, φs, φe in resonance map table; affects directional factors when FRF is used for blim.
- **Chip load (fz)** — with Nt and RPM gives feed (mm/min); used in optimal loads and reference point.
- **Material** — labels and (when available) cutting coefficients for blim(Ω).

---

## 6. Flow Diagram (Conceptual)

```
Tap CSV (t, ax, ay, az [, Fx, Fy ])
         │
         ▼
   ┌─────────────┐
   │ Analyze FFT │ ──► f_n ± uncertainty, magnitude_axis
   └─────────────┘
         │
         ├──────────────────────────────────────────────────┐
         ▼                                                  ▼
   ┌──────────────┐   (if force)                    ┌──────────────┐
   │ Time signal  │   ─────────────►   FRF          │ RPM chart    │
   │ Spectrum     │                    (Hxx, Hyy)    │ Optimal loads│
   └──────────────┘                         │       │ Resonance map│
                                            ▼       └──────────────┘
                                     ┌─────────────┐         │
                                     │ blim(Ω)    │         ▼
                                     │ (if Ks,β)  │   ┌─────────────────┐
                                     └─────────────┘   │ Milling dynamics│
                                                       └─────────────────┘
```

---

## 7. Summary Table: Order and CLI

| Order | Output | CLI | Needs FRF? |
|-------|--------|-----|------------|
| 1 | Time signal map | `--plot-time-signal` | No |
| 2 | Spectrum | `--plot-spectrum` | No |
| 3 | FRF | `--plot-frf` | Force in CSV |
| 4 | RPM chart | default / `--plot` | No |
| 4 | Optimal loads | default / `--plot-optimal-loads` | No |
| 4 | Resonance map | `--plot-resonance-map` + `--helix` | No |
| 5 | Milling dynamics | `--plot-milling-dynamics` | No |
| 6 | Stability lobe blim(Ω) | (FRF + Ks, β; not in default CLI) | Yes |

Using `--workflow` (or `--plot-all`) produces steps 1–5 in this order for accurate, efficient optimal feeds and speeds with the current tool, including helix and other cutting factors.
