# Chatter identification from cutting recordings

Chatter is self-excited vibration at or near a structural natural frequency. This doc describes how to use the accelerometer and the analyzer to **identify chatter** in a recording made **during a cut**, then get suggested spindle speeds to try.

## Workflow

1. **Get natural frequency (tap test)**  
   Run a normal tap test and analyze it to get the tool–holder natural frequency \(f_n\) (Hz). For example:
   ```bash
   python -m tap_testing.record_tap -o data/tap_001.csv
   python -m tap_testing.analyze data/tap_001.csv --flutes 4
   ```
   Note the printed **natural frequency** (e.g. 920 Hz).

2. **Record during a cut**  
   Use the same CSV format (`t_s`, `ax_g`, `ay_g`, `az_g`). Start recording, run your cut, then stop. Save to e.g. `data/cut_001.csv`. The same `record_tap` script can be used with a fixed duration that covers the cut, or use any pipeline that writes the same columns.

3. **Run chatter assessment**  
   Pass the cutting CSV and the natural frequency from step 1:
   ```bash
   python -m tap_testing.analyze data/cut_001.csv --chatter --natural-freq 920 --flutes 4
   ```
   Optional: `--natural-freq-uncertainty 25` (Hz band around \(f_n\); default 3% of \(f_n\)), `--rpm 18000` (spindle RPM during the cut, for tooth-passing context), `--chatter-plot-out file.png` (save spectrum with \(f_n\) band and peaks).

4. **Interpret and act**  
   - **Chatter likely: Yes** — A significant peak in the spectrum falls inside the natural-frequency band. Try one of the **suggested spindle speeds** (stability lobe formula using the detected or given \(f_n\)). Reduce depth or width if needed.  
   - **Chatter likely: No** — No strong peak in the \(f_n\) band; cut may be stable, or chatter is weak. You can still inspect **top peaks** and compare to tooth-passing (RPM × flutes / 60) if you passed `--rpm`.

## How it works

- The tool computes the FFT magnitude spectrum of the cutting signal (magnitude \(\sqrt{a_x^2 + a_y^2 + a_z^2}\)).
- It finds **peaks** (local maxima) and checks whether any fall inside **\(f_n \pm\) uncertainty**.
- If a peak lies in that band, the result is **chatter likely**.
- **Suggested RPM** values are from the stability-lobe relation \(\Omega = f_n \cdot 60 / ((N+1) \cdot N_t)\) for lobe indices \(N = 0, 1, \ldots\), using the detected chatter frequency (or \(f_n\)) and the given flute count.

## Tips

- Use a **prior tap test** on the same tool–holder setup to get \(f_n\); don’t use a cutting CSV to derive \(f_n\).
- For **marginal** cases, record a short **no-cut** (spindle only or air cut) and compare spectrum level; set “significant” by experience or by raising the bar above that baseline.
- Suggested RPMs are targets to try; stability also depends on depth and width of cut. Iterate with new recordings if needed.
