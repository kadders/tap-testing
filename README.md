# Tap Testing for Milo

Tap testing application for measuring tool vibrations using an accelerometer mounted at the tool tip. A hammer tap excites the structure; the ADXL345 records the response for analysis (e.g. frequency content, decay).

## Scope

- **Hardware**: ADXL345 accelerometer + Raspberry Pi
- **Use case**: Mount accelerometer on tool, tap with hammer, capture and store vibration data
- **Stack**: Python 3, I2C (or SPI) to the ADXL345

## Hardware

- **Raspberry Pi** (3B+, 4, or 5)
- **ADXL345** (±2g / ±4g / ±8g, I2C or SPI)
- Wiring (I2C example):
  - ADXL345 `VCC` → 3.3 V
  - ADXL345 `GND` → GND
  - ADXL345 `SDA` → Pi GPIO 2 (SDA)
  - ADXL345 `SCL` → Pi GPIO 3 (SCL)
  - Optional: `CS` high for I2C (or tie to 3.3 V); see ADXL345 datasheet for I2C vs SPI

Enable I2C on the Pi: `sudo raspi-config` → Interface Options → I2C → Enable.

**Status LED (optional)**  
The ADXL345 breakout has no user-controllable LED. For the **run_cycle** script you can connect an LED (with a suitable series resistor, e.g. 330 Ω) to a Raspberry Pi GPIO pin. Default: **GPIO 17 (BCM)**. LED **ON** = recording (tap now); **OFF** = waiting between taps. On the Pi, install `RPi.GPIO` if needed: `pip install RPi.GPIO`. Use `--no-led` to run without a LED.

## Installation

### System (Raspberry Pi)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev
# Enable I2C: sudo raspi-config → Interface Options → I2C
```

### Project

```bash
cd tap-testing
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Testing (optional)

Tests use **pytest** and synthetic data only (no Pi or accelerometer required):

```bash
cd tap-testing
pip install -r requirements-dev.txt   # adds pytest
pytest tests/ -v
```

See `tests/` for what is covered: CSV load/save, FFT dominant frequency, RPM avoid/suggested ranges, combine logic, config defaults, and chart figure generation.

## Usage

- **Check accelerometer**: `python -m tap_testing.check_sensor` — prints live X,Y,Z and magnitude until Ctrl+C.
- **Record a tap**: `python -m tap_testing.record_tap` — records for a fixed duration, saves CSV to `data/tap_001.csv` by default. Use `-o`, `-d`, `-r` for output path, duration, and sample rate.
- **Run 3-tap cycle (recommended)**: `python -m tap_testing.run_cycle` — runs 3 tap tests 15 s apart, combines the data, analyzes, and shows/saves the RPM chart. Optional **status LED** on a Pi GPIO: **ON = tap now**, **OFF = wait**. Output goes to `data/cycle/<timestamp>/`.
- **Tap cycle with GUI (Pi)**: `python -m tap_testing.cycle_gui` or `python -m tap_testing.run_cycle --gui` — opens a window showing **live status** (e.g. "Tap 1/3 — TAP the tool now", "Waiting 15 s…") and, when done, the **RPM band chart** (red = avoid, green = optimal) in the same window. Same LED and output dir as run_cycle.
- **Analyze and get speed/feed guidance**: `python -m tap_testing.analyze data/tap_001.csv --flutes 4 --max-rpm 24000 --tool-diameter 6` — prints natural frequency, RPMs to avoid, and a suggested stable RPM range. Use `--plot` to show a chart (red = avoid, green = optimal) or `--plot-out file.png` to save it. Use `--chip-load` for example feed.

---

## From tap test to spindle speed and feed

### Why it matters

The tap test excites the **tool–holder–spindle** structure. The dominant frequency in the response is (approximately) the **first natural frequency** of that assembly. During milling, if the **tooth passing frequency**—or one of its harmonics—matches this natural frequency, you get **chatter** (resonance). So we use the tap test to find that frequency, then choose spindle speeds that avoid it.

### How we get spindle speed from the tap test

1. **Record** — Mount the accelerometer, tap the tool, run `record_tap` to get a CSV (time, ax, ay, az).
2. **Analyze** — Run `analyze` on the CSV. It:
   - Computes an FFT of the vibration (magnitude or a single axis).
   - Picks the **dominant natural frequency** (peak in the FFT).
   - Uses the relation **tooth passing frequency = (RPM/60) × N_teeth**.
   - **RPMs to avoid**: resonance when `(RPM/60) × N_teeth × k = natural_freq_hz` ⇒  
     `RPM = 60 × natural_freq_hz / (N_teeth × k)` for k = 1, 2, 3, …
   - Suggests a **stable RPM range**: a “pocket” between those critical RPMs where you are less likely to chatter.
3. **Choose RPM** — Pick a spindle speed inside the suggested range (or at least away from the “avoid” list).
4. **Feed** — Feed is not set by the tap test; it’s set by **chip load** (mm or in per tooth), number of teeth, and RPM:  
   **Feed (mm/min) = chip_load_mm × N_teeth × RPM**.  
   Use material/tool guidelines for chip load; the tap test only tells you which RPM range is safe. The analyzer can print an example feed for a given chip load with `--chip-load`.

### Parameters

When analyzing, you can define:

- **`--max-rpm`** — Maximum spindle RPM (default **24000**, standard LDO Milo spindle). The avoid/suggested ranges and the chart use min RPM 4000 up to this max. Set a different max when testing another spindle.
- **`--tool-diameter`** — Tool diameter in mm (for reference and chart title; optional).
- **`--flutes`** (or `-n`) — Flute (tooth) count (default 4). Used for tooth-passing frequency.

### Chart (red = avoid, green = optimal)

Use **`--plot`** to display the RPM band chart in a window, or **`--plot-out file.png`** to save it. Red shaded bands are RPMs to avoid (tooth-pass resonance); green bands are optimal (stable) ranges.

### Example

```bash
# Record a tap (on the Pi with sensor)
python -m tap_testing.record_tap -o data/tap_001.csv -d 1 -r 800

# Analyze with your tool and spindle limits; show chart
python -m tap_testing.analyze data/tap_001.csv --flutes 4 --max-rpm 10000 --tool-diameter 6 --plot

# Same, save chart to file
python -m tap_testing.analyze data/tap_001.csv --flutes 4 --max-rpm 10000 --tool-diameter 6 --plot-out data/tap_001_chart.png

# Full cycle: 3 taps (15 s apart), combine, analyze, and show chart (LED on GPIO 17 = tap now)
python -m tap_testing.run_cycle --plot

# On the Pi: window with status + chart (no separate plot window)
python -m tap_testing.cycle_gui

# With options: different spacing, no LED, save chart only
python -m tap_testing.run_cycle -s 20 --no-led --plot-out data/my_chart.png

# Example feed at suggested RPM with 0.05 mm/tooth chip load
python -m tap_testing.analyze data/tap_001.csv --flutes 4 --chip-load 0.05
# At 3000 RPM, chip load 0.05 mm/tooth → feed ≈ 600.0 mm/min
```

So: **tap test → natural frequency → avoid those RPMs, use a stable pocket → set feed from chip load and chosen RPM.**

## Project layout

```
tap-testing/
  README.md           # This file
  requirements.txt    # Python dependencies
  tap_testing/        # Main package
    __init__.py
    config.py         # Defaults (sample rate, cycle iterations/spacing, LED GPIO)
    accelerometer.py  # ADXL345 init and streaming
    record_tap.py     # Recording logic and output
    analyze.py        # FFT, natural frequency, spindle speed and feed guidance
    run_cycle.py      # 3-tap cycle: record → combine → analyze → plot (optional LED)
    cycle_gui.py      # Same cycle with Pi window: status + embedded RPM chart
  tests/              # Pytest suite (synthetic data; no hardware)
  data/               # Recorded tap CSVs; data/cycle/<timestamp>/ for run_cycle
```

## Data format

Recorded CSVs have header `t_s, ax_g, ay_g, az_g`, a comment line `# sample_rate_hz, <value>`, then one row per sample. Analysis uses this to infer sample rate and run the FFT.

## License

Same as parent Milo-Code repository.
