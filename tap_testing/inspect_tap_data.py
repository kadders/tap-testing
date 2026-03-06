"""
Inspect tap-test CSV data: signal stats and validity check.

Usage:
  python -m tap_testing.inspect_tap_data <path>
  where <path> is a single .csv file or a cycle directory (e.g. data/cycle/20260302_164843).

Prints per-file: sample count, duration, sample rate, per-axis min/max/std,
magnitude std, and whether the data has usable signal (not all zeros).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .analyze import check_tap_signal, load_tap_csv


def _stats(path: Path) -> None:
    """Load one CSV and print stats + signal check."""
    try:
        t, data, sr = load_tap_csv(path)
    except Exception as e:
        print(f"  ERROR loading {path.name}: {e}")
        return
    n = data.shape[1]
    duration_s = float(t[-1] - t[0]) if n > 1 else 0.0
    ax, ay, az = data[0], data[1], data[2]
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    ok, msg = check_tap_signal(data)
    signal_status = "OK (usable)" if ok else f"NO SIGNAL: {msg}"
    print(f"  {path.name}:")
    print(f"    samples={n}, duration={duration_s:.3f} s, sample_rate={sr:.1f} Hz")
    print(f"    ax_g: min={ax.min():.4f}, max={ax.max():.4f}, std={ax.std():.6f}")
    print(f"    ay_g: min={ay.min():.4f}, max={ay.max():.4f}, std={ay.std():.6f}")
    print(f"    az_g: min={az.min():.4f}, max={az.max():.4f}, std={az.std():.6f}")
    print(f"    magnitude std={mag.std():.6f}")
    print(f"    Signal: {signal_status}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m tap_testing.inspect_tap_data <path>")
        print("  <path> = single .csv file or cycle directory (e.g. data/cycle/20260302_164843)")
        sys.exit(1)
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            print("Expected a .csv file or a directory containing tap_*.csv / combined.csv")
            sys.exit(1)
        print(f"Inspecting: {path}\n")
        _stats(path)
        return
    # Directory: find tap_*.csv and combined.csv
    csvs = sorted(path.glob("tap_*.csv")) + ([path / "combined.csv"] if (path / "combined.csv").exists() else [])
    if not csvs:
        print(f"No tap_*.csv or combined.csv in {path}")
        sys.exit(1)
    print(f"Inspecting {len(csvs)} file(s) in: {path}\n")
    for f in csvs:
        _stats(f)


if __name__ == "__main__":
    main()
