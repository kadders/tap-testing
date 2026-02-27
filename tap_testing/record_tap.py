"""
Record a tap test: stream accelerometer data for a fixed duration and save to file.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .accelerometer import open_accelerometer, stream_samples
from .config import get_config


def record_tap(
    output_path: str | Path,
    duration_s: float | None = None,
    sample_rate_hz: float | None = None,
) -> Path:
    """
    Record accelerometer data for a fixed duration and write CSV.

    Columns: t_s, ax_g, ay_g, az_g.

    Args:
        output_path: Path for output CSV.
        duration_s: Recording duration in seconds (default from config).
        sample_rate_hz: Target sample rate (default from config).

    Returns:
        Path to the written file.
    """
    cfg = get_config()
    duration_s = duration_s if duration_s is not None else cfg.record_duration_s
    sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else cfg.sample_rate_hz
    interval_s = 1.0 / sample_rate_hz
    n_samples = int(round(duration_s * sample_rate_hz))

    accel = open_accelerometer(address=cfg.i2c_address, range_g=cfg.range_g)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        writer.writerow(("# sample_rate_hz", sample_rate_hz, "", ""))
        for t, x, y, z in stream_samples(accel, interval_s, stop_after_n=n_samples):
            writer.writerow((t, x, y, z))

    return path


def main() -> None:
    """CLI entry: record one tap to tap_testing/data/ by default."""
    import argparse
    parser = argparse.ArgumentParser(description="Record a tap test (fixed duration).")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tap_001.csv",
        help="Output CSV path",
    )
    parser.add_argument("-d", "--duration", type=float, default=None, help="Duration (s)")
    parser.add_argument("-r", "--rate", type=float, default=None, help="Sample rate (Hz)")
    args = parser.parse_args()
    path = record_tap(args.output, duration_s=args.duration, sample_rate_hz=args.rate)
    print(f"Recorded to {path}")


if __name__ == "__main__":
    main()
