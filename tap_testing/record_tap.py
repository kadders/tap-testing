"""
Record a tap test: stream accelerometer data for a fixed duration and save to file.
Optionally report when a tap (impact) is detected during the recording.

Also supports live spindle recording: start/stop a single ADXL stream (e.g. for
homing calibration or tool cycle) without tap detection or fixed duration.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable

from .accelerometer import open_accelerometer, stream_samples
from .config import get_config

# CSV columns are ax_g, ay_g, az_g (g). Driver returns m/s²; 1 g = 9.80665 m/s².
_MPS2_TO_G = 1.0 / 9.80665

# Minimum samples before we consider baseline valid and start checking for impact.
_BASELINE_SAMPLES_MIN = 10
# Default seconds of data used for baseline (pre-tap) before impact detection.
_BASELINE_DURATION_S = 0.1


def record_tap(
    output_path: str | Path,
    duration_s: float | None = None,
    sample_rate_hz: float | None = None,
    sample_callback: Callable[[float, float, float, float], None] | None = None,
    callback_interval_s: float = 0.05,
    on_tap_detected: Callable[[float], None] | None = None,
    impact_threshold_g: float | None = None,
) -> Path:
    """
    Record accelerometer data for a fixed duration and write CSV.

    Columns: t_s, ax_g, ay_g, az_g.

    Args:
        output_path: Path for output CSV.
        duration_s: Recording duration in seconds (default from config).
        sample_rate_hz: Target sample rate (default from config).
        sample_callback: Optional (t, x_g, y_g, z_g) callback for live display; called at callback_interval_s.
        callback_interval_s: Minimum time between sample_callback calls (default 0.05 s).
        on_tap_detected: If set, called once with t_s when impact is detected (magnitude above baseline + threshold).
        impact_threshold_g: Magnitude (g) above baseline to consider a tap; default from config.

    Returns:
        Path to the written file.
    """
    cfg = get_config()
    duration_s = duration_s if duration_s is not None else cfg.record_duration_s
    sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else cfg.sample_rate_hz
    threshold_g = impact_threshold_g if impact_threshold_g is not None else cfg.impact_threshold_g
    interval_s = 1.0 / sample_rate_hz
    n_samples = int(round(duration_s * sample_rate_hz))
    n_baseline = max(
        _BASELINE_SAMPLES_MIN,
        int(round(_BASELINE_DURATION_S * sample_rate_hz)),
    )
    n_baseline = min(n_baseline, n_samples - 1)

    accel = open_accelerometer(
        address=cfg.i2c_address,
        range_g=cfg.range_g,
        bus=cfg.i2c_bus,
        interface=cfg.adxl345_interface,
        spi_cs_pin=cfg.spi_cs_pin,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_callback_t = 0.0
    baseline_mag = 0.0
    mag_buffer: list[float] = []
    tap_reported = False

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        writer.writerow(("# sample_rate_hz", sample_rate_hz, "", ""))
        for idx, (t, x, y, z) in enumerate(stream_samples(accel, interval_s, stop_after_n=n_samples)):
            x_g = x * _MPS2_TO_G
            y_g = y * _MPS2_TO_G
            z_g = z * _MPS2_TO_G
            writer.writerow((t, x_g, y_g, z_g))

            mag = (x_g * x_g + y_g * y_g + z_g * z_g) ** 0.5
            if idx < n_baseline:
                mag_buffer.append(mag)
                if idx == n_baseline - 1 and mag_buffer:
                    baseline_mag = sum(mag_buffer) / len(mag_buffer)
            elif not tap_reported and on_tap_detected is not None:
                if mag >= baseline_mag + threshold_g:
                    tap_reported = True
                    on_tap_detected(t)

            if sample_callback is not None and (t - last_callback_t) >= callback_interval_s:
                sample_callback(t, x_g, y_g, z_g)
                last_callback_t = t

    return path


def record_stream(
    output_path: str | Path,
    stop_event: threading.Event,
    sample_rate_hz: float | None = None,
    sample_callback: Callable[[float, float, float, float], None] | None = None,
    callback_interval_s: float = 0.05,
    accel: object | None = None,
) -> Path:
    """
    Record a single continuous stream of ADXL data until stop_event is set.
    Same CSV format as record_tap: t_s, ax_g, ay_g, az_g (no tap detection).

    Use for live spindle tests (e.g. homing calibration, tool cycle): start
    recording, run the machine operation, then stop recording.

    Args:
        output_path: Path for output CSV.
        stop_event: When set, recording stops and the file is closed.
        sample_rate_hz: Target sample rate (default from config).
        sample_callback: Optional (t, x_g, y_g, z_g) callback for live display.
        callback_interval_s: Minimum time between sample_callback calls.
        accel: Optional pre-opened accelerometer instance (e.g. for shared use with live plot).
              If None, opens and uses an internal instance.

    Returns:
        Path to the written file.
    """
    cfg = get_config()
    sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else cfg.sample_rate_hz
    interval_s = 1.0 / sample_rate_hz

    if accel is None:
        accel = open_accelerometer(
            address=cfg.i2c_address,
            range_g=cfg.range_g,
            bus=cfg.i2c_bus,
            interface=cfg.adxl345_interface,
            spi_cs_pin=cfg.spi_cs_pin,
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    last_callback_t = 0.0
    n = 0

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        writer.writerow(("# sample_rate_hz", sample_rate_hz, "", ""))
        while not stop_event.is_set():
            t = time.perf_counter() - t0
            x, y, z = accel.acceleration
            x_g = x * _MPS2_TO_G
            y_g = y * _MPS2_TO_G
            z_g = z * _MPS2_TO_G
            writer.writerow((t, x_g, y_g, z_g))
            n += 1

            if sample_callback is not None and (t - last_callback_t) >= callback_interval_s:
                sample_callback(t, x_g, y_g, z_g)
                last_callback_t = t

            next_t = t0 + (n + 1) * interval_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                if stop_event.wait(timeout=sleep_s):
                    break

    return path


def main() -> None:
    """CLI entry: record one tap to tap_testing/data/ by default."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Record a tap test (fixed duration).")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tap_001.csv",
        help="Output CSV path",
    )
    parser.add_argument("-d", "--duration", type=float, default=None, help="Duration (s)")
    parser.add_argument("-r", "--rate", type=float, default=None, help="Sample rate (Hz)")
    parser.add_argument(
        "--no-tap-report",
        action="store_true",
        help="Do not print when a tap is detected during recording",
    )
    args = parser.parse_args()

    def on_tap(t_s: float) -> None:
        print(f"Tap detected at t={t_s:.2f} s", file=sys.stderr, flush=True)

    path = record_tap(
        args.output,
        duration_s=args.duration,
        sample_rate_hz=args.rate,
        on_tap_detected=None if args.no_tap_report else on_tap,
    )
    print(f"Recorded to {path}")


if __name__ == "__main__":
    main()
