"""
Live accelerometer check: print X,Y,Z (and optional magnitude) until Ctrl+C.
Useful to verify wiring and that the ADXL345 is readable.
"""
from __future__ import annotations

import math
import time

from .accelerometer import open_accelerometer, stream_samples
from .config import get_config


def magnitude_g(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def main() -> None:
    cfg = get_config()
    accel = open_accelerometer(address=cfg.i2c_address, range_g=cfg.range_g)
    print("ADXL345 live (Ctrl+C to stop). Static Z should be ~1g if sensor is level.\n")
    try:
        for t, x, y, z in stream_samples(accel, interval_s=0.05, stop_after_n=None):
            mag = magnitude_g(x, y, z)
            print(f"t={t:.2f}s  x={x:+.2f} y={y:+.2f} z={z:+.2f}  |a|={mag:.2f}g")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
