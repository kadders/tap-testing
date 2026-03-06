"""
Live accelerometer check: print X,Y,Z in g until Ctrl+C.
Useful to verify wiring and that the ADXL345 is readable.
"""
from __future__ import annotations

import math

from .accelerometer import open_accelerometer, stream_samples
from .config import get_config

# Driver returns m/s²; convert to g for display
_MPS2_TO_G = 1.0 / 9.80665


def main() -> None:
    cfg = get_config()
    accel = open_accelerometer(
        address=cfg.i2c_address,
        range_g=cfg.range_g,
        bus=cfg.i2c_bus,
        interface=cfg.adxl345_interface,
        spi_cs_pin=cfg.spi_cs_pin,
    )
    print("ADXL345 live (Ctrl+C to stop). Values in g; static Z should be ~1.0 if level.\n")
    try:
        for t, x, y, z in stream_samples(accel, interval_s=0.05, stop_after_n=None):
            x_g, y_g, z_g = x * _MPS2_TO_G, y * _MPS2_TO_G, z * _MPS2_TO_G
            mag = math.sqrt(x_g * x_g + y_g * y_g + z_g * z_g)
            print(f"t={t:.2f}s  x={x_g:+.2f} y={y_g:+.2f} z={z_g:+.2f}  |a|={mag:.2f} g")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
