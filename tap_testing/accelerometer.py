"""
ADXL345 accelerometer interface for tap testing.
Uses Adafruit CircuitPython driver via Blinka on Raspberry Pi.
"""
from __future__ import annotations

import time
from typing import Generator, Tuple

try:
    import board
    import adafruit_adxl34x
except ImportError as e:
    raise ImportError(
        "tap_testing.accelerometer requires adafruit-blinka and adafruit-circuitpython-adxl34x. "
        "Install with: pip install -r requirements.txt"
    ) from e


def open_accelerometer(
    address: int = 0x53,
    range_g: int = 4,
) -> adafruit_adxl34x.ADXL345:
    """
    Open ADXL345 over I2C.

    Uses the system default I2C bus (bus 1 on Raspberry Pi). Enable I2C via
    raspi-config if needed.

    Args:
        address: I2C address (0x53 or 0x1D depending on ALT ADDRESS pin).
        range_g: Full-scale range in g (2, 4, or 8).

    Returns:
        ADXL345 instance.
    """
    i2c = board.I2C()
    accel = adafruit_adxl34x.ADXL345(i2c, address=address)

    if range_g == 2:
        accel.range = adafruit_adxl34x.Range.RANGE_2_G
    elif range_g == 4:
        accel.range = adafruit_adxl34x.Range.RANGE_4_G
    elif range_g == 8:
        accel.range = adafruit_adxl34x.Range.RANGE_8_G
    else:
        raise ValueError("range_g must be 2, 4, or 8")

    return accel


def stream_samples(
    accel: adafruit_adxl34x.ADXL345,
    interval_s: float,
    stop_after_n: int | None = None,
) -> Generator[Tuple[float, float, float, float], None, None]:
    """
    Stream (timestamp, x, y, z) in g at roughly the given interval.

    Args:
        accel: ADXL345 instance.
        interval_s: Target time between samples in seconds.
        stop_after_n: If set, yield only this many samples then stop.

    Yields:
        (t, x, y, z) with t in seconds since first sample.
    """
    t0 = time.perf_counter()
    n = 0
    while True:
        t = time.perf_counter() - t0
        x, y, z = accel.acceleration
        yield (t, x, y, z)
        n += 1
        if stop_after_n is not None and n >= stop_after_n:
            break
        next_t = t0 + (n + 1) * interval_s
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
