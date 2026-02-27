"""
Default configuration for tap testing (ADXL345 on Raspberry Pi).
Override via environment or a config file later if needed.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TapTestConfig:
    """Settings for accelerometer and recording."""

    # I2C (default bus on Raspberry Pi)
    i2c_bus: int = 1
    i2c_address: int = 0x53  # ADXL345 default when ALT ADDRESS pin is low

    # Sensor range: 2, 4, or 8 (g)
    range_g: int = 4

    # Sample rate (Hz). ADXL345 supports up to 3200 Hz in 400 Hz bandwidth mode.
    # 800–1600 Hz is often enough for tap testing.
    sample_rate_hz: float = 800.0

    # Recording: duration in seconds to capture per tap
    record_duration_s: float = 1.0

    # Trigger (optional): record when magnitude exceeds this (g) instead of fixed window
    trigger_threshold_g: Optional[float] = None  # None = no trigger, use fixed window

    # Pre-trigger buffer: seconds to keep before trigger (if using trigger)
    pre_trigger_s: float = 0.05

    # Multi-tap cycle (run_cycle script)
    cycle_iterations: int = 3
    cycle_spacing_s: float = 15.0
    # Status LED: GPIO pin (BCM) for "tap now" indicator. None = no LED (ADXL345 has no user LED; use Pi GPIO).
    status_led_gpio: Optional[int] = 17


def get_config() -> TapTestConfig:
    """Return default config. Can later read from env or file."""
    return TapTestConfig()
