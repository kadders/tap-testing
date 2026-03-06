"""
Default configuration for tap testing (ADXL345 on Raspberry Pi).
Override via environment or a config file later if needed.
"""
import os
from dataclasses import dataclass
from typing import Optional

# Default force excitation type: we use impulse (hammer tap) by default.
# Other options (sine-sweep, random) could be supported with different hardware.
DEFAULT_EXCITATION_TYPE = "impulse"


@dataclass
class TapTestConfig:
    """Settings for accelerometer and recording."""

    # Force excitation: "impulse" (hammer tap), "sine_sweep", "random" (future).
    excitation_type: str = DEFAULT_EXCITATION_TYPE

    # Interface: "i2c" or "spi". Default SPI (Klipper-style); see docs/ADXL345_WIRING.md.
    adxl345_interface: str = "spi"
    # I2C (when adxl345_interface == "i2c")
    i2c_bus: int = 1
    i2c_address: int = 0x53  # ADXL345 default when ALT ADDRESS pin is low
    # SPI (when adxl345_interface == "spi"): CS pin BCM number; spidev bus/device when using kernel CS.
    spi_cs_pin: int = 8
    spi_bus: int = 0      # spidev bus (e.g. 0 → /dev/spidev0.x)
    spi_device: int = 0   # spidev device: 0=CE0, 1=CE1 (e.g. 0 → /dev/spidev0.0)
    # If True, use first received byte as register data (some boards); default False = use second byte (datasheet).
    spi_first_byte_data: bool = False

    # Sensor range: 2, 4, or 8 (g)
    range_g: int = 4

    # Sample rate (Hz). ADXL345 supports up to 3200 Hz in 400 Hz bandwidth mode.
    # 800–1600 Hz is often enough for tap testing.
    sample_rate_hz: float = 800.0

    # Recording: duration in seconds to capture per tap
    record_duration_s: float = 1.0

    # Tap cycle and background subtraction (for cleaner FFT analysis)
    # Duration of the "tap phase" from impact (s); used when extracting tap cycle from recording.
    tap_cycle_duration_s: float = 1.0
    # Seconds of recording to use as background (pre-tap) for baseline subtraction. None = use up to impact.
    background_duration_s: Optional[float] = 0.2
    # If no impact detected, fraction of recording length used as background (0.0–1.0).
    background_fraction: float = 0.2
    # Magnitude threshold (g) above baseline to detect impact.
    impact_threshold_g: float = 0.5
    # When True, analysis preprocesses: detect impact, extract tap cycle, subtract background.
    subtract_background: bool = True

    # Trigger (optional): record when magnitude exceeds this (g) instead of fixed window
    trigger_threshold_g: Optional[float] = None  # None = no trigger, use fixed window

    # Pre-trigger buffer: seconds to keep before trigger (if using trigger)
    pre_trigger_s: float = 0.05

    # Multi-tap cycle (run_cycle script)
    cycle_iterations: int = 3
    cycle_spacing_s: float = 5.0  # Seconds between taps (shorter for FRF data collection)
    # Status LED: GPIO pin (BCM) for "tap now" indicator. None = no LED (ADXL345 has no user LED; use Pi GPIO).
    status_led_gpio: Optional[int] = 17

    # Workpiece material for charts and milling guidance (e.g. "6061 aluminum", "7075 aluminum", "A36 steel")
    material_name: str = "6061 aluminum"

    # Tool (cutter) material for chart labels and future speed limits (e.g. "carbide", "HSS", "ceramic")
    tool_material: str = "carbide"

    # Spindle default operating frequency (Hz, revolutions per second). Used as default reference
    # operating point: rpm = spindle_operating_frequency_hz * 60 (e.g. 400 Hz → 24000 rpm).
    spindle_operating_frequency_hz: float = 400.0

    # Optional: hammer/tapper mass (kg) for rough FRF when no force sensor is present.
    # With this set, analysis can estimate force from F ≈ 2*m*v (impulse) and produce a rough FRF.
    hammer_mass_kg: Optional[float] = None
    # Typical impact velocity (m/s) for the tap, used with hammer_mass_kg for synthetic force. Default 0.5 m/s (hand tap).
    impact_velocity_m_s: Optional[float] = 0.5


def get_config() -> TapTestConfig:
    """Return default config. Can later read from env or file."""
    c = TapTestConfig()
    bus_env = os.environ.get("TAP_I2C_BUS")
    if bus_env is not None:
        try:
            c.i2c_bus = int(bus_env)
        except ValueError:
            pass
    if_env = os.environ.get("TAP_ADXL345_INTERFACE")
    if if_env in ("i2c", "spi"):
        c.adxl345_interface = if_env
    bus_env = os.environ.get("TAP_SPI_BUS")
    if bus_env is not None:
        try:
            c.spi_bus = int(bus_env)
        except ValueError:
            pass
    dev_env = os.environ.get("TAP_SPI_DEVICE")
    if dev_env is not None:
        try:
            c.spi_device = int(dev_env)
        except ValueError:
            pass
    first_byte_env = os.environ.get("TAP_SPI_FIRST_BYTE_DATA", "").strip().lower()
    if first_byte_env in ("1", "true", "yes"):
        c.spi_first_byte_data = True
    hammer_env = os.environ.get("TAP_HAMMER_MASS_KG")
    if hammer_env is not None:
        try:
            c.hammer_mass_kg = float(hammer_env)
        except ValueError:
            pass
    vel_env = os.environ.get("TAP_IMPACT_VELOCITY_M_S")
    if vel_env is not None:
        try:
            c.impact_velocity_m_s = float(vel_env)
        except ValueError:
            pass
    # Tap cycle and background subtraction
    tap_dur = os.environ.get("TAP_CYCLE_DURATION_S")
    if tap_dur is not None:
        try:
            c.tap_cycle_duration_s = float(tap_dur)
        except ValueError:
            pass
    bg_dur = os.environ.get("TAP_BACKGROUND_DURATION_S")
    if bg_dur is not None:
        try:
            c.background_duration_s = float(bg_dur)
        except ValueError:
            pass
    bg_frac = os.environ.get("TAP_BACKGROUND_FRACTION")
    if bg_frac is not None:
        try:
            c.background_fraction = float(bg_frac)
        except ValueError:
            pass
    thresh = os.environ.get("TAP_IMPACT_THRESHOLD_G")
    if thresh is not None:
        try:
            c.impact_threshold_g = float(thresh)
        except ValueError:
            pass
    sub_bg = os.environ.get("TAP_SUBTRACT_BACKGROUND", "").strip().lower()
    if sub_bg in ("0", "false", "no", "off"):
        c.subtract_background = False
    elif sub_bg in ("1", "true", "yes", "on"):
        c.subtract_background = True
    spindle_freq = os.environ.get("TAP_SPINDLE_OPERATING_FREQUENCY_HZ")
    if spindle_freq is not None:
        try:
            c.spindle_operating_frequency_hz = float(spindle_freq)
        except ValueError:
            pass
    tool_mat = os.environ.get("TAP_TOOL_MATERIAL")
    if tool_mat is not None and tool_mat.strip():
        c.tool_material = tool_mat.strip().lower()
    return c


def rpm_from_spindle_frequency_hz(freq_hz: float) -> float:
    """Convert spindle frequency (rev/s, Hz) to RPM: rpm = freq_hz * 60."""
    return freq_hz * 60.0
