"""
ADXL345 accelerometer interface for tap testing.
Supports I2C (Adafruit driver) and SPI (Klipper-style wiring) on Raspberry Pi.

Protocol aligned with Klipper klippy/extras/adxl345.py where applicable:
- read_reg: send [reg | 0x80, 0x00], use second byte (response[1]); some boards use first byte (see use_first_byte_data).
- set_reg: send [reg, val] (no read bit).
- SPI mode 3, 5 MHz; DEVID 0xE5 before any write. We poll DATAX0..DATAZ1; Klipper uses FIFO/bulk for high-rate.
See docs/ADXL345_WIRING.md and https://www.klipper3d.org/Measuring_Resonances.html#adxl345
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Generator, Tuple

try:
    import board
    import adafruit_adxl34x
    import busio
    import digitalio
except ImportError as e:
    raise ImportError(
        "tap_testing.accelerometer requires adafruit-blinka and adafruit-circuitpython-adxl34x. "
        "Install with: pip install -r requirements.txt"
    ) from e

# ADXL345 register addresses and protocol (aligned with Klipper klippy/extras/adxl345.py)
_REG_DEVID = 0x00
_REG_BW_RATE = 0x2C   # Klipper sets for bulk rate; we poll so optional
_REG_POWER_CTL = 0x2D
_REG_DATA_FORMAT = 0x31
_REG_DATAX0 = 0x32
_REG_FIFO_CTL = 0x38  # Klipper uses for bulk; we poll so optional
_REG_MOD_READ = 0x80  # Klipper: read single register
_REG_MOD_MULTI = 0x40 # Klipper: read multiple (0xC0 = 0x80 | 0x40)
_ADXL345_MG2G = 0.004
_STANDARD_GRAVITY = 9.80665
_ADXL345_DEVID = 0xE5  # Klipper ADXL345_DEV_ID


class _SpidevWrapper:
    """Thin wrapper so spidev (with mode 3) can be used like busio.SPI for ADXL345.
    Uses a lock so only one transfer runs at a time (avoids polling issues with shared fd)."""

    def __init__(self, spidev_obj) -> None:
        self._spi = spidev_obj
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self._spi.close()
        except Exception:
            pass

    def write(self, buf: bytes) -> None:
        with self._lock:
            self._spi.xfer2(list(buf))

    def write_readinto(self, buf_out: bytearray | bytes, buf_in: bytearray) -> None:
        n = len(buf_out)
        # Pass a copy so xfer2 (which may overwrite in-place) does not touch buf_out
        tx = list(buf_out[:n])
        with self._lock:
            rx = self._spi.xfer2(tx)
        # Copy received bytes; rx may be tx overwritten by kernel, so copy into buf_in
        for i in range(min(len(rx), n, len(buf_in))):
            buf_in[i] = rx[i] & 0xFF


def _open_spi_mode3(bus: int, device: int):
    """Open SPI on /dev/spidev{bus}.{device} with mode 3 for ADXL345. Returns (spi_like, None) or (None, None) if unavailable."""
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(bus, device)
        spi.max_speed_hz = 5000000
        spi.mode = 3  # ADXL345 requires CPOL=1, CPHA=1
        return (_SpidevWrapper(spi), None)
    except Exception:
        return (None, None)


class _ADXL345SPI:
    """Minimal ADXL345 over SPI (same .acceleration/.range interface as Adafruit I2C driver)."""

    def __init__(
        self,
        spi: busio.SPI,
        cs: digitalio.DigitalInOut | None,
        use_first_byte_data: bool = False,
    ) -> None:
        self._spi = spi
        self._cs = cs  # None = kernel-managed CS (CE0)
        self._use_first_byte_data = use_first_byte_data
        self._buf_out = bytearray(7)
        self._buf_in = bytearray(7)
        # ADXL345 requires SPI Mode 3 (CPOL=1, CPHA=1)
        try:
            if spi.try_lock():
                try:
                    spi.configure(polarity=1, phase=1)
                finally:
                    spi.unlock()
        except (AttributeError, NotImplementedError):
            pass
        # Match Klipper order: verify DEVID before writing any register (chip in power-on state).
        # https://github.com/Klipper3d/klipper/blob/master/klippy/extras/adxl345.py
        # On some spidev setups the first transfer returns zeros; discard one read then use the next.
        # When using first-byte mode, do only one read (same as raw diagnostic that works on these boards).
        self._cs_low()
        try:
            if not self._use_first_byte_data:
                self._read_reg(_REG_DEVID, 1)  # dummy read
            devid = self._read_reg(_REG_DEVID, 1)[0]
            if devid != _ADXL345_DEVID:
                hint = (
                    "0x00 often means wrong SPI mode (need Mode 3) or MISO/wiring. "
                    "Connect ADXL345 CS to Pi Pin 24 (CE0). See docs/ADXL345_WIRING.md."
                    if devid == 0x00
                    else ""
                )
                raise ValueError(
                    f"ADXL345 DEVID register returned 0x{devid:02X} (expected 0xE5). "
                    f"Check wiring and SPI mode. {hint}"
                )
            # Enable measurement (Klipper: set_reg REG_POWER_CTL 0x08)
            self._write_reg(_REG_POWER_CTL, 0x08)
        finally:
            self._cs_high()

    def _cs_low(self) -> None:
        if self._cs is not None:
            self._cs.value = False

    def _cs_high(self) -> None:
        if self._cs is not None:
            self._cs.value = True

    def _write_reg(self, reg: int, value: int) -> None:
        self._cs_low()
        try:
            self._spi.write(bytes([reg & 0x3F, value & 0xFF]))
        finally:
            self._cs_high()

    def _read_reg(self, reg: int, length: int) -> bytes:
        if not hasattr(self._spi, "write_readinto"):
            raise ValueError(
                "SPI write_readinto required for ADXL345. "
                "Update adafruit-blinka: pip install -U adafruit-blinka"
            )
        n = length + 1
        # Klipper read_reg: spi_transfer([reg | REG_MOD_READ, 0x00]), use response[1]
        self._buf_out[0] = (_REG_MOD_READ if length == 1 else _REG_MOD_READ | _REG_MOD_MULTI) | (reg & 0x3F)
        for i in range(1, n):
            self._buf_out[i] = 0
        self._cs_low()
        try:
            # Pass full buf_in so the wrapper writes into our buffer (slices are copies; would leave _buf_in unchanged)
            self._spi.write_readinto(self._buf_out[:n], self._buf_in)
            # Datasheet: first byte received is junk, second is data. Some boards use first byte.
            if self._use_first_byte_data:
                return bytes(self._buf_in[0:length])
            return bytes(self._buf_in[1:n])
        finally:
            self._cs_high()

    @property
    def acceleration(self) -> Tuple[float, float, float]:
        """X, Y, Z in m/s² (same units as Adafruit driver)."""
        raw = self._read_reg(_REG_DATAX0, 6)
        x, y, z = struct.unpack("<hhh", raw)
        scale = _ADXL345_MG2G * _STANDARD_GRAVITY
        return (x * scale, y * scale, z * scale)

    @property
    def range(self) -> int:
        return self._read_reg(_REG_DATA_FORMAT, 1)[0] & 0x03

    @range.setter
    def range(self, val: int) -> None:
        fmt = self._read_reg(_REG_DATA_FORMAT, 1)[0]
        fmt = (fmt & ~0x0F) | (val & 0x03) | 0x08  # FULL_RES
        self._write_reg(_REG_DATA_FORMAT, fmt)


def open_accelerometer(
    address: int = 0x53,
    range_g: int = 4,
    bus: int | None = None,
    interface: str = "i2c",
    spi_cs_pin: int = 8,
) -> adafruit_adxl34x.ADXL345 | _ADXL345SPI:
    """
    Open ADXL345 over I2C or SPI (Klipper-style wiring; see docs/ADXL345_WIRING.md).

    Args:
        address: I2C address (0x53 or 0x1D). Ignored when interface=="spi".
        range_g: Full-scale range in g (2, 4, or 8).
        bus: I2C bus number when interface=="i2c" (e.g. 1, 20, 21).
        interface: "i2c" or "spi". SPI recommended on RPi (Klipper).
        spi_cs_pin: BCM GPIO for SPI CS when interface=="spi" (default 8 = CE0).

    Returns:
        Accelerometer instance with .acceleration and .range (same interface for I2C/SPI).
    """
    if interface == "spi":
        cs = None
        # Prefer spidev with mode 3 when using kernel CS (spi_cs_pin 8 = CE0). Config can set spi_bus/spi_device.
        from .config import get_config
        cfg = get_config()
        if spi_cs_pin == 8:
            spi_obj, _ = _open_spi_mode3(cfg.spi_bus, cfg.spi_device)
            if spi_obj is not None:
                spi = spi_obj
            else:
                spi = board.SPI()
        else:
            spi = board.SPI()
            cs = digitalio.DigitalInOut(getattr(board, f"D{spi_cs_pin}"))
            cs.direction = digitalio.Direction.OUTPUT
            cs.value = True
        last_err = None
        for use_first_byte in (cfg.spi_first_byte_data, True):
            try:
                accel = _ADXL345SPI(spi, cs, use_first_byte)
                break
            except OSError as e:
                raise ValueError(
                    f"ADXL345 SPI failed on CS GPIO {spi_cs_pin}: {e}. "
                    "Check: SPI enabled (raspi-config), wiring (see docs/ADXL345_WIRING.md)."
                ) from e
            except ValueError as e:
                last_err = e
                if use_first_byte is True:
                    raise
                # First fd often returns zeros on some Pis; close and reopen before retry with first-byte mode
                if spi_cs_pin == 8 and hasattr(spi, "close"):
                    spi.close()
                    spi_obj, _ = _open_spi_mode3(cfg.spi_bus, cfg.spi_device)
                    if spi_obj is not None:
                        spi = spi_obj
        else:
            if last_err is not None:
                raise last_err
        if range_g == 2:
            accel.range = 0
        elif range_g == 4:
            accel.range = 1
        elif range_g == 8:
            accel.range = 2
        else:
            raise ValueError("range_g must be 2, 4, or 8")
        return accel

    # I2C path
    if bus is not None and bus != 1:
        try:
            from adafruit_extended_bus import ExtendedI2C
            i2c = ExtendedI2C(bus)
        except ImportError:
            raise ImportError(
                "Selecting I2C bus by number (other than 1) requires adafruit-extended-bus. "
                "Install with: pip install adafruit-extended-bus"
            ) from None
        bus_label = f"-y {bus}"
    else:
        i2c = board.I2C()
        bus_label = "-y 1 (default)"
    try:
        accel = adafruit_adxl34x.ADXL345(i2c, address=address)
    except OSError as e:
        hint = "Remote I/O (121) or I/O (5) usually means loose wiring, poor connection, or power."
        raise ValueError(
            f"I2C communication failed at 0x{address:02X}: {e}. {hint} "
            f"Check: wiring (SDA, SCL, 3.3V, GND), I2C enabled (raspi-config). Try: i2cdetect {bus_label}"
        ) from e
    except ValueError as e:
        if "No I2C device" in str(e) or "0x" in str(e).lower():
            msg = (
                f"ADXL345 not found at I2C address 0x{address:02X}. "
                "Check: (1) I2C enabled (raspi-config → Interface Options → I2C), "
                "(2) wiring (SDA, SCL, 3.3V, GND), "
                "(3) address—use 0x1D if ALT ADDRESS pin is high. "
                f"List devices with: i2cdetect {bus_label}"
            )
            cause = e.__cause__
            if cause is not None and isinstance(cause, OSError):
                msg += " (Log may show Remote I/O or I/O error—check wiring and power.)"
            raise ValueError(msg) from e
        raise

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
    accel: adafruit_adxl34x.ADXL345 | _ADXL345SPI,
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
