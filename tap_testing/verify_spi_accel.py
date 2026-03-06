"""
Verify accelerometer data over SPI (spidev or board.SPI).

Streams a short burst of samples and prints X,Y,Z in g. Use this to confirm
the ADXL345 is responding and data is non-zero (e.g. Z ≈ 1g at rest).

Usage:
  python -m tap_testing.verify_spi_accel [--samples N] [--rate HZ]

Example:
  python -m tap_testing.verify_spi_accel --samples 20 --rate 100
"""
from __future__ import annotations

import argparse
import sys

from .accelerometer import (
    _ADXL345SPI,
    _open_spi_mode3,
    open_accelerometer,
    stream_samples,
)
from .check_spi_mode import discover_spidev_devices
from .config import get_config

_MPS2_TO_G = 1.0 / 9.80665


def _raw_read_devid(spidev_module, bus: int, device: int) -> list[int] | None:
    """Perform raw DEVID read; return rx bytes or None on failure."""
    try:
        spi = spidev_module.SpiDev()
        spi.open(bus, device)
        spi.max_speed_hz = 5000000
        spi.mode = 3
        rx = spi.xfer2([0x80, 0x00])  # read reg 0x00 (DEVID)
        spi.close()
        return list(rx)
    except Exception:
        return None


def _print_raw_devid(spidev_module, bus: int, device: int) -> list[int] | None:
    """Print raw DEVID read (tx/rx) for debugging when DEVID comes back 0x00. Returns rx bytes or None."""
    rx = _raw_read_devid(spidev_module, bus, device)
    if rx is None:
        print("      Raw read failed")
        return None
    rx_hex = [f"0x{b:02X}" for b in rx]
    print(f"      Raw DEVID: tx=[0x80, 0x00]  rx=[{', '.join(rx_hex)}]")
    if len(rx) >= 1 and rx[0] == 0xE5:
        print("      → First byte is 0xE5; this board may use first byte as data (not second).")
    elif all(b == 0 for b in rx):
        print("      → All zeros: MISO likely floating or no device on this CS. Check MISO/SDO and CS pin (CE0=24, CE1=26).")
    return rx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ADXL345 SPI: stream samples and print X,Y,Z in g."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to stream (default 20)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Sample rate in Hz (default 100)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Try all spidev devices and report which one has ADXL345 (DEVID 0xE5)",
    )
    args = parser.parse_args()

    cfg = get_config()
    if cfg.adxl345_interface != "spi" and not args.probe:
        print(f"Config has adxl345_interface={cfg.adxl345_interface!r}. For SPI verification use SPI.", file=sys.stderr)
        print("Set TAP_ADXL345_INTERFACE=spi or change config to use SPI.", file=sys.stderr)
        sys.exit(1)

    # --probe: try all spidev devices and report which has ADXL345
    if args.probe:
        try:
            import spidev
        except ImportError:
            print("spidev required for --probe. pip install spidev", file=sys.stderr)
            sys.exit(1)
        devices = discover_spidev_devices()
        if not devices:
            print("No spidev devices found. Enable SPI (raspi-config).", file=sys.stderr)
            sys.exit(1)
        print(f"Probing {len(devices)} spidev device(s) for ADXL345 (DEVID 0xE5)...\n")
        found = []
        for bus, device in sorted(devices):
            spi_obj, _ = _open_spi_mode3(bus, device)
            if spi_obj is None:
                continue
            try:
                accel = _ADXL345SPI(spi_obj, None, cfg.spi_first_byte_data)
                found.append((bus, device))
                print(f"  /dev/spidev{bus}.{device}: ADXL345 found (DEVID 0xE5)")
            except ValueError as e:
                err_str = str(e)
                print(f"  /dev/spidev{bus}.{device}: no ADXL345 — {e}")
                # When DEVID is 0x00: close this fd and reopen fresh, then retry with first-byte mode (same protocol as Klipper read_reg but first byte = data).
                if "0x00" in err_str and "expected 0xE5" in err_str:
                    try:
                        spi_obj.close()
                    except Exception:
                        pass
                    fresh_spi, _ = _open_spi_mode3(bus, device)
                    if fresh_spi is not None:
                        try:
                            accel = _ADXL345SPI(fresh_spi, None, True)
                            found.append((bus, device))
                            print(f"  /dev/spidev{bus}.{device}: ADXL345 found (DEVID 0xE5) using first-byte mode (fresh open).")
                            print("      Set TAP_SPI_FIRST_BYTE_DATA=1 for other commands (or we will auto-detect when opening).")
                        except Exception as ex:
                            fresh_spi.close()
                            print(f"      Retry with first-byte mode failed: {ex}")
                    # Show raw DEVID so user can confirm hardware returns 0xE5
                    rx = _print_raw_devid(spidev, bus, device)
            except Exception as e:
                print(f"  /dev/spidev{bus}.{device}: error — {e}")
        if found:
            print(f"\nUse one of: TAP_SPI_BUS={found[0][0]} TAP_SPI_DEVICE={found[0][1]}")
        else:
            print("\nNo ADXL345 found on any spidev device. Check wiring and SPI mode.", file=sys.stderr)
            sys.exit(1)
        return

    backend = "unknown"
    accel = None

    # Prefer spidev (Mode 3) when on CE0 so we can report it
    if cfg.spi_cs_pin == 8:
        spi_obj, _ = _open_spi_mode3(cfg.spi_bus, cfg.spi_device)
        if spi_obj is not None:
            backend = f"spidev (Mode 3), /dev/spidev{cfg.spi_bus}.{cfg.spi_device}"
            accel = None
            try:
                accel = _ADXL345SPI(spi_obj, None, cfg.spi_first_byte_data)
                if cfg.range_g == 2:
                    accel.range = 0
                elif cfg.range_g == 4:
                    accel.range = 1
                elif cfg.range_g == 8:
                    accel.range = 2
            except ValueError:
                spi_obj.close()
                spi_obj, _ = _open_spi_mode3(cfg.spi_bus, cfg.spi_device)
                if spi_obj is not None:
                    try:
                        accel = _ADXL345SPI(spi_obj, None, True)
                        if cfg.range_g == 2:
                            accel.range = 0
                        elif cfg.range_g == 4:
                            accel.range = 1
                        elif cfg.range_g == 8:
                            accel.range = 2
                    except Exception as e:
                        print(f"spidev open failed: {e}", file=sys.stderr)
                        spi_obj.close()
            except Exception as e:
                print(f"spidev open failed: {e}", file=sys.stderr)
                accel = None

    if accel is None:
        backend = "board.SPI() (config)"
        try:
            accel = open_accelerometer(
                address=cfg.i2c_address,
                range_g=cfg.range_g,
                bus=cfg.i2c_bus,
                interface="spi",
                spi_cs_pin=cfg.spi_cs_pin,
            )
        except Exception as e:
            print(f"Failed to open accelerometer: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"SPI backend: {backend}")
    print(f"Streaming {args.samples} samples at {args.rate} Hz (values in g). Static Z should be ~1.0.\n")

    interval_s = 1.0 / args.rate
    samples = []
    for t, x, y, z in stream_samples(accel, interval_s=interval_s, stop_after_n=args.samples):
        x_g = x * _MPS2_TO_G
        y_g = y * _MPS2_TO_G
        z_g = z * _MPS2_TO_G
        samples.append((x_g, y_g, z_g))
        print(f"  t={t:.3f}s  x={x_g:+.3f}  y={y_g:+.3f}  z={z_g:+.3f} g")

    if not samples:
        print("No samples received.", file=sys.stderr)
        sys.exit(1)

    # Check for non-zero signal and gravity sanity (at rest, |a| ≈ 1 g)
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    zs = [s[2] for s in samples]
    mags = [(xs[i]**2 + ys[i]**2 + zs[i]**2)**0.5 for i in range(len(xs))]
    mean_mag = sum(mags) / len(mags)
    tol = 1e-4
    has_signal = (
        max(xs) - min(xs) > tol or max(ys) - min(ys) > tol or max(zs) - min(zs) > tol
        or abs(sum(zs) / len(zs)) > 0.01
    )
    # At rest, magnitude should be ~1 g (gravity). 0.85–1.15 g is a reasonable band.
    gravity_ok = 0.85 <= mean_mag <= 1.15
    if has_signal:
        print("\nPASS: Accelerometer data is present (non-zero signal).")
        if gravity_ok:
            print(f"  Gravity check: mean |a| = {mean_mag:.3f} g (expected ~1.0 at rest). Data looks valid.")
        else:
            print(f"  Note: mean |a| = {mean_mag:.3f} g (expected ~1.0 at rest). Check orientation or calibration.")
    else:
        print("\nFAIL: All samples are near zero. Check wiring and SPI mode (ADXL345 needs Mode 3).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
