"""
Report SPI mode for spidev device(s).

SPI mode is (CPOL, CPHA): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
ADXL345 requires Mode 3 (CPOL=1, CPHA=1).

Usage:
  python -m tap_testing.check_spi_mode              # check default spidev0.0
  python -m tap_testing.check_spi_mode --all        # discover and list all spidev devices + mode
  python -m tap_testing.check_spi_mode --bus 0 --device 1
"""
from __future__ import annotations

import argparse
import sys

_SPI_MODES = {
    0: "0 (CPOL=0, CPHA=0)",
    1: "1 (CPOL=0, CPHA=1)",
    2: "2 (CPOL=1, CPHA=0)",
    3: "3 (CPOL=1, CPHA=1) — required for ADXL345",
}


def discover_spidev_devices(max_bus: int = 4, max_device: int = 4) -> list[tuple[int, int]]:
    """Return list of (bus, device) that can be opened. Requires spidev."""
    import spidev
    found = []
    for bus in range(max_bus):
        for device in range(max_device):
            try:
                spi = spidev.SpiDev()
                spi.open(bus, device)
                spi.close()
                found.append((bus, device))
            except Exception:
                pass
    return found


def check_one(spidev_module, bus: int, device: int) -> tuple[int, str] | None:
    """Open spidev bus.device and return (mode, description) or None on failure."""
    try:
        spi = spidev_module.SpiDev()
        spi.open(bus, device)
        mode = spi.mode
        spi.close()
        desc = _SPI_MODES.get(mode, f"{mode} (unknown)")
        return (mode, desc)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check SPI mode for spidev device(s). Use --all to list every available device."
    )
    parser.add_argument("--bus", type=int, default=0, help="SPI bus number (default 0)")
    parser.add_argument("--device", type=int, default=0, help="SPI device number, 0=CE0, 1=CE1 (default 0)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Discover and report mode for all /dev/spidev* devices",
    )
    args = parser.parse_args()

    try:
        import spidev
    except ImportError:
        print("spidev is required. Install with: pip install spidev", file=sys.stderr)
        sys.exit(1)

    if args.all:
        devices = discover_spidev_devices()
        if not devices:
            print("No spidev devices found. Enable SPI: raspi-config → Interface Options → SPI.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(devices)} spidev device(s):\n")
        for bus, device in sorted(devices):
            result = check_one(spidev, bus, device)
            if result is not None:
                mode, desc = result
                ok = " (ADXL345 OK)" if mode == 3 else ""
                print(f"  /dev/spidev{bus}.{device}: mode = {desc}{ok}")
            else:
                print(f"  /dev/spidev{bus}.{device}: (could not read mode)")
        return

    result = check_one(spidev, args.bus, args.device)
    if result is None:
        print(f"Cannot open /dev/spidev{args.bus}.{args.device}.", file=sys.stderr)
        print("Ensure SPI is enabled (raspi-config → Interface Options → SPI).", file=sys.stderr)
        print("Use --all to list all available devices.", file=sys.stderr)
        sys.exit(1)
    mode, desc = result
    print(f"/dev/spidev{args.bus}.{args.device}: mode = {desc}")
    if mode != 3:
        print("ADXL345 needs mode 3. Set with: spi.mode = 3 (or use tap_testing with spidev).", file=sys.stderr)


if __name__ == "__main__":
    main()
