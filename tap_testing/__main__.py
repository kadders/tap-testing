"""
Top-level entry point for tap_testing. Run:

  python -m tap_testing
  python -m tap_testing --help
  python -m tap_testing -h

to see available commands. Each command is run as:

  python -m tap_testing.<module> [args...]
"""
from __future__ import annotations

import argparse
import sys


def _make_help_text() -> str:
    return """Tap testing - accelerometer-based vibration capture for tool tap tests.

Usage: run a command by invoking its module, e.g.:

  python -m tap_testing.record_tap [options]
  python -m tap_testing.run_cycle [options]
  python -m tap_testing.analyze <csv> [options]

Recording and analysis
  record_tap           Record a single tap to CSV (output path, duration, sample rate)
  run_cycle            Run 3-tap cycle: record, combine, analyze (optional LED)
  cycle_gui            Same as run_cycle with GUI: live status + RPM chart
  homing_gui           Record ADXL during machine homing (Z then X/Y) for spindle calibration
  analyze              Analyze CSV: natural frequency, RPM bands, charts (--plot, --workflow)

Diagnostics and inspection
  check_sensor         Live X,Y,Z and magnitude stream until Ctrl+C
  verify_spi_accel     Probe for ADXL345 (--probe), stream samples, gravity check
  check_spi_mode       List spidev devices and SPI mode (ADXL345 needs Mode 3)
  inspect_tap_data    Per-file or per-cycle stats and signal check (OK / NO SIGNAL)

Documentation helpers
  docs.generate_example_chart   Generate example chart images (synthetic data)

Examples
  python -m tap_testing.record_tap -o data/tap_001.csv
  python -m tap_testing.run_cycle --plot
  python -m tap_testing.analyze data/tap_001.csv --flutes 4 --max-rpm 24000 --plot

See README.md and docs/ for full usage and workflows.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tap_testing",
        description="Tap testing: accelerometer-based vibration capture. Show available commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_make_help_text(),
    )
    # No args: show command list; --help/-h is handled by argparse and exits after printing
    if len(sys.argv) == 1:
        parser.print_help()
        return
    parser.parse_known_args()
    parser.print_help()


if __name__ == "__main__":
    main()
