"""
Run a multi-tap test cycle: 3 successive tap tests (e.g. 15 s apart), then combine
the data and run analysis + visualization. Optional status LED on Raspberry Pi GPIO:
ON = recording (tap now), OFF = waiting.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .analyze import analyze_tap_data, load_tap_csv, plot_result
from .config import get_config
from .record_tap import record_tap


def _led_available(gpio_bcm: int | None) -> bool:
    """Try to init GPIO for LED; return True if we can use it."""
    if gpio_bcm is None:
        return False
    try:
        import RPi.GPIO as GPIO  # noqa: F401
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(gpio_bcm, GPIO.OUT)
        GPIO.output(gpio_bcm, GPIO.LOW)
        return True
    except Exception:
        return False


def _led_on(gpio_bcm: int) -> None:
    try:
        import RPi.GPIO as GPIO
        GPIO.output(gpio_bcm, GPIO.HIGH)
    except Exception:
        pass


def _led_off(gpio_bcm: int) -> None:
    try:
        import RPi.GPIO as GPIO
        GPIO.output(gpio_bcm, GPIO.LOW)
    except Exception:
        pass


def _combine_tap_csvs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load multiple tap CSVs and concatenate (t, ax, ay, az) with continuous time.

    Returns:
        t: combined time (s)
        data: (3, N) combined [ax, ay, az]
        sample_rate_hz
    """
    if not paths:
        raise ValueError("No tap files to combine")
    all_t: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    sample_rate_hz = 0.0
    offset = 0.0
    dt = 0.0
    for p in paths:
        t, data, sr = load_tap_csv(p)
        if sample_rate_hz <= 0:
            sample_rate_hz = sr
        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
        all_t.append(t + offset)
        all_data.append(data)
        offset = offset + float(t[-1]) - float(t[0]) + dt
    t_combined = np.concatenate(all_t)
    data_combined = np.concatenate(all_data, axis=1)
    return t_combined, data_combined, sample_rate_hz


def run_cycle(
    output_dir: str | Path | None = None,
    iterations: int = 3,
    spacing_s: float = 15.0,
    duration_s: float | None = None,
    sample_rate_hz: float | None = None,
    led_gpio: int | None = None,
    flute_count: int = 4,
    max_rpm: float = 24000.0,
    tool_diameter_mm: float | None = None,
    plot: bool = True,
    plot_out: str | Path | None = None,
) -> tuple[list[Path], Path | None]:
    """
    Run multiple tap tests in sequence, combine data, analyze and optionally plot.

    LED (if led_gpio set): ON during each recording, OFF during wait.
    Saves per-tap CSVs and one combined CSV; analysis uses the combined data.

    Returns:
        (list of per-tap CSV paths, path to combined CSV or None)
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "cycle"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if led_gpio is None:
        led_gpio = cfg.status_led_gpio
    use_led = _led_available(led_gpio)
    if use_led:
        print(f"Status LED on GPIO {led_gpio}: ON = tap now, OFF = wait")
    else:
        print("No status LED (set status_led_gpio or connect LED to Pi GPIO). Use console prompts.")

    duration_s = duration_s or cfg.record_duration_s
    sample_rate_hz = sample_rate_hz or cfg.sample_rate_hz
    tap_paths: list[Path] = []

    for i in range(1, iterations + 1):
        tap_path = run_dir / f"tap_{i}.csv"
        print(f"\n--- Tap test {i}/{iterations} ---")
        if use_led:
            print("LED ON — TAP the tool now.")
        else:
            print("TAP the tool now.")
        if use_led:
            _led_on(led_gpio)
        try:
            record_tap(tap_path, duration_s=duration_s, sample_rate_hz=sample_rate_hz)
        finally:
            if use_led:
                _led_off(led_gpio)
        tap_paths.append(tap_path)
        print(f"Saved {tap_path.name}")

        if i < iterations:
            print(f"Waiting {spacing_s:.0f} s until next tap...")
            time.sleep(spacing_s)

    print("\nCombining data and analyzing...")
    t_combined, data_combined, sr = _combine_tap_csvs(tap_paths)
    combined_csv = run_dir / "combined.csv"
    with combined_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        writer.writerow(("# sample_rate_hz", sr, "", ""))
        for j in range(len(t_combined)):
            writer.writerow((t_combined[j], data_combined[0, j], data_combined[1, j], data_combined[2, j]))

    result = analyze_tap_data(
        t_combined, data_combined, sr,
        flute_count=flute_count,
        max_rpm=max_rpm,
        tool_diameter_mm=tool_diameter_mm,
    )

    print(f"\nNatural frequency: {result.natural_freq_hz:.1f} Hz")
    print(f"Avoid RPM: {result.avoid_rpm}")
    print(f"Suggested RPM range: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f}")

    # Save and optionally show chart (RPM bands + 3 tap traces + average)
    chart_path = plot_out if plot_out is not None else run_dir / "rpm_chart.png"
    from .analyze import load_tap_csv as load_csv, plot_cycle_result_figure
    tap_series_list = [(load_csv(p)[0], load_csv(p)[1]) for p in tap_paths]
    fig = plot_cycle_result_figure(result, tap_series_list, output_path=chart_path)
    print(f"Chart saved to {chart_path} (view anytime)")
    if plot:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close(fig)

    return tap_paths, combined_csv


def main() -> None:
    import argparse
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Run 3 tap tests (15 s apart), combine data, analyze and plot. LED: ON=tap, OFF=wait.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for run (default: data/cycle/<timestamp>)",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=cfg.cycle_iterations,
        help=f"Number of tap tests (default {cfg.cycle_iterations})",
    )
    parser.add_argument(
        "-s", "--spacing",
        type=float,
        default=cfg.cycle_spacing_s,
        help=f"Seconds between tap tests (default {cfg.cycle_spacing_s})",
    )
    parser.add_argument("-d", "--duration", type=float, default=None, help="Recording duration per tap (s)")
    parser.add_argument("-r", "--rate", type=float, default=None, help="Sample rate (Hz)")
    parser.add_argument(
        "--led-pin",
        type=int,
        default=None,
        metavar="GPIO",
        help=f"GPIO (BCM) for status LED; ON=record, OFF=wait (default {cfg.status_led_gpio}). Use --no-led to disable.",
    )
    parser.add_argument("--no-led", action="store_true", help="Do not use status LED")
    parser.add_argument("--flutes", type=int, default=4, help="Flute count for analysis")
    parser.add_argument("--max-rpm", type=float, default=24000, help="Max spindle RPM")
    parser.add_argument("--tool-diameter", type=float, default=None, metavar="MM", help="Tool diameter (mm)")
    parser.add_argument("--plot", action="store_true", help="Show RPM chart after run")
    parser.add_argument("--plot-out", type=Path, default=None, help="Save chart to file (default: run_dir/rpm_chart.png)")
    parser.add_argument("--no-plot", action="store_true", help="Do not show or save chart")
    parser.add_argument("--gui", action="store_true", help="Show status window and chart on Pi (same as cycle_gui)")
    args = parser.parse_args()

    if args.gui:
        from .cycle_gui import run_cycle_gui
        run_cycle_gui(
            output_dir=args.output_dir,
            iterations=args.iterations,
            spacing_s=args.spacing,
            duration_s=args.duration,
            sample_rate_hz=args.rate,
            led_gpio=None if args.no_led else (args.led_pin if args.led_pin is not None else cfg.status_led_gpio),
            flute_count=args.flutes,
            max_rpm=args.max_rpm,
            tool_diameter_mm=args.tool_diameter,
        )
        return

    led_gpio = None if args.no_led else (args.led_pin if args.led_pin is not None else cfg.status_led_gpio)
    do_plot = (args.plot or args.plot_out is not None) and not args.no_plot

    run_cycle(
        output_dir=args.output_dir,
        iterations=args.iterations,
        spacing_s=args.spacing,
        duration_s=args.duration,
        sample_rate_hz=args.rate,
        led_gpio=led_gpio,
        flute_count=args.flutes,
        max_rpm=args.max_rpm,
        tool_diameter_mm=args.tool_diameter,
        plot=do_plot and args.plot_out is None,
        plot_out=args.plot_out,
    )


if __name__ == "__main__":
    main()
