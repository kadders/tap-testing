"""
GUI for the tap test cycle on the Pi: window with live status and RPM chart when done.
Run with: python -m tap_testing.cycle_gui
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable

# Use TkAgg so matplotlib figures can be embedded in tkinter
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import font as tkfont

from .analyze import analyze_tap_data, load_tap_csv, plot_result_figure
from .config import get_config
from .record_tap import record_tap
from .run_cycle import _combine_tap_csvs, _led_available, _led_off, _led_on


def _run_cycle_worker(
    run_dir: Path,
    iterations: int,
    spacing_s: float,
    duration_s: float,
    sample_rate_hz: float,
    led_gpio: int | None,
    flute_count: int,
    max_rpm: float,
    tool_diameter_mm: float | None,
    root: tk.Tk,
    status_var: tk.StringVar,
    chart_frame: tk.Frame,
    result_holder: list,
    save_chart_path: Path,
    on_done: Callable[[], None] | None,
) -> None:
    """Run the tap cycle in a background thread; update GUI via root.after()."""
    def status(msg: str) -> None:
        root.after(0, lambda: status_var.set(msg))

    def done() -> None:
        if on_done:
            root.after(0, on_done)

    use_led = _led_available(led_gpio)
    tap_paths: list[Path] = []

    try:
        for i in range(1, iterations + 1):
            status(f"Tap {i}/{iterations} — TAP the tool now.")
            if use_led:
                _led_on(led_gpio)
            try:
                tap_path = run_dir / f"tap_{i}.csv"
                record_tap(tap_path, duration_s=duration_s, sample_rate_hz=sample_rate_hz)
                tap_paths.append(tap_path)
            finally:
                if use_led:
                    _led_off(led_gpio)

            if i < iterations:
                status(f"Saved tap {i}. Waiting {int(spacing_s)} s until next tap...")
                time.sleep(spacing_s)

        status("Combining and analyzing...")
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
        result_holder.append(result)

        summary = (
            f"Done. Natural freq: {result.natural_freq_hz:.1f} Hz  |  "
            f"Suggested RPM: {result.suggested_rpm_min:.0f} – {result.suggested_rpm_max:.0f}\n"
            f"Chart saved to: {save_chart_path} (view anytime)"
        )
        status(summary)

        # Embed chart on main thread and save to run dir (with tap paths for traces)
        root.after(0, lambda: _embed_chart(chart_frame, result, save_chart_path, tap_paths))
        done()
    except Exception as e:
        root.after(0, lambda: status_var.set(f"Error: {e}"))
        done()


def _embed_chart(
    parent: tk.Frame,
    result,
    save_path: Path | None = None,
    tap_paths: list | None = None,
) -> None:
    """Create RPM chart + tap traces figure and embed (call on main thread)."""
    for w in parent.winfo_children():
        w.destroy()
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from .analyze import load_tap_csv, plot_cycle_result_figure, plot_result_figure
        if tap_paths and len(tap_paths) >= 1:
            tap_series_list = [(load_tap_csv(p)[0], load_tap_csv(p)[1]) for p in tap_paths]
            fig = plot_cycle_result_figure(
                result, tap_series_list, output_path=save_path, figsize=(8, 5)
            )
        else:
            fig = plot_result_figure(result, output_path=save_path, figsize=(8, 2.2))
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    except Exception:
        for w in parent.winfo_children():
            w.destroy()
        label = tk.Label(parent, text="Could not draw chart.", fg="red")
        label.pack(expand=True)


def run_cycle_gui(
    output_dir: str | Path | None = None,
    iterations: int = 3,
    spacing_s: float = 15.0,
    duration_s: float | None = None,
    sample_rate_hz: float | None = None,
    led_gpio: int | None = None,
    flute_count: int = 4,
    max_rpm: float = 24000.0,
    tool_diameter_mm: float | None = None,
) -> None:
    """Open the cycle GUI window and run the tap cycle with status + chart."""
    cfg = get_config()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "cycle"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_s = duration_s or cfg.record_duration_s
    sample_rate_hz = sample_rate_hz or cfg.sample_rate_hz
    if led_gpio is None:
        led_gpio = cfg.status_led_gpio

    root = tk.Tk()
    root.title("Tap Test Cycle — Milo")
    root.minsize(500, 400)
    root.geometry("800x500")

    # Status area
    status_var = tk.StringVar(value="Initializing...")
    status_font = tkfont.Font(size=18, weight="bold")
    status_label = tk.Label(root, textvariable=status_var, font=status_font, wraplength=700)
    status_label.pack(pady=15, padx=10, fill=tk.X)

    # Buttons: Start new test cycle
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5, padx=10, fill=tk.X)
    new_cycle_btn = tk.Button(
        button_frame,
        text="Start new test cycle",
        font=tkfont.Font(size=14),
        command=lambda: None,  # set below
    )
    new_cycle_btn.pack(side=tk.LEFT, padx=5)

    # Chart area (filled when cycle completes)
    chart_frame = tk.Frame(root, relief=tk.GROOVE, bd=2)
    chart_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    placeholder = tk.Label(chart_frame, text="Chart will appear after 3 tap tests complete.", fg="gray")
    placeholder.pack(expand=True)

    result_holder: list = []

    def start_worker() -> None:
        # Clear chart area and show placeholder
        for w in chart_frame.winfo_children():
            w.destroy()
        pl = tk.Label(chart_frame, text="Chart will appear after 3 tap tests complete.", fg="gray")
        pl.pack(expand=True)
        new_cycle_btn.config(state=tk.DISABLED, text="Running…")

        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        save_chart_path = run_dir / "rpm_chart.png"

        status_var.set("Tap 1/3 — TAP the tool now.")

        def on_done() -> None:
            new_cycle_btn.config(state=tk.NORMAL, text="Start new test cycle")

        thread = threading.Thread(
            target=_run_cycle_worker,
            args=(
                run_dir,
                iterations,
                spacing_s,
                duration_s,
                sample_rate_hz,
                led_gpio,
                flute_count,
                max_rpm,
                tool_diameter_mm,
                root,
                status_var,
                chart_frame,
                result_holder,
                save_chart_path,
                on_done,
            ),
            daemon=True,
        )
        thread.start()

    new_cycle_btn.config(command=start_worker)
    root.after(500, start_worker)
    root.mainloop()


def main() -> None:
    import argparse
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Tap test cycle with status window and RPM chart on Pi.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-n", "--iterations", type=int, default=cfg.cycle_iterations)
    parser.add_argument("-s", "--spacing", type=float, default=cfg.cycle_spacing_s)
    parser.add_argument("-d", "--duration", type=float, default=None)
    parser.add_argument("-r", "--rate", type=float, default=None)
    parser.add_argument("--led-pin", type=int, default=None)
    parser.add_argument("--no-led", action="store_true", help="Do not use status LED")
    parser.add_argument("--flutes", type=int, default=4)
    parser.add_argument("--max-rpm", type=float, default=24000)
    parser.add_argument("--tool-diameter", type=float, default=None, metavar="MM")
    args = parser.parse_args()

    led_gpio = None if args.no_led else (args.led_pin or cfg.status_led_gpio)

    run_cycle_gui(
        output_dir=args.output_dir,
        iterations=args.iterations,
        spacing_s=args.spacing,
        duration_s=args.duration,
        sample_rate_hz=args.rate,
        led_gpio=led_gpio,
        flute_count=args.flutes,
        max_rpm=args.max_rpm,
        tool_diameter_mm=args.tool_diameter,
    )


if __name__ == "__main__":
    main()
