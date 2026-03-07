"""
GUI for the tap test cycle on the Pi: window with live status and RPM chart when done.
Run with: python -m tap_testing.cycle_gui
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Use TkAgg so matplotlib figures can be embedded in tkinter
import matplotlib
matplotlib.use("TkAgg")

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ModuleNotFoundError as e:
    if "tkinter" in str(e).lower():
        raise ModuleNotFoundError(
            "tkinter is required for the cycle GUI but is not installed. "
            "Install the Tcl/Tk package for your system, then use the same Python:\n"
            "  Debian/Ubuntu/Raspberry Pi OS: sudo apt install python3-tk\n"
            "  Fedora: sudo dnf install python3-tkinter\n"
            "  macOS (Homebrew): brew install python-tk\n"
            "  Windows: tkinter is usually included with the Python installer."
        ) from e
    raise

from .analyze import (
    analyze_tap_data,
    format_milling_guidance_for_cycle,
    load_tap_csv,
    plot_result_figure,
)
from .accelerometer import open_accelerometer
from .config import get_config, rpm_from_spindle_frequency_hz
from .record_tap import record_tap
from .run_cycle import _combine_tap_csvs, _led_available, _led_off, _led_on


def _friendly_i2c_error(err_msg: str) -> str | None:
    """Return a short GUI message for I2C/accelerometer errors, or None if not applicable."""
    lower = err_msg.lower()
    if any(
        x in lower or x in err_msg
        for x in (
            "no i2c device",
            "adxl345 not found",
            "adxl345 spi",
            "0x53",
            "0x1d",
            "remote i/o",
            "i/o error",
            "errno 121",
            "errno 5",
            "communication failed",
            "i2c",
            "spi",
        )
    ):
        return (
            "Accelerometer error. Check wiring and I2C/SPI enabled (raspi-config). "
            "I2C: try i2cdetect -y 1. SPI: see docs/ADXL345_WIRING.md. See log for details."
        )
    if "no usable signal" in lower or "all zeros" in lower or "constant" in lower:
        return (
            "Tap data has no signal (all zeros). Check accelerometer connection. "
            "Run: python -m tap_testing.inspect_tap_data <cycle_dir> to inspect saved data."
        )
    return None


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
    tool_material: str | None,
    root: tk.Tk,
    status_var: tk.StringVar,
    live_var: tk.StringVar,
    chart_frame: tk.Frame,
    result_holder: list,
    save_chart_path: Path,
    on_done: Callable[[], None] | None,
    spindle_operating_frequency_hz: float | None = None,
) -> None:
    """Run the tap cycle in a background thread; update GUI via root.after()."""
    def status(msg: str) -> None:
        logger.info(msg)
        root.after(0, lambda: status_var.set(msg))

    def done() -> None:
        if on_done:
            root.after(0, on_done)

    def sample_callback(t: float, x_g: float, y_g: float, z_g: float) -> None:
        mag = (x_g * x_g + y_g * y_g + z_g * z_g) ** 0.5
        s = f"t={t:.2f}s  x={x_g:+.3f}  y={y_g:+.3f}  z={z_g:+.3f}  |a|={mag:.3f} g"
        root.after(0, lambda: live_var.set(s))

    use_led = _led_available(led_gpio)
    tap_paths: list[Path] = []

    try:
        for i in range(1, iterations + 1):
            status(f"Tap {i}/{iterations} — TAP the tool now.")
            root.after(0, lambda: live_var.set("Live: streaming…"))
            if use_led:
                _led_on(led_gpio)
            try:
                tap_path = run_dir / f"tap_{i}.csv"
                cfg = get_config()

                def on_tap_detected(t_s: float) -> None:
                    status(f"Tap detected at t={t_s:.2f} s — recording…")

                record_tap(
                    tap_path,
                    duration_s=duration_s,
                    sample_rate_hz=sample_rate_hz,
                    sample_callback=sample_callback,
                    callback_interval_s=0.05,
                    on_tap_detected=on_tap_detected,
                    impact_threshold_g=cfg.impact_threshold_g,
                )
                tap_paths.append(tap_path)
            finally:
                if use_led:
                    _led_off(led_gpio)

            if i < iterations:
                root.after(0, lambda: live_var.set("Live: — (waiting for next tap)"))
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

        spindle_hz = spindle_operating_frequency_hz if spindle_operating_frequency_hz is not None else get_config().spindle_operating_frequency_hz
        result = analyze_tap_data(
            t_combined, data_combined, sr,
            flute_count=flute_count,
            max_rpm=max_rpm,
            tool_diameter_mm=tool_diameter_mm,
            tool_material=tool_material,
            spindle_operating_frequency_hz=spindle_hz,
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
        logger.exception("Cycle failed")
        friendly = _friendly_i2c_error(str(e)) or f"Error: {e}"
        root.after(0, lambda msg=friendly: status_var.set(msg))
        done()


def _embed_chart(
    parent: tk.Frame,
    result,
    save_path: Path | None = None,
    tap_paths: list | None = None,
) -> None:
    """Create RPM chart + tap traces figure and embed; show milling guidance text (call on main thread)."""
    for w in parent.winfo_children():
        w.destroy()
    material_name = get_config().material_name
    guidance_text = format_milling_guidance_for_cycle(result, material_name)
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from .analyze import load_tap_csv, plot_cycle_result_figure, plot_result_figure

        if tap_paths and len(tap_paths) >= 1:
            tap_series_list = [(load_tap_csv(p)[0], load_tap_csv(p)[1]) for p in tap_paths]
            ref_hz = result.spindle_operating_frequency_hz or get_config().spindle_operating_frequency_hz
            reference_rpm = rpm_from_spindle_frequency_hz(ref_hz) if ref_hz > 0 else None
            fig = plot_cycle_result_figure(
                result, tap_series_list, output_path=save_path, figsize=(8, 5),
                material_name=material_name,
                reference_rpm=reference_rpm, reference_chip_load=0.05,
            )
        else:
            fig = plot_result_figure(result, output_path=save_path, figsize=(8, 2.2), material_name=material_name)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Milling guidance text for near real-time reading (same content as chart panel)
        guidance_frame = tk.Frame(parent)
        guidance_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(guidance_frame, text="Milling guidance (from tap result)", font=tkfont.Font(weight="bold")).pack(anchor=tk.W)
        guidance_label = tk.Label(
            guidance_frame,
            text=guidance_text,
            justify=tk.LEFT,
            wraplength=700,
            font=("TkDefaultFont", 9),
            fg="black",
        )
        guidance_label.pack(anchor=tk.W, fill=tk.X)
    except Exception as e:
        logger.exception("Could not draw chart")
        for w in parent.winfo_children():
            w.destroy()
        err_str = str(e).strip()
        is_pil = "ImageTk" in err_str or "PIL" in err_str or "Pillow" in err_str.lower()
        if is_pil:
            run_dir = save_path.parent if save_path else Path(".")
            combined = run_dir / "combined.csv"
            msg = (
                "Chart was saved to disk but could not be shown in the window.\n\n"
                "Install Pillow for in-window chart:  pip install Pillow\n\n"
                f"Chart file: {save_path}\n\n"
                "To re-analyze this run: python -m tap_testing.analyze "
                f"{combined} --flutes 4 --max-rpm 24000\n\n"
                "Milling guidance (from tap result):\n" + guidance_text
            )
        else:
            msg = err_str or "Could not draw chart."
            if len(msg) > 120:
                msg = msg[:117] + "..."
        label = tk.Label(parent, text=msg, fg="red" if not is_pil else "black", wraplength=600, justify=tk.LEFT)
        label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)


def run_cycle_gui(
    output_dir: str | Path | None = None,
    iterations: int = 3,
    spacing_s: float = 5.0,
    duration_s: float | None = None,
    sample_rate_hz: float | None = None,
    led_gpio: int | None = None,
    flute_count: int = 3,
    max_rpm: float = 24000.0,
    tool_diameter_mm: float | None = 6.0,
    tool_material: str | None = None,
    spindle_operating_frequency_hz: float | None = None,
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

    try:
        root = tk.Tk()
    except tk.TclError as e:
        if "display" in str(e).lower():
            raise RuntimeError(
                "No display available for the GUI. This usually means you're on a headless "
                "session (e.g. SSH without a display). Options:\n"
                "  • Run locally on the Pi with a monitor, or\n"
                "  • Use X11 forwarding: ssh -X or ssh -Y to the Pi, then run the GUI, or\n"
                "  • Use a virtual display: xvfb-run python3 -m tap_testing.cycle_gui"
            ) from e
        raise

    root.title("Tap Test Cycle — Milo")
    root.minsize(500, 400)
    root.geometry("800x500")
    root.resizable(True, True)

    # Status area
    status_var = tk.StringVar(value="Initializing...")
    status_font = tkfont.Font(size=18, weight="bold")
    status_label = tk.Label(root, textvariable=status_var, font=status_font, wraplength=700)
    status_label.pack(pady=15, padx=10, fill=tk.X)

    # Live stream (updates during each tap recording)
    live_var = tk.StringVar(value="Live: — (data appears during each tap)")
    live_frame = tk.Frame(root)
    live_frame.pack(pady=2, padx=10, fill=tk.X)
    tk.Label(live_frame, text="Accelerometer:", font=tkfont.Font(weight="bold")).pack(side=tk.LEFT, padx=(0, 5))
    live_label = tk.Label(live_frame, textvariable=live_var, font=("TkFixedFont", 10), fg="darkgreen")
    live_label.pack(side=tk.LEFT)

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

        # Pre-flight: fail fast with clear error if accelerometer is not reachable
        status_var.set("Checking accelerometer...")
        root.update_idletasks()
        try:
            cfg = get_config()
            open_accelerometer(
                address=cfg.i2c_address,
                range_g=cfg.range_g,
                bus=cfg.i2c_bus,
                interface=cfg.adxl345_interface,
                spi_cs_pin=cfg.spi_cs_pin,
            )
        except Exception as e:
            logger.exception("Accelerometer pre-flight check failed")
            friendly = _friendly_i2c_error(str(e)) or f"Error: {e}"
            status_var.set(friendly)
            new_cycle_btn.config(state=tk.NORMAL, text="Start new test cycle")
            return

        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        save_chart_path = run_dir / "rpm_chart.png"
        logger.info("Starting tap cycle: %d iterations -> %s", iterations, run_dir)

        status_var.set("Tap 1/3 — TAP the tool now.")

        def on_done() -> None:
            live_var.set("Live: — (cycle complete)")
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
                tool_material,
                root,
                status_var,
                live_var,
                chart_frame,
                result_holder,
                save_chart_path,
                on_done,
                spindle_operating_frequency_hz,
            ),
            daemon=True,
        )
        thread.start()

    new_cycle_btn.config(command=start_worker)
    root.after(500, start_worker)
    root.mainloop()


def _setup_logging(log_dir: Path | None = None) -> Path:
    """Configure logging to console and a file. Returns the log file path."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cycle_gui.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(console)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return log_file


def main() -> None:
    import argparse

    cfg = get_config()
    log_path = _setup_logging()
    logger.info("Logging to file: %s", log_path)
    parser = argparse.ArgumentParser(description="Tap test cycle with status window and RPM chart on Pi.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-n", "--iterations", type=int, default=cfg.cycle_iterations)
    parser.add_argument("-s", "--spacing", type=float, default=cfg.cycle_spacing_s)
    parser.add_argument("-d", "--duration", type=float, default=None)
    parser.add_argument("-r", "--rate", type=float, default=None)
    parser.add_argument("--led-pin", type=int, default=None)
    parser.add_argument("--no-led", action="store_true", help="Do not use status LED")
    parser.add_argument("--flutes", type=int, default=3)
    parser.add_argument("--max-rpm", type=float, default=24000)
    parser.add_argument("--tool-diameter", type=float, default=6.0, metavar="MM")
    from .material import list_tool_material_names
    parser.add_argument(
        "--tool-material",
        type=str,
        default=None,
        metavar="NAME",
        dest="tool_material",
        help=f"Tool (cutter) material for chart (default: carbide). Choices: {', '.join(list_tool_material_names())}",
    )
    parser.add_argument(
        "--spindle-frequency",
        type=float,
        default=None,
        metavar="HZ",
        dest="spindle_frequency_hz",
        help=f"Spindle operating frequency in Hz (rev/s); used as reference on chart. rpm = HZ × 60 (default {cfg.spindle_operating_frequency_hz}).",
    )
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
        tool_material=getattr(args, "tool_material", None),
        spindle_operating_frequency_hz=getattr(args, "spindle_frequency_hz", None),
    )


if __name__ == "__main__":
    main()
