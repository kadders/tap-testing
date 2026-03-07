"""
GUI for homing calibration: record a single stream of ADXL data while the machine
runs its homing sequence (Z to 0 upward, then X and Y to home). Used to capture
motion and direction during homing for spindle-mounted accelerometer calibration.

Shows a live X/Y plot (ax vs ay in g) from the accelerometer whether or not
recording is active. Does not rely on Moonraker or Klippy.
Run with: python -m tap_testing.homing_calibration_gui
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ModuleNotFoundError as e:
    if "tkinter" in str(e).lower():
        raise ModuleNotFoundError(
            "tkinter is required for the homing calibration GUI. "
            "Install the Tcl/Tk package for your system (e.g. sudo apt install python3-tk)."
        ) from e
    raise

# Use TkAgg so matplotlib embeds in tkinter
import matplotlib
matplotlib.use("TkAgg")

# TkAgg needs PIL.ImageTk; system PIL on Linux often omits it (use pip Pillow or python3-pil.imagetk)
try:
    from PIL import ImageTk  # noqa: F401
except ImportError as e:
    raise ImportError(
        "PIL.ImageTk is required for the homing calibration GUI (matplotlib TkAgg backend). "
        "Install Pillow with:  pip install Pillow  "
        "On Debian/Ubuntu you can instead run:  sudo apt install python3-pil.imagetk"
    ) from e

from .accelerometer import open_accelerometer
from .config import get_config
from .record_tap import record_stream

# CSV columns: driver returns m/s²; 1 g = 9.80665 m/s²
_MPS2_TO_G = 1.0 / 9.80665

# Max points to show in the X/Y trail (longer trail = better motion visual)
_PLOT_TRAIL_LEN = 8000
# Axis range for live plot (g); homing motion often under ±0.5 g, 1.5 g scale for visibility
_PLOT_RANGE_G = 1.5

logger = logging.getLogger(__name__)

# Homing sequence description and G-code hint (for user to run on machine; we do not send it).
HOMING_INSTRUCTIONS = (
    "Homing sequence (run on your machine — not sent by this app):\n"
    "  1. Move Z to 0 (upward).\n"
    "  2. Home X and Y to their home positions.\n"
    "Typical G-code:  G28 Z   then  G28 X Y\n"
    "Start recording below, then run the homing on the machine; stop when done."
)


def _friendly_accel_error(err_msg: str) -> str | None:
    """Return a short GUI message for accelerometer errors, or None."""
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
            "communication failed",
            "i2c",
            "spi",
        )
    ):
        return (
            "Accelerometer error. Check wiring and I2C/SPI (raspi-config). "
            "See docs/ADXL345_WIRING.md and log for details."
        )
    return None


def _run_recording_worker(
    output_path: Path,
    stop_event: threading.Event,
    sample_rate_hz: float,
    root: tk.Tk,
    status_var: tk.StringVar,
    live_var: tk.StringVar,
    on_done: Callable[[], None] | None,
    accel: object | None = None,
    on_plot_point: Callable[[float, float], None] | None = None,
) -> None:
    """Run record_stream in a background thread; update GUI via root.after()."""
    def status(msg: str) -> None:
        logger.info(msg)
        root.after(0, lambda: status_var.set(msg))

    def done() -> None:
        if on_done:
            root.after(0, on_done)

    def sample_callback(t: float, x_g: float, y_g: float, z_g: float) -> None:
        # x_g, y_g, z_g are sensor X,Y,Z in g (same order as accelerometer.acceleration → CSV ax_g, ay_g, az_g)
        mag = (x_g * x_g + y_g * y_g + z_g * z_g) ** 0.5
        s = f"t={t:.2f}s  x={x_g:+.3f}  y={y_g:+.3f}  z={z_g:+.3f}  |a|={mag:.3f} g"
        root.after(0, lambda: live_var.set(s))
        if on_plot_point is not None:
            # Plot horizontal plane: ax = sensor X (g), ay = sensor Y (g)
            root.after(0, lambda ax=x_g, ay=y_g: on_plot_point(ax, ay))

    try:
        record_stream(
            output_path,
            stop_event,
            sample_rate_hz=sample_rate_hz,
            sample_callback=sample_callback,
            callback_interval_s=0.05,
            accel=accel,
        )
        status(f"Recording stopped. Saved to: {output_path}")
    except Exception as e:
        logger.exception("Recording failed")
        friendly = _friendly_accel_error(str(e)) or f"Error: {e}"
        root.after(0, lambda msg=friendly: status_var.set(msg))
    finally:
        done()


def run_homing_calibration_gui(
    output_dir: str | Path | None = None,
    sample_rate_hz: float | None = None,
) -> None:
    """Open the homing calibration GUI: design parameters, Start/Stop recording, no Moonraker/Klippy."""
    cfg = get_config()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "live_spindle" / "homing"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else cfg.sample_rate_hz

    try:
        root = tk.Tk()
    except tk.TclError as e:
        if "display" in str(e).lower():
            raise RuntimeError(
                "No display available. Run with a monitor, X11 forwarding (ssh -X), or xvfb-run."
            ) from e
        raise

    root.title("Homing calibration — live spindle recording")
    root.minsize(520, 520)
    root.geometry("720x620")
    root.resizable(True, True)

    # Shared accelerometer (opened once; used for live plot and recording)
    shared_accel: object | None = None
    # Trail of (ax_g, ay_g) for X/Y plot
    plot_trail_x: deque = deque(maxlen=_PLOT_TRAIL_LEN)
    plot_trail_y: deque = deque(maxlen=_PLOT_TRAIL_LEN)
    # Timer id for live plot updates when not recording
    live_plot_timer_id: str | None = None
    # Timer id for "clear plot every 5 s" option
    clear_plot_timer_id: str | None = None
    recording_active = False

    # --- Design parameters frame ---
    params_frame = tk.LabelFrame(root, text="Design parameters", padx=10, pady=8)
    params_frame.pack(pady=8, padx=10, fill=tk.X)

    row = 0
    tk.Label(params_frame, text="Output directory:").grid(row=row, column=0, sticky=tk.W, pady=2)
    output_dir_var = tk.StringVar(value=str(output_dir))
    output_dir_entry = tk.Entry(params_frame, textvariable=output_dir_var, width=55)
    output_dir_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
    row += 1

    tk.Label(params_frame, text="Sample rate (Hz):").grid(row=row, column=0, sticky=tk.W, pady=2)
    rate_var = tk.StringVar(value=str(int(sample_rate_hz)))
    tk.Spinbox(params_frame, from_=100, to=3200, increment=100, textvariable=rate_var, width=8).grid(
        row=row, column=1, sticky=tk.W, padx=5, pady=2
    )
    row += 1

    clear_every_5s_var = tk.BooleanVar(value=False)
    clear_5s_cb = tk.Checkbutton(
        params_frame,
        text="Clear plot every 5 s",
        variable=clear_every_5s_var,
    )
    clear_5s_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
    row += 1

    params_frame.columnconfigure(1, weight=1)

    # --- Instructions (homing sequence; run on machine) ---
    inst_frame = tk.LabelFrame(root, text="Homing sequence (run on your machine)", padx=10, pady=8)
    inst_frame.pack(pady=8, padx=10, fill=tk.X)
    inst_text = tk.Label(
        inst_frame,
        text=HOMING_INSTRUCTIONS,
        justify=tk.LEFT,
        wraplength=580,
        font=("TkDefaultFont", 9),
        fg="gray20",
    )
    inst_text.pack(anchor=tk.W)

    # --- Status ---
    status_var = tk.StringVar(value="Set parameters, then Start recording. Run homing on the machine; Stop when done.")
    status_font = tkfont.Font(size=12, weight="bold")
    status_label = tk.Label(root, textvariable=status_var, font=status_font, wraplength=600)
    status_label.pack(pady=8, padx=10, fill=tk.X)

    # --- Live stream ---
    live_var = tk.StringVar(value="Live: — (data appears when accelerometer is connected)")
    live_frame = tk.Frame(root)
    live_frame.pack(pady=2, padx=10, fill=tk.X)
    tk.Label(live_frame, text="Accelerometer:", font=tkfont.Font(weight="bold")).pack(side=tk.LEFT, padx=(0, 5))
    live_label = tk.Label(live_frame, textvariable=live_var, font=("TkFixedFont", 10), fg="darkgreen")
    live_label.pack(side=tk.LEFT)

    # --- Start / Stop recording (packed before plot so buttons stay visible) ---
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, padx=10, fill=tk.X)

    stop_event: threading.Event | None = None

    def start_recording() -> None:
        nonlocal stop_event, recording_active, live_plot_timer_id, clear_plot_timer_id
        if shared_accel is None:
            status_var.set("Accelerometer not available. Check connection and retry.")
            return
        out_dir = Path(output_dir_var.get().strip() or str(output_dir))
        try:
            rate = float(rate_var.get())
        except ValueError:
            rate = sample_rate_hz
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "homing.csv"

        recording_active = True
        if live_plot_timer_id is not None:
            root.after_cancel(live_plot_timer_id)
            live_plot_timer_id = None
        if clear_plot_timer_id is not None:
            root.after_cancel(clear_plot_timer_id)
            clear_plot_timer_id = None

        status_var.set(f"Recording… Saving to {run_dir.name}/homing.csv — Run homing on the machine (Z up, then X Y home). Click Stop when done.")
        start_btn.config(state=tk.DISABLED, text="Recording…")
        stop_btn.config(state=tk.NORMAL)
        live_var.set("Live: streaming…")

        stop_event = threading.Event()

        def on_done() -> None:
            nonlocal recording_active, live_plot_timer_id, clear_plot_timer_id
            recording_active = False
            start_btn.config(state=tk.NORMAL, text="Start recording")
            stop_btn.config(state=tk.DISABLED)
            if shared_accel is not None:
                live_plot_timer_id = root.after(50, live_update_plot)
            if clear_every_5s_var.get():
                clear_plot_timer_id = root.after(5000, _clear_plot_every_5s)

        def on_plot_point(ax_g: float, ay_g: float) -> None:
            append_plot_point(ax_g, ay_g)
            redraw_plot()

        thread = threading.Thread(
            target=_run_recording_worker,
            args=(
                output_path,
                stop_event,
                rate,
                root,
                status_var,
                live_var,
                on_done,
                shared_accel,
                on_plot_point,
            ),
            daemon=True,
        )
        thread.start()

    def stop_recording() -> None:
        if stop_event is not None:
            stop_event.set()

    start_btn = tk.Button(
        button_frame,
        text="Start recording",
        font=tkfont.Font(size=14),
        command=start_recording,
    )
    start_btn.pack(side=tk.LEFT, padx=5)

    stop_btn = tk.Button(
        button_frame,
        text="Stop recording",
        font=tkfont.Font(size=14),
        state=tk.DISABLED,
        command=stop_recording,
    )
    stop_btn.pack(side=tk.LEFT, padx=5)

    # --- Live X/Y plot (ax vs ay in g); ax = sensor X, ay = sensor Y (ADXL345 DATAX/DATAY) ---
    plot_frame = tk.LabelFrame(root, text="Live X/Y (ax vs ay, g) — from accelerometer", padx=4, pady=4)
    plot_frame.pack(pady=6, padx=10, fill=tk.BOTH, expand=True)

    fig = None
    line_xy = None
    canvas = None

    def _init_plot() -> None:
        nonlocal fig, line_xy, canvas
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_xlabel("ax (g)")
        ax.set_ylabel("ay (g)")
        ax.set_title("Horizontal plane acceleration")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        line_xy, = ax.plot([], [], "b-", linewidth=0.8, alpha=0.8)
        ax.set_xlim(-_PLOT_RANGE_G, _PLOT_RANGE_G)
        ax.set_ylim(-_PLOT_RANGE_G, _PLOT_RANGE_G)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def append_plot_point(ax_g: float, ay_g: float) -> None:
        plot_trail_x.append(ax_g)
        plot_trail_y.append(ay_g)

    def redraw_plot() -> None:
        if line_xy is None or canvas is None:
            return
        if not plot_trail_x:
            return
        line_xy.set_data(list(plot_trail_x), list(plot_trail_y))
        canvas.draw_idle()

    def live_update_plot() -> None:
        """Read accelerometer and update X/Y plot; reschedule when not recording."""
        nonlocal live_plot_timer_id
        if recording_active or shared_accel is None:
            if not recording_active and shared_accel is not None:
                live_plot_timer_id = root.after(50, live_update_plot)
            return
        try:
            # acceleration is (X, Y, Z) in m/s²; convert to g for plot (same mapping as recording CSV)
            x, y, _ = shared_accel.acceleration
            ax_g = x * _MPS2_TO_G
            ay_g = y * _MPS2_TO_G
            append_plot_point(ax_g, ay_g)
            redraw_plot()
        except Exception:
            pass
        live_plot_timer_id = root.after(50, live_update_plot)

    _init_plot()

    def _clear_plot_every_5s() -> None:
        nonlocal clear_plot_timer_id
        plot_trail_x.clear()
        plot_trail_y.clear()
        if line_xy is not None and canvas is not None:
            line_xy.set_data([], [])
            canvas.draw_idle()
        if clear_every_5s_var.get():
            clear_plot_timer_id = root.after(5000, _clear_plot_every_5s)

    def _on_clear_5s_toggled() -> None:
        nonlocal clear_plot_timer_id
        if clear_every_5s_var.get():
            if clear_plot_timer_id is not None:
                root.after_cancel(clear_plot_timer_id)
            clear_plot_timer_id = root.after(5000, _clear_plot_every_5s)
        else:
            if clear_plot_timer_id is not None:
                root.after_cancel(clear_plot_timer_id)
                clear_plot_timer_id = None

    clear_5s_cb.config(command=_on_clear_5s_toggled)

    # Pre-flight: open accelerometer once for live plot and recording
    def check_accel() -> None:
        nonlocal shared_accel, live_plot_timer_id
        try:
            shared_accel = open_accelerometer(
                address=cfg.i2c_address,
                range_g=cfg.range_g,
                bus=cfg.i2c_bus,
                interface=cfg.adxl345_interface,
                spi_cs_pin=cfg.spi_cs_pin,
            )
            status_var.set("Accelerometer OK. X/Y plot is live. Set parameters, then Start recording when ready.")
            live_var.set("Live: streaming to X/Y plot…")
            if not recording_active:
                live_plot_timer_id = root.after(50, live_update_plot)
        except Exception as e:
            shared_accel = None
            friendly = _friendly_accel_error(str(e)) or str(e)
            status_var.set(friendly)

    root.after(300, check_accel)
    root.mainloop()


def main() -> None:
    import argparse

    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Homing calibration: record ADXL stream while machine homes (Z then X Y). No Moonraker/Klippy."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Base output directory; each run is saved as <output-dir>/<YYYYmmdd_HHMMSS>/homing.csv (default: data/live_spindle/homing)",
    )
    parser.add_argument("-r", "--rate", type=float, default=None, help="Sample rate (Hz)")
    args = parser.parse_args()

    run_homing_calibration_gui(
        output_dir=args.output_dir,
        sample_rate_hz=args.rate or cfg.sample_rate_hz,
    )


if __name__ == "__main__":
    main()
