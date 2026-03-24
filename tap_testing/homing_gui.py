"""
GUI for homing calibration: record a single stream of ADXL data while the machine
runs its homing sequence (Z to 0 upward, then X and Y to home). Used to capture
motion and direction during homing for spindle-mounted accelerometer calibration.

Shows a live X/Y plot (ax vs ay in g) from the accelerometer whether or not
recording is active. Does not rely on Moonraker or Klippy.

Optional Modbus (``--modbus``): polls holding/input registers (and optional discrete
inputs / coils) via pymodbus during recording, saves ``modbus.csv`` next to ``homing.csv``,
and shows live register traces. Configure RTU/TCP via env; H100 VFD: ``TAP_MODBUS_PROFILE=h100`` or ``--modbus-profile h100``
(see docs/MODBUS_H100_VFD.md, docs/MODBUS_RASPBERRY_PI.md).

Optional Duet / RepRapFirmware HTTP job sync: poll rr_model (RRF REST API / OpenAPI) so
recording starts when a print or simulate job is active and stops when it finishes or
cancels. Configure with TAP_RRF_BASE, TAP_RRF_POLL_S, TAP_RRF_DISCOVER_HOSTS, TAP_RRF_PASSWORD.

Run with: python -m tap_testing.homing_gui
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
except ModuleNotFoundError as e:
    if "tkinter" in str(e).lower():
        raise ModuleNotFoundError(
            "tkinter is required for the homing GUI. "
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
        "PIL.ImageTk is required for the homing GUI (matplotlib TkAgg backend). "
        "Install Pillow with:  pip install Pillow  "
        "On Debian/Ubuntu you can instead run:  sudo apt install python3-pil.imagetk"
    ) from e

from .accelerometer import open_accelerometer
from .config import get_config
from .modbus_logging import (
    ModbusLoggingConfig,
    apply_modbus_profile,
    modbus_logging_config_from_env,
    run_modbus_poll_loop,
)
from .record_tap import record_stream
from .rrf_http import (
    RrfClient,
    RrfHttpError,
    discover_rrf_base,
    infer_print_job_active,
    probe_rrf_base,
    rrf_default_base_url,
    rrf_default_poll_interval_s,
)

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


def run_homing_gui(
    output_dir: str | Path | None = None,
    sample_rate_hz: float | None = None,
    *,
    modbus: bool = False,
    modbus_config: ModbusLoggingConfig | None = None,
    rrf_job_sync: bool = False,
    rrf_base_url: str | None = None,
    rrf_poll_interval_s: float | None = None,
    rrf_password: str | None = None,
) -> None:
    """Open the homing GUI: record ADXL during machine homing (Z then X/Y). No Moonraker/Klippy.

    If modbus is True, pymodbus must be installed; during recording we poll Modbus and save modbus.csv.

    If rrf_job_sync is True, the GUI enables Duet HTTP polling to start/stop recording with print jobs.
    """
    cfg = get_config()
    mb_cfg: ModbusLoggingConfig | None = None
    if modbus:
        mb_cfg = modbus_config if modbus_config is not None else modbus_logging_config_from_env()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "live_spindle" / "homing"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = sample_rate_hz if sample_rate_hz is not None else cfg.sample_rate_hz
    rrf_base_init = (rrf_base_url or rrf_default_base_url()).strip()
    rrf_poll_init = (
        float(rrf_poll_interval_s)
        if rrf_poll_interval_s is not None
        else rrf_default_poll_interval_s()
    )
    rrf_pw_init = (
        rrf_password if rrf_password is not None else os.environ.get("TAP_RRF_PASSWORD", "")
    )

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

    # --- RepRapFirmware (Duet) HTTP job sync ---
    rrf_poller_stop = threading.Event()
    rrf_poller_thread: list[threading.Thread | None] = [None]
    rrf_last_job_active: list[bool | None] = [None]

    rrf_frame = tk.LabelFrame(root, text="Duet / RepRapFirmware — print job sync (rr_model)", padx=10, pady=8)
    rrf_frame.pack(pady=8, padx=10, fill=tk.X)
    rrf_sync_var = tk.BooleanVar(value=bool(rrf_job_sync))
    rrf_sync_cb = tk.Checkbutton(
        rrf_frame,
        text="Poll controller: start ADXL recording when a print/sim job runs, stop when it ends",
        variable=rrf_sync_var,
    )
    rrf_sync_cb.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 4))
    tk.Label(rrf_frame, text="Base URL:").grid(row=1, column=0, sticky=tk.W, pady=2)
    rrf_base_var = tk.StringVar(value=rrf_base_init)
    tk.Entry(rrf_frame, textvariable=rrf_base_var, width=42).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
    tk.Label(rrf_frame, text="Poll (s):").grid(row=1, column=2, sticky=tk.W, pady=2)
    rrf_poll_var = tk.StringVar(value=str(rrf_poll_init))
    tk.Spinbox(
        rrf_frame,
        from_=0.25,
        to=10.0,
        increment=0.25,
        textvariable=rrf_poll_var,
        width=6,
    ).grid(row=1, column=3, sticky=tk.W, pady=2)
    tk.Label(rrf_frame, text="Password (optional):").grid(row=2, column=0, sticky=tk.W, pady=2)
    rrf_password_var = tk.StringVar(value=rrf_pw_init)
    tk.Entry(rrf_frame, textvariable=rrf_password_var, width=24, show="*").grid(
        row=2, column=1, sticky=tk.W, padx=4, pady=2
    )
    rrf_btn_row = tk.Frame(rrf_frame)
    rrf_btn_row.grid(row=2, column=2, columnspan=2, sticky=tk.E, pady=2)

    rrf_status_var = tk.StringVar(value="RRF: sync off")

    def _rrf_probe_worker() -> None:
        try:
            msg = probe_rrf_base(
                rrf_base_var.get().strip(),
                password=rrf_password_var.get(),
                timeout_s=4.0,
            )
            root.after(0, lambda m=msg: rrf_status_var.set(f"RRF: {m}"))
        except RrfHttpError as e:
            root.after(0, lambda err=str(e): rrf_status_var.set(f"RRF probe failed: {err}"))

    def _on_rrf_probe() -> None:
        rrf_status_var.set("RRF: probing…")
        threading.Thread(target=_rrf_probe_worker, daemon=True).start()

    def _rrf_discover_worker() -> None:
        try:
            found = discover_rrf_base(password=rrf_password_var.get(), timeout_s=2.5)
            if found:

                def _apply(u: str = found) -> None:
                    rrf_base_var.set(u)
                    rrf_status_var.set(f"RRF: discovered {u}")

                root.after(0, _apply)
            else:
                root.after(
                    0,
                    lambda: rrf_status_var.set(
                        "RRF: discovery found no host (set URL or TAP_RRF_DISCOVER_HOSTS)"
                    ),
                )
        except Exception as e:
            root.after(0, lambda err=str(e): rrf_status_var.set(f"RRF discovery error: {err}"))

    def _on_rrf_discover() -> None:
        rrf_status_var.set("RRF: discovering…")
        threading.Thread(target=_rrf_discover_worker, daemon=True).start()

    tk.Button(rrf_btn_row, text="Test connection", command=_on_rrf_probe).pack(side=tk.LEFT, padx=4)
    tk.Button(rrf_btn_row, text="Discover on LAN", command=_on_rrf_discover).pack(side=tk.LEFT, padx=4)
    tk.Label(
        rrf_frame,
        textvariable=rrf_status_var,
        font=("TkDefaultFont", 9),
        fg="gray30",
        wraplength=560,
        justify=tk.LEFT,
    ).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
    rrf_frame.columnconfigure(1, weight=1)

    if modbus:
        modbus_info = (
            f"Modbus: {mb_cfg.transport.upper()} "
            + (
                f"{mb_cfg.host}:{mb_cfg.port} unit={mb_cfg.unit_id}"
                if mb_cfg.transport.lower() == "tcp"
                else f"{mb_cfg.serial_port} @ {mb_cfg.baudrate} unit={mb_cfg.unit_id}"
            )
            + f"  poll≈{mb_cfg.poll_hz} Hz"
            + (f"  [{mb_cfg.profile}]" if mb_cfg.profile else "")
        )
        tk.Label(
            params_frame,
            text=modbus_info,
            font=("TkDefaultFont", 9),
            fg="darkblue",
            wraplength=560,
            justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=4)
        row += 1

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

    modbus_live_var: tk.StringVar | None = None
    if modbus:
        modbus_live_var = tk.StringVar(value="Modbus: enabled — data appears when recording starts")
        modbus_row2 = tk.Frame(root)
        modbus_row2.pack(pady=2, padx=10, fill=tk.X)
        tk.Label(modbus_row2, text="Modbus:", font=tkfont.Font(weight="bold")).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(
            modbus_row2,
            textvariable=modbus_live_var,
            font=("TkFixedFont", 9),
            fg="darkblue",
            wraplength=620,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT)

    # --- Start / Stop recording (packed before plot so buttons stay visible) ---
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, padx=10, fill=tk.X)

    stop_event: threading.Event | None = None

    _MODBUS_PLOT_LEN = 1200
    modbus_t0_holder: list[float | None] = [None]
    modbus_t_deque: deque = deque(maxlen=_MODBUS_PLOT_LEN)
    modbus_h_deques: list[deque] = [deque(maxlen=_MODBUS_PLOT_LEN) for _ in range(6)]
    modbus_i_deques: list[deque] = [deque(maxlen=_MODBUS_PLOT_LEN) for _ in range(6)]
    modbus_plot: dict = {
        "lines_h": [],
        "lines_i": [],
        "canvas": None,
        "ax_h": None,
        "ax_i": None,
    }

    def reset_modbus_buffers() -> None:
        modbus_t0_holder[0] = None
        modbus_t_deque.clear()
        for d in modbus_h_deques:
            d.clear()
        for d in modbus_i_deques:
            d.clear()

    def redraw_modbus_plot() -> None:
        canvas = modbus_plot.get("canvas")
        lines_h = modbus_plot.get("lines_h") or []
        lines_i = modbus_plot.get("lines_i") or []
        if canvas is None or not lines_h:
            return
        t_list = list(modbus_t_deque)
        if not t_list:
            return
        for i, ln in enumerate(lines_h):
            ln.set_data(t_list, list(modbus_h_deques[i]))
        for i, ln in enumerate(lines_i):
            ln.set_data(t_list, list(modbus_i_deques[i]))
        ax_h = modbus_plot.get("ax_h")
        ax_i = modbus_plot.get("ax_i")
        if ax_h is not None:
            ax_h.relim()
            ax_h.autoscale_view()
        if ax_i is not None:
            ax_i.relim()
            ax_i.autoscale_view()
        canvas.draw_idle()

    def apply_modbus_row(row: dict) -> None:
        ts = row.get("t_s")
        if ts is None:
            return
        if modbus_t0_holder[0] is None:
            modbus_t0_holder[0] = float(ts)
        t_rel = float(ts) - modbus_t0_holder[0]
        modbus_t_deque.append(t_rel)
        for i in range(6):
            k = f"hr_{i}"
            modbus_h_deques[i].append(float(row[k]) if k in row else float("nan"))
            k2 = f"ir_{i}"
            modbus_i_deques[i].append(float(row[k2]) if k2 in row else float("nan"))
        if modbus_live_var is not None:
            parts = []
            for i in range(4):
                kk = f"hr_{i}"
                if kk in row:
                    parts.append(f"{kk}={row[kk]}")
            modbus_live_var.set("Modbus: " + " ".join(parts) if parts else "Modbus: polling…")
        redraw_modbus_plot()

    def start_recording() -> None:
        nonlocal stop_event, recording_active, live_plot_timer_id, clear_plot_timer_id
        if recording_active:
            return
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

        if modbus:
            reset_modbus_buffers()
            if modbus_live_var is not None:
                modbus_live_var.set("Modbus: connecting / polling…")

        recording_active = True
        if live_plot_timer_id is not None:
            root.after_cancel(live_plot_timer_id)
            live_plot_timer_id = None
        if clear_plot_timer_id is not None:
            root.after_cancel(clear_plot_timer_id)
            clear_plot_timer_id = None

        rec_msg = f"Recording… Saving to {run_dir.name}/homing.csv"
        if modbus:
            rec_msg += " + modbus.csv"
        rec_msg += " — Run homing on the machine (Z up, then X Y home). Click Stop when done."
        status_var.set(rec_msg)
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

        if modbus:
            assert mb_cfg is not None

            def on_mb_row(row: dict) -> None:
                root.after(0, lambda r=dict(row): apply_modbus_row(r))

            def on_mb_fail(msg: str) -> None:
                if modbus_live_var is not None:
                    root.after(0, lambda m=msg: modbus_live_var.set(f"Modbus: connect failed — {m}"))

            modbus_path = run_dir / "modbus.csv"
            modbus_thread = threading.Thread(
                target=run_modbus_poll_loop,
                args=(modbus_path, stop_event, mb_cfg, on_mb_row, on_mb_fail),
                daemon=True,
            )
            modbus_thread.start()

    def stop_recording() -> None:
        if stop_event is not None:
            stop_event.set()

    def rrf_on_poll_result(active: bool, status_s: str) -> None:
        prev = rrf_last_job_active[0]
        rrf_status_var.set(f"RRF: print job active={active}  state.status={status_s}")
        if prev is None:
            if active:
                start_recording()
        else:
            if not prev and active:
                start_recording()
            elif prev and not active:
                stop_recording()
        # If a job is active but recording did not start (e.g. no accelerometer), stay "not tracking"
        # so the next poll can retry the rising edge.
        rrf_last_job_active[0] = active if (recording_active or not active) else False

    def rrf_poll_worker() -> None:
        base = rrf_base_var.get().strip()
        try:
            poll_s = float(rrf_poll_var.get())
        except ValueError:
            poll_s = rrf_default_poll_interval_s()
        poll_s = max(0.25, min(60.0, poll_s))
        pw = rrf_password_var.get()
        if not base:

            def _rrf_no_base() -> None:
                rrf_status_var.set("RRF: set Base URL first (or use Discover)")
                rrf_sync_var.set(False)

            root.after(0, _rrf_no_base)
            return
        client = RrfClient(base, password=pw, timeout_s=5.0)
        try:
            client.connect()
        except RrfHttpError as e:

            def _fail(msg: str = str(e)) -> None:
                rrf_status_var.set(f"RRF: connect failed — {msg}")
                rrf_sync_var.set(False)

            root.after(0, _fail)
            return
        while not rrf_poller_stop.is_set():
            try:
                state, job = client.fetch_state_and_job()
                active = infer_print_job_active(state, job)
                st = (state or {}).get("status", "?")
                if not isinstance(st, str):
                    st = repr(st)
                root.after(0, lambda a=active, s=st: rrf_on_poll_result(a, s))
            except RrfHttpError as e:
                err_msg = str(e)
                try:
                    client.connect()
                except RrfHttpError:
                    pass

                def _err(m: str = err_msg) -> None:
                    rrf_status_var.set(f"RRF poll error: {m}")

                root.after(0, _err)
            if rrf_poller_stop.wait(poll_s):
                break

    def stop_rrf_poller() -> None:
        rrf_poller_stop.set()
        t = rrf_poller_thread[0]
        if t is not None:
            t.join(timeout=5.0)
        rrf_poller_thread[0] = None
        rrf_poller_stop.clear()
        rrf_last_job_active[0] = None

    def start_rrf_poller() -> None:
        stop_rrf_poller()
        if not rrf_sync_var.get():
            return
        rrf_status_var.set("RRF: sync on — polling…")
        th = threading.Thread(target=rrf_poll_worker, daemon=True)
        rrf_poller_thread[0] = th
        th.start()

    def on_rrf_sync_toggled() -> None:
        if rrf_sync_var.get():
            start_rrf_poller()
        else:
            stop_rrf_poller()
            rrf_status_var.set("RRF: sync off")

    rrf_sync_cb.config(command=on_rrf_sync_toggled)

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

    # --- Live charts: notebook (accel + Modbus) or accelerometer only ---
    if modbus:
        chart_nb = ttk.Notebook(root)
        chart_nb.pack(pady=6, padx=10, fill=tk.BOTH, expand=True)
        tab_accel = tk.Frame(chart_nb)
        tab_modbus = tk.Frame(chart_nb)
        chart_nb.add(tab_accel, text="Accelerometer (X/Y)")
        chart_nb.add(tab_modbus, text="Modbus (live)")
        plot_parent = tab_accel
        _modbus_plot_caption = (
            "H100: ir_0/ir_1 = output & set frequency (raw u16); hr_5/hr_11/hr_513 in CSV — docs/MODBUS_H100_VFD.md"
            if mb_cfg.profile == "h100"
            else "Holding / input registers 0–5 vs time (raw u16); full map in modbus.csv"
        )
        modbus_plot_frame = tk.LabelFrame(
            tab_modbus,
            text=_modbus_plot_caption,
            padx=4,
            pady=4,
        )
        modbus_plot_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    else:
        plot_parent = root
        modbus_plot_frame = None

    plot_frame = tk.LabelFrame(
        plot_parent,
        text="Live X/Y (ax vs ay, g) — from accelerometer",
        padx=4,
        pady=4,
    )
    if modbus:
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    else:
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

    def _init_modbus_plot() -> None:
        if not modbus or modbus_plot_frame is None:
            return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig_m, (ax_h, ax_i) = plt.subplots(2, 1, figsize=(5, 5), height_ratios=[1, 1])
        colors = ("C0", "C1", "C2", "C3", "C4", "C5")
        lines_h = [
            ax_h.plot([], [], lw=0.9, label=f"hr_{i}", color=colors[i % len(colors)])[0] for i in range(6)
        ]
        lines_i = [
            ax_i.plot([], [], lw=0.9, label=f"ir_{i}", color=colors[i % len(colors)])[0] for i in range(6)
        ]
        ax_h.set_ylabel("Holding (raw)")
        if mb_cfg.profile == "h100":
            ax_h.set_title("H100 holding 0–5 (note: hr_5=max freq, hr_11=min — see CSV)")
        else:
            ax_h.set_title("Holding registers 0–5 (scaling is device-specific)")
        ax_h.grid(True, alpha=0.3)
        ax_h.legend(loc="upper right", fontsize=7, ncol=2)
        ax_i.set_ylabel("Input (raw)")
        ax_i.set_xlabel("Time from first Modbus sample (s)")
        if mb_cfg.profile == "h100":
            ax_i.set_title("H100 input 0–5 (ir_0/ir_1: frequency-related per FluidNC H100 map)")
        else:
            ax_i.set_title("Input registers 0–5")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(loc="upper right", fontsize=7, ncol=2)
        fig_m.tight_layout()
        canvas_m = FigureCanvasTkAgg(fig_m, master=modbus_plot_frame)
        canvas_m.draw()
        canvas_m.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        modbus_plot["lines_h"] = lines_h
        modbus_plot["lines_i"] = lines_i
        modbus_plot["canvas"] = canvas_m
        modbus_plot["ax_h"] = ax_h
        modbus_plot["ax_i"] = ax_i

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
    _init_modbus_plot()

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
            msg = "Accelerometer OK. X/Y plot is live. Set parameters, then Start recording when ready."
            if modbus:
                msg += " Modbus logging when recording (see docs/MODBUS_RASPBERRY_PI.md)."
            status_var.set(msg)
            live_var.set("Live: streaming to X/Y plot…")
            if not recording_active:
                live_plot_timer_id = root.after(50, live_update_plot)
        except Exception as e:
            shared_accel = None
            friendly = _friendly_accel_error(str(e)) or str(e)
            status_var.set(friendly)

    def on_window_close() -> None:
        stop_rrf_poller()
        if stop_event is not None:
            stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_window_close)

    root.after(300, check_accel)
    if rrf_job_sync:
        root.after(900, start_rrf_poller)

    root.mainloop()


def main() -> None:
    import argparse
    import sys

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
    parser.add_argument(
        "--modbus",
        action="store_true",
        help="Poll Modbus during recording (pymodbus); saves modbus.csv and shows live register plots. Default transport is RTU; use TAP_MODBUS_* and optional --modbus-profile h100 (docs/MODBUS_H100_VFD.md).",
    )
    parser.add_argument(
        "--modbus-host",
        default=None,
        help="Override TAP_MODBUS_HOST (TCP only)",
    )
    parser.add_argument(
        "--modbus-port",
        type=int,
        default=None,
        help="Override TAP_MODBUS_PORT (TCP only)",
    )
    parser.add_argument(
        "--modbus-unit",
        type=int,
        default=None,
        help="Override TAP_MODBUS_UNIT (slave / device id)",
    )
    parser.add_argument(
        "--modbus-transport",
        choices=("tcp", "rtu"),
        default=None,
        help="Override TAP_MODBUS_TRANSPORT",
    )
    parser.add_argument(
        "--modbus-serial-port",
        default=None,
        help="Override TAP_MODBUS_SERIAL_PORT (RTU)",
    )
    parser.add_argument(
        "--modbus-baudrate",
        type=int,
        default=None,
        help="Override TAP_MODBUS_BAUDRATE (RTU)",
    )
    parser.add_argument(
        "--modbus-poll-hz",
        type=float,
        default=None,
        help="Override TAP_MODBUS_POLL_HZ",
    )
    parser.add_argument(
        "--modbus-profile",
        choices=("h100",),
        default=None,
        help="Device preset (RTU span/baud). Overrides env if set. See docs/MODBUS_H100_VFD.md for h100.",
    )
    parser.add_argument(
        "--rrf-job-sync",
        action="store_true",
        help="Enable Duet/RepRapFirmware HTTP polling: start/stop ADXL recording with print job (rr_model). See TAP_RRF_* env vars.",
    )
    parser.add_argument(
        "--rrf-base",
        default=None,
        help="Controller base URL (default: TAP_RRF_BASE or http://duet3.local)",
    )
    parser.add_argument(
        "--rrf-poll-s",
        type=float,
        default=None,
        help="Seconds between rr_model polls (default: TAP_RRF_POLL_S or 1.0)",
    )
    parser.add_argument(
        "--rrf-password",
        default=None,
        help="rr_connect password if required (default: TAP_RRF_PASSWORD)",
    )
    args = parser.parse_args()

    if args.modbus:
        try:
            import pymodbus  # noqa: F401
        except ImportError:
            print("Homing GUI --modbus requires pymodbus. Install with:  pip install pymodbus", file=sys.stderr)
            raise SystemExit(1) from None

    mb_cfg = modbus_logging_config_from_env()
    if args.modbus_profile is not None:
        apply_modbus_profile(mb_cfg, args.modbus_profile)
    if args.modbus_host is not None:
        mb_cfg.host = args.modbus_host
    if args.modbus_port is not None:
        mb_cfg.port = int(args.modbus_port)
    if args.modbus_unit is not None:
        mb_cfg.unit_id = int(args.modbus_unit)
    if args.modbus_transport is not None:
        mb_cfg.transport = args.modbus_transport
    if args.modbus_serial_port is not None:
        mb_cfg.serial_port = args.modbus_serial_port
    if args.modbus_baudrate is not None:
        mb_cfg.baudrate = int(args.modbus_baudrate)
    if args.modbus_poll_hz is not None:
        mb_cfg.poll_hz = float(args.modbus_poll_hz)

    run_homing_gui(
        output_dir=args.output_dir,
        sample_rate_hz=args.rate or cfg.sample_rate_hz,
        modbus=args.modbus,
        modbus_config=mb_cfg if args.modbus else None,
        rrf_job_sync=args.rrf_job_sync,
        rrf_base_url=args.rrf_base,
        rrf_poll_interval_s=args.rrf_poll_s,
        rrf_password=args.rrf_password,
    )


if __name__ == "__main__":
    main()
