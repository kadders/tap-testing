"""
Headless live-spindle telemetry service for RRF SBC / DuetPi.

Streams ADXL via ``record_stream``, optionally gated by print-job state from
``rr_model``, publishes MQTT (SBC-gated), and writes a status JSON for a tray UI.

Examples::

  python -m tap_testing.live_spindle_service --job-sync
  python -m tap_testing.live_spindle_service --always-on
  python -m tap_testing.live_spindle_service --job-sync --tray
  python -m tap_testing.live_spindle_service --video-test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import get_config
from .gcode_summary import summarize_gcode_bytes
from .motion_sample import MotionPublishFilter, build_motion_sample
from .mqtt_telemetry import try_create_publisher
from .record_tap import record_stream
from .rrf_http import (
    RrfClient,
    RrfHttpError,
    infer_print_job_active,
    infer_job_sync_recording_active,
    infer_job_sync_stop_reason,
    parse_current_tool,
    parse_job_duration_s,
    parse_job_file_name,
    parse_job_file_position,
    parse_job_times_left_s,
    parse_axis_positions_mm,
    parse_requested_feed_mm_min,
    rrf_default_base_url,
    rrf_default_poll_interval_s,
)
from .spindle_telemetry import SpindleTelemetryRecorder, parse_arborctl_sample
from .tool_telemetry import ToolEventRecorder
from .video_recording import VideoSessionRecorder, run_video_test, try_create_video_recorder

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _tool_name_from_table(tools: list[dict[str, Any]] | None, tool_n: int | None) -> str | None:
    """Return tool name from an RRF tools list, ``""`` if none, or None if unknown."""
    if tool_n is None:
        return None
    if tool_n < 0:
        return ""
    if not tools:
        return None
    for t in tools:
        try:
            if int(t.get("number")) == int(tool_n):
                name = t.get("name")
                return name if isinstance(name, str) else ""
        except (TypeError, ValueError):
            continue
    return ""


def collect_video_overlay_fields(client: RrfClient) -> dict[str, Any]:
    """Snapshot RRF fields for the video HUD. Missing bits are omitted or None."""
    out: dict[str, Any] = {}
    try:
        state, job = client.fetch_state_and_job()
    except RrfHttpError:
        state, job = None, None
    tool_n = parse_current_tool(state)
    if tool_n is not None:
        out["tool_number"] = tool_n
    job_file = parse_job_file_name(job) or ""
    if job_file:
        out["job_file"] = job_file
    out["job_duration_s"] = parse_job_duration_s(job)
    out["job_times_left_s"] = parse_job_times_left_s(job)
    if isinstance(state, dict):
        st = state.get("status")
        if isinstance(st, str) and st:
            out["rrf_status"] = st
    tools: list[dict[str, Any]] | None = None
    try:
        tools = client.fetch_tools()
    except RrfHttpError:
        tools = None
    tool_name = _tool_name_from_table(tools, tool_n)
    if tool_name is not None:
        out["tool_name"] = tool_name
    try:
        move = client.fetch_move()
        out["feed_mm_min"] = parse_requested_feed_mm_min(move)
        out["axis_positions_mm"] = parse_axis_positions_mm(move)
    except RrfHttpError:
        out["feed_mm_min"] = None
        out["axis_positions_mm"] = {}
    try:
        g_om = client.fetch_globals()
    except RrfHttpError:
        g_om = None
    try:
        spindles = client.fetch_spindles()
    except RrfHttpError:
        spindles = None
    sample = parse_arborctl_sample(
        g_om,
        spindles=spindles,
        tools=tools,
        current_tool=tool_n,
    )
    if sample:
        rpm = sample.get("rpm")
        load = sample.get("load_percent")
        if rpm is not None:
            out["spindle_rpm"] = rpm
        if sample.get("power_available") and load is not None:
            out["spindle_load_percent"] = load
    return out


def run_live_video_test(
    *,
    duration_s: float,
    output_dir: Path,
    rrf_base: str,
    rrf_password: str = "",
    sample_rate_hz: float = 800.0,
) -> int:
    """Video smoke test with the same live overlay as a job (RRF + ADXL)."""
    client: RrfClient | None = None
    try:
        client = RrfClient(rrf_base, password=rrf_password, timeout_s=5.0)
        client.connect()
        logger.info("Video test: RRF connected at %s — live overlay", rrf_base)
    except RrfHttpError as e:
        logger.warning(
            "Video test: RRF unavailable (%s) — overlay will lack tool/feed/XYZA/RPM",
            e,
        )
        client = None

    adxl_stop = threading.Event()
    adxl_thread: threading.Thread | None = None

    def on_started(
        run_dir: Path,
        t0: float,
        _session_id: str,
        recorder: VideoSessionRecorder,
    ) -> None:
        nonlocal adxl_thread

        def _on_sample(_t: float, x: float, y: float, z: float) -> None:
            if recorder.active:
                recorder.update_overlay(ax_g=x, ay_g=y, az_g=z)

        def worker() -> None:
            csv_path = Path(run_dir) / "homing.csv"
            try:
                record_stream(
                    csv_path,
                    adxl_stop,
                    sample_rate_hz=sample_rate_hz,
                    sample_callback=_on_sample,
                    callback_interval_s=0.25,
                    mqtt_manage_session=False,
                    recording_t0_mono=t0,
                    mqtt_idle_timeout_s=0.0,
                )
            except Exception:
                logger.warning(
                    "Video test: ADXL unavailable — overlay will lack accel bars",
                    exc_info=True,
                )

        adxl_thread = threading.Thread(target=worker, name="video-test-adxl", daemon=True)
        adxl_thread.start()

    def overlay_tick(recorder: VideoSessionRecorder) -> None:
        if client is None:
            return
        fields = collect_video_overlay_fields(client)
        if not fields.get("job_file"):
            fields["job_file"] = "video-test"
        recorder.update_overlay(**fields)

    def on_stopped() -> None:
        adxl_stop.set()
        if adxl_thread is not None:
            adxl_thread.join(timeout=3.0)

    return run_video_test(
        duration_s=duration_s,
        output_dir=output_dir,
        overlay_tick=overlay_tick,
        on_started=on_started,
        on_stopped=on_stopped,
        force_overlay=True,
    )


def rrf_default_tool_poll_interval_s() -> float:
    """Faster poll while recording so tool changes align better with ADXL time."""
    return float(os.environ.get("TAP_RRF_TOOL_POLL_S", "0.25"))


def rrf_default_tool_table_poll_interval_s() -> float:
    """Slower tool-table refresh for offset-change detection while recording."""
    return float(os.environ.get("TAP_RRF_TOOL_TABLE_POLL_S", "1.0"))


@dataclass
class ServiceStatus:
    state: str = "idle"  # idle | recording | error | stopping
    recording: bool = False
    session_id: str = ""
    mqtt_enabled: bool = False
    mqtt_detail: str = ""
    rrf_connected: bool = False
    rrf_job_active: bool | None = None
    rrf_status: str = ""
    current_tool: int | None = None
    job_file: str = ""
    sample_rate_hz: float = 800.0
    output_path: str = ""
    last_error: str = ""
    stop_reason: str = ""
    spindle_rpm: float | None = None
    spindle_load_percent: float | None = None
    video_recording: bool = False
    video_path: str = ""
    youtube_live: bool = False
    video_stream_remote: bool = False
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["updated_at"] = self.updated_at
        return d


class LiveSpindleService:
    """Start/stop continuous ADXL capture with optional RRF job sync + MQTT."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sample_rate_hz: float,
        status_path: Path,
        rrf_base: str,
        rrf_password: str = "",
        rrf_poll_s: float = 1.0,
        rrf_tool_poll_s: float | None = None,
        rrf_tool_table_poll_s: float | None = None,
        job_sync: bool = True,
        always_on: bool = False,
        mqtt_mode: str = "live_spindle",
        on_status: Callable[[ServiceStatus], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate_hz = float(sample_rate_hz)
        self.status_path = Path(status_path)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.rrf_base = rrf_base.strip()
        self.rrf_password = rrf_password
        self.rrf_poll_s = max(0.25, float(rrf_poll_s))
        self.rrf_tool_poll_s = max(
            0.1,
            float(rrf_tool_poll_s if rrf_tool_poll_s is not None else rrf_default_tool_poll_interval_s()),
        )
        self.rrf_tool_table_poll_s = max(
            0.25,
            float(
                rrf_tool_table_poll_s
                if rrf_tool_table_poll_s is not None
                else rrf_default_tool_table_poll_interval_s()
            ),
        )
        self.job_sync = bool(job_sync) and not always_on
        self.always_on = bool(always_on)
        self.mqtt_mode = mqtt_mode
        self.on_status = on_status

        self._lock = threading.Lock()
        self._stop_service = threading.Event()
        self._rec_stop: threading.Event | None = None
        self._rec_thread: threading.Thread | None = None
        self._rrf_thread: threading.Thread | None = None
        self._last_job_active: bool | None = None
        self._mqtt = None
        self._tool_recorder: ToolEventRecorder | None = None
        self._spindle_recorder: SpindleTelemetryRecorder | None = None
        self._rrf_client: RrfClient | None = None
        self._session_id: str | None = None
        self._axis_letters: list[str] = []
        self._last_tools_fetch_mono: float = 0.0
        self._last_adxl_mono: float = 0.0
        self._session_idle_timeout_s = float(
            os.environ.get("MQTT_SESSION_IDLE_TIMEOUT_S", "25") or "25"
        )
        self._job_sync_stop_grace_s = float(
            os.environ.get("TAP_JOB_SYNC_STOP_GRACE_S", "3") or "3"
        )
        self._rrf_disconnect_stop_s = float(
            os.environ.get("TAP_RRF_DISCONNECT_STOP_S", "3") or "3"
        )
        self._job_sync_inactive_since: float | None = None
        self._rrf_disconnect_since: float | None = None
        self._pending_stop_reason: str = ""
        self._video_recorder: VideoSessionRecorder | None = try_create_video_recorder()
        self._motion_filter = MotionPublishFilter()
        self.status = ServiceStatus(sample_rate_hz=self.sample_rate_hz)

    def _set_status(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.status, k, v)
            self.status.updated_at = time.time()
            snap = self.status.to_dict()
        try:
            self.status_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        except OSError as e:
            logger.debug("status write failed: %s", e)
        if self.on_status is not None:
            try:
                self.on_status(self.status)
            except Exception:
                logger.exception("on_status callback failed")

    def start(self) -> None:
        self._mqtt = try_create_publisher()
        mqtt_ok = self._mqtt is not None
        if mqtt_ok:
            mqtt_detail = "connected"
        elif not os.environ.get("TAP_MQTT_HOST", "").strip():
            mqtt_detail = "TAP_MQTT_HOST unset"
        else:
            mqtt_detail = "SBC gate blocked or broker unreachable"
        self._set_status(
            mqtt_enabled=mqtt_ok,
            mqtt_detail=mqtt_detail,
            state="idle",
        )
        # RRF poller: job sync edges + tool timeline while recording
        self._rrf_thread = threading.Thread(target=self._rrf_poll_loop, name="rrf-poll", daemon=True)
        self._rrf_thread.start()
        if self.always_on:
            self.start_recording()
        logger.info(
            "live_spindle_service started (job_sync=%s always_on=%s mqtt=%s tool_poll=%.2fs)",
            self.job_sync,
            self.always_on,
            mqtt_ok,
            self.rrf_tool_poll_s,
        )

    def stop(self) -> None:
        self._stop_service.set()
        self.stop_recording(reason="service_shutdown")
        if self._rrf_thread is not None:
            self._rrf_thread.join(timeout=5.0)
            self._rrf_thread = None
        if self._mqtt is not None:
            try:
                self._mqtt.close()
            except Exception:
                pass
            self._mqtt = None
        self._set_status(state="idle", recording=False, rrf_job_active=None)

    def _fetch_tool_context(self, client: RrfClient) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
        list[str],
    ]:
        state, job = client.fetch_state_and_job()
        tools: list[dict[str, Any]] = []
        axes: list[str] = []
        try:
            tools = client.fetch_tools()
        except RrfHttpError as e:
            logger.debug("fetch_tools failed: %s", e)
        try:
            axes = client.fetch_axis_letters()
        except RrfHttpError as e:
            logger.debug("fetch_axis_letters failed: %s", e)
        return state, job, tools, axes

    def start_recording(self) -> None:
        with self._lock:
            if self._rec_thread is not None and self._rec_thread.is_alive():
                return
            stop_ev = threading.Event()
            self._rec_stop = stop_ev
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "homing.csv"

        state: dict[str, Any] | None = None
        job: dict[str, Any] | None = None
        tools: list[dict[str, Any]] = []
        axes: list[str] = []
        client = self._rrf_client
        if client is not None:
            try:
                state, job, tools, axes = self._fetch_tool_context(client)
            except RrfHttpError as e:
                logger.warning("RRF context at recording start failed: %s", e)

        job_file = parse_job_file_name(job)
        tool_n = parse_current_tool(state)
        tool_name = None
        for t in tools:
            try:
                if int(t.get("number")) == tool_n and isinstance(t.get("name"), str):
                    tool_name = t["name"]
                    break
            except (TypeError, ValueError):
                continue

        # Hash + summarize machine G-code for Jarvis CAM resolution (bytes discarded).
        # Top-level gcode_sha256 for CAM lookup; nested summary omits duplicated keys.
        mqtt_extra: dict[str, Any] | None = None
        if client is not None and job_file:
            try:
                raw = client.download_file(job_file)
                summary = summarize_gcode_bytes(raw, job_file=job_file)
                digest = summary.get("gcode_sha256")
                compact = {
                    k: v
                    for k, v in summary.items()
                    if k not in ("gcode_sha256", "job_file")
                }
                mqtt_extra = {
                    "gcode_sha256": digest,
                    "gcode_summary": compact,
                }
                del raw
            except Exception as e:
                logger.warning("G-code download/summary failed: %s", e)

        mqtt = self._mqtt
        if mqtt is not None:
            try:
                mqtt.session_start(
                    mode=self.mqtt_mode,
                    sample_rate_hz=self.sample_rate_hz,
                    session_id=timestamp,
                    job_file=job_file,
                    tool_number=tool_n,
                    tool_name=tool_name,
                    extra=mqtt_extra,
                )
            except Exception as e:
                logger.warning("MQTT session_start failed: %s", e)

        # Shared monotonic origin for ADXL + tool events (set again when stream starts)
        recording_t0 = time.monotonic()
        recorder = ToolEventRecorder(
            run_dir=run_dir,
            session_id=timestamp,
            mqtt=mqtt,
            axis_letters=axes,
        )
        recorder.begin(
            tools=tools,
            state=state,
            job=job,
            sample_rate_hz=self.sample_rate_hz,
            axis_letters=axes,
            recording_t0_mono=recording_t0,
            extra_meta={"mode": self.mqtt_mode, "rrf_base": self.rrf_base},
        )
        spindle_rec = SpindleTelemetryRecorder(
            run_dir=run_dir,
            session_id=timestamp,
            mqtt=mqtt,
        )
        spindle_rec.begin(recording_t0_mono=recording_t0)
        with self._lock:
            self._tool_recorder = recorder
            self._spindle_recorder = spindle_rec
            self._session_id = timestamp
            self._axis_letters = axes
            self._last_tools_fetch_mono = time.monotonic()
            self._last_adxl_mono = time.monotonic()
        self._motion_filter = MotionPublishFilter()

        self._set_status(
            current_tool=tool_n,
            job_file=job_file or "",
            session_id=timestamp,
        )

        video_rec = self._video_recorder
        video_ok = False
        if video_rec is not None:
            video_ok = video_rec.start(
                run_dir,
                recording_t0_mono=recording_t0,
                session_id=timestamp,
                job_file=job_file,
                tool_number=tool_n,
                tool_name=tool_name,
                mqtt_publisher=mqtt,
            )
            vpath = video_rec.output_path
            if video_ok:
                self._patch_run_meta_video(
                    run_dir,
                    enabled=True,
                    video_path=str(vpath) if vpath else "",
                )
            self._set_status(
                video_recording=video_ok and video_rec.active,
                video_path=str(vpath) if vpath else "",
                youtube_live=video_rec.youtube_live,
                video_stream_remote=video_rec.video_stream_remote,
            )

        def worker() -> None:
            self._set_status(
                state="recording",
                recording=True,
                output_path=str(out_path),
                last_error="",
                session_id=timestamp,
            )
            if mqtt is not None:
                try:
                    mqtt.publish_status(
                        "recording",
                        {
                            "session_id": timestamp,
                            "mode": self.mqtt_mode,
                            "tool_number": tool_n,
                            "job_file": job_file,
                            "sample_rate_hz": self.sample_rate_hz,
                        },
                    )
                except Exception:
                    pass
            try:
                overlay_last_t = [0.0]
                idle_timeout_ev = threading.Event()

                def _on_sample(t: float, x: float, y: float, z: float) -> None:
                    with self._lock:
                        self._last_adxl_mono = time.monotonic()
                    if video_rec is not None and video_rec.active:
                        if t - overlay_last_t[0] >= 0.25:
                            video_rec.update_overlay(ax_g=x, ay_g=y, az_g=z)
                            overlay_last_t[0] = t

                def _on_origin(t0: float) -> None:
                    recorder.set_recording_origin(t0)
                    spindle_rec.set_recording_origin(t0)

                record_stream(
                    out_path,
                    stop_ev,
                    sample_rate_hz=self.sample_rate_hz,
                    sample_callback=_on_sample,
                    callback_interval_s=0.0,
                    mqtt=mqtt,
                    mqtt_mode=self.mqtt_mode,
                    mqtt_session_id=timestamp,
                    mqtt_manage_session=False,
                    recording_t0_mono=recording_t0,
                    on_recording_origin=_on_origin,
                    mqtt_idle_timeout_s=self._session_idle_timeout_s,
                    idle_timeout_event=idle_timeout_ev,
                )
                if idle_timeout_ev.is_set():
                    self._pending_stop_reason = "idle_timeout"
                    logger.warning(
                        "ADXL idle timeout (%.1fs) — session %s ended",
                        self._session_idle_timeout_s,
                        timestamp,
                    )
                    self._set_status(
                        state="error",
                        last_error=f"idle_timeout ({self._session_idle_timeout_s:.0f}s)",
                        stop_reason="idle_timeout",
                    )
            except Exception as e:
                logger.exception("recording failed")
                self._pending_stop_reason = "worker_exception"
                self._set_status(
                    state="error",
                    last_error=str(e),
                    recording=False,
                    stop_reason="worker_exception",
                )
                if mqtt is not None:
                    try:
                        mqtt.publish_status(
                            "error",
                            {
                                "session_id": timestamp,
                                "last_error": str(e),
                                "stop_reason": "worker_exception",
                            },
                        )
                    except Exception:
                        pass
            finally:
                if video_rec is not None and video_rec.active:
                    try:
                        video_rec.stop()
                    except Exception:
                        logger.exception("video stop failed")
                with self._lock:
                    rec = self._tool_recorder
                    spindle_rec_end = self._spindle_recorder
                    self._tool_recorder = None
                    self._spindle_recorder = None
                    final_tool = rec.last_tool if rec is not None else tool_n
                if rec is not None:
                    try:
                        rec.end()
                    except Exception:
                        pass
                if spindle_rec_end is not None:
                    try:
                        spindle_rec_end.end()
                    except Exception:
                        pass
                if mqtt is not None:
                    try:
                        stop_extra: dict[str, Any] = {}
                        if self._pending_stop_reason:
                            stop_extra["stop_reason"] = self._pending_stop_reason
                        mqtt.session_stop(
                            tool_number=final_tool,
                            job_file=job_file,
                            extra=stop_extra or None,
                        )
                    except Exception:
                        pass
                self._pending_stop_reason = ""
                if not self._stop_service.is_set():
                    self._set_status(
                        state="idle",
                        recording=False,
                        session_id="",
                        video_recording=False,
                        video_path="",
                        youtube_live=False,
                        video_stream_remote=False,
                    )

        t = threading.Thread(target=worker, name="adxl-record", daemon=True)
        with self._lock:
            self._rec_thread = t
        t.start()
        logger.info("recording started → %s (tool=%s job=%s)", out_path, tool_n, job_file)

    def _recording_join_timeout_s(self) -> float:
        """Wait long enough for video_recording.stop() graceful + terminate/kill."""
        rec = self._video_recorder
        if rec is None:
            return 15.0
        if getattr(rec, "remote_mode", False):
            return 15.0
        cfg = getattr(rec, "cfg", None)
        if cfg is None:
            return 15.0
        grace = cfg.stop_timeout_s + (5.0 if rec.youtube_live else 0.0)
        return max(15.0, grace + 15.0)

    def stop_recording(
        self,
        *,
        reason: str = "manual",
        rrf_status: str = "",
    ) -> None:
        self._pending_stop_reason = reason
        with self._lock:
            ev = self._rec_stop
            thr = self._rec_thread
        if ev is not None:
            ev.set()
        join_s = self._recording_join_timeout_s()
        if thr is not None:
            thr.join(timeout=join_s)
            if thr.is_alive():
                logger.warning(
                    "recording worker still running after %.0fs (stop_reason=%s)",
                    join_s,
                    reason,
                )
        with self._lock:
            self._rec_stop = None
            self._rec_thread = None
            self._job_sync_inactive_since = None
            self._rrf_disconnect_since = None
        self._set_status(
            recording=False,
            state="idle" if not self._stop_service.is_set() else "stopping",
            video_recording=False,
            video_path="",
            youtube_live=False,
            video_stream_remote=False,
            stop_reason=reason,
        )
        logger.info(
            "recording stopped (reason=%s rrf_status=%s)",
            reason,
            rrf_status or self.status.rrf_status,
        )

    def _patch_run_meta_video(
        self, run_dir: Path, *, enabled: bool, video_path: str = ""
    ) -> None:
        path = run_dir / "run-meta.json"
        try:
            meta: dict[str, Any] = {}
            if path.exists():
                meta = json.loads(path.read_text(encoding="utf-8"))
            meta["video_enabled"] = enabled
            if video_path:
                meta["video_path"] = video_path
            path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.debug("run-meta video patch failed: %s", e)

    def _update_video_overlay(
        self,
        *,
        tool_n: int | None = None,
        tool_name: str | None = None,
        job_file: str | None = None,
        rrf_status: str | None = None,
        spindle_rpm: float | None = None,
        spindle_load_percent: float | None = None,
        feed_mm_min: float | None = None,
        axis_positions_mm: dict[str, float] | None = None,
        job_duration_s: float | None = None,
        job_times_left_s: float | None = None,
        set_feed: bool = False,
        set_positions: bool = False,
        set_job_times: bool = False,
    ) -> None:
        rec = self._video_recorder
        if rec is None or not rec.active:
            return
        kwargs: dict[str, Any] = {}
        if tool_n is not None:
            kwargs["tool_number"] = tool_n
        if tool_name is not None:
            kwargs["tool_name"] = tool_name
        if job_file:
            kwargs["job_file"] = job_file
        if rrf_status:
            kwargs["rrf_status"] = rrf_status
        if spindle_rpm is not None:
            kwargs["spindle_rpm"] = spindle_rpm
        if spindle_load_percent is not None:
            kwargs["spindle_load_percent"] = spindle_load_percent
        if set_feed:
            kwargs["feed_mm_min"] = feed_mm_min
        if set_positions:
            kwargs["axis_positions_mm"] = dict(axis_positions_mm or {})
        if set_job_times:
            kwargs["job_duration_s"] = job_duration_s
            kwargs["job_times_left_s"] = job_times_left_s
        if kwargs:
            rec.update_overlay(**kwargs)

    def _note_rrf_poll_success(self) -> None:
        self._rrf_disconnect_since = None

    def _note_rrf_poll_failure(self, err: str) -> None:
        now = time.monotonic()
        if self._rrf_disconnect_since is None:
            self._rrf_disconnect_since = now
            logger.warning("RRF poll failed: %s — disconnect grace started", err)
        with self._lock:
            recording = self.status.recording
        if not recording:
            return
        elapsed = now - self._rrf_disconnect_since
        if self._rrf_disconnect_stop_s <= 0 or elapsed >= self._rrf_disconnect_stop_s:
            self.stop_recording(reason="rrf_disconnect", rrf_status="disconnected")

    def _rrf_poll_loop(self) -> None:
        client = RrfClient(self.rrf_base, password=self.rrf_password, timeout_s=5.0)
        self._rrf_client = client
        try:
            client.connect()
            self._set_status(rrf_connected=True, last_error="")
        except RrfHttpError as e:
            self._set_status(rrf_connected=False, last_error=f"RRF connect: {e}", state="error")
            logger.error("RRF connect failed: %s", e)
            return

        while not self._stop_service.is_set():
            try:
                state, job = client.fetch_state_and_job()
                active = infer_print_job_active(state, job)
                st = (state or {}).get("status", "?")
                if not isinstance(st, str):
                    st = repr(st)
                tool_n = parse_current_tool(state)
                job_file = parse_job_file_name(job) or ""
                job_duration_s = parse_job_duration_s(job)
                job_times_left_s = parse_job_times_left_s(job)
                if self.job_sync:
                    self._on_job(state, job)
                with self._lock:
                    rec = self._tool_recorder
                    recording = self.status.recording
                    last_tools = self._last_tools_fetch_mono
                    prev_tool = rec.last_tool if rec is not None else None
                tools: list[dict[str, Any]] | None = None
                now = time.monotonic()
                need_tools = False
                if rec is not None and rec.active:
                    # Immediate refresh on selection change; periodic otherwise
                    if tool_n != prev_tool:
                        need_tools = True
                    elif (now - last_tools) >= self.rrf_tool_table_poll_s:
                        need_tools = True
                if need_tools:
                    try:
                        tools = client.fetch_tools()
                        with self._lock:
                            self._last_tools_fetch_mono = now
                    except RrfHttpError as e:
                        logger.debug("fetch_tools during poll failed: %s", e)
                        tools = None
                if rec is not None and rec.active:
                    emitted = rec.observe(state=state, job=job, tools=tools)
                    # Selection change with tools already applied above; if tools
                    # fetch failed, still emit selection-only change.
                    if tools is None and tool_n != prev_tool:
                        pass
                    _ = emitted
                with self._lock:
                    spindle_rec = self._spindle_recorder
                    recording = self.status.recording
                if spindle_rec is not None and spindle_rec.active:
                    try:
                        g_om = client.fetch_globals()
                    except RrfHttpError as e:
                        logger.debug("fetch_globals during poll failed: %s", e)
                        g_om = None
                    spindles: list[dict[str, Any]] | None = None
                    try:
                        spindles = client.fetch_spindles()
                    except RrfHttpError as e:
                        logger.debug("fetch_spindles during poll failed: %s", e)
                    try:
                        sample = spindle_rec.observe(
                            globals_om=g_om,
                            spindles=spindles,
                            tools=tools,
                            current_tool=tool_n,
                        )
                        if sample:
                            rpm = sample.get("rpm")
                            load = sample.get("load_percent")
                            extra: dict[str, Any] = {}
                            if rpm is not None:
                                extra["spindle_rpm"] = rpm
                            if sample.get("power_available") and load is not None:
                                extra["spindle_load_percent"] = load
                            if extra:
                                self._set_status(**extra)
                    except Exception:
                        logger.debug("spindle telemetry observe failed", exc_info=True)
                feed_mm_min: float | None = None
                axis_positions_mm: dict[str, float] = {}
                if recording:
                    try:
                        move = client.fetch_move()
                        feed_mm_min = parse_requested_feed_mm_min(move)
                        axis_positions_mm = parse_axis_positions_mm(move)
                    except RrfHttpError as e:
                        logger.debug("fetch_move failed: %s", e)
                tool_name = _tool_name_from_table(tools, tool_n)
                self._update_video_overlay(
                    tool_n=tool_n,
                    tool_name=tool_name,
                    job_file=job_file,
                    rrf_status=st,
                    spindle_rpm=self.status.spindle_rpm,
                    spindle_load_percent=self.status.spindle_load_percent,
                    feed_mm_min=feed_mm_min,
                    axis_positions_mm=axis_positions_mm,
                    job_duration_s=job_duration_s,
                    job_times_left_s=job_times_left_s,
                    set_feed=True,
                    set_positions=True,
                    set_job_times=True,
                )
                if recording:
                    mqtt = self._mqtt
                    rec = self._tool_recorder
                    if mqtt is not None and rec is not None and rec.active:
                        sid = self._session_id or ""
                        if sid:
                            file_pos = parse_job_file_position(job)
                            sample = build_motion_sample(
                                session_id=sid,
                                device_id=mqtt.device_id,
                                t_s=rec.elapsed_s(),
                                axis_positions_mm=axis_positions_mm,
                                feed_mm_min=feed_mm_min,
                                job_file=job_file or None,
                                file_position=file_pos,
                                rrf_status=st,
                                tool_number=tool_n,
                            )
                            if sample is not None and self._motion_filter.should_publish(sample):
                                try:
                                    mqtt.publish_motion_sample(sample)
                                except Exception:
                                    logger.debug("motion MQTT publish failed", exc_info=True)
                self._set_status(
                    rrf_connected=True,
                    rrf_status=st,
                    rrf_job_active=active,
                    current_tool=tool_n,
                    job_file=job_file,
                    session_id=self._session_id or "",
                )
                self._note_rrf_poll_success()
            except RrfHttpError as e:
                self._set_status(rrf_connected=False, last_error=f"RRF poll: {e}")
                self._note_rrf_poll_failure(str(e))
                try:
                    client.connect()
                except RrfHttpError:
                    pass
            with self._lock:
                recording = self.status.recording
            wait_s = self.rrf_tool_poll_s if recording else self.rrf_poll_s
            if self._stop_service.wait(wait_s):
                break

    def _on_job(self, state: dict[str, Any] | None, job: dict[str, Any] | None) -> None:
        active = infer_print_job_active(state, job)
        st = (state or {}).get("status", "?")
        if not isinstance(st, str):
            st = repr(st)
        with self._lock:
            recording = self.status.recording
        track_active = (
            infer_job_sync_recording_active(state, job) if recording else active
        )
        prev = self._last_job_active
        if prev is None:
            if active:
                self.start_recording()
        else:
            if not prev and active:
                self.start_recording()
            elif recording and not track_active:
                # Use ``recording``, not ``prev`` — ``_last_job_active`` may already
                # be False while grace is counting down (homing_gui edge-tracking).
                stop_reason = infer_job_sync_stop_reason(state, job)
                now = time.monotonic()
                if self._job_sync_stop_grace_s <= 0:
                    self.stop_recording(reason=stop_reason, rrf_status=st)
                elif self._job_sync_inactive_since is None:
                    self._job_sync_inactive_since = now
                    logger.info(
                        "RRF job ended (status=%s reason=%s) — grace %.1fs before stop",
                        st,
                        stop_reason,
                        self._job_sync_stop_grace_s,
                    )
                elif (now - self._job_sync_inactive_since) >= self._job_sync_stop_grace_s:
                    self.stop_recording(reason=stop_reason, rrf_status=st)
            elif track_active:
                self._job_sync_inactive_since = None
        # Keep tracking only if recording started, same as homing_gui
        with self._lock:
            recording = self.status.recording
        if self._job_sync_inactive_since is not None and recording:
            self._last_job_active = True
        else:
            self._last_job_active = track_active if (recording or not track_active) else False


def _try_run_tray(service: LiveSpindleService, stop_event: threading.Event) -> bool:
    """Optional system tray; returns False if unavailable."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        logger.info("No DISPLAY/WAYLAND_DISPLAY — tray disabled")
        return False
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("pystray/Pillow not installed — tray disabled (pip install pystray Pillow)")
        return False

    def _icon_image(color: str) -> Image.Image:
        img = Image.new("RGB", (64, 64), color)
        d = ImageDraw.Draw(img)
        d.ellipse((8, 8, 56, 56), fill="white")
        return img

    colors = {"idle": "#555555", "recording": "#1b8a1b", "error": "#a11", "stopping": "#888"}

    def make_title() -> str:
        s = service.status
        bits = [f"tap-spindle: {s.state}"]
        if s.rrf_status:
            bits.append(f"rrf={s.rrf_status}")
        if s.current_tool is not None:
            bits.append(f"T{s.current_tool}")
        if s.mqtt_enabled:
            bits.append("mqtt")
        return " | ".join(bits)

    icon_holder: list[Any] = []

    def on_status(_st: ServiceStatus) -> None:
        if not icon_holder:
            return
        icon = icon_holder[0]
        try:
            icon.title = make_title()
            icon.icon = _icon_image(colors.get(service.status.state, "#555555"))
        except Exception:
            pass

    service.on_status = on_status

    def quit_action(icon, _item) -> None:
        stop_event.set()
        icon.stop()

    menu = pystray.Menu(pystray.MenuItem("Quit", quit_action))
    icon = pystray.Icon("tap-spindle", _icon_image("gray"), make_title(), menu)
    icon_holder.append(icon)

    def run_icon() -> None:
        icon.run()

    t = threading.Thread(target=run_icon, name="tray", daemon=True)
    t.start()
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("TAP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Live spindle ADXL → MQTT service (RRF SBC)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "live_spindle" / "service",
        help="Directory for timestamped recording folders",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path(os.environ.get("TAP_LIVE_STATUS_PATH", "/tmp/tap-spindle-status.json")),
        help="JSON status file for tray / monitoring",
    )
    parser.add_argument("-r", "--rate", type=float, default=None, help="Sample rate Hz")
    parser.add_argument(
        "--rrf-base",
        default=os.environ.get("TAP_RRF_BASE", rrf_default_base_url()),
        help="RRF HTTP base (default TAP_RRF_BASE or http://127.0.0.1 on SBC)",
    )
    parser.add_argument("--rrf-password", default=os.environ.get("TAP_RRF_PASSWORD", ""))
    parser.add_argument(
        "--rrf-poll-s",
        type=float,
        default=float(os.environ.get("TAP_RRF_POLL_S", str(rrf_default_poll_interval_s()))),
    )
    parser.add_argument(
        "--rrf-tool-poll-s",
        type=float,
        default=float(os.environ.get("TAP_RRF_TOOL_POLL_S", str(rrf_default_tool_poll_interval_s()))),
        help="RRF poll interval while recording (tool-change timeline; default 0.25s)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--job-sync",
        dest="mode",
        action="store_const",
        const="job_sync",
        help="Start/stop recording with RRF print job",
    )
    mode.add_argument(
        "--always-on",
        dest="mode",
        action="store_const",
        const="always_on",
        help="Record continuously from start",
    )
    mode.add_argument(
        "--idle",
        dest="mode",
        action="store_const",
        const="idle",
        help="Do not auto-start recording (status only / tray)",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        default=os.environ.get("TAP_LIVE_TRAY", "").strip().lower() in ("1", "true", "yes"),
        help="Show system tray icon when a display is available",
    )
    parser.add_argument(
        "--video-test",
        action="store_true",
        help="Record a timed video with live RRF/ADXL overlay (and optional YouTube) then exit",
    )
    parser.add_argument(
        "--video-test-s",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Duration for --video-test (default 30)",
    )
    args = parser.parse_args(argv)

    if args.video_test:
        rrf_base = args.rrf_base
        if rrf_base.rstrip("/") in ("http://milo.local", "http://milo") and Path("/run/dsf").exists():
            rrf_base = "http://127.0.0.1"
        test_dir = args.output_dir / "video-test"
        rate = args.rate if args.rate is not None else cfg.sample_rate_hz
        return run_live_video_test(
            duration_s=args.video_test_s,
            output_dir=test_dir,
            rrf_base=rrf_base,
            rrf_password=args.rrf_password,
            sample_rate_hz=rate,
        )

    if args.mode is None:
        if os.environ.get("TAP_LIVE_ALWAYS_ON", "").strip().lower() in ("1", "true", "yes"):
            args.mode = "always_on"
        elif os.environ.get("TAP_LIVE_JOB_SYNC", "1").strip().lower() in ("0", "false", "no"):
            args.mode = "idle"
        else:
            args.mode = "job_sync"

    job_sync = args.mode == "job_sync"
    always_on = args.mode == "always_on"
    # Prefer localhost on SBC when env still points at milo.local
    rrf_base = args.rrf_base
    if rrf_base.rstrip("/") in ("http://milo.local", "http://milo") and Path("/run/dsf").exists():
        rrf_base = "http://127.0.0.1"

    service = LiveSpindleService(
        output_dir=args.output_dir,
        sample_rate_hz=args.rate if args.rate is not None else cfg.sample_rate_hz,
        status_path=args.status_path,
        rrf_base=rrf_base,
        rrf_password=args.rrf_password,
        rrf_poll_s=args.rrf_poll_s,
        rrf_tool_poll_s=args.rrf_tool_poll_s,
        job_sync=job_sync,
        always_on=always_on,
    )

    stop_event = threading.Event()

    def _handle_sig(_signum, _frame) -> None:
        logger.info("signal received — shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    service.start()
    if args.tray:
        _try_run_tray(service, stop_event)

    try:
        while not stop_event.wait(1.0):
            pass
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
