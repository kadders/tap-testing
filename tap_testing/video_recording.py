"""
Session video recording from ustreamer MJPEG with optional telemetry overlay and YouTube Live.

Enabled when ``TAP_VIDEO_ENABLED=1``. Ingests ``TAP_USTREAMER_URL`` (default
``http://127.0.0.1:8081/stream``), writes ``session.mp4`` beside live-spindle
run artifacts, and optionally tees to YouTube RTMP when preflight passes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_USTREAMER_URL = "http://127.0.0.1:8081/stream"
DEFAULT_YOUTUBE_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2"
DEFAULT_FPS = 15.0
DEFAULT_BITRATE = "1800k"
DEFAULT_GOP = 30
DEFAULT_MAX_WIDTH = 640  # Pi local encode: keep under ~720p after rotate
YOUTUBE_TEST_DEFAULT_BITRATE = "1200k"
YOUTUBE_TEST_DEFAULT_MAX_WIDTH = 720
YOUTUBE_TEST_DEFAULT_FPS = 20.0
YOUTUBE_TEST_INPUT_QUEUE_SIZE = "32"
OVERLAY_MIN_INTERVAL_S = 0.2  # ~5 Hz
DEFAULT_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf",
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def _redact_stream_key(key: str) -> str:
    k = key.strip()
    if len(k) <= 4:
        return "****"
    return f"****{k[-4:]}"


def parse_rotate_deg(raw: str) -> int:
    """Parse TAP_VIDEO_ROTATE: 0, 90, 180, or 270 (clockwise). Invalid → 0."""
    s = (raw or "").strip().lower()
    if not s:
        return 0
    aliases = {
        "0": 0,
        "none": 0,
        "90": 90,
        "cw": 90,
        "clockwise": 90,
        "180": 180,
        "270": 270,
        "ccw": 270,
        "counterclockwise": 270,
        "anticlockwise": 270,
    }
    if s in aliases:
        return aliases[s]
    try:
        deg = int(float(s)) % 360
    except ValueError:
        logger.warning("TAP_VIDEO_ROTATE=%r invalid — using 0", raw)
        return 0
    if deg in (0, 90, 180, 270):
        return deg
    logger.warning("TAP_VIDEO_ROTATE=%r not 0/90/180/270 — using 0", raw)
    return 0


def parse_flip(raw: str) -> str:
    """Parse TAP_VIDEO_FLIP: '', 'h', 'v', or 'hv'. Invalid → ''."""
    s = (raw or "").strip().lower()
    if not s or s in ("0", "none", "off", "false", "no"):
        return ""
    aliases = {
        "h": "h",
        "hflip": "h",
        "horizontal": "h",
        "x": "h",
        "v": "v",
        "vflip": "v",
        "vertical": "v",
        "y": "v",
        "hv": "hv",
        "vh": "hv",
        "both": "hv",
        "hvflip": "hv",
    }
    if s in aliases:
        return aliases[s]
    logger.warning("TAP_VIDEO_FLIP=%r invalid (use h, v, or hv) — ignoring", raw)
    return ""


def parse_output_dir(raw: str) -> Path | None:
    """Parse TAP_VIDEO_DIR. Empty → None (use telemetry run directory)."""
    s = (raw or "").strip()
    if not s:
        return None
    return Path(os.path.expandvars(os.path.expanduser(s))).resolve()


def parse_video_mode(raw: str) -> str:
    """Parse TAP_VIDEO_MODE: ``local`` (default) or ``remote`` (cluster ffmpeg)."""
    s = (raw or "").strip().lower()
    if s in ("", "local", "pi", "on-device"):
        return "local"
    if s in ("remote", "cluster", "k8s", "kubernetes"):
        return "remote"
    logger.warning("TAP_VIDEO_MODE=%r invalid — using local", raw)
    return "local"


def resolve_video_session_dir(
    cfg: VideoRecordingConfig,
    run_dir: Path,
    session_id: str,
) -> Path:
    """
    Directory for session.mp4 / overlay / video-meta.

    When ``TAP_VIDEO_DIR`` is set, files go under ``{dir}/{session_id}/``.
    Otherwise they stay in the telemetry run directory.
    """
    if cfg.output_dir is None:
        return Path(run_dir)
    dest = Path(cfg.output_dir) / session_id
    try:
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    except OSError as e:
        logger.warning(
            "TAP_VIDEO_DIR=%s unusable (%s) — falling back to run dir %s. "
            "Use a path that exists on this host (the Pi), not a desktop path.",
            cfg.output_dir,
            e,
            run_dir,
        )
        return Path(run_dir)


def orientation_filters(rotate_deg: int, flip: str) -> list[str]:
    """ffmpeg vf chain for clockwise rotate then optional h/v flip (before overlay)."""
    parts: list[str] = []
    deg = int(rotate_deg) % 360
    if deg == 90:
        parts.append("transpose=1")
    elif deg == 180:
        parts.append("transpose=1,transpose=1")
    elif deg == 270:
        parts.append("transpose=2")
    mode = (flip or "").strip().lower()
    if mode in ("h", "hv"):
        parts.append("hflip")
    if mode in ("v", "hv"):
        parts.append("vflip")
    return parts


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` atomically so ffmpeg drawtext reload never reads a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class VideoRecordingConfig:
    enabled: bool = False
    ustreamer_url: str = DEFAULT_USTREAMER_URL
    overlay_enabled: bool = True
    fps: float = DEFAULT_FPS
    bitrate: str = DEFAULT_BITRATE
    encoder: str = ""  # empty = auto (h264_v4l2m2m then libx264)
    rotate_deg: int = 0  # 0, 90, 180, 270 clockwise
    flip: str = ""  # "", "h", "v", "hv"
    output_dir: Path | None = None  # TAP_VIDEO_DIR; None = telemetry run dir
    fontfile: str = ""
    youtube_enabled: bool = False
    youtube_stream_key: str = ""
    youtube_rtmp_url: str = DEFAULT_YOUTUBE_RTMP_URL
    ffmpeg_path: str = "ffmpeg"
    stop_timeout_s: float = 10.0
    accel_bar_scale_g: float = 2.0
    overlay_min_interval_s: float = OVERLAY_MIN_INTERVAL_S
    video_mode: str = "local"  # local | remote
    max_width: int = DEFAULT_MAX_WIDTH  # 0 = no downscale


def video_config_from_env(*, force_enabled: bool = False) -> VideoRecordingConfig | None:
    """Return config when ``TAP_VIDEO_ENABLED`` is truthy, else None.

    ``force_enabled`` builds config anyway (used by ``--video-test``).
    """
    if not force_enabled and not _env_truthy("TAP_VIDEO_ENABLED", default=False):
        return None
    fps_s = _env_str("TAP_VIDEO_FPS", str(DEFAULT_FPS))
    try:
        fps = max(1.0, float(fps_s))
    except ValueError:
        fps = DEFAULT_FPS
    rtmp = _env_str("TAP_YOUTUBE_RTMP_URL", DEFAULT_YOUTUBE_RTMP_URL) or DEFAULT_YOUTUBE_RTMP_URL
    ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
    try:
        accel_scale = float(_env_str("TAP_VIDEO_ACCEL_SCALE_G", "2") or "2")
    except ValueError:
        accel_scale = 2.0
    accel_scale = max(0.1, accel_scale)
    try:
        max_w = int(_env_str("TAP_VIDEO_MAX_WIDTH", str(DEFAULT_MAX_WIDTH)) or str(DEFAULT_MAX_WIDTH))
    except ValueError:
        max_w = DEFAULT_MAX_WIDTH
    # 0 disables downscale; otherwise clamp to a sane Pi-friendly range.
    if max_w < 0:
        max_w = DEFAULT_MAX_WIDTH
    if max_w > 0:
        max_w = max(320, min(max_w, 1920))
    return VideoRecordingConfig(
        enabled=True,
        ustreamer_url=_env_str("TAP_USTREAMER_URL", DEFAULT_USTREAMER_URL) or DEFAULT_USTREAMER_URL,
        overlay_enabled=_env_truthy("TAP_VIDEO_OVERLAY", default=True),
        fps=fps,
        bitrate=_env_str("TAP_VIDEO_BITRATE", DEFAULT_BITRATE) or DEFAULT_BITRATE,
        encoder=_env_str("TAP_VIDEO_ENCODER", ""),
        rotate_deg=parse_rotate_deg(_env_str("TAP_VIDEO_ROTATE", "0")),
        flip=parse_flip(_env_str("TAP_VIDEO_FLIP", "")),
        output_dir=parse_output_dir(_env_str("TAP_VIDEO_DIR", "")),
        fontfile=_env_str("TAP_VIDEO_FONT", ""),
        youtube_enabled=_env_truthy("TAP_YOUTUBE_ENABLED", default=False),
        youtube_stream_key=_env_str("TAP_YOUTUBE_STREAM_KEY", ""),
        youtube_rtmp_url=rtmp.rstrip("/"),
        ffmpeg_path=ffmpeg_path,
        accel_bar_scale_g=accel_scale,
        video_mode=parse_video_mode(_env_str("TAP_VIDEO_MODE", "local")),
        max_width=max_w,
    )


def youtube_preflight(cfg: VideoRecordingConfig) -> tuple[bool, str]:
    """
    Return ``(ok, reason)`` for YouTube RTMP tee.

    RTMP is never attempted unless every required setting is present.
    """
    if not cfg.enabled:
        return False, "TAP_VIDEO_ENABLED is not set (local recording required)"
    if not cfg.youtube_enabled:
        return False, "TAP_YOUTUBE_ENABLED is not set"
    if not cfg.youtube_stream_key.strip():
        return False, "TAP_YOUTUBE_STREAM_KEY is empty"
    if not cfg.youtube_rtmp_url.strip():
        return False, "TAP_YOUTUBE_RTMP_URL is empty"
    if not shutil.which(cfg.ffmpeg_path):
        return False, "ffmpeg not found on PATH"
    return True, ""


def youtube_rtmp_destination(cfg: VideoRecordingConfig) -> str:
    """Full RTMP URL: ``{ingest_base}/{stream_key}``."""
    base = cfg.youtube_rtmp_url.rstrip("/")
    key = cfg.youtube_stream_key.strip()
    return f"{base}/{key}"


@dataclass
class TelemetryOverlayState:
    recording_t0_mono: float = 0.0
    session_id: str = ""
    job_file: str = ""
    tool_number: int | None = None
    tool_name: str = ""
    spindle_rpm: float | None = None
    spindle_load_percent: float | None = None
    feed_mm_min: float | None = None
    axis_positions_mm: dict[str, float] = field(default_factory=dict)
    job_duration_s: float | None = None
    job_times_left_s: float | None = None
    rrf_status: str = ""
    ax_g: float | None = None
    ay_g: float | None = None
    az_g: float | None = None
    accel_bar_scale_g: float = 2.0

    def elapsed_s(self) -> float:
        if self.recording_t0_mono <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self.recording_t0_mono)

    def accel_mag_g(self) -> float | None:
        if self.ax_g is None or self.ay_g is None or self.az_g is None:
            return None
        return math.sqrt(self.ax_g ** 2 + self.ay_g ** 2 + self.az_g ** 2)


def format_hms(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


_OVERLAY_AXIS_ORDER = ("X", "Y", "Z", "A")


def format_axis_positions(positions: dict[str, float] | None) -> str:
    """Compact work-coordinate line, XYZA first then any extra letters in map order."""
    if not positions:
        return ""
    bits: list[str] = []
    seen: set[str] = set()
    for letter in _OVERLAY_AXIS_ORDER:
        if letter in positions:
            bits.append(f"{letter} {positions[letter]:.3f}")
            seen.add(letter)
    for letter, val in positions.items():
        if letter in seen:
            continue
        bits.append(f"{letter} {val:.3f}")
    return "  ".join(bits)


def format_axis_bar(value_g: float, *, scale: float = 2.0, width: int = 16) -> str:
    """Centered level meter: negative left of ``|``, positive right. Clamped to ±scale."""
    half = max(2, int(width) // 2)
    span = max(0.1, float(scale))
    clamped = max(-span, min(span, float(value_g)))
    n = int(round(abs(clamped) / span * half))
    n = min(half, max(0, n))
    if clamped >= 0:
        return " " * half + "|" + "█" * n + " " * (half - n)
    return " " * (half - n) + "█" * n + "|" + " " * half


def format_overlay_text(state: TelemetryOverlayState) -> str:
    """Multi-line overlay for ffmpeg drawtext (human-readable; caller escapes ``%`` before writing)."""
    lines: list[str] = []
    t_s = state.elapsed_s()
    time_bits = [f"t={t_s:.1f}s"]
    if state.job_duration_s is not None:
        time_bits.append(f"job={format_hms(state.job_duration_s)}")
    if state.job_times_left_s is not None:
        time_bits.append(f"left={format_hms(state.job_times_left_s)}")
    lines.append("  ".join(time_bits))

    if state.tool_number is not None and state.tool_number < 0:
        lines.append("T-1 (no tool)")
    else:
        tool_bits: list[str] = []
        if state.tool_number is not None:
            tool_bits.append(f"T{state.tool_number}")
        if state.tool_name:
            tool_bits.append(state.tool_name)
        if tool_bits:
            lines.append("  ".join(tool_bits))

    rpm_bits: list[str] = []
    if state.spindle_rpm is not None:
        rpm_bits.append(f"RPM {state.spindle_rpm:.0f}")
    if state.spindle_load_percent is not None:
        rpm_bits.append(f"Load {state.spindle_load_percent:.1f}%")
    if state.feed_mm_min is not None:
        rpm_bits.append(f"F {state.feed_mm_min:.0f} mm/min")
    if rpm_bits:
        lines.append("  ".join(rpm_bits))

    pos_line = format_axis_positions(state.axis_positions_mm)
    if pos_line:
        lines.append(pos_line)

    if state.ax_g is not None and state.ay_g is not None and state.az_g is not None:
        mag = state.accel_mag_g()
        mag_s = f"{mag:.3f}" if mag is not None else "?"
        lines.append(f"|a|={mag_s}g")
        scale = state.accel_bar_scale_g if state.accel_bar_scale_g > 0 else 2.0
        for label, val in (("X", state.ax_g), ("Y", state.ay_g), ("Z", state.az_g)):
            bar = format_axis_bar(val, scale=scale)
            lines.append(f"{label} {val:+.3f} [{bar}]")

    if state.job_file:
        name = state.job_file.rsplit("/", 1)[-1]
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(name)
    return "\n".join(lines)


def resolve_fontfile(cfg: VideoRecordingConfig) -> str | None:
    """Return a TTF path for drawtext, or None if overlay must be skipped."""
    if cfg.fontfile:
        p = Path(os.path.expandvars(os.path.expanduser(cfg.fontfile)))
        if p.is_file():
            return str(p)
        logger.warning("TAP_VIDEO_FONT=%s not found — searching system fonts", cfg.fontfile)
    for candidate in DEFAULT_FONTS:
        if Path(candidate).is_file():
            return candidate
    return None


def _escape_drawtext_textfile(text: str) -> str:
    """Escape ``%`` for ffmpeg drawtext textfile mode (``%`` → ``%%``)."""
    return text.replace("%", "%%")


def _ffmpeg_escape_filter_path(path: Path | str) -> str:
    """Escape ``:`` and ``\\`` for ffmpeg filter arguments."""
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def _redact_cmd(cmd: list[str], stream_key: str) -> str:
    text = " ".join(cmd)
    key = (stream_key or "").strip()
    if key:
        text = text.replace(key, _redact_stream_key(key))
    return text


def _resolve_encoder(cfg: VideoRecordingConfig) -> str:
    if cfg.encoder:
        return cfg.encoder
    if shutil.which(cfg.ffmpeg_path):
        try:
            proc = subprocess.run(
                [cfg.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "h264_v4l2m2m" in (proc.stdout or ""):
                return "h264_v4l2m2m"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "libx264"


def build_ffmpeg_command(
    cfg: VideoRecordingConfig,
    *,
    output_mp4: Path,
    overlay_path: Path | None,
    youtube_live: bool,
    ffmpeg_duration_s: float | None = None,
    include_local_mp4: bool = True,
) -> list[str]:
    """Build ffmpeg argv for local MP4 (+ optional YouTube tee)."""
    encoder = _resolve_encoder(cfg)
    input_url = cfg.ustreamer_url
    youtube_only_test = (
        youtube_live
        and ffmpeg_duration_s is not None
        and not include_local_mp4
    )
    # YouTube live adds ingest buffering; keep ffmpeg side latency low-ish by
    # reducing thread queues and using shorter GOPs. YouTube-only test uses a
    # tiny input queue so stale MJPEG frames are dropped when encode falls behind.
    if youtube_only_test:
        thread_queue_size = YOUTUBE_TEST_INPUT_QUEUE_SIZE
    else:
        thread_queue_size = "128" if youtube_live else "512"
    youtube_gop = str(int(cfg.fps)) if int(cfg.fps) > 0 else str(DEFAULT_GOP)
    local_gop = str(max(15, int(round(float(cfg.fps) * 2)))) if float(cfg.fps) > 0 else str(DEFAULT_GOP)
    cmd: list[str] = [
        cfg.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    if youtube_only_test:
        cmd.extend(["-fflags", "+nobuffer+discardcorrupt"])
    cmd.extend(["-thread_queue_size", thread_queue_size])
    if not youtube_only_test:
        # Wall-clock PTS so sparse encode frames still span real job duration
        # (passthrough alone stamped ~25 fps and collapsed a 2+ min job into ~11 s).
        cmd.extend(
            [
                "-fflags",
                "+genpts+discardcorrupt",
                "-use_wallclock_as_timestamps",
                "1",
            ]
        )
    cmd.extend(["-f", "mjpeg", "-i", input_url])
    if not youtube_live:
        # Job recordings: tolerate occasional corrupt MJPEG frames from ustreamer.
        cmd.extend(["-err_detect", "ignore_err", "-ignore_io_errors", "1"])
    if youtube_live:
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])

    if ffmpeg_duration_s is not None:
        # Video-test only: tolerate slightly truncated MJPEG frames.
        cmd.extend(["-err_detect", "ignore_err", "-ignore_io_errors", "1"])
        if youtube_live and not include_local_mp4:
            # YouTube-only smoke test: Python stop() ends ffmpeg (no `-t`, which
            # can cut mid-frame and crash h264_v4l2m2m on corrupt MJPEG).
            pass
        elif youtube_live:
            frames = max(
                1,
                int(round(float(ffmpeg_duration_s) * float(cfg.fps))) - 2,
            )
            cmd.extend(["-frames:v", str(frames)])
        else:
            cmd.extend(["-t", str(float(ffmpeg_duration_s))])

    vf_parts: list[str] = []
    if youtube_only_test:
        # Full-range MJPEG (yuvj*) → limited yuv420p before transpose/drawtext.
        vf_parts.append("scale=iw:ih:in_range=full:out_range=tv,format=yuv420p")
    vf_parts.extend(orientation_filters(cfg.rotate_deg, cfg.flip))
    if youtube_only_test:
        max_w = int(_env_str("TAP_VIDEO_TEST_MAX_WIDTH", str(YOUTUBE_TEST_DEFAULT_MAX_WIDTH)) or str(YOUTUBE_TEST_DEFAULT_MAX_WIDTH))
        max_w = max(320, min(max_w, 1920))
        vf_parts.append(f"scale={max_w}:-2:flags=fast_bilinear,format=yuv420p")
    if cfg.overlay_enabled and overlay_path is not None:
        font = resolve_fontfile(cfg)
        escaped = _ffmpeg_escape_filter_path(overlay_path)
        draw = (
            f"drawtext=textfile={escaped}:reload=1"
                f":expansion=none"
            f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55:x=10:y=10"
        )
        if font:
            draw += f":fontfile={_ffmpeg_escape_filter_path(font)}"
        else:
            logger.warning(
                "No TTF font found for overlay (install fonts-dejavu-core or set TAP_VIDEO_FONT) "
                "— recording without telemetry overlay"
            )
            draw = ""
        if draw:
            vf_parts.append(draw)
    if vf_parts:
        if not youtube_only_test:
            # Colorspace convert + optional downscale for Pi CPU budget.
            if int(cfg.max_width) > 0:
                vf_parts.append(
                    f"scale={int(cfg.max_width)}:-2:flags=fast_bilinear:"
                    f"in_range=full:out_range=limited,format=yuv420p"
                )
            else:
                vf_parts.append("scale=in_range=full:out_range=limited,format=yuv420p")
        cmd.extend(["-vf", ",".join(vf_parts)])
    elif not youtube_live and int(cfg.max_width) > 0:
        cmd.extend(
            [
                "-vf",
                f"scale={int(cfg.max_width)}:-2:flags=fast_bilinear:"
                f"in_range=full:out_range=limited,format=yuv420p",
            ]
        )

    if youtube_live:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])

    cmd.extend(
        [
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            cfg.bitrate,
            "-g",
            youtube_gop if youtube_live else local_gop,
        ]
    )
    if youtube_live:
        cmd.extend(["-r", str(int(cfg.fps))])
    else:
        # CFR at TAP_VIDEO_FPS with wall-clock PTS: duration matches wall time;
        # duplicates fill gaps when encode falls behind.
        cmd.extend(["-r", str(int(cfg.fps)), "-fps_mode", "cfr"])
    if encoder == "libx264":
        cmd.extend(["-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0"])
    elif encoder == "h264_v4l2m2m" and youtube_only_test:
        cmd.extend(["-bf", "0"])
    if youtube_only_test:
        # Cap peak bitrate on motion-heavy scenes so RTMP writes stay steadier.
        cmd.extend(["-maxrate", cfg.bitrate, "-bufsize", "2400k"])

    if youtube_live:
        audio_bitrate = "96k" if ffmpeg_duration_s is not None and not include_local_mp4 else "128k"
        cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", audio_bitrate])
        # YouTube-only smoke test: Python stop() ends ffmpeg; don't tie shutdown to `-shortest`.
        if not (ffmpeg_duration_s is not None and not include_local_mp4):
            cmd.extend(["-shortest"])
        dest = youtube_rtmp_destination(cfg)
        if include_local_mp4:
            rtmp_leg = (
                f"[f=fifo:onfail=ignore:fifo_format=flv:queue_size=240:"
                f"drop_pkts_on_overflow=1:attempt_recovery=1:recovery_wait_time=1:"
                f"restart_with_keyframe=1:max_recovery_attempts=5]{dest}"
            )
            # Fragmented MP4 so tee can write a live file; RTMP leg is fifo-wrapped.
            tee_spec = (
                f"[f=mp4:movflags=frag_keyframe+empty_moov+default_base_moof]{output_mp4}"
                f"|{rtmp_leg}"
            )
            cmd.extend(["-y", "-f", "tee", tee_spec])
        else:
            # YouTube-only smoke test: fifo muxer drops/recovers on RTMP stalls
            # instead of exiting immediately like bare `-f flv`.
            cmd.extend(
                [
                    "-y",
                    "-f",
                    "fifo",
                    "-fifo_format",
                    "flv",
                    "-queue_size",
                    "128",
                    "-drop_pkts_on_overflow",
                    "1",
                    "-attempt_recovery",
                    "1",
                    "-recovery_wait_time",
                    "1",
                    "-restart_with_keyframe",
                    "1",
                    "-max_recovery_attempts",
                    "5",
                    dest,
                ]
            )
    else:
        cmd.extend(
            [
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                "-y",
                str(output_mp4),
            ]
        )

    return cmd


class VideoSessionRecorder:
    """Start/stop ffmpeg for one live-spindle job session."""

    def __init__(
        self,
        cfg: VideoRecordingConfig,
        *,
        popen: Callable[..., subprocess.Popen] | None = None,
    ) -> None:
        self.cfg = cfg
        self._popen = popen or subprocess.Popen
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._overlay_state = TelemetryOverlayState()
        self._overlay_path: Path | None = None
        self._run_dir: Path | None = None
        self._telemetry_run_dir: Path | None = None
        self._output_path: Path | None = None
        self._video_started_at_mono: float | None = None
        self._youtube_live = False
        self._youtube_only_test = False
        self._youtube_skip_reason = ""
        self._encoder = ""
        self._last_overlay_write = 0.0
        self._mqtt_publisher: Any | None = None
        self._stderr_file: Any = None
        self._logged_ffmpeg_death = False
        self.active = False

    @property
    def remote_mode(self) -> bool:
        return self.cfg.video_mode == "remote"

    @property
    def video_stream_remote(self) -> bool:
        return self.remote_mode and self.active

    @property
    def youtube_live(self) -> bool:
        return self._youtube_live

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def process_running(self) -> bool:
        if self.remote_mode:
            return self.active
        proc = self._proc
        if proc is None:
            return False
        return proc.poll() is None

    def overlay_state(self) -> TelemetryOverlayState:
        with self._lock:
            return TelemetryOverlayState(**vars(self._overlay_state))

    def update_overlay(self, **kwargs: Any) -> None:
        """Update overlay fields; rewrite overlay file or publish MQTT at most ~5 Hz."""
        self._warn_if_ffmpeg_dead()
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._overlay_state, k):
                    setattr(self._overlay_state, k, v)
            if not self.active or not self.cfg.overlay_enabled:
                return
            now = time.monotonic()
            if now - self._last_overlay_write < self.cfg.overlay_min_interval_s:
                return
            self._last_overlay_write = now
            text = format_overlay_text(self._overlay_state)
            if self.remote_mode:
                pub = self._mqtt_publisher
                if pub is not None and hasattr(pub, "publish_video_overlay"):
                    try:
                        pub.publish_video_overlay(
                            text=text,
                            session_id=self._overlay_state.session_id,
                            t_s=self._overlay_state.elapsed_s(),
                        )
                    except Exception as e:
                        logger.debug("video_overlay MQTT publish failed: %s", e)
                return
            if self._overlay_path is None:
                return
            try:
                _atomic_write_text(self._overlay_path, text)
            except OSError as e:
                logger.debug("overlay write failed: %s", e)

    def start(
        self,
        run_dir: Path,
        *,
        recording_t0_mono: float,
        session_id: str,
        job_file: str | None = None,
        tool_number: int | None = None,
        tool_name: str | None = None,
        ffmpeg_duration_s: float | None = None,
        include_local_mp4: bool = True,
        mqtt_publisher: Any | None = None,
    ) -> bool:
        """
        Start ffmpeg for this session. Returns True if recording started.

        On failure, logs a warning and returns False (ADXL capture continues).
        """
        if self.active:
            return True
        self._mqtt_publisher = mqtt_publisher

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        video_dir = resolve_video_session_dir(self.cfg, run_dir, session_id)
        output_mp4 = video_dir / "session.mp4"
        overlay_path = video_dir / "overlay.txt" if self.cfg.overlay_enabled else None

        if self.remote_mode:
            if not self.cfg.enabled:
                logger.warning("Remote video skipped: TAP_VIDEO_ENABLED is not set")
                return False
            self._telemetry_run_dir = run_dir
            self._run_dir = video_dir
            # Cluster tap-stream writes this path on shared NFS; Pi does not encode.
            self._output_path = output_mp4
            self._overlay_path = overlay_path
            self._youtube_live = False
            self._youtube_only_test = False
            with self._lock:
                self._overlay_state = TelemetryOverlayState(
                    recording_t0_mono=recording_t0_mono,
                    session_id=session_id,
                    job_file=job_file or "",
                    tool_number=tool_number,
                    tool_name=tool_name or "",
                    accel_bar_scale_g=self.cfg.accel_bar_scale_g,
                )
            logger.info(
                "Remote video stream enabled (cluster ffmpeg → %s via MQTT session + video_overlay)",
                output_mp4,
            )
            self.active = True
            self._write_video_meta(
                session_id=session_id,
                recording_t0_mono=recording_t0_mono,
                job_file=job_file,
            )
            if self.cfg.overlay_enabled:
                self.update_overlay()
            return True

        if not shutil.which(self.cfg.ffmpeg_path):
            logger.warning("Video recording skipped: ffmpeg not found on PATH")
            return False

        yt_ok, yt_reason = youtube_preflight(self.cfg)
        if yt_ok:
            logger.info("YouTube live enabled (ingest %s, key %s)", self.cfg.youtube_rtmp_url, _redact_stream_key(self.cfg.youtube_stream_key))
        else:
            if self.cfg.youtube_enabled or self.cfg.youtube_stream_key:
                logger.info("YouTube live skipped: %s", yt_reason)
            self._youtube_skip_reason = yt_reason

        self._encoder = _resolve_encoder(self.cfg)
        self._youtube_live = yt_ok
        self._youtube_only_test = bool(
            yt_ok and ffmpeg_duration_s is not None and not include_local_mp4
        )
        self._telemetry_run_dir = run_dir
        self._run_dir = video_dir
        self._output_path = output_mp4
        self._overlay_path = overlay_path

        with self._lock:
            self._overlay_state = TelemetryOverlayState(
                recording_t0_mono=recording_t0_mono,
                session_id=session_id,
                job_file=job_file or "",
                tool_number=tool_number,
                tool_name=tool_name or "",
                accel_bar_scale_g=self.cfg.accel_bar_scale_g,
            )
        if overlay_path is not None:
            overlay_path.write_text(
                format_overlay_text(self._overlay_state),
                encoding="utf-8",
            )

        cmd = build_ffmpeg_command(
            self.cfg,
            output_mp4=output_mp4,
            overlay_path=overlay_path,
            youtube_live=yt_ok,
            ffmpeg_duration_s=ffmpeg_duration_s,
            include_local_mp4=include_local_mp4,
        )
        if yt_ok and not include_local_mp4:
            logger.info("Starting YouTube live stream")
        else:
            logger.info("Starting session video → %s", output_mp4)
        logger.info("ffmpeg: %s", _redact_cmd(cmd, self.cfg.youtube_stream_key))

        self._logged_ffmpeg_death = False
        log_path = video_dir / "ffmpeg.log"
        try:
            self._stderr_file = log_path.open("w", encoding="utf-8")
            self._proc = self._popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
        except OSError as e:
            logger.warning("Video recording failed to start: %s", e)
            self._close_stderr_file()
            self._reset_session()
            return False

        self._video_started_at_mono = time.monotonic()
        self.active = True
        self._write_video_meta(
            session_id=session_id,
            recording_t0_mono=recording_t0_mono,
            job_file=job_file,
        )
        return True

    def stop(self) -> None:
        """Gracefully stop ffmpeg and finalize video-meta.json."""
        if self.remote_mode:
            self._finalize_video_meta()
            self._reset_session()
            logger.info("Remote video stream stopped")
            return
        proc = self._proc
        if proc is None or not self.active:
            self._reset_session()
            return

        still_running = proc.poll() is None
        if still_running and proc.stdin is not None:
            try:
                # Ask ffmpeg to quit gracefully; fragmented MP4 needs time to flush.
                proc.stdin.write(b"q")
                proc.stdin.flush()
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        elif proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

        grace = self.cfg.stop_timeout_s + (5.0 if self._youtube_live else 0.0)
        if still_running:
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not exit after q — terminating")
                proc.terminate()
                self._wait_ffmpeg_exit(proc, timeout_s=15.0)
                if proc.poll() is None:
                    proc.kill()
                    self._wait_ffmpeg_exit(proc, timeout_s=10.0)
                if proc.poll() is None:
                    logger.error("ffmpeg pid %s did not die after SIGKILL", proc.pid)

        code = proc.returncode
        teardown_ok = self._youtube_only_test and code in (255, -1)
        level = logging.DEBUG if code in (0, None) or teardown_ok else logging.ERROR
        self._consume_stderr(level=level)
        if code not in (0, None):
            if teardown_ok:
                logger.warning(
                    "ffmpeg exited with code %s (YouTube RTMP teardown — expected after live test)",
                    code,
                )
            elif code == -7:
                logger.error(
                    "ffmpeg exited with code %s (SIGBUS — check ffmpeg.log; "
                    "fragmented MP4 should still be partially playable)",
                    code,
                )
            else:
                logger.error("ffmpeg exited with code %s", code)

        self._close_stderr_file()
        self._finalize_video_meta()
        self._reset_session()
        logger.info("Session video stopped")

    def _warn_if_ffmpeg_dead(self) -> None:
        proc = self._proc
        if proc is None or not self.active or self._logged_ffmpeg_death:
            return
        code = proc.poll()
        if code is None:
            return
        self._logged_ffmpeg_death = True
        logger.error(
            "ffmpeg died during session (exit %s) — capture file %s",
            code,
            self._output_path,
        )

    def _wait_ffmpeg_exit(self, proc: subprocess.Popen, *, timeout_s: float) -> int | None:
        deadline = time.monotonic() + max(0.1, timeout_s)
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        return proc.poll()

    def _close_stderr_file(self) -> None:
        fh = self._stderr_file
        self._stderr_file = None
        if fh is None:
            return
        try:
            fh.flush()
            fh.close()
        except OSError:
            pass

    def _consume_stderr(self, *, level: int = logging.DEBUG) -> str:
        if self._stderr_file is not None:
            try:
                self._stderr_file.flush()
            except OSError:
                pass
        log_path = self._run_dir / "ffmpeg.log" if self._run_dir is not None else None
        text = ""
        if log_path is not None and log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                text = ""
        if text:
            logger.log(level, "ffmpeg stderr:\n%s", text[-4000:])
        return text

    def _write_video_meta(
        self,
        *,
        session_id: str,
        recording_t0_mono: float,
        job_file: str | None,
    ) -> None:
        if self._run_dir is None:
            return
        meta = {
            "session_id": session_id,
            "time_basis": "recording_monotonic",
            "recording_t0_mono": recording_t0_mono,
            "video_started_at_mono": self._video_started_at_mono,
            "video_started_at_wall": time.time(),
            "ustreamer_url": self.cfg.ustreamer_url,
            "output_file": "session.mp4",
            "output_dir": str(self._run_dir),
            "video_path": str(self._output_path) if self._output_path else "",
            "telemetry_run_dir": str(self._telemetry_run_dir or ""),
            "overlay_enabled": self.cfg.overlay_enabled,
            "encoder": self._encoder,
            "fps": self.cfg.fps,
            "bitrate": self.cfg.bitrate,
            "max_width": self.cfg.max_width,
            "rotate_deg": self.cfg.rotate_deg,
            "flip": self.cfg.flip,
            "youtube_enabled": self._youtube_live,
            "youtube_skip_reason": self._youtube_skip_reason if not self._youtube_live else "",
            "youtube_rtmp_url": self.cfg.youtube_rtmp_url if self._youtube_live else "",
            "youtube_stream_key_hint": _redact_stream_key(self.cfg.youtube_stream_key) if self._youtube_live else "",
            "job_file": job_file or "",
            "video_mode": self.cfg.video_mode,
        }
        if self._video_started_at_mono is not None:
            meta["video_start_skew_s"] = self._video_started_at_mono - recording_t0_mono
        path = self._run_dir / "video-meta.json"
        try:
            path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError as e:
            logger.debug("video-meta write failed: %s", e)

    def _finalize_video_meta(self) -> None:
        if self._run_dir is None:
            return
        path = self._run_dir / "video-meta.json"
        if not path.exists():
            return
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        meta["video_stopped_at_mono"] = time.monotonic()
        meta["video_stopped_at_wall"] = time.time()
        if self._output_path is not None and self._output_path.exists():
            meta["output_size_bytes"] = self._output_path.stat().st_size
        try:
            path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError as e:
            logger.debug("video-meta finalize failed: %s", e)

    def _reset_session(self) -> None:
        self._close_stderr_file()
        with self._lock:
            self._proc = None
            self.active = False
            self._overlay_path = None
            self._run_dir = None
            self._telemetry_run_dir = None
            self._output_path = None
            self._video_started_at_mono = None
            self._youtube_live = False
            self._last_overlay_write = 0.0
            self._mqtt_publisher = None
            self._logged_ffmpeg_death = False


def try_create_video_recorder() -> VideoSessionRecorder | None:
    """Factory: None when video is disabled via env."""
    cfg = video_config_from_env()
    if cfg is None:
        return None
    return VideoSessionRecorder(cfg)


def run_video_test(
    *,
    duration_s: float = 30.0,
    output_dir: Path | None = None,
    recorder: VideoSessionRecorder | None = None,
    sleep: Callable[[float], None] = time.sleep,
    overlay_tick: Callable[[VideoSessionRecorder], None] | None = None,
    on_started: Callable[[Path, float, str, VideoSessionRecorder], None] | None = None,
    on_stopped: Callable[[], None] | None = None,
    force_overlay: bool = False,
) -> int:
    """
    Timed YouTube live smoke test (no local ``session.mp4``).

    Uses the same env as the daemon. If ``TAP_VIDEO_ENABLED`` is unset, video is
    enabled for this process only. Returns 0 on success, 1 on failure.

    Requires YouTube preflight to pass (``TAP_YOUTUBE_ENABLED`` + stream key).
    Local MP4 is not written during ``--video-test``; overlay artifacts
    (``overlay.txt``, ``video-meta.json``) still land in the run directory.

    ``overlay_tick`` (if given) owns HUD updates — typically live RRF + ADXL
    from ``live_spindle_service``. ``force_overlay`` turns drawtext on for this
    process even when ``TAP_VIDEO_OVERLAY=0``.
    """
    duration_s = max(1.0, float(duration_s))
    if recorder is None:
        cfg = video_config_from_env(force_enabled=True)
        if cfg is None:
            logger.error("Could not build video config")
            return 1
        if not _env_truthy("TAP_VIDEO_ENABLED", default=False):
            logger.info("TAP_VIDEO_ENABLED is not set — enabling for this test only")
        if force_overlay and not cfg.overlay_enabled:
            logger.info("TAP_VIDEO_OVERLAY is off — enabling overlay for this test only")
            cfg.overlay_enabled = True
        cfg.overlay_min_interval_s = 0.2  # 5 Hz HUD for live stream test
        cfg.stop_timeout_s = 5.0
        if _env_str("TAP_VIDEO_BITRATE", "") == "":
            cfg.bitrate = YOUTUBE_TEST_DEFAULT_BITRATE
            logger.info("Video test: defaulting TAP_VIDEO_BITRATE to %s", cfg.bitrate)
        if _env_str("TAP_VIDEO_FPS", "") == "":
            cfg.fps = YOUTUBE_TEST_DEFAULT_FPS
            logger.info("Video test: defaulting TAP_VIDEO_FPS to %.0f", cfg.fps)
        if cfg.video_mode == "remote":
            logger.info(
                "Video test: TAP_VIDEO_MODE=remote — Pi publishes overlay; "
                "cluster tap-stream owns YouTube RTMP"
            )
            recorder = VideoSessionRecorder(cfg)
            video_test_youtube_only = True
        else:
            if not cfg.encoder:
                # h264_v4l2m2m can SIGBUS (-7) on corrupt/truncated MJPEG under load.
                cfg.encoder = "libx264"
                logger.info(
                    "Video test: using libx264 encoder "
                    "(set TAP_VIDEO_ENCODER to override; v4l2m2m is fragile here)"
                )
            if not shutil.which(cfg.ffmpeg_path):
                logger.error("Video test failed: ffmpeg not found on PATH")
                return 1

            yt_ok, yt_reason = youtube_preflight(cfg)
            if not yt_ok:
                logger.error("Video test failed: YouTube live required: %s", yt_reason)
                return 1
            logger.info("Video test: YouTube live only (no local session.mp4)")

            recorder = VideoSessionRecorder(cfg)
            video_test_youtube_only = True
    else:
        # Test harness / unit tests: keep existing single-process behavior.
        video_test_youtube_only = False
        cfg = recorder.cfg
        if force_overlay and not cfg.overlay_enabled:
            logger.info("TAP_VIDEO_OVERLAY is off — enabling overlay for this test only")
            cfg.overlay_enabled = True
        if not shutil.which(cfg.ffmpeg_path):
            logger.error("Video test failed: ffmpeg not found on PATH")
            return 1

        yt_ok, yt_reason = youtube_preflight(cfg)
        if yt_ok:
            logger.info("Video test: YouTube live will be attempted")
        else:
            logger.info("Video test: YouTube live skipped: %s", yt_reason)

    base = Path(output_dir) if output_dir is not None else Path.cwd() / "data" / "live_spindle" / "video-test"
    session_id = time.strftime("test_%Y%m%d_%H%M%S")
    run_dir = base / session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ok = recorder.start(
        run_dir,
        recording_t0_mono=t0,
        session_id=session_id,
        job_file="video-test",
        ffmpeg_duration_s=duration_s,
        include_local_mp4=not video_test_youtube_only,
    )
    if not ok:
        logger.error("Video test failed: video session did not start")
        return 1

    logger.info(
        "Video test running for %.0fs → %s%s",
        duration_s,
        "remote cluster stream" if recorder.remote_mode else (
            "YouTube live" if video_test_youtube_only else (recorder.output_path or run_dir)
        ),
        " (live overlay)" if overlay_tick is not None else "",
    )
    if on_started is not None:
        try:
            on_started(run_dir, t0, session_id, recorder)
        except Exception:
            logger.warning("Video test on_started failed", exc_info=True)

    # YouTube-only test: Python stop() after duration; allow RTMP teardown grace.
    deadline = t0 + duration_s + (5.0 if video_test_youtube_only else 8.0)
    failed = False
    ran_full_duration = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                ran_full_duration = True
                break
            if not recorder.process_running():
                elapsed = time.monotonic() - t0
                # ffmpeg may exit during RTMP teardown after a full-duration run.
                natural_stop = video_test_youtube_only and elapsed >= (duration_s - 1.5)
                if remaining > 1.0 and not natural_stop:
                    logger.error("Video test failed: ffmpeg exited early")
                    failed = True
                break
            if overlay_tick is not None:
                try:
                    overlay_tick(recorder)
                except Exception:
                    logger.debug("Video test overlay tick failed", exc_info=True)
            else:
                recorder.update_overlay(
                    rrf_status=f"video-test {remaining:.0f}s left",
                    job_file="video-test",
                )
            sleep(min(0.25, remaining))
    except KeyboardInterrupt:
        logger.info("Video test interrupted")
    finally:
        if on_stopped is not None:
            try:
                on_stopped()
            except Exception:
                logger.debug("Video test on_stopped failed", exc_info=True)
        recorder.stop()

    if failed:
        return 1
    if video_test_youtube_only and ran_full_duration:
        logger.info(
            "Video test complete (YouTube live, %.0fs — RTMP teardown may log warnings)",
            duration_s,
        )
        return 0
    if video_test_youtube_only:
        logger.info("Video test complete (YouTube live, %.0fs)", duration_s)
    else:
        logger.info("Video test complete (file: %s)", recorder.output_path or run_dir)
    return 0
