"""Unit tests for session video recording (no hardware / ffmpeg)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tap_testing.video_recording import (
    TelemetryOverlayState,
    VideoRecordingConfig,
    VideoSessionRecorder,
    build_ffmpeg_command,
    format_overlay_text,
    format_axis_bar,
    format_axis_positions,
    format_hms,
    orientation_filters,
    parse_flip,
    parse_output_dir,
    parse_rotate_deg,
    parse_video_mode,
    resolve_video_session_dir,
    try_create_video_recorder,
    video_config_from_env,
    youtube_preflight,
    youtube_rtmp_destination,
    run_video_test,
)


def _base_cfg(**kwargs) -> VideoRecordingConfig:
    defaults = dict(
        enabled=True,
        ustreamer_url="http://127.0.0.1:8081/stream",
        overlay_enabled=True,
        fps=30.0,
        bitrate="2500k",
        encoder="libx264",
        youtube_enabled=False,
        youtube_stream_key="",
        youtube_rtmp_url="rtmp://a.rtmp.youtube.com/live2",
        ffmpeg_path="ffmpeg",
    )
    defaults.update(kwargs)
    return VideoRecordingConfig(**defaults)


def test_video_config_from_env_disabled(monkeypatch):
    monkeypatch.delenv("TAP_VIDEO_ENABLED", raising=False)
    assert video_config_from_env() is None


def test_video_config_from_env_enabled(monkeypatch):
    monkeypatch.setenv("TAP_VIDEO_ENABLED", "1")
    monkeypatch.setenv("TAP_USTREAMER_URL", "http://cam.local/stream")
    monkeypatch.setenv("TAP_VIDEO_FPS", "25")
    monkeypatch.setenv("TAP_VIDEO_BITRATE", "1500k")
    monkeypatch.setenv("TAP_VIDEO_ROTATE", "90")
    monkeypatch.setenv("TAP_VIDEO_FLIP", "h")
    monkeypatch.setenv("TAP_VIDEO_DIR", "/tmp/tap-video")
    cfg = video_config_from_env()
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.ustreamer_url == "http://cam.local/stream"
    assert cfg.fps == 25.0
    assert cfg.bitrate == "1500k"
    assert cfg.rotate_deg == 90
    assert cfg.flip == "h"
    assert cfg.output_dir == Path("/tmp/tap-video").resolve()


def test_parse_rotate_and_flip():
    assert parse_rotate_deg("") == 0
    assert parse_rotate_deg("90") == 90
    assert parse_rotate_deg("cw") == 90
    assert parse_rotate_deg("180") == 180
    assert parse_rotate_deg("270") == 270
    assert parse_rotate_deg("ccw") == 270
    assert parse_rotate_deg("45") == 0
    assert parse_rotate_deg("nope") == 0
    assert parse_flip("") == ""
    assert parse_flip("h") == "h"
    assert parse_flip("hflip") == "h"
    assert parse_flip("v") == "v"
    assert parse_flip("both") == "hv"
    assert parse_flip("none") == ""
    assert parse_flip("diagonal") == ""


def test_orientation_filters():
    assert orientation_filters(0, "") == []
    assert orientation_filters(90, "") == ["transpose=1"]
    assert orientation_filters(180, "") == ["transpose=1,transpose=1"]
    assert orientation_filters(270, "") == ["transpose=2"]
    assert orientation_filters(0, "h") == ["hflip"]
    assert orientation_filters(90, "v") == ["transpose=1", "vflip"]
    assert orientation_filters(0, "hv") == ["hflip", "vflip"]


def test_parse_output_dir(tmp_path: Path):
    assert parse_output_dir("") is None
    assert parse_output_dir("  ") is None
    got = parse_output_dir(str(tmp_path / "videos"))
    assert got == (tmp_path / "videos").resolve()


def test_resolve_video_session_dir_default(tmp_path: Path):
    cfg = _base_cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert resolve_video_session_dir(cfg, run_dir, "sess") == run_dir


def test_resolve_video_session_dir_custom(tmp_path: Path):
    dest = tmp_path / "videos"
    cfg = _base_cfg(output_dir=dest)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = resolve_video_session_dir(cfg, run_dir, "sess1")
    assert out == dest / "sess1"
    assert out.is_dir()


def test_youtube_preflight_skip_reasons():
    cfg = _base_cfg()
    ok, reason = youtube_preflight(cfg)
    assert ok is False
    assert "TAP_YOUTUBE_ENABLED" in reason

    cfg = _base_cfg(youtube_enabled=True)
    ok, reason = youtube_preflight(cfg)
    assert ok is False
    assert "TAP_YOUTUBE_STREAM_KEY" in reason

    cfg = _base_cfg(enabled=False, youtube_enabled=True, youtube_stream_key="abc")
    ok, reason = youtube_preflight(cfg)
    assert ok is False
    assert "TAP_VIDEO_ENABLED" in reason


def test_youtube_preflight_passes(monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    cfg = _base_cfg(youtube_enabled=True, youtube_stream_key="secret-key-1234")
    ok, reason = youtube_preflight(cfg)
    assert ok is True
    assert reason == ""


def test_youtube_rtmp_destination():
    cfg = _base_cfg(
        youtube_enabled=True,
        youtube_stream_key="my-key",
        youtube_rtmp_url="rtmp://a.rtmp.youtube.com/live2",
    )
    assert youtube_rtmp_destination(cfg) == "rtmp://a.rtmp.youtube.com/live2/my-key"


def test_format_overlay_text():
    state = TelemetryOverlayState(
        recording_t0_mono=100.0,
        session_id="20260801_141516",
        job_file="0:/gcodes/part.gcode",
        tool_number=3,
        tool_name="Single flute 12mm",
        spindle_rpm=24000.0,
        spindle_load_percent=23.3,
        rrf_status="processing",
        ax_g=0.01,
        ay_g=-0.02,
        az_g=1.01,
    )
    import time

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", lambda: 112.5)
        text = format_overlay_text(state)
    assert "t=12.5s" in text
    assert "T3" in text
    assert "Single flute 12mm" in text
    assert "RPM 24000" in text
    assert "Load 23.3%" in text
    assert "|a|=" in text
    assert "part.gcode" in text
    assert "X " in text
    assert "Y " in text
    assert "Z " in text
    assert "[" in text


def test_format_overlay_text_feed_job_time_and_no_tool():
    state = TelemetryOverlayState(
        recording_t0_mono=100.0,
        tool_number=-1,
        spindle_rpm=12000.0,
        feed_mm_min=1800.0,
        job_duration_s=3862.0,
        job_times_left_s=90.0,
        ax_g=1.5,
        ay_g=-0.5,
        az_g=1.0,
        accel_bar_scale_g=2.0,
    )
    import time

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", lambda: 100.0)
        text = format_overlay_text(state)
    assert "t=0.0s" in text
    assert "job=1:04:22" in text
    assert "left=1:30" in text
    assert "T-1 (no tool)" in text
    assert "F 1800 mm/min" in text
    assert "RPM 12000" in text


def test_format_overlay_omits_job_and_feed_when_missing():
    state = TelemetryOverlayState(recording_t0_mono=1.0, tool_number=2, tool_name="EM")
    import time

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", lambda: 1.0)
        text = format_overlay_text(state)
    assert "job=" not in text
    assert "left=" not in text
    assert "F " not in text
    assert "T2" in text
    assert "X " not in text


def test_format_overlay_xyza_positions():
    state = TelemetryOverlayState(
        recording_t0_mono=1.0,
        axis_positions_mm={"X": 12.345, "Y": -3.2, "Z": 1.0, "A": 90.0},
    )
    import time

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", lambda: 1.0)
        text = format_overlay_text(state)
    assert "X 12.345" in text
    assert "Y -3.200" in text
    assert "Z 1.000" in text
    assert "A 90.000" in text


def test_format_axis_positions_order_and_omit():
    assert format_axis_positions({}) == ""
    assert format_axis_positions(None) == ""
    assert format_axis_positions({"Z": 1.5, "X": 0.0}) == "X 0.000  Z 1.500"
    assert format_axis_positions({"B": 10.0, "X": 1.0, "A": 45.0}) == "X 1.000  A 45.000  B 10.000"


def test_format_axis_bar_grows_and_z_rest_is_positive():
    z_rest = format_axis_bar(1.0, scale=2.0, width=16)
    x_zero = format_axis_bar(0.0, scale=2.0, width=16)
    x_pos = format_axis_bar(2.0, scale=2.0, width=16)
    x_neg = format_axis_bar(-2.0, scale=2.0, width=16)
    assert z_rest.index("|") < z_rest.rfind("█")  # fill is to the right of center
    assert x_zero.count("█") == 0
    assert x_pos.count("█") > x_zero.count("█")
    assert x_neg.index("█") < x_neg.index("|")
    assert format_hms(3862) == "1:04:22"
    assert format_hms(90) == "1:30"


def test_build_ffmpeg_command_local_only(tmp_path: Path):
    cfg = _base_cfg(encoder="libx264", fps=15.0, max_width=640)
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("hello", encoding="utf-8")
    cmd = build_ffmpeg_command(
        cfg,
        output_mp4=tmp_path / "session.mp4",
        overlay_path=overlay,
        youtube_live=False,
    )
    joined = " ".join(cmd)
    assert "rtmp://" not in joined
    assert str(tmp_path / "session.mp4") in cmd
    assert "frag_keyframe+empty_moov+default_base_moof" in joined
    assert cmd[cmd.index("-r") + 1] == "15"
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert "use_wallclock_as_timestamps" in joined
    assert "mpegts" not in joined
    assert "zerolatency" in joined
    assert "ignore_err" in joined
    assert "drawtext" in joined
    assert "scale=640:-2" in joined
    assert "http://127.0.0.1:8081/stream" in cmd


def test_build_ffmpeg_command_rotate_and_flip(tmp_path: Path):
    cfg = _base_cfg(encoder="libx264", rotate_deg=90, flip="h")
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("hello", encoding="utf-8")
    cmd = build_ffmpeg_command(
        cfg,
        output_mp4=tmp_path / "session.mp4",
        overlay_path=overlay,
        youtube_live=False,
    )
    joined = " ".join(cmd)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("transpose=1,hflip,drawtext=")
    assert "drawtext" in joined


def test_build_ffmpeg_command_no_rtmp_when_youtube_disabled(tmp_path: Path):
    cfg = _base_cfg(encoder="libx264", youtube_enabled=False)
    cmd = build_ffmpeg_command(
        cfg,
        output_mp4=tmp_path / "session.mp4",
        overlay_path=None,
        youtube_live=False,
    )
    assert "rtmp://" not in " ".join(cmd)
    assert "anullsrc" not in " ".join(cmd)


def test_build_ffmpeg_command_youtube_tee(tmp_path: Path):
    cfg = _base_cfg(
        encoder="libx264",
        youtube_enabled=True,
        youtube_stream_key="test-key",
    )
    cmd = build_ffmpeg_command(
        cfg,
        output_mp4=tmp_path / "session.mp4",
        overlay_path=None,
        youtube_live=True,
    )
    joined = " ".join(cmd)
    assert "anullsrc" in joined
    assert "rtmp://a.rtmp.youtube.com/live2/test-key" in joined
    assert "-f tee" in joined
    assert "f=fifo" in joined
    assert "fifo_format=flv" in joined
    assert "attempt_recovery=1" in joined
    assert "frag_keyframe" in joined
    assert "-map" in cmd


def test_build_ffmpeg_command_youtube_only_test(tmp_path: Path):
    cfg = _base_cfg(
        encoder="libx264",
        youtube_enabled=True,
        youtube_stream_key="test-key",
    )
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("hello", encoding="utf-8")
    cmd = build_ffmpeg_command(
        cfg,
        output_mp4=tmp_path / "session.mp4",
        overlay_path=overlay,
        youtube_live=True,
        ffmpeg_duration_s=30.0,
        include_local_mp4=False,
    )
    joined = " ".join(cmd)
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale=iw:ih:in_range=full:out_range=tv" in vf
    assert "scale=720:-2" in vf
    assert vf.index("scale=iw:ih") < vf.index("drawtext=")
    assert cmd[cmd.index("-thread_queue_size") + 1] == "32"
    assert "-t" not in cmd
    assert "-use_wallclock_as_timestamps" not in cmd
    assert "-f tee" not in joined
    assert "-f fifo" in joined
    assert "-maxrate" in cmd
    assert "2400k" in joined
    assert str(tmp_path / "session.mp4") not in joined


class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdin = MagicMock()
        self.stderr = MagicMock()
        self.stderr.read.return_value = b""
        self.returncode = 0
        FakePopen.instances.append(self)

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None

    def send_signal(self, sig):
        pass

    def terminate(self):
        pass

    def kill(self):
        pass


def test_video_session_recorder_start_stop(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    FakePopen.instances.clear()
    cfg = _base_cfg(encoder="libx264")
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    run_dir = tmp_path / "run1"
    ok = rec.start(
        run_dir,
        recording_t0_mono=1000.0,
        session_id="sess1",
        job_file="0:/gcodes/test.gcode",
        tool_number=1,
        tool_name="Endmill",
    )
    assert ok is True
    assert rec.active is True
    assert (run_dir / "overlay.txt").exists()
    assert (run_dir / "video-meta.json").exists()
    meta = json.loads((run_dir / "video-meta.json").read_text())
    assert meta["session_id"] == "sess1"
    assert meta["youtube_enabled"] is False
    assert meta["rotate_deg"] == 0
    assert meta["flip"] == ""
    assert "TAP_YOUTUBE" in meta["youtube_skip_reason"]
    assert FakePopen.instances
    assert "rtmp://" not in " ".join(FakePopen.instances[0].cmd)
    rec.update_overlay(spindle_rpm=12000.0, spindle_load_percent=23.3, ax_g=0.1, ay_g=0.0, az_g=1.0)
    overlay_text = (run_dir / "overlay.txt").read_text(encoding="utf-8")
    assert "Load 23.3%" in overlay_text
    assert "Load 23.3%%" not in overlay_text
    rec.stop()
    assert rec.active is False
    meta = json.loads((run_dir / "video-meta.json").read_text())
    assert "video_stopped_at_mono" in meta


def test_video_session_recorder_youtube_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    FakePopen.instances.clear()
    cfg = _base_cfg(
        encoder="libx264",
        youtube_enabled=True,
        youtube_stream_key="abcd-efgh",
    )
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    run_dir = tmp_path / "run2"
    rec.start(run_dir, recording_t0_mono=1.0, session_id="s2")
    assert rec.youtube_live is True
    cmd = " ".join(FakePopen.instances[0].cmd)
    assert "rtmp://a.rtmp.youtube.com/live2/abcd-efgh" in cmd
    meta = json.loads((run_dir / "video-meta.json").read_text())
    assert meta["youtube_enabled"] is True
    assert meta["youtube_skip_reason"] == ""
    rec.stop()


def test_video_session_recorder_custom_output_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    FakePopen.instances.clear()
    video_root = tmp_path / "videos"
    cfg = _base_cfg(encoder="libx264", output_dir=video_root)
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    run_dir = tmp_path / "run3"
    rec.start(run_dir, recording_t0_mono=1.0, session_id="s3")
    video_dir = video_root / "s3"
    assert rec.output_path == video_dir / "session.mp4"
    assert (video_dir / "overlay.txt").exists()
    assert (video_dir / "video-meta.json").exists()
    assert not (run_dir / "session.mp4").exists()
    meta = json.loads((video_dir / "video-meta.json").read_text())
    assert meta["output_dir"] == str(video_dir)
    assert meta["telemetry_run_dir"] == str(run_dir)
    rec.stop()


def test_try_create_video_recorder_none(monkeypatch):
    monkeypatch.delenv("TAP_VIDEO_ENABLED", raising=False)
    assert try_create_video_recorder() is None


def test_try_create_video_recorder_enabled(monkeypatch):
    monkeypatch.setenv("TAP_VIDEO_ENABLED", "1")
    rec = try_create_video_recorder()
    assert rec is not None


def test_run_video_test_success(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    FakePopen.instances.clear()
    clock = {"t": 1000.0}

    def fake_mono() -> float:
        return clock["t"]

    monkeypatch.setattr("tap_testing.video_recording.time.monotonic", fake_mono)
    cfg = _base_cfg(encoder="libx264")
    rec = VideoSessionRecorder(cfg, popen=FakePopen)

    def fake_sleep(s: float) -> None:
        clock["t"] += max(s, 0.25)

    rc = run_video_test(
        duration_s=1.0,
        output_dir=tmp_path,
        recorder=rec,
        sleep=fake_sleep,
    )
    assert rc == 0
    assert FakePopen.instances
    assert rec.active is False


def test_run_video_test_live_overlay_tick(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: "/usr/bin/ffmpeg")
    FakePopen.instances.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr("tap_testing.video_recording.time.monotonic", lambda: clock["t"])
    cfg = _base_cfg(encoder="libx264", overlay_enabled=False)
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    ticks: list[str] = []
    started: list[Path] = []

    def tick(recorder: VideoSessionRecorder) -> None:
        ticks.append("tick")
        recorder.update_overlay(
            tool_number=3,
            tool_name="live tool",
            spindle_rpm=24000.0,
            axis_positions_mm={"X": 1.0, "Y": 2.0, "Z": 3.0, "A": 4.0},
            ax_g=0.01,
            ay_g=-0.02,
            az_g=1.0,
        )

    def on_started(run_dir: Path, _t0: float, _sid: str, _rec: VideoSessionRecorder) -> None:
        started.append(run_dir)

    def fake_sleep(s: float) -> None:
        clock["t"] += max(s, 0.25)

    rc = run_video_test(
        duration_s=1.0,
        output_dir=tmp_path,
        recorder=rec,
        sleep=fake_sleep,
        overlay_tick=tick,
        on_started=on_started,
        force_overlay=True,
    )
    assert rc == 0
    assert ticks
    assert started
    overlay = list(tmp_path.rglob("overlay.txt"))
    assert overlay
    text = overlay[0].read_text(encoding="utf-8")
    assert "live tool" in text
    assert "RPM 24000" in text
    assert "A 4.000" in text
    assert "|a|=" in text


def test_run_video_test_ffmpeg_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: None)
    cfg = _base_cfg(encoder="libx264")
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    assert run_video_test(duration_s=1.0, output_dir=tmp_path, recorder=rec) == 1


def test_parse_video_mode():
    assert parse_video_mode("") == "local"
    assert parse_video_mode("remote") == "remote"
    assert parse_video_mode("cluster") == "remote"
    assert parse_video_mode("bogus") == "local"


def test_remote_video_session_no_ffmpeg(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tap_testing.video_recording.shutil.which", lambda _p: None)
    FakePopen.instances.clear()
    video_root = tmp_path / "mnt_recordings"
    cfg = _base_cfg(video_mode="remote", output_dir=video_root)
    rec = VideoSessionRecorder(cfg, popen=FakePopen)
    pub = MagicMock()
    ok = rec.start(
        tmp_path / "run1",
        recording_t0_mono=1000.0,
        session_id="sess-remote",
        job_file="test.gcode",
        mqtt_publisher=pub,
    )
    assert ok is True
    assert rec.active is True
    assert rec.remote_mode is True
    assert rec.video_stream_remote is True
    expected = video_root / "sess-remote" / "session.mp4"
    assert rec.output_path == expected
    meta = json.loads((video_root / "sess-remote" / "video-meta.json").read_text(encoding="utf-8"))
    assert meta["video_path"] == str(expected)
    assert meta["video_mode"] == "remote"
    assert not FakePopen.instances
    rec.update_overlay(spindle_rpm=24000.0, ax_g=0.1, ay_g=0.0, az_g=1.0)
    assert pub.publish_video_overlay.called
    rec.stop()
    assert rec.active is False

