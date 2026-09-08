"""Unit tests for live_spindle_service job-edge logic (no hardware)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from tap_testing.live_spindle_service import LiveSpindleService, ServiceStatus, main


def test_service_status_json(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=False,
        always_on=False,
    )
    svc._set_status(state="idle", mqtt_enabled=False)
    data = json.loads(status_path.read_text())
    assert data["state"] == "idle"
    assert data["sample_rate_hz"] == 100.0
    assert "spindle_rpm" in data
    assert "spindle_load_percent" in data
    svc._set_status(spindle_rpm=24000.0, spindle_load_percent=23.3)
    data = json.loads(status_path.read_text())
    assert data["spindle_rpm"] == 24000.0
    assert data["spindle_load_percent"] == 23.3
    assert "video_recording" in data
    assert "youtube_live" in data


def test_job_edge_starts_and_stops(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=True,
        always_on=False,
    )
    started = []
    stopped = []

    def fake_start():
        started.append(1)
        with svc._lock:
            svc.status.recording = True
            svc.status.state = "recording"

    def fake_stop(**kwargs):
        stopped.append(kwargs.get("reason", "manual"))
        with svc._lock:
            svc.status.recording = False
            svc.status.state = "idle"

    svc.start_recording = fake_start  # type: ignore[method-assign]
    svc.stop_recording = fake_stop  # type: ignore[method-assign]
    svc._job_sync_stop_grace_s = 0.0

    proc = {"status": "processing"}
    idle = {"status": "idle"}
    svc._on_job(proc, None)
    assert started == [1]
    svc._on_job(proc, None)
    assert started == [1]  # no re-start
    svc._on_job(idle, None)
    assert stopped == ["job_no_file"]


def test_job_sync_busy_stops_while_recording(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=True,
        always_on=False,
    )
    svc._job_sync_stop_grace_s = 0.0
    stopped: list[str] = []
    svc.start_recording = lambda: None  # type: ignore[method-assign]
    svc.stop_recording = lambda **kw: stopped.append(kw.get("reason", "")) or setattr(svc.status, "recording", False)  # type: ignore[method-assign]
    with svc._lock:
        svc.status.recording = True
    job = {"file": {"fileName": "0:/gcodes/part.gcode"}}
    svc._last_job_active = True
    svc._on_job({"status": "busy"}, job)
    assert stopped == ["job_sync_inactive"]
    stopped.clear()
    with svc._lock:
        svc.status.recording = True
    svc._last_job_active = True
    svc._on_job({"status": "completed"}, job)
    assert stopped == ["job_completed"]


def test_job_sync_grace_period_stops_recording(tmp_path: Path, monkeypatch):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=True,
        always_on=False,
    )
    svc._job_sync_stop_grace_s = 0.1
    stopped: list[str] = []
    svc.start_recording = lambda: None  # type: ignore[method-assign]
    svc.stop_recording = lambda **kw: stopped.append(kw.get("reason", "")) or setattr(svc.status, "recording", False)  # type: ignore[method-assign]
    with svc._lock:
        svc.status.recording = True
    svc._last_job_active = True

    t = [1000.0]

    def mono() -> float:
        t[0] += 0.05
        return t[0]

    monkeypatch.setattr(time, "monotonic", mono)

    idle: dict = {"status": "idle"}
    svc._on_job(idle, None)
    assert stopped == []
    assert svc._job_sync_inactive_since is not None

    svc._on_job(idle, None)
    assert stopped == []
    svc._on_job(idle, None)
    assert stopped == []
    svc._on_job(idle, None)  # 4th poll: 0.05*4 >= 0.1 grace (float-safe)
    assert stopped == ["job_no_file"]


def test_always_on_calls_start(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=False,
        always_on=True,
    )
    called = []
    svc.start_recording = lambda: called.append(1)  # type: ignore[method-assign]
    with patch("tap_testing.live_spindle_service.try_create_publisher", return_value=None):
        svc.start()
    assert called == [1]
    svc.stop()


def test_start_recording_invokes_video_recorder(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TAP_VIDEO_ENABLED", raising=False)
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=False,
        always_on=False,
    )
    mock_video = MagicMock()
    mock_video.configure_mock(
        active=True,
        output_path=tmp_path / "out" / "x" / "session.mp4",
        youtube_live=False,
        video_stream_remote=False,
    )
    mock_video.start.return_value = True
    svc._video_recorder = mock_video

    with patch("tap_testing.live_spindle_service.record_stream") as mock_stream:
        mock_stream.side_effect = lambda *a, **k: None
        svc.start_recording()
        time.sleep(0.05)
        svc.stop_recording()

    mock_video.start.assert_called_once()
    mock_video.stop.assert_called_once()
    call_kw = mock_video.start.call_args.kwargs
    assert "recording_t0_mono" in call_kw
    assert call_kw["session_id"]


def test_main_video_test_invokes_runner(tmp_path: Path):
    with patch("tap_testing.live_spindle_service.run_live_video_test", return_value=0) as mock_run:
        rc = main(["--video-test", "--video-test-s", "12", "--output-dir", str(tmp_path)])
    assert rc == 0
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["duration_s"] == 12.0
    assert kwargs["output_dir"] == tmp_path / "video-test"
    assert "rrf_base" in kwargs
    assert kwargs["sample_rate_hz"] > 0


def test_collect_video_overlay_fields():
    from tap_testing.live_spindle_service import collect_video_overlay_fields

    client = MagicMock()
    client.fetch_state_and_job.return_value = (
        {"status": "idle", "currentTool": 3},
        {"file": {"fileName": "0:/gcodes/part.gcode"}, "duration": 12.0},
    )
    client.fetch_tools.return_value = [{"number": 3, "name": "Single flute 12mm"}]
    client.fetch_move.return_value = {
        "currentMove": {"requestedSpeed": 30},
        "axes": [
            {"letter": "X", "userPosition": 1.5},
            {"letter": "Y", "userPosition": -2.0},
            {"letter": "Z", "userPosition": 10.0},
            {"letter": "A", "userPosition": 45.0},
        ],
    }
    client.fetch_globals.return_value = {}
    client.fetch_spindles.return_value = [{"current": 24000.0, "active": 24000.0}]

    fields = collect_video_overlay_fields(client)
    assert fields["tool_number"] == 3
    assert fields["tool_name"] == "Single flute 12mm"
    assert fields["job_file"] == "0:/gcodes/part.gcode"
    assert fields["feed_mm_min"] == 1800.0
    assert fields["axis_positions_mm"]["A"] == 45.0
    assert fields["spindle_rpm"] == 24000.0


def test_job_sync_halted_stop_reason(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=True,
        always_on=False,
    )
    svc._job_sync_stop_grace_s = 0.0
    stopped: list[str] = []
    svc.start_recording = lambda: None  # type: ignore[method-assign]
    svc.stop_recording = lambda **kw: stopped.append(kw.get("reason", "")) or setattr(svc.status, "recording", False)  # type: ignore[method-assign]
    with svc._lock:
        svc.status.recording = True
    job = {"file": {"fileName": "0:/gcodes/part.gcode"}}
    svc._last_job_active = True
    svc._on_job({"status": "halted"}, job)
    assert stopped == ["rrf_halted"]


def test_rrf_disconnect_stops_recording_after_grace(tmp_path: Path, monkeypatch):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
        job_sync=False,
        always_on=True,
    )
    svc._rrf_disconnect_stop_s = 0.1
    stopped: list[str] = []
    svc.stop_recording = lambda **kw: stopped.append(kw.get("reason", "")) or setattr(svc.status, "recording", False)  # type: ignore[method-assign]
    with svc._lock:
        svc.status.recording = True

    t = [1000.0]

    def mono() -> float:
        return t[0]

    monkeypatch.setattr(time, "monotonic", mono)

    svc._note_rrf_poll_failure("connection refused")
    assert stopped == []
    assert svc._rrf_disconnect_since == 1000.0

    t[0] += 0.05
    svc._note_rrf_poll_failure("connection refused")
    assert stopped == []

    t[0] += 0.06
    svc._note_rrf_poll_failure("connection refused")
    assert stopped == ["rrf_disconnect"]


def test_rrf_poll_success_clears_disconnect_timer(tmp_path: Path):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
    )
    svc._rrf_disconnect_since = 123.0
    svc._note_rrf_poll_success()
    assert svc._rrf_disconnect_since is None


def test_rrf_disconnect_no_stop_when_not_recording(tmp_path: Path, monkeypatch):
    status_path = tmp_path / "status.json"
    svc = LiveSpindleService(
        output_dir=tmp_path / "out",
        sample_rate_hz=100.0,
        status_path=status_path,
        rrf_base="http://127.0.0.1",
    )
    svc._rrf_disconnect_stop_s = 0.0
    stopped: list[str] = []
    svc.stop_recording = lambda **kw: stopped.append(kw.get("reason", ""))  # type: ignore[method-assign]
    with svc._lock:
        svc.status.recording = False
    svc._note_rrf_poll_failure("connection refused")
    assert stopped == []


def test_run_live_video_test_wires_live_overlay(tmp_path: Path):
    from tap_testing.live_spindle_service import run_live_video_test

    rec = MagicMock()
    rec.active = True

    with patch("tap_testing.live_spindle_service.run_video_test", return_value=0) as mock_rvt, patch(
        "tap_testing.live_spindle_service.RrfClient"
    ) as mock_client_cls, patch(
        "tap_testing.live_spindle_service.record_stream"
    ) as mock_stream:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_state_and_job.return_value = ({"currentTool": 1}, {})
        mock_client.fetch_tools.return_value = [{"number": 1, "name": "EM"}]
        mock_client.fetch_move.return_value = {
            "axes": [{"letter": "X", "userPosition": 12.3}]
        }
        mock_client.fetch_globals.return_value = {}
        mock_client.fetch_spindles.return_value = []

        rc = run_live_video_test(
            duration_s=5.0,
            output_dir=tmp_path,
            rrf_base="http://127.0.0.1",
            sample_rate_hz=100.0,
        )
        assert rc == 0
        kwargs = mock_rvt.call_args.kwargs
        assert kwargs["force_overlay"] is True
        assert kwargs["overlay_tick"] is not None
        kwargs["overlay_tick"](rec)
        rec.update_overlay.assert_called()
        overlay_kwargs = rec.update_overlay.call_args.kwargs
        assert overlay_kwargs["tool_number"] == 1
        assert overlay_kwargs["tool_name"] == "EM"
        assert overlay_kwargs["axis_positions_mm"]["X"] == 12.3
        kwargs["on_started"](tmp_path, 1.0, "sess", rec)
        kwargs["on_stopped"]()
        mock_stream.assert_called_once()
        mock_client.connect.assert_called_once()
