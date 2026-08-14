"""Unit tests for live_spindle_service job-edge logic (no hardware)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from tap_testing.live_spindle_service import LiveSpindleService, ServiceStatus


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

    def fake_stop():
        stopped.append(1)
        with svc._lock:
            svc.status.recording = False
            svc.status.state = "idle"

    svc.start_recording = fake_start  # type: ignore[method-assign]
    svc.stop_recording = fake_stop  # type: ignore[method-assign]

    svc._on_job(True, "processing")
    assert started == [1]
    svc._on_job(True, "processing")
    assert started == [1]  # no re-start
    svc._on_job(False, "idle")
    assert stopped == [1]


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
