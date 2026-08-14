"""Unit tests for ArborCTL spindle telemetry (no hardware)."""

from __future__ import annotations

import json
from pathlib import Path

from tap_testing.mqtt_telemetry import MqttTelemetryConfig, MqttTelemetryPublisher
from tap_testing.rrf_http import normalize_rrf_globals
from tap_testing.spindle_telemetry import (
    H100_TYPE_INDEX,
    SpindleTelemetryRecorder,
    hz_to_rpm,
    parse_arborctl_sample,
    rpm_to_hz,
    select_spindle_index,
)
from test_mqtt_telemetry import FakeMqttClient


def _h100_globals(
    *,
    hz: float = 400.0,
    rpm: float = 24000.0,
    poles: float = 2,
    watts: float = 350.0,
    load_percent: float = 23.3,
    fc4_count: int = 13,
    running: bool = True,
    type_index: int = H100_TYPE_INDEX,
) -> dict:
    return {
        "arborctlLdd": True,
        "arborVFDConfig": [[type_index, 2, 1]],
        "arborMotorSpec": [[1.5, poles, 220.0, 400.0, 7.0, 24000.0]],
        "arborVFDStatus": [[running, 1, hz, rpm, True]],
        "arborVFDPower": [[watts, load_percent]],
        "arborVFDCommReady": [True],
        "h100Fc4Count": [fc4_count],
        "arborState": [[None, False, True, None, False]],
    }


def test_hz_rpm_identity_2_and_4_pole():
    assert hz_to_rpm(400.0, poles=2) == 24000.0
    assert hz_to_rpm(400.0, poles=4) == 12000.0
    assert rpm_to_hz(24000.0, poles=2) == 400.0
    assert rpm_to_hz(12000.0, poles=4) == 400.0
    assert abs(hz_to_rpm(rpm_to_hz(18000.0, 2), 2) - 18000.0) < 1e-6


def test_h100_fc4_long_includes_load():
    sample = parse_arborctl_sample(_h100_globals())
    assert sample is not None
    assert sample["source"] == "arborctl"
    assert sample["hz"] == 400.0
    assert sample["rpm"] == 24000.0
    assert sample["poles"] == 2.0
    assert sample["power_available"] is True
    assert sample["watts"] == 350.0
    assert sample["load_percent"] == 23.3
    assert sample["h100_fc4_count"] == 13
    assert "watts" in sample


def test_h100_fc4_idle_zero_load_still_published():
    sample = parse_arborctl_sample(
        _h100_globals(watts=0.0, load_percent=0.0, running=True)
    )
    assert sample is not None
    assert sample["power_available"] is True
    assert sample["load_percent"] == 0.0
    assert sample["watts"] == 0.0


def test_h100_fc4_short_omits_load():
    sample = parse_arborctl_sample(
        _h100_globals(fc4_count=2, watts=0.0, load_percent=0.0)
    )
    assert sample is not None
    assert sample["power_available"] is False
    assert "watts" not in sample
    assert "load_percent" not in sample
    assert sample["h100_fc4_count"] == 2
    assert sample["rpm"] == 24000.0


def test_four_pole_status_kept_as_is():
    sample = parse_arborctl_sample(
        _h100_globals(hz=400.0, rpm=12000.0, poles=4, type_index=0)
    )
    assert sample is not None
    assert sample["hz"] == 400.0
    assert sample["rpm"] == 12000.0
    assert sample["poles"] == 4.0
    # Non-H100 with a power vector still publishes load.
    assert sample["power_available"] is True


def test_commanded_from_rrf_spindles_active():
    sample = parse_arborctl_sample(
        _h100_globals(),
        spindles=[{"state": "forward", "active": 18000.0, "current": 17950.0}],
    )
    assert sample is not None
    assert sample["commanded_rpm"] == 18000.0
    assert sample["commanded_hz"] == 300.0  # 18000 * 2 / 120


def test_unloaded_falls_back_to_rrf_spindle():
    sample = parse_arborctl_sample(
        {},
        spindles=[{"state": "forward", "active": 12000.0, "current": 11900.0}],
    )
    assert sample is not None
    assert sample["source"] == "rrf_spindle"
    assert sample["rpm"] == 11900.0
    assert sample["power_available"] is False
    assert sample["commanded_rpm"] == 12000.0


def test_nothing_loaded_returns_none():
    assert parse_arborctl_sample({}) is None
    assert parse_arborctl_sample({"arborctlLdd": False}) is None


def test_select_spindle_from_tool():
    om = {
        "arborVFDConfig": [None, [5, 2, 1]],
        "arborctlLdd": True,
    }
    idx = select_spindle_index(
        om,
        tools=[{"number": 3, "spindle": 1}],
        current_tool=3,
    )
    assert idx == 1
    assert select_spindle_index({"arborVFDConfig": [None, [5, 2, 1]]}) == 1


def test_normalize_rrf_globals_list_and_dict():
    assert normalize_rrf_globals({"arborctlLdd": True})["arborctlLdd"] is True
    pairs = [{"name": "arborctlLdd", "value": True}, {"name": "arborMaxLoad", "value": 80}]
    g = normalize_rrf_globals(pairs)
    assert g["arborctlLdd"] is True
    assert g["arborMaxLoad"] == 80


def test_recorder_jsonl_and_mqtt(tmp_path: Path):
    client = FakeMqttClient()
    cfg = MqttTelemetryConfig(host="localhost", device_id="testdev", batch_ms=50.0)
    pub = MqttTelemetryPublisher(cfg, client=client, auto_connect=True)
    pub.session_start(mode="live_spindle", sample_rate_hz=800.0, session_id="run1")

    rec = SpindleTelemetryRecorder(
        run_dir=tmp_path / "run1",
        session_id="run1",
        mqtt=pub,
    )
    rec.begin(recording_t0_mono=0.0)
    sample = rec.observe(
        globals_om=_h100_globals(),
        spindles=[{"active": 24000.0}],
        current_tool=1,
    )
    rec.end()
    assert sample is not None
    path = tmp_path / "run1" / "spindle-telemetry.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["rpm"] == 24000.0
    assert lines[0]["load_percent"] == 23.3
    assert lines[0]["session_id"] == "run1"
    frames = client.payloads_for("spindle")
    assert frames and frames[-1]["hz"] == 400.0
    assert frames[-1]["power_available"] is True
    pub.session_stop()
