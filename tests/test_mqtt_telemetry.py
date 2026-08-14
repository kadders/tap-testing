"""Unit tests for optional MQTT telemetry publisher (fake client, no broker)."""
from __future__ import annotations

import json
import os

import pytest

from tap_testing.mqtt_telemetry import (
    MqttTelemetryConfig,
    MqttTelemetryPublisher,
    analysis_payload_from_result,
    mqtt_config_from_env,
    mqtt_require_sbc,
    mqtt_sbc_gate_allows,
    try_create_publisher,
)


class FakeMqttClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, int]] = []
        self.started = False
        self.disconnected = False

    def publish(self, topic: str, payload: str, qos: int = 0):
        self.messages.append((topic, payload, qos))
        return True

    def connect(self, host: str, port: int, keepalive: int = 60):
        return 0

    def loop_start(self) -> None:
        self.started = True

    def loop_stop(self) -> None:
        self.started = False

    def disconnect(self) -> None:
        self.disconnected = True

    def username_pw_set(self, username: str, password: str | None = None) -> None:
        pass

    def payloads_for(self, leaf: str) -> list[dict]:
        out = []
        for topic, body, _qos in self.messages:
            if topic.endswith("/" + leaf) or topic.endswith(leaf):
                out.append(json.loads(body))
        return out


@pytest.fixture
def pub_and_client(monkeypatch):
    monkeypatch.setenv("TAP_MQTT_HOST", "localhost")
    monkeypatch.setenv("TAP_MQTT_FORCE", "1")
    client = FakeMqttClient()
    cfg = MqttTelemetryConfig(host="localhost", device_id="testdev", batch_ms=50.0)
    pub = MqttTelemetryPublisher(cfg, client=client, auto_connect=True)
    return pub, client


def test_mqtt_config_from_env_disabled(monkeypatch):
    monkeypatch.delenv("TAP_MQTT_HOST", raising=False)
    assert mqtt_config_from_env() is None


def test_mqtt_config_from_env_enabled(monkeypatch):
    monkeypatch.setenv("TAP_MQTT_HOST", "broker.local")
    monkeypatch.setenv("TAP_MQTT_PORT", "1884")
    monkeypatch.setenv("TAP_MQTT_DEVICE_ID", "milo-pi")
    monkeypatch.setenv("TAP_MQTT_BATCH_MS", "80")
    cfg = mqtt_config_from_env()
    assert cfg is not None
    assert cfg.host == "broker.local"
    assert cfg.port == 1884
    assert cfg.device_id == "milo-pi"
    assert cfg.batch_ms == 80.0


def test_session_start_stop_topics(pub_and_client):
    pub, client = pub_and_client
    sid = pub.session_start(mode="tap", sample_rate_hz=800.0)
    assert sid
    starts = client.payloads_for("session")
    assert starts[-1]["event"] == "start"
    assert starts[-1]["event_type"] == "start_session"
    assert starts[-1]["session_id"] == sid
    assert starts[-1]["mode"] == "tap"
    assert starts[-1]["device_id"] == "testdev"
    statuses = client.payloads_for("status")
    assert any(s.get("state") == "recording" and s.get("session_id") == sid for s in statuses)
    pub.session_stop(tool_number=3, n_samples=100)
    stops = [p for p in client.payloads_for("session") if p["event"] == "stop"]
    assert stops and stops[-1]["session_id"] == sid
    assert stops[-1]["event_type"] == "end_session"
    assert stops[-1]["tool_number"] == 3
    assert stops[-1]["n_samples"] == 100


def test_session_stop_echoes_job_file(pub_and_client):
    pub, client = pub_and_client
    sid = pub.session_start(
        mode="live_spindle",
        sample_rate_hz=800.0,
        job_file="0:/gcodes/part.gcode",
        extra={"gcode_sha256": "abc123"},
    )
    starts = [p for p in client.payloads_for("session") if p["event"] == "start"]
    assert starts[-1]["job_file"] == "0:/gcodes/part.gcode"
    assert starts[-1]["gcode_sha256"] == "abc123"
    pub.session_stop(tool_number=2)
    stops = [p for p in client.payloads_for("session") if p["event"] == "stop"]
    assert stops[-1]["session_id"] == sid
    assert stops[-1]["job_file"] == "0:/gcodes/part.gcode"
    idle = [s for s in client.payloads_for("status") if s.get("state") == "idle"]
    assert idle and idle[-1].get("job_file") == "0:/gcodes/part.gcode"


def test_batching_and_seq(pub_and_client):
    pub, client = pub_and_client
    # 50 ms batch @ 800 Hz → ~40 samples triggers flush
    pub.session_start(mode="tap", sample_rate_hz=800.0, session_id="s1")
    dt = 1.0 / 800.0
    for i in range(45):
        pub.emit_sample(i * dt, 0.01 * i, 0.0, 1.0)
    batches = client.payloads_for("accel/batch")
    assert len(batches) >= 1
    assert batches[0]["session_id"] == "s1"
    assert batches[0]["seq"] == 0
    assert len(batches[0]["ax"]) >= 1
    assert batches[0]["dt_s"] > 0
    # Second batch after more samples
    for i in range(45, 90):
        pub.emit_sample(i * dt, 0.01 * i, 0.0, 1.0)
    batches = client.payloads_for("accel/batch")
    assert len(batches) >= 2
    assert batches[1]["seq"] == 1
    pub.session_stop()


def test_flush_on_stop(pub_and_client):
    pub, client = pub_and_client
    pub.session_start(mode="stream", sample_rate_hz=100.0, session_id="s2")
    pub.emit_sample(0.0, 0.1, 0.2, 0.3)
    pub.emit_sample(0.01, 0.1, 0.2, 0.3)
    # Not enough for batch window yet
    assert client.payloads_for("accel/batch") == []
    pub.session_stop()
    batches = client.payloads_for("accel/batch")
    assert len(batches) == 1
    assert batches[0]["seq"] == 0
    assert len(batches[0]["ax"]) == 2


def test_publish_analysis_and_modbus(pub_and_client):
    pub, client = pub_and_client
    pub.session_start(mode="homing", sample_rate_hz=800.0, session_id="s3")
    pub.publish_analysis(
        {
            "session_id": "s3",
            "source": "pi",
            "fn_hz": 900.0,
            "avoid_rpm": [13500],
            "suggested_rpm_min": 15000,
            "suggested_rpm_max": 18000,
            "n_teeth": 4,
            "sample_rate_hz": 800.0,
        }
    )
    analyses = client.payloads_for("analysis")
    assert analyses[-1]["fn_hz"] == 900.0
    assert analyses[-1]["source"] == "pi"
    pub.publish_modbus_row(1.5, {"t_s": 1.5, "ts": 1710000000.5, "hr_0": 42})
    mb = client.payloads_for("modbus")
    assert mb[-1]["registers"]["hr_0"] == 42
    assert "t_s" not in mb[-1]["registers"]
    assert "ts" not in mb[-1]["registers"]
    assert mb[-1]["t_s"] == 1.5
    assert mb[-1]["ts"] == 1710000000.5
    pub.publish_spindle_sample(
        {
            "t_s": 1.5,
            "hz": 400.0,
            "rpm": 24000.0,
            "poles": 2,
            "source": "arborctl",
            "power_available": True,
            "load_percent": 20.0,
        }
    )
    sp = client.payloads_for("spindle")
    assert sp[-1]["rpm"] == 24000.0
    assert sp[-1]["session_id"] == "s3"
    pub.session_stop()


def test_analysis_payload_from_result():
    class R:
        natural_freq_hz = 920.5
        natural_freq_hz_uncertainty = 2.0
        avoid_rpm = [13800.0]
        suggested_rpm_min = 15000.0
        suggested_rpm_max = 18000.0
        n_teeth_used = 4
        sample_rate_hz = 800.0

    payload = analysis_payload_from_result(R(), "abc", source="pi")
    assert payload["session_id"] == "abc"
    assert payload["fn_hz"] == 920.5
    assert payload["source"] == "pi"


def test_tap_detected(pub_and_client):
    pub, client = pub_and_client
    pub.session_start(mode="tap", sample_rate_hz=800.0, session_id="s4")
    pub.publish_tap_detected(0.12)
    events = [p for p in client.payloads_for("session") if p["event"] == "tap_detected"]
    assert events and events[-1]["t_s"] == 0.12


def test_mqtt_require_sbc_defaults(monkeypatch):
    monkeypatch.delenv("TAP_MQTT_FORCE", raising=False)
    monkeypatch.delenv("TAP_MQTT_REQUIRE_SBC", raising=False)
    assert mqtt_require_sbc() is True
    monkeypatch.setenv("TAP_MQTT_REQUIRE_SBC", "0")
    assert mqtt_require_sbc() is False
    monkeypatch.setenv("TAP_MQTT_FORCE", "1")
    assert mqtt_require_sbc() is False


def test_sbc_gate_force_skips_probe(monkeypatch):
    monkeypatch.setenv("TAP_MQTT_FORCE", "1")
    allowed, reason, sbc = mqtt_sbc_gate_allows()
    assert allowed is True
    assert "FORCE" in reason
    assert sbc is None


def test_sbc_gate_blocks_without_sbc(monkeypatch):
    monkeypatch.delenv("TAP_MQTT_FORCE", raising=False)
    monkeypatch.setenv("TAP_MQTT_REQUIRE_SBC", "1")
    monkeypatch.setenv("TAP_MQTT_HOST", "broker")

    def fake_probe(*_a, **_k):
        return False, None, "Standalone / non-SBC mode"

    monkeypatch.setattr(
        "tap_testing.rrf_http.probe_rrf_sbc_mode",
        fake_probe,
    )
    allowed, reason, _ = mqtt_sbc_gate_allows()
    assert allowed is False
    assert "Standalone" in reason
    assert try_create_publisher() is None


def test_sbc_gate_allows_when_sbc(monkeypatch):
    monkeypatch.delenv("TAP_MQTT_FORCE", raising=False)
    monkeypatch.setenv("TAP_MQTT_REQUIRE_SBC", "1")
    monkeypatch.setenv("TAP_MQTT_HOST", "broker")

    sbc_obj = {"distribution": "DuetPi", "dsf": {"version": "3.5.0"}}

    def fake_probe(*_a, **_k):
        return True, sbc_obj, "SBC mode active"

    monkeypatch.setattr(
        "tap_testing.rrf_http.probe_rrf_sbc_mode",
        fake_probe,
    )
    allowed, reason, sbc = mqtt_sbc_gate_allows()
    assert allowed is True
    assert sbc == sbc_obj
