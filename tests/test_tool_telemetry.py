"""Tests for tool-change timeline recorder."""

from __future__ import annotations

import json
from pathlib import Path

from tap_testing.mqtt_telemetry import MqttTelemetryConfig, MqttTelemetryPublisher
from tap_testing.rrf_http import (
    named_offsets_mm,
    normalize_tool_snapshot,
    parse_current_tool,
    parse_job_file_name,
    parse_job_file_position,
    summarize_tools,
)
from tap_testing.tool_telemetry import ToolEventRecorder
from test_mqtt_telemetry import FakeMqttClient


def test_parse_current_tool():
    assert parse_current_tool(None) is None
    assert parse_current_tool({}) is None
    assert parse_current_tool({"currentTool": -1}) == -1
    assert parse_current_tool({"currentTool": 5}) == 5
    assert parse_current_tool({"currentTool": "3"}) == 3


def test_parse_job_helpers():
    assert parse_job_file_name({"file": {"fileName": "a.gcode"}}) == "a.gcode"
    assert parse_job_file_name({"fileName": "b.g"}) == "b.g"
    assert parse_job_file_position({"filePosition": 1234}) == 1234
    assert parse_job_file_position({}) is None


def test_summarize_tools():
    tools = [{"number": 1, "name": "1/4 EM", "spindleRpm": 18000, "extra": "drop"}]
    out = summarize_tools(tools)
    assert out == [{"number": 1, "name": "1/4 EM", "spindleRpm": 18000}]


def test_named_offsets_and_snapshot():
    tool = {
        "number": 5,
        "name": "1/4 EM",
        "offsets": [0.0, 0.1, -31.742],
        "offsetsProbed": 4,
        "spindleRpm": 18000,
    }
    snap = normalize_tool_snapshot(tool, axis_letters=["X", "Y", "Z"])
    assert snap is not None
    assert snap["offsets_mm"] == {"X": 0.0, "Y": 0.1, "Z": -31.742}
    assert snap["offsets_probed"] == 4
    assert named_offsets_mm([1, 2], ["A", "B"]) == {"A": 1.0, "B": 2.0}


def test_tool_event_recorder_jsonl_and_mqtt(tmp_path: Path):
    client = FakeMqttClient()
    cfg = MqttTelemetryConfig(host="localhost", device_id="testdev", batch_ms=50.0)
    pub = MqttTelemetryPublisher(cfg, client=client, auto_connect=True)
    pub.session_start(mode="live_spindle", sample_rate_hz=800.0, session_id="run1")

    rec = ToolEventRecorder(
        run_dir=tmp_path / "run1",
        session_id="run1",
        mqtt=pub,
        axis_letters=["X", "Y", "Z"],
    )
    start = rec.begin(
        tools=[
            {"number": 1, "name": "EM1", "offsets": [0, 0, -10.0]},
            {"number": 3, "name": "EM3", "offsets": [0, 0, -20.0]},
        ],
        state={"currentTool": 1},
        job={"file": {"fileName": "part.gcode"}, "filePosition": 10},
        sample_rate_hz=800.0,
        axis_letters=["X", "Y", "Z"],
    )
    assert start is not None
    assert start["event"] == "start"
    assert start["event_type"] == "tool_selected"
    assert start["tool_number"] == 1
    assert start["tool_name"] == "EM1"
    assert start["schema_version"] == 2
    assert start["tool_snapshot"]["offsets_mm"]["Z"] == -10.0
    assert (tmp_path / "run1" / "tools-snapshot.json").is_file()
    assert (tmp_path / "run1" / "run-meta.json").is_file()
    snap = json.loads((tmp_path / "run1" / "tools-snapshot.json").read_text())
    assert snap["schema_version"] == 2
    assert snap["axis_letters"] == ["X", "Y", "Z"]

    # No change
    assert rec.observe(state={"currentTool": 1}, job={"filePosition": 20}) == []

    # Offset-only change on active tool (no selection change)
    offset_events = rec.observe(
        state={"currentTool": 1},
        job={"filePosition": 50},
        tools=[
            {"number": 1, "name": "EM1", "offsets": [0, 0, -10.5]},
            {"number": 3, "name": "EM3", "offsets": [0, 0, -20.0]},
        ],
    )
    assert len(offset_events) == 1
    assert offset_events[0]["event_type"] == "tool_offset_changed"
    assert offset_events[0]["offsets"]["delta"]["Z"] == -0.5

    change = rec.observe(
        state={"currentTool": 3},
        job={"file": {"fileName": "part.gcode"}, "filePosition": 99},
        tools=[
            {"number": 1, "name": "EM1", "offsets": [0, 0, -10.5]},
            {"number": 3, "name": "EM3", "offsets": [0, 0, -20.0]},
        ],
    )
    assert len(change) == 1
    assert change[0]["event"] == "change"
    assert change[0]["event_type"] == "tool_selected"
    assert change[0]["previous_tool"] == 1
    assert change[0]["tool_number"] == 3
    assert change[0]["tool_name"] == "EM3"
    assert change[0]["file_position"] == 99

    stop = rec.end()
    assert stop is not None
    assert stop["event"] == "stop"
    assert stop["event_type"] == "tool_table_snapshot"

    lines = (tmp_path / "run1" / "tool-events.jsonl").read_text().strip().splitlines()
    events = [json.loads(x) for x in lines]
    types = [e["event_type"] for e in events]
    assert "tool_table_snapshot" in types
    assert "tool_selected" in types
    assert "tool_offset_changed" in types

    tool_msgs = client.payloads_for("tool")
    assert any(m.get("event_type") == "tool_offset_changed" for m in tool_msgs)
    assert any(m.get("tool_number") == 3 and m.get("previous_tool") == 1 for m in tool_msgs)
    pub.session_stop()


def test_session_start_includes_tool_and_job(tmp_path: Path):
    client = FakeMqttClient()
    cfg = MqttTelemetryConfig(host="localhost", device_id="testdev")
    pub = MqttTelemetryPublisher(cfg, client=client, auto_connect=True)
    pub.session_start(
        mode="live_spindle",
        sample_rate_hz=800.0,
        session_id="s-job",
        job_file="x.gcode",
        tool_number=2,
    )
    starts = [p for p in client.payloads_for("session") if p["event"] == "start"]
    assert starts[-1]["job_file"] == "x.gcode"
    assert starts[-1]["tool_number"] == 2
    assert starts[-1]["event_type"] == "start_session"
    pub.session_stop()
    stops = [p for p in client.payloads_for("session") if p["event"] == "stop"]
    assert stops[-1]["event_type"] == "end_session"
