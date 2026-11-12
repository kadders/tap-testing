#!/usr/bin/env python3
"""Unit tests for NormalizeMqttTelemetry (no Fusion runtime)."""
from __future__ import annotations

import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_src = (_HERE / "PostProcessAll.py").read_text(encoding="utf-8")
_start = _src.index("def NormalizeMqttTelemetry")
_end = _src.index("\ndef ApplyMqttTelemetryToFile")
_ns: dict = {"re": __import__("re")}
exec(_src[_start:_end], _ns)  # noqa: S102
NormalizeMqttTelemetry = _ns["NormalizeMqttTelemetry"]

SAMPLE = (
    '(MQTT: duet tool change active session)\n'
    'M118 P6 S"{""event"":""start"",""tool_number"":1,""job_file"":""8910.gcode""}" '
    'T"duet/milo/tool"\n'
    'T2\n'
    'M118 P6 S"{""event"":""start"",""tool_number"":2,""job_file"":""job.gcode""}" '
    'T"duet/milo/tool"\n'
)


class TestNormalizeMqttTelemetry(unittest.TestCase):
    def test_job_file_and_events(self):
        out = NormalizeMqttTelemetry(SAMPLE, "01 Putter - 3 axis - Op1.gcode")
        self.assertIn('job_file"":""01 Putter - 3 axis - Op1.gcode""', out)
        self.assertNotIn("8910.gcode", out)
        self.assertNotIn("job.gcode", out)
        events = []
        for line in out.splitlines():
            if "M118 P6" in line:
                if 'event"":""start""' in line:
                    events.append("start")
                elif 'event"":""change""' in line:
                    events.append("change")
        self.assertEqual(events, ["start", "change"])

    def test_noop_without_mqtt(self):
        text = "G0 X0\nM30\n"
        self.assertEqual(NormalizeMqttTelemetry(text, "x.gcode"), text)


if __name__ == "__main__":
    unittest.main()
