"""Unit tests for G-code summary + rr_download wiring (mocked)."""
from __future__ import annotations

from tap_testing.gcode_summary import sha256_bytes, summarize_gcode_bytes
from tap_testing.rrf_http import RrfClient, RrfHttpError


def test_summarize_gcode_bytes_tools_and_moves():
    src = b"""
;OPERATION: Face
T1
M3 S18000
G0 X0 Y0
G1 X10 F1350
T2
G1 X20 F800
M5
"""
    out = summarize_gcode_bytes(src, job_file="0:/gcodes/part.gcode")
    assert out["gcode_sha256"] == sha256_bytes(src)
    assert out["tools_used"] == [1, 2]
    assert out["move_stats"]["n_g0"] >= 1
    assert out["move_stats"]["n_g1"] >= 1
    assert any(tc["tool_number"] == 1 for tc in out["tool_change_lines"])
    assert out["source"] == "rrf_download"
    # Bytes are not retained — only the summary object
    assert isinstance(out["gcode_sha256"], str)


def test_rrf_download_file_mocked(monkeypatch):
    client = RrfClient("http://example.invalid")
    payload = b"T3\nG1 X1 F100\n"

    class Resp:
        def read(self, n=-1):
            data = getattr(self, "_data", payload)
            self._data = b""
            return data if data else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=5.0):
        assert "rr_download" in req.full_url
        assert "name=" in req.full_url
        return Resp()

    monkeypatch.setattr(client._opener, "open", fake_open)
    raw = client.download_file("0:/gcodes/part.gcode")
    assert raw == payload
    summary = summarize_gcode_bytes(raw, job_file="0:/gcodes/part.gcode")
    assert summary["tools_used"] == [3]
    del raw  # discard after hash/summary


def test_rrf_download_http_error(monkeypatch):
    import urllib.error

    client = RrfClient("http://example.invalid")

    def boom(req, timeout=5.0):
        raise urllib.error.HTTPError(
            req.full_url, 404, "missing", hdrs=None, fp=None  # type: ignore[arg-type]
        )

    monkeypatch.setattr(client._opener, "open", boom)
    try:
        client.download_file("missing.gcode")
        assert False, "expected RrfHttpError"
    except RrfHttpError as e:
        assert "404" in str(e)
