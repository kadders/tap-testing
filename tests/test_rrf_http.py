"""Tests for ``tap_testing.rrf_http`` (mocked unit tests + optional live Duet at milo.local).

**Live tests are diagnostic-only:** they use ``GET /rr_connect`` and ``GET /rr_model`` only.
They do **not** call ``/rr_gcode``, upload files, or any other endpoint that could queue motion,
change spindle output, or alter machine state beyond opening an HTTP session.

Live HTTP tests are gated on ``TAP_RRF_TEST=1`` so CI does not wait on ``milo.local`` DNS.

Run everything (including live tests against your Duet). Use ``-s`` so response dumps
appear on stdout::

    set TAP_RRF_TEST=1
    pytest tests/test_rrf_http.py -m rrf -s -v

Or only offline unit tests (default)::

    pytest tests/test_rrf_http.py -m "not rrf"
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from tap_testing.rrf_http import (
    RrfClient,
    RrfHttpError,
    infer_print_job_active,
    is_sbc_mode,
    normalize_rrf_globals,
    probe_rrf_base,
    rrf_discovery_hosts,
)

# Live Duet integration: default board hostname from user setup.
_MILO_BASE = os.environ.get("TAP_RRF_BASE", "http://milo.local").rstrip("/")
_RRF_PASSWORD = os.environ.get("TAP_RRF_PASSWORD", "")


def _live_rrf_enabled() -> bool:
    return os.environ.get("TAP_RRF_TEST", "").strip().lower() in ("1", "yes", "true")


skip_if_no_live_rrf = pytest.mark.skipif(
    not _live_rrf_enabled(),
    reason="Set TAP_RRF_TEST=1 to run live RRF tests (milo.local or TAP_RRF_BASE)",
)


def _rrf_print_response(title: str, data: object) -> None:
    """Pretty-print a response body (use ``pytest -s`` to see stdout)."""
    bar = "=" * 64
    print(f"\n{bar}\nRRF ← {title}\n{bar}")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        print(repr(data))


@pytest.mark.parametrize(
    ("state", "job", "expected"),
    [
        (None, None, False),
        ({}, None, False),
        ({"status": "idle"}, None, False),
        ({"status": "busy"}, None, False),
        ({"status": "processing"}, None, True),
        ({"status": "simulating"}, None, True),
        ({"status": "paused"}, None, True),
        ({"status": "cancelling"}, None, True),
        ({"status": "starting"}, None, False),
        ({"status": "starting"}, {"file": {"fileName": "x.gcode"}}, True),
        ({"status": "starting"}, {"fileName": "y.gcode"}, True),
        ({"status": "processing"}, {"file": {"fileName": ""}}, True),
    ],
)
def test_infer_print_job_active(
    state: dict | None,
    job: dict | None,
    expected: bool,
) -> None:
    assert infer_print_job_active(state, job) is expected


def test_infer_job_sync_recording_active() -> None:
    from tap_testing.rrf_http import infer_job_sync_recording_active

    job = {"file": {"fileName": "0:/gcodes/part.gcode"}}
    assert infer_job_sync_recording_active({"status": "processing"}, job) is True
    assert infer_job_sync_recording_active({"status": "paused"}, job) is True
    assert infer_job_sync_recording_active({"status": "completed"}, job) is False
    assert infer_job_sync_recording_active({"status": "idle"}, job) is False
    assert infer_job_sync_recording_active({"status": "busy"}, job) is False
    assert infer_job_sync_recording_active({"status": "processing"}, None) is False
    assert infer_job_sync_recording_active({"status": "idle"}, None) is False


def test_infer_job_sync_stop_reason() -> None:
    from tap_testing.rrf_http import infer_job_sync_stop_reason

    job = {"file": {"fileName": "0:/gcodes/part.gcode"}}
    assert infer_job_sync_stop_reason({"status": "completed"}, job) == "job_completed"
    assert infer_job_sync_stop_reason({"status": "idle"}, None) == "job_no_file"
    assert infer_job_sync_stop_reason({"status": "busy"}, job) == "job_sync_inactive"
    assert infer_job_sync_stop_reason({"status": "halted"}, job) == "rrf_halted"
    assert infer_job_sync_stop_reason({"status": "off"}, job) == "rrf_shutdown"
    assert infer_job_sync_stop_reason({"status": "shutdown"}, job) == "rrf_shutdown"


def test_parse_current_tool_and_job_file() -> None:
    from tap_testing.rrf_http import (
        parse_current_tool,
        parse_job_file_name,
        parse_job_file_position,
        summarize_tool,
    )

    assert parse_current_tool({"currentTool": 4}) == 4
    assert parse_current_tool({"currentTool": -1}) == -1
    assert parse_job_file_name({"file": {"fileName": "x.gcode"}}) == "x.gcode"
    assert parse_job_file_position({"filePosition": 50}) == 50
    assert summarize_tool({"number": 1, "name": "a", "ignore": True}) == {
        "number": 1,
        "name": "a",
    }


def test_parse_job_duration_times_left_and_feed() -> None:
    from tap_testing.rrf_http import (
        parse_job_duration_s,
        parse_job_times_left_s,
        parse_requested_feed_mm_min,
    )

    assert parse_job_duration_s(None) is None
    assert parse_job_duration_s({}) is None
    assert parse_job_duration_s({"duration": 3862}) == 3862.0
    assert parse_job_times_left_s(None) is None
    assert parse_job_times_left_s({"timesLeft": {"file": 90}}) == 90.0
    assert parse_job_times_left_s({"timesLeft": {"file": None}}) is None
    assert parse_requested_feed_mm_min(None) is None
    assert parse_requested_feed_mm_min({"requestedSpeed": 30}) == 1800.0
    assert parse_requested_feed_mm_min({"currentMove": {"requestedSpeed": 25}}) == 1500.0
    assert parse_requested_feed_mm_min({"requestedSpeed": 0}) is None
    assert parse_requested_feed_mm_min({}) is None


def test_parse_axis_positions_mm() -> None:
    from tap_testing.rrf_http import parse_axis_positions_mm

    assert parse_axis_positions_mm(None) == {}
    assert parse_axis_positions_mm({}) == {}
    axes = [
        {"letter": "X", "userPosition": 12.345, "machinePosition": 0.0},
        {"letter": "Y", "userPosition": -3.2},
        {"letter": "Z", "machinePosition": 1.0},
        {"letter": "A", "userPosition": 90},
        {"letter": "B"},
    ]
    assert parse_axis_positions_mm({"axes": axes}) == {
        "X": 12.345,
        "Y": -3.2,
        "Z": 1.0,
        "A": 90.0,
    }
    assert parse_axis_positions_mm(axes)["A"] == 90.0
    assert parse_axis_positions_mm([{"letter": "X", "userPosition": "nan"}]) == {}


def test_fetch_current_move_narrow_then_fallback() -> None:
    client = RrfClient("http://dummy")
    calls: list[str] = []

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 0}
        key = query.get("key", "")
        calls.append(key)
        if key == "move.currentMove":
            return {"result": {}}
        if key == "move":
            return {"result": {"currentMove": {"requestedSpeed": 10}}}
        raise AssertionError(f"unexpected {path} {query}")

    with patch.object(client, "_request_json", side_effect=fake):
        client.connect()
        cm = client.fetch_current_move()
    assert cm == {"requestedSpeed": 10}
    assert calls == ["move.currentMove", "move"]

@pytest.mark.parametrize(
    ("sbc", "expected"),
    [
        (None, False),
        ("", False),
        ([], False),
        ({}, False),
        ({"distribution": "DuetPi"}, True),
        ({"dsf": {"version": "3.5.0"}}, True),
    ],
)
def test_is_sbc_mode(sbc: object, expected: bool) -> None:
    assert is_sbc_mode(sbc) is expected


def test_rrf_client_fetch_sbc() -> None:
    client = RrfClient("http://dummy")

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 0}
        if path == "/rr_model" and query.get("key") == "sbc":
            return {"key": "sbc", "flags": "v", "result": {"distribution": "DuetPi"}}
        raise AssertionError(f"unexpected {path} {query}")

    with patch.object(client, "_request_json", side_effect=fake):
        client.connect()
        sbc = client.fetch_sbc()
        assert sbc == {"distribution": "DuetPi"}
        assert is_sbc_mode(sbc) is True


def test_rrf_client_fetch_globals_and_spindles() -> None:
    client = RrfClient("http://dummy")

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 0}
        if path == "/rr_model" and query.get("key") == "global":
            return {
                "key": "global",
                "flags": "v",
                "result": {
                    "arborctlLdd": True,
                    "arborVFDStatus": [[True, 1, 400.0, 24000.0, True]],
                    "h100Fc4Count": [13],
                },
            }
        if path == "/rr_model" and query.get("key") == "spindles":
            return {
                "key": "spindles",
                "flags": "v",
                "result": [{"state": "forward", "active": 24000, "current": 23980}],
            }
        raise AssertionError(f"unexpected {path} {query}")

    with patch.object(client, "_request_json", side_effect=fake):
        client.connect()
        g = client.fetch_globals()
        spindles = client.fetch_spindles()
    assert g["arborctlLdd"] is True
    assert g["h100Fc4Count"] == [13]
    assert spindles[0]["active"] == 24000
    assert normalize_rrf_globals([{"name": "x", "value": 1}]) == {"x": 1}


def test_rrf_client_connect_invalid_password() -> None:
    client = RrfClient("http://dummy")

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 1}
        raise AssertionError(f"unexpected {path}")

    with patch.object(client, "_request_json", side_effect=fake):
        with pytest.raises(RrfHttpError, match="invalid password"):
            client.connect()


def test_rrf_client_fetch_state_and_job() -> None:
    client = RrfClient("http://dummy")

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 0}
        if path == "/rr_model" and query.get("key") == "state":
            return {"key": "state", "flags": "v", "result": {"status": "processing"}}
        if path == "/rr_model" and query.get("key") == "job":
            return {"key": "job", "flags": "v", "result": {"file": {"fileName": "a.g"}}}
        raise AssertionError(f"unexpected {path} {query}")

    with patch.object(client, "_request_json", side_effect=fake):
        client.connect()
        state, job = client.fetch_state_and_job()
    assert infer_print_job_active(state, job) is True


def test_rrf_client_send_gcode() -> None:
    """Unit-test ``send_gcode`` wiring only (mocked — never hits hardware)."""
    client = RrfClient("http://dummy")
    gcode_calls: list[str] = []

    def fake(path: str, query: dict[str, str]) -> dict:
        if path == "/rr_connect":
            return {"err": 0}
        if path == "/rr_gcode":
            g = query.get("gcode")
            gcode_calls.append(g or "")
            if g == "M115":
                return {"bufferSpace": 100}
            if g == "M122":
                return {"bufferSpace": 120}
            raise AssertionError(f"unexpected gcode {g!r}")
        raise AssertionError(path)

    with patch.object(client, "_request_json", side_effect=fake):
        client.connect()
        out115 = client.send_gcode("M115")
        out122 = client.send_gcode("M122")
    assert out115["bufferSpace"] == 100
    assert out122["bufferSpace"] == 120
    assert gcode_calls == ["M115", "M122"]


def test_rrf_discovery_hosts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAP_RRF_DISCOVER_HOSTS", "a.local, b.local")
    assert rrf_discovery_hosts() == ("a.local", "b.local")


# --- Live integration (milo.local or TAP_RRF_BASE) ---
# Allowed HTTP paths: /rr_connect, /rr_model only (read-only diagnostics).


@skip_if_no_live_rrf
@pytest.mark.rrf
def test_milo_local_probe_and_object_model() -> None:
    """Reachable Duet must answer rr_connect + rr_model for ``state`` (no motion, no spindle)."""
    try:
        msg = probe_rrf_base(_MILO_BASE, password=_RRF_PASSWORD, timeout_s=5.0)
    except RrfHttpError as e:
        pytest.skip(f"Duet not reachable at {_MILO_BASE}: {e}")
    _rrf_print_response(f"probe_rrf_base({_MILO_BASE!r})", msg)
    assert "OK" in msg
    client = RrfClient(_MILO_BASE, password=_RRF_PASSWORD, timeout_s=5.0)
    conn = client.connect()
    _rrf_print_response("rr_connect", conn)
    state, job = client.fetch_state_and_job()
    _rrf_print_response("rr_model key=state (result)", state)
    _rrf_print_response("rr_model key=job (result)", job)
    assert isinstance(state, dict)
    assert "status" in state
    assert isinstance(job, dict)


@skip_if_no_live_rrf
@pytest.mark.rrf
def test_milo_local_read_only_model_poll() -> None:
    """
    Repeated ``rr_model`` reads only (same contract as homing GUI polling).

    Does not send ``/rr_gcode`` or any command that could move axes or run the spindle.
    """
    client = RrfClient(_MILO_BASE, password=_RRF_PASSWORD, timeout_s=5.0)
    try:
        conn = client.connect()
    except RrfHttpError as e:
        pytest.skip(f"Duet not reachable at {_MILO_BASE}: {e}")
    _rrf_print_response("rr_connect", conn)

    prev_status: object = object()
    for poll_n in range(1, 4):
        state, job = client.fetch_state_and_job()
        _rrf_print_response(f"rr_model poll #{poll_n} key=state (result)", state)
        _rrf_print_response(f"rr_model poll #{poll_n} key=job (result)", job)
        st = (state or {}).get("status")
        assert isinstance(state, dict)
        assert "status" in state
        assert isinstance(job, dict)
        if st != prev_status:
            print(f"  (state.status transition {prev_status!r} → {st!r})")
            prev_status = st
        if poll_n < 3:
            time.sleep(0.3)


@skip_if_no_live_rrf
@pytest.mark.rrf
def test_milo_local_arborctl_globals_read_only() -> None:
    """Dump ArborCTL globals (status / power / FC4 count) — no G-code."""
    client = RrfClient(_MILO_BASE, password=_RRF_PASSWORD, timeout_s=5.0)
    try:
        conn = client.connect()
    except RrfHttpError as e:
        pytest.skip(f"Duet not reachable at {_MILO_BASE}: {e}")
    _rrf_print_response("rr_connect", conn)
    try:
        g = client.fetch_globals()
    except RrfHttpError as e:
        pytest.skip(f"rr_model key=global failed: {e}")
    _rrf_print_response("rr_model key=global (normalized)", {
        k: g.get(k)
        for k in (
            "arborctlLdd",
            "arborctlVer",
            "arborVFDStatus",
            "arborVFDPower",
            "h100Fc4Count",
            "arborVFDCommReady",
            "arborMotorSpec",
        )
        if k in g
    })
    assert isinstance(g, dict)
    try:
        spindles = client.fetch_spindles()
    except RrfHttpError as e:
        pytest.skip(f"rr_model key=spindles failed: {e}")
    _rrf_print_response("rr_model key=spindles (result)", spindles)
    assert isinstance(spindles, list)
