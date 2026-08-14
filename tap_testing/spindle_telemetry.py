"""
Live spindle speed/load from ArborCTL object-model globals.

Persists ``spindle-telemetry.jsonl`` beside ``homing.csv`` and optionally publishes
MQTT ``tap/{device}/spindle``. Does **not** open a second Modbus master — ArborCTL
owns RS-485; this module only reads ``/rr_model``.

Identity (same as ArborCTL nameplate check)::

    RPM = 120 * Hz / poles
    Hz  = |RPM| * poles / 120

ArborCTL already applies that when writing ``arborVFDStatus``. We publish those
values as-is plus ``poles`` so collectors never re-scale with FluidNC ``rpm*60/10``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SPINDLE_SCHEMA_VERSION = 1
H100_TYPE_INDEX = 5
DEFAULT_POLES = 2.0

logger = logging.getLogger(__name__)


class _MqttSpindlePublisher(Protocol):
    def publish_spindle_sample(self, payload: dict[str, Any]) -> None: ...


def hz_to_rpm(hz: float, poles: float = DEFAULT_POLES) -> float:
    """Electrical Hz → mechanical RPM. ``RPM = 120 * Hz / poles``."""
    p = float(poles) if poles and float(poles) > 0 else DEFAULT_POLES
    return float(hz) * 120.0 / p


def rpm_to_hz(rpm: float, poles: float = DEFAULT_POLES) -> float:
    """Mechanical RPM → electrical Hz. ``Hz = |RPM| * poles / 120``."""
    p = float(poles) if poles and float(poles) > 0 else DEFAULT_POLES
    return abs(float(rpm)) * p / 120.0


def deci_hz_to_hz(raw: float) -> float:
    """H100 FC4 / holding words are deci-Hz (value × 10)."""
    return float(raw) / 10.0


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    f = _as_float(v)
    if f is None:
        return None
    return int(f)


def _as_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    return None


def _seq_at(seq: Any, index: int) -> Any:
    if not isinstance(seq, list) or index < 0 or index >= len(seq):
        return None
    return seq[index]


def _vec_float(seq: Any, index: int) -> float | None:
    return _as_float(_seq_at(seq, index))


def _vec_int(seq: Any, index: int) -> int | None:
    return _as_int(_seq_at(seq, index))


def _vec_bool(seq: Any, index: int) -> bool | None:
    return _as_bool(_seq_at(seq, index))


def arborctl_loaded(globals_om: dict[str, Any] | None) -> bool:
    if not globals_om:
        return False
    flag = _as_bool(globals_om.get("arborctlLdd"))
    if flag is True:
        return True
    st = globals_om.get("arborVFDStatus")
    if isinstance(st, list) and any(x is not None for x in st):
        return True
    return False


def select_spindle_index(
    globals_om: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    current_tool: int | None = None,
) -> int:
    """Prefer the active tool's RRF spindle index, else first configured VFD."""
    if current_tool is not None and current_tool >= 0 and tools:
        for t in tools:
            try:
                if int(t.get("number")) != int(current_tool):
                    continue
            except (TypeError, ValueError):
                continue
            sp = t.get("spindle")
            n = _as_int(sp)
            if n is not None and n >= 0:
                return n
    cfg = globals_om.get("arborVFDConfig")
    if isinstance(cfg, list):
        for i, row in enumerate(cfg):
            if row is not None:
                return i
    return 0


def parse_arborctl_sample(
    globals_om: dict[str, Any] | None,
    *,
    spindles: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    current_tool: int | None = None,
    t_s: float = 0.0,
    ts: float | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Build one spindle telemetry sample from RRF globals (+ optional spindles[]).

    Returns None when ArborCTL is not loaded and there is no RRF spindle fallback.
    """
    om = globals_om or {}
    idx = select_spindle_index(om, tools=tools, current_tool=current_tool)
    wall = float(ts) if ts is not None else time.time()
    status = _seq_at(om.get("arborVFDStatus"), idx)
    loaded = arborctl_loaded(om)

    if loaded and isinstance(status, list) and len(status) >= 4:
        return _sample_from_arborctl(
            om,
            idx=idx,
            status=status,
            spindles=spindles,
            t_s=t_s,
            ts=wall,
            session_id=session_id,
        )

    # Fallback: RRF spindles[n].current as reference only (not observed).
    if spindles and 0 <= idx < len(spindles) and isinstance(spindles[idx], dict):
        sp = spindles[idx]
        current = _as_float(sp.get("current"))
        active = _as_float(sp.get("active"))
        if current is None and active is None:
            return None
        rpm = current if current is not None else active
        poles = DEFAULT_POLES
        spec = _seq_at(om.get("arborMotorSpec"), idx)
        if isinstance(spec, list):
            p = _vec_float(spec, 1)
            if p is not None and p > 0:
                poles = p
        payload: dict[str, Any] = {
            "schema_version": SPINDLE_SCHEMA_VERSION,
            "t_s": float(t_s),
            "ts": wall,
            "source": "rrf_spindle",
            "spindle_index": idx,
            "rpm": float(rpm) if rpm is not None else None,
            "hz": rpm_to_hz(float(rpm), poles) if rpm is not None else None,
            "poles": float(poles),
            "running": bool(rpm is not None and rpm > 0),
            "power_available": False,
            "time_basis": "recording_monotonic",
        }
        if active is not None:
            payload["commanded_rpm"] = float(active)
            payload["commanded_hz"] = rpm_to_hz(float(active), poles)
        if session_id:
            payload["session_id"] = session_id
        return payload
    return None


def _sample_from_arborctl(
    om: dict[str, Any],
    *,
    idx: int,
    status: list[Any],
    spindles: list[dict[str, Any]] | None,
    t_s: float,
    ts: float,
    session_id: str | None,
) -> dict[str, Any]:
    hz = _vec_float(status, 2)
    rpm = _vec_float(status, 3)
    running = _vec_bool(status, 0)
    direction = _vec_int(status, 1)
    stable = _vec_bool(status, 4)
    comm = _as_bool(_seq_at(om.get("arborVFDCommReady"), idx))

    spec = _seq_at(om.get("arborMotorSpec"), idx)
    poles = DEFAULT_POLES
    if isinstance(spec, list):
        p = _vec_float(spec, 1)
        if p is not None and p > 0:
            poles = p

    cfg = _seq_at(om.get("arborVFDConfig"), idx)
    type_index = _vec_int(cfg, 0) if isinstance(cfg, list) else None

    fc4_count = _as_int(_seq_at(om.get("h100Fc4Count"), idx))
    if fc4_count is None:
        fc4_count = _as_int(om.get("h100ReadMonCount"))

    power = _seq_at(om.get("arborVFDPower"), idx)
    watts = _vec_float(power, 0) if isinstance(power, list) else None
    load_percent = _vec_float(power, 1) if isinstance(power, list) else None

    power_available = _power_available(
        type_index=type_index,
        fc4_count=fc4_count,
        power_vec=power if isinstance(power, list) else None,
        watts=watts,
        load_percent=load_percent,
    )

    commanded_rpm = None
    if spindles and 0 <= idx < len(spindles) and isinstance(spindles[idx], dict):
        commanded_rpm = _as_float(spindles[idx].get("active"))

    state_row = _seq_at(om.get("arborState"), idx)
    fault = _vec_bool(state_row, 4) if isinstance(state_row, list) else None

    payload: dict[str, Any] = {
        "schema_version": SPINDLE_SCHEMA_VERSION,
        "t_s": float(t_s),
        "ts": float(ts),
        "source": "arborctl",
        "spindle_index": idx,
        "hz": hz,
        "rpm": rpm,
        "poles": float(poles),
        "running": running,
        "dir": direction,
        "stable": stable,
        "comm_ready": comm,
        "power_available": power_available,
        "time_basis": "recording_monotonic",
    }
    if type_index is not None:
        payload["vfd_type_index"] = type_index
    if fc4_count is not None:
        payload["h100_fc4_count"] = fc4_count
    if commanded_rpm is not None:
        payload["commanded_rpm"] = commanded_rpm
        payload["commanded_hz"] = rpm_to_hz(commanded_rpm, poles)
    if power_available:
        payload["watts"] = watts if watts is not None else 0.0
        payload["load_percent"] = load_percent if load_percent is not None else 0.0
    if fault is not None:
        payload["fault"] = fault
    if session_id:
        payload["session_id"] = session_id
    return payload


def _power_available(
    *,
    type_index: int | None,
    fc4_count: int | None,
    power_vec: list[Any] | None,
    watts: float | None,
    load_percent: float | None,
) -> bool:
    """H100: long FC4 monitor. Other drivers: any power vector. Short H100 clones: no."""
    if type_index == H100_TYPE_INDEX:
        if fc4_count is not None:
            return fc4_count > 2
        # Older firmware without h100Fc4Count: only claim power if OM is non-zero.
        return bool(
            (watts is not None and watts != 0.0)
            or (load_percent is not None and load_percent != 0.0)
        )
    return power_vec is not None


@dataclass
class SpindleTelemetryRecorder:
    """
    Record ArborCTL spindle samples for one recording session.

    Call :meth:`begin` when ADXL capture starts, :meth:`observe` on each RRF poll,
    and :meth:`end` when capture stops.
    """

    run_dir: Path
    session_id: str
    mqtt: _MqttSpindlePublisher | None = None
    events_filename: str = "spindle-telemetry.jsonl"

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _t0_mono: float | None = field(default=None, init=False)
    _active: bool = field(default=False, init=False)
    _seq: int = field(default=0, init=False)
    _events_path: Path | None = field(default=None, init=False)
    _last_sample: dict[str, Any] | None = field(default=None, init=False)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_sample(self) -> dict[str, Any] | None:
        return self._last_sample

    def set_recording_origin(self, t0_mono: float) -> None:
        with self._lock:
            self._t0_mono = float(t0_mono)

    def begin(self, *, recording_t0_mono: float | None = None) -> None:
        with self._lock:
            if self._active:
                return
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._t0_mono = (
                float(recording_t0_mono) if recording_t0_mono is not None else time.monotonic()
            )
            self._active = True
            self._seq = 0
            self._last_sample = None
            self._events_path = self.run_dir / self.events_filename
            self._events_path.write_text("", encoding="utf-8")

    def observe(
        self,
        *,
        globals_om: dict[str, Any] | None,
        spindles: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        current_tool: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            if not self._active:
                return None
            t0 = self._t0_mono if self._t0_mono is not None else time.monotonic()
            t_s = time.monotonic() - float(t0)
            sample = parse_arborctl_sample(
                globals_om,
                spindles=spindles,
                tools=tools,
                current_tool=current_tool,
                t_s=t_s,
                session_id=self.session_id,
            )
            if sample is None:
                return None
            sample["seq"] = self._seq
            self._seq += 1
            self._last_sample = sample
            self._append_unlocked(sample)
            mqtt = self.mqtt
        if mqtt is not None:
            try:
                mqtt.publish_spindle_sample(sample)
            except Exception:
                logger.debug("MQTT spindle publish failed", exc_info=True)
        return sample

    def end(self) -> None:
        with self._lock:
            self._active = False

    def _append_unlocked(self, sample: dict[str, Any]) -> None:
        if self._events_path is None:
            return
        try:
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")
        except OSError as e:
            logger.debug("spindle-telemetry.jsonl write failed: %s", e)
