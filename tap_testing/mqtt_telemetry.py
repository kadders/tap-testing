"""
Optional MQTT publisher for tap-testing telemetry (Jarvis tap_collector).

Enabled when TAP_MQTT_HOST is set. Uses paho-mqtt when available; otherwise
logs a warning and becomes a no-op so SPI capture is never blocked.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MqttClientLike(Protocol):
    """Minimal client surface used by the publisher (real or fake)."""

    def publish(self, topic: str, payload: str, qos: int = 0) -> Any: ...

    def connect(self, host: str, port: int, keepalive: int = 60) -> Any: ...

    def loop_start(self) -> None: ...

    def loop_stop(self) -> None: ...

    def disconnect(self) -> None: ...

    def username_pw_set(self, username: str, password: str | None = None) -> None: ...


@dataclass
class MqttTelemetryConfig:
    host: str
    port: int = 1883
    device_id: str = ""
    batch_ms: float = 100.0
    username: str | None = None
    password: str | None = None
    client_id: str = ""

    def __post_init__(self) -> None:
        if not self.device_id:
            self.device_id = socket.gethostname()
        if not self.client_id:
            self.client_id = f"tap-{self.device_id}"


def mqtt_config_from_env() -> MqttTelemetryConfig | None:
    """Return config if TAP_MQTT_HOST is set, else None (MQTT disabled)."""
    host = os.environ.get("TAP_MQTT_HOST", "").strip()
    if not host:
        return None
    port_s = os.environ.get("TAP_MQTT_PORT", "1883")
    try:
        port = int(port_s)
    except ValueError:
        port = 1883
    batch_s = os.environ.get("TAP_MQTT_BATCH_MS", "100")
    try:
        batch_ms = float(batch_s)
    except ValueError:
        batch_ms = 100.0
    return MqttTelemetryConfig(
        host=host,
        port=port,
        device_id=os.environ.get("TAP_MQTT_DEVICE_ID", "").strip() or socket.gethostname(),
        batch_ms=max(10.0, batch_ms),
        username=os.environ.get("TAP_MQTT_USERNAME") or None,
        password=os.environ.get("TAP_MQTT_PASSWORD") or None,
        client_id=os.environ.get("TAP_MQTT_CLIENT_ID", "").strip(),
    )


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def mqtt_require_sbc() -> bool:
    """
    When True (default), MQTT only enables if RRF reports SBC/DSF mode.

    Override with ``TAP_MQTT_FORCE=1`` (skip probe) or ``TAP_MQTT_REQUIRE_SBC=0``.
    """
    if _env_truthy("TAP_MQTT_FORCE", default=False):
        return False
    return _env_truthy("TAP_MQTT_REQUIRE_SBC", default=True)


def mqtt_sbc_gate_allows() -> tuple[bool, str, dict[str, Any] | None]:
    """
    Gate MQTT for RRF SBC setups.

    Returns:
        (allowed, reason, sbc_object_or_none)
    """
    if not mqtt_require_sbc():
        if _env_truthy("TAP_MQTT_FORCE", default=False):
            return True, "TAP_MQTT_FORCE=1 (SBC probe skipped)", None
        return True, "TAP_MQTT_REQUIRE_SBC=0 (SBC probe skipped)", None
    try:
        from .rrf_http import probe_rrf_sbc_mode
    except ImportError:
        from tap_testing.rrf_http import probe_rrf_sbc_mode  # type: ignore
    is_sbc, sbc, msg = probe_rrf_sbc_mode()
    if is_sbc:
        return True, msg, sbc
    return False, msg, None


def analysis_payload_from_result(
    result: Any,
    session_id: str,
    *,
    source: str = "pi",
) -> dict[str, Any]:
    """Build an analysis MQTT payload from a TapTestResult-like object."""
    return {
        "session_id": session_id,
        "source": source,
        "fn_hz": float(getattr(result, "natural_freq_hz", 0.0)),
        "fn_hz_uncertainty": getattr(result, "natural_freq_hz_uncertainty", None),
        "avoid_rpm": [float(x) for x in getattr(result, "avoid_rpm", []) or []],
        "suggested_rpm_min": float(getattr(result, "suggested_rpm_min", 0.0)),
        "suggested_rpm_max": float(getattr(result, "suggested_rpm_max", 0.0)),
        "n_teeth": int(getattr(result, "n_teeth_used", 0) or 0),
        "sample_rate_hz": float(getattr(result, "sample_rate_hz", 0.0)),
        "ts": time.time(),
    }


@dataclass
class _BatchBuffer:
    t: list[float] = field(default_factory=list)
    ax: list[float] = field(default_factory=list)
    ay: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)


class MqttTelemetryPublisher:
    """
    Non-blocking MQTT publisher. Drops publishes on backpressure / errors so
    accelerometer capture is never stalled.
    """

    def __init__(
        self,
        config: MqttTelemetryConfig,
        client: MqttClientLike | None = None,
        *,
        auto_connect: bool = True,
    ) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._mode: str = "tap"
        self._sample_rate_hz: float = 800.0
        self._job_file: str | None = None
        self._seq = 0
        self._buf = _BatchBuffer()
        self._batch_s = config.batch_ms / 1000.0
        self._connected = False
        self._client: MqttClientLike | None = client
        self._owns_client = client is None
        if auto_connect:
            self.connect()

    @property
    def device_id(self) -> str:
        return self.config.device_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def topic(self, leaf: str) -> str:
        return f"tap/{self.config.device_id}/{leaf}"

    def connect(self) -> bool:
        if self._client is not None and not self._owns_client:
            self._connected = True
            self._publish_status("connected", {"injected": True})
            return True
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError:
            logger.warning(
                "TAP_MQTT_HOST is set but paho-mqtt is not installed; MQTT disabled. "
                "Install with: pip install paho-mqtt"
            )
            self._connected = False
            return False
        try:
            try:
                client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION1,
                    client_id=self.config.client_id,
                )
            except (AttributeError, TypeError):
                client = mqtt.Client(
                    client_id=self.config.client_id,
                    protocol=mqtt.MQTTv311,
                )
            if self.config.username:
                client.username_pw_set(self.config.username, self.config.password)
            client.connect(self.config.host, self.config.port, keepalive=60)
            client.loop_start()
            self._client = client
            self._connected = True
            self._publish_status("connected", {"host": self.config.host, "port": self.config.port})
            return True
        except Exception as e:
            logger.warning("MQTT connect failed: %s", e)
            self._connected = False
            self._client = None
            return False

    def close(self) -> None:
        with self._lock:
            self._flush_batch_unlocked()
        if self._owns_client and self._client is not None:
            try:
                self._publish_status("disconnected", {})
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False

    def _publish(self, leaf: str, payload: dict[str, Any], qos: int = 0) -> None:
        if not self._connected or self._client is None:
            return
        topic = self.topic(leaf)
        try:
            body = json.dumps(payload, separators=(",", ":"))
            self._client.publish(topic, body, qos=qos)
        except Exception as e:
            logger.debug("MQTT publish dropped (%s): %s", leaf, e)

    def _publish_status(self, state: str, detail: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "device_id": self.config.device_id,
            "ts": time.time(),
            **detail,
        }
        if self._session_id and "session_id" not in payload:
            payload["session_id"] = self._session_id
        self._publish("status", payload, qos=1)

    def publish_status(self, state: str, detail: dict[str, Any] | None = None) -> None:
        """Publish a recording/publisher status frame (QoS 1)."""
        self._publish_status(state, detail or {})

    def session_start(
        self,
        mode: str = "tap",
        sample_rate_hz: float = 800.0,
        session_id: str | None = None,
        *,
        job_file: str | None = None,
        tool_number: int | None = None,
        tool_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            self._flush_batch_unlocked()
            self._session_id = session_id or str(uuid.uuid4())
            self._mode = mode
            self._sample_rate_hz = float(sample_rate_hz)
            self._job_file = job_file
            self._seq = 0
            self._buf = _BatchBuffer()
            sid = self._session_id
        payload: dict[str, Any] = {
            "event": "start",
            "event_type": "start_session",
            "session_id": sid,
            "device_id": self.config.device_id,
            "mode": mode,
            "sample_rate_hz": float(sample_rate_hz),
            "ts": time.time(),
        }
        if job_file is not None:
            payload["job_file"] = job_file
        if tool_number is not None:
            payload["tool_number"] = tool_number
        if tool_name is not None:
            payload["tool_name"] = tool_name
        if extra:
            payload.update(extra)
        self._publish("session", payload, qos=1)
        self._publish_status(
            "recording",
            {
                "session_id": sid,
                "mode": mode,
                "sample_rate_hz": float(sample_rate_hz),
                "job_file": job_file,
                "tool_number": tool_number,
            },
        )
        return sid

    def session_stop(
        self,
        *,
        tool_number: int | None = None,
        n_batches: int | None = None,
        n_samples: int | None = None,
        job_file: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._flush_batch_unlocked()
            sid = self._session_id
            mode = self._mode
            sr = self._sample_rate_hz
            batch_seq = self._seq
            cached_job = self._job_file
            self._session_id = None
            self._job_file = None
        if sid is None:
            return
        job = job_file if job_file is not None else cached_job
        payload: dict[str, Any] = {
            "event": "stop",
            "event_type": "end_session",
            "session_id": sid,
            "device_id": self.config.device_id,
            "mode": mode,
            "sample_rate_hz": sr,
            "ts": time.time(),
            "n_batches": int(n_batches if n_batches is not None else batch_seq),
        }
        if tool_number is not None:
            payload["tool_number"] = tool_number
        if n_samples is not None:
            payload["n_samples"] = int(n_samples)
        if job is not None:
            payload["job_file"] = job
        if extra:
            payload.update(extra)
        self._publish("session", payload, qos=1)
        self._publish_status(
            "idle",
            {
                "session_id": sid,
                "mode": mode,
                "n_batches": payload["n_batches"],
                "tool_number": tool_number,
                "job_file": job,
            },
        )

    def publish_tap_detected(self, t_s: float) -> None:
        sid = self._session_id
        if sid is None:
            return
        self._publish(
            "session",
            {
                "event": "tap_detected",
                "session_id": sid,
                "mode": self._mode,
                "sample_rate_hz": self._sample_rate_hz,
                "t_s": float(t_s),
                "ts": time.time(),
            },
            qos=1,
        )

    def emit_sample(self, t_s: float, ax_g: float, ay_g: float, az_g: float) -> None:
        with self._lock:
            if self._session_id is None:
                return
            self._buf.t.append(float(t_s))
            self._buf.ax.append(float(ax_g))
            self._buf.ay.append(float(ay_g))
            self._buf.az.append(float(az_g))
            if len(self._buf.t) >= 2:
                span = self._buf.t[-1] - self._buf.t[0]
            else:
                span = 0.0
            if span >= self._batch_s or (
                self._sample_rate_hz > 0
                and len(self._buf.t) >= max(1, int(round(self._batch_s * self._sample_rate_hz)))
            ):
                self._flush_batch_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._flush_batch_unlocked()

    def _flush_batch_unlocked(self) -> None:
        if not self._buf.t or self._session_id is None:
            self._buf = _BatchBuffer()
            return
        t0 = self._buf.t[0]
        if len(self._buf.t) >= 2:
            dt = (self._buf.t[-1] - self._buf.t[0]) / (len(self._buf.t) - 1)
        elif self._sample_rate_hz > 0:
            dt = 1.0 / self._sample_rate_hz
        else:
            dt = 0.0
        payload = {
            "session_id": self._session_id,
            "seq": self._seq,
            "t0_s": t0,
            "dt_s": dt,
            "ax": list(self._buf.ax),
            "ay": list(self._buf.ay),
            "az": list(self._buf.az),
        }
        self._seq += 1
        self._buf = _BatchBuffer()
        # publish outside holding? still under lock — keep short
        self._publish("accel/batch", payload, qos=0)

    def publish_analysis(self, payload: dict[str, Any]) -> None:
        if "session_id" not in payload and self._session_id:
            payload = {**payload, "session_id": self._session_id}
        if "ts" not in payload:
            payload = {**payload, "ts": time.time()}
        self._publish("analysis", payload, qos=1)

    def publish_modbus_row(self, t_s: float, registers: dict[str, Any]) -> None:
        sid = self._session_id
        if sid is None:
            return
        # Drop empty / non-scalar noise; keep raw register map
        regs = {
            k: v
            for k, v in registers.items()
            if k not in ("t_s", "ts") and v is not None and v != ""
        }
        payload: dict[str, Any] = {
            "session_id": sid,
            "t_s": float(t_s),
            "registers": regs,
        }
        # Wall-clock optional; recording-relative t_s is the correlation key
        if registers.get("ts") is not None:
            try:
                payload["ts"] = float(registers["ts"])
            except (TypeError, ValueError):
                payload["ts"] = time.time()
        else:
            payload["ts"] = time.time()
        self._publish("modbus", payload, qos=0)

    def publish_spindle_sample(self, payload: dict[str, Any]) -> None:
        """Publish a decoded ArborCTL spindle sample on ``tap/{device}/spindle`` (QoS 0)."""
        sid = self._session_id
        if sid is None:
            return
        body = dict(payload)
        body.setdefault("session_id", sid)
        if "ts" not in body:
            body["ts"] = time.time()
        self._publish("spindle", body, qos=0)

    def publish_tool_event(
        self,
        *,
        t_s: float,
        tool_number: int | None,
        previous_tool: int | None = None,
        seq: int | None = None,
        job_file: str | None = None,
        file_position: int | None = None,
        tool_name: str | None = None,
        event: str = "change",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Publish a tool-change / tool-snapshot event on ``tap/{device}/tool``.

        ``t_s`` should use the same time base as ADXL CSV (seconds since recording start).
        """
        sid = self._session_id
        if sid is None:
            return
        payload: dict[str, Any] = {
            "session_id": sid,
            "event": event,
            "t_s": float(t_s),
            "ts": time.time(),
            "tool_number": tool_number,
            "previous_tool": previous_tool,
        }
        if seq is not None:
            payload["seq"] = int(seq)
        if job_file is not None:
            payload["job_file"] = job_file
        if file_position is not None:
            payload["file_position"] = int(file_position)
        if tool_name is not None:
            payload["tool_name"] = tool_name
        if extra:
            payload.update(extra)
        self._publish("tool", payload, qos=1)


def try_create_publisher(
    client: MqttClientLike | None = None,
    *,
    skip_sbc_gate: bool = False,
) -> MqttTelemetryPublisher | None:
    """
    Create a publisher from env, or None if MQTT is disabled / unavailable / gated.

    By default requires RRF SBC mode (``rr_model.sbc`` non-null) unless
    ``TAP_MQTT_FORCE=1`` or ``TAP_MQTT_REQUIRE_SBC=0``. Pass ``skip_sbc_gate=True``
    in unit tests with an injected client.
    """
    cfg = mqtt_config_from_env()
    if cfg is None:
        return None
    sbc_meta: dict[str, Any] | None = None
    if client is not None or skip_sbc_gate:
        # Injected client or explicit skip: no RRF probe (unit tests / forced local).
        pass
    else:
        allowed, reason, sbc_meta = mqtt_sbc_gate_allows()
        if not allowed:
            logger.warning("MQTT disabled (SBC gate): %s", reason)
            return None
        logger.info("MQTT SBC gate passed: %s", reason)
    pub = MqttTelemetryPublisher(cfg, client=client, auto_connect=True)
    if not pub._connected and client is None:
        return None
    if sbc_meta is not None:
        dsf = sbc_meta.get("dsf") if isinstance(sbc_meta.get("dsf"), dict) else {}
        pub._publish_status(
            "sbc",
            {
                "distribution": sbc_meta.get("distribution"),
                "dsf_version": dsf.get("version") if isinstance(dsf, dict) else None,
            },
        )
    return pub
