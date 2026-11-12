"""Structured motion sample builder + publish delta filter (RRF poll → MQTT motion)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_motion_sample(
    *,
    session_id: str,
    device_id: str,
    t_s: float,
    axis_positions_mm: dict[str, float] | None,
    feed_mm_min: float | None = None,
    job_file: str | None = None,
    file_position: int | None = None,
    rrf_status: str | None = None,
    tool_number: int | None = None,
    coord_frame: str = "user",
    source: str = "tap_rrf_poll",
) -> dict[str, Any] | None:
    """Build a motion sample dict; returns None when XYZ cannot be resolved."""
    axes = dict(axis_positions_mm or {})
    x = axes.get("X")
    y = axes.get("Y")
    z = axes.get("Z")
    if x is None or y is None or z is None:
        return None
    body: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "device_id": device_id,
        "t_s": float(t_s),
        "ts": time.time(),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "axis_positions_mm": axes,
        "source": source,
        "coord_frame": coord_frame,
    }
    if "A" in axes:
        body["a"] = float(axes["A"])
    if feed_mm_min is not None:
        body["feed_mm_min"] = float(feed_mm_min)
    if job_file:
        body["job_file"] = job_file
    if file_position is not None:
        body["file_position"] = int(file_position)
    if rrf_status:
        body["rrf_status"] = rrf_status
    if tool_number is not None:
        body["tool_number"] = int(tool_number)
    return body


@dataclass
class MotionPublishFilter:
    """Throttle motion MQTT: delta on XYZ/A, file_position/status change, heartbeat."""

    pos_eps_mm: float = field(default_factory=lambda: _env_float("TAP_MOTION_POS_EPS_MM", 0.05))
    rot_eps_deg: float = field(default_factory=lambda: _env_float("TAP_MOTION_ROT_EPS_DEG", 0.5))
    heartbeat_s: float = field(default_factory=lambda: _env_float("TAP_MOTION_HEARTBEAT_S", 2.0))
    _last: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_publish_mono: float = field(default=0.0, init=False, repr=False)

    def should_publish(self, sample: dict[str, Any], *, now_mono: float | None = None) -> bool:
        now = now_mono if now_mono is not None else time.monotonic()
        if self._last is None:
            self._last = dict(sample)
            self._last_publish_mono = now
            return True
        if (now - self._last_publish_mono) >= self.heartbeat_s:
            self._last = dict(sample)
            self._last_publish_mono = now
            return True
        for key in ("file_position", "rrf_status", "tool_number"):
            if sample.get(key) != self._last.get(key):
                self._last = dict(sample)
                self._last_publish_mono = now
                return True
        for axis, eps in (("x", self.pos_eps_mm), ("y", self.pos_eps_mm), ("z", self.pos_eps_mm)):
            cur = sample.get(axis)
            prev = self._last.get(axis)
            if cur is None or prev is None:
                continue
            if abs(float(cur) - float(prev)) >= eps:
                self._last = dict(sample)
                self._last_publish_mono = now
                return True
        if "a" in sample or "a" in self._last:
            cur_a = sample.get("a")
            prev_a = self._last.get("a")
            if cur_a is None or prev_a is None:
                if cur_a != prev_a:
                    self._last = dict(sample)
                    self._last_publish_mono = now
                    return True
            elif abs(float(cur_a) - float(prev_a)) >= self.rot_eps_deg:
                self._last = dict(sample)
                self._last_publish_mono = now
                return True
        return False
