"""
Tool-change timeline for live spindle / RRF recordings.

Persists ``tool-events.jsonl`` + ``tools-snapshot.json`` beside ``homing.csv``,
and optionally publishes MQTT ``tap/{device}/tool`` events so Jarvis can segment
ADXL data by active tool and retain offset history.

Time base: ``t_s`` is seconds since recording start (monotonic), matching ADXL CSV.
RRF ``state.currentTool`` is the machine slot number (−1 = none). On this machine
RRF ``Tn`` maps one-to-one to Fusion ``post-process.number``; Jarvis resolves the
Fusion GUID from that number.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .rrf_http import (
    normalize_tool_snapshot,
    offset_delta as compute_offset_delta,
    offsets_equal,
    parse_current_tool,
    parse_job_file_name,
    parse_job_file_position,
    summarize_tools,
)

logger = logging.getLogger(__name__)

TOOL_SCHEMA_VERSION = 2


class _MqttToolPublisher(Protocol):
    def publish_tool_event(self, **kwargs: Any) -> None: ...


@dataclass
class ToolEventRecorder:
    """
    Record RRF tool changes / offset updates for one recording session.

    Call :meth:`begin` when ADXL capture starts, :meth:`observe` on each RRF poll,
    :meth:`refresh_tools` when the tool table is re-fetched, and :meth:`end` when
    capture stops.
    """

    run_dir: Path
    session_id: str
    mqtt: _MqttToolPublisher | None = None
    events_filename: str = "tool-events.jsonl"
    snapshot_filename: str = "tools-snapshot.json"
    meta_filename: str = "run-meta.json"
    axis_letters: list[str] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _t0_mono: float | None = field(default=None, init=False)
    _active: bool = field(default=False, init=False)
    _last_tool: int | None = field(default=None, init=False)
    _seq: int = field(default=0, init=False)
    _job_file: str | None = field(default=None, init=False)
    _tool_names: dict[int, str] = field(default_factory=dict, init=False)
    _tools_by_number: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _last_snapshots: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _events_path: Path | None = field(default=None, init=False)

    def set_recording_origin(self, t0_mono: float) -> None:
        """Align tool-event ``t_s`` with the ADXL recording monotonic origin."""
        with self._lock:
            self._t0_mono = float(t0_mono)

    def begin(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
        job: dict[str, Any] | None = None,
        sample_rate_hz: float | None = None,
        axis_letters: list[str] | None = None,
        extra_meta: dict[str, Any] | None = None,
        recording_t0_mono: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Start a recording timeline. Writes tools snapshot + run meta, emits initial tool event.

        Returns the initial tool event payload (or None if already active).
        """
        with self._lock:
            if self._active:
                return None
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._t0_mono = float(recording_t0_mono) if recording_t0_mono is not None else time.monotonic()
            self._active = True
            self._seq = 0
            self._last_tool = None
            self._events_path = self.run_dir / self.events_filename
            self._events_path.write_text("", encoding="utf-8")
            if axis_letters:
                self.axis_letters = [str(a).upper() for a in axis_letters]

            self._ingest_tools_unlocked(tools or [])
            self._job_file = parse_job_file_name(job)
            self._write_snapshot_unlocked(source="rrf_tools_start")
            meta: dict[str, Any] = {
                "session_id": self.session_id,
                "started_at": time.time(),
                "job_file": self._job_file,
                "sample_rate_hz": sample_rate_hz,
                "initial_tool": parse_current_tool(state),
                "axis_letters": list(self.axis_letters),
                "schema_version": TOOL_SCHEMA_VERSION,
                "time_basis": "recording_monotonic",
            }
            if extra_meta:
                meta.update(extra_meta)
            (self.run_dir / self.meta_filename).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            tool_n = parse_current_tool(state)
            file_pos = parse_job_file_position(job)
            # Table snapshot event (session start)
            self._emit_unlocked(
                tool_number=tool_n,
                previous_tool=None,
                file_position=file_pos,
                event="start",
                event_type="tool_table_snapshot",
                include_table=True,
            )
            # Selection event
            return self._emit_unlocked(
                tool_number=tool_n,
                previous_tool=None,
                file_position=file_pos,
                event="start",
                event_type="tool_selected" if tool_n is not None and tool_n >= 0 else "tool_deselected",
            )

    def observe(
        self,
        *,
        state: dict[str, Any] | None,
        job: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Emit change/offset events.

        When ``tools`` is provided, diffs the tool table for offset-only changes
        even if ``state.currentTool`` is unchanged.
        """
        with self._lock:
            if not self._active:
                return []
            emitted: list[dict[str, Any]] = []
            fn = parse_job_file_name(job)
            if fn:
                self._job_file = fn
            file_pos = parse_job_file_position(job)
            tool_n = parse_current_tool(state)

            if tools is not None:
                emitted.extend(self._diff_tools_unlocked(tools, file_position=file_pos))

            if tool_n != self._last_tool:
                previous = self._last_tool
                event_type = (
                    "tool_deselected"
                    if tool_n is None or tool_n < 0
                    else "tool_selected"
                )
                payload = self._emit_unlocked(
                    tool_number=tool_n,
                    previous_tool=previous,
                    file_position=file_pos,
                    event="change",
                    event_type=event_type,
                )
                emitted.append(payload)
            return emitted

    def refresh_tools(
        self,
        tools: list[dict[str, Any]],
        *,
        file_position: int | None = None,
        force_snapshot: bool = False,
    ) -> list[dict[str, Any]]:
        """Refresh cached tool table and emit offset-change events when needed."""
        with self._lock:
            if not self._active:
                return []
            emitted = self._diff_tools_unlocked(tools, file_position=file_position)
            if force_snapshot:
                self._write_snapshot_unlocked(source="rrf_tools_refresh")
            return emitted

    def end(self) -> dict[str, Any] | None:
        """Mark recording end (optional final event)."""
        with self._lock:
            if not self._active:
                return None
            self._write_snapshot_unlocked(source="rrf_tools_end")
            payload = self._emit_unlocked(
                tool_number=self._last_tool,
                previous_tool=self._last_tool,
                file_position=None,
                event="stop",
                event_type="tool_table_snapshot",
                include_table=True,
                force=True,
            )
            self._active = False
            # Keep _t0_mono until after final emit; clear now
            self._t0_mono = None
            return payload

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def last_tool(self) -> int | None:
        with self._lock:
            return self._last_tool

    def elapsed_s(self) -> float:
        with self._lock:
            if self._t0_mono is None:
                return 0.0
            return max(0.0, time.monotonic() - self._t0_mono)

    def _ingest_tools_unlocked(self, tools: list[dict[str, Any]]) -> None:
        summarized = summarize_tools(tools)
        self._tool_names = {}
        self._tools_by_number = {}
        self._last_snapshots = {}
        for t in summarized:
            num = t.get("number")
            try:
                num_i = int(num) if num is not None else None
            except (TypeError, ValueError):
                num_i = None
            if num_i is None:
                continue
            self._tools_by_number[num_i] = t
            name = t.get("name")
            if isinstance(name, str) and name.strip():
                self._tool_names[num_i] = name.strip()
            snap = normalize_tool_snapshot(t, axis_letters=self.axis_letters)
            if snap:
                self._last_snapshots[num_i] = snap

    def _write_snapshot_unlocked(self, *, source: str) -> None:
        tools = [
            normalize_tool_snapshot(t, axis_letters=self.axis_letters)
            for t in self._tools_by_number.values()
        ]
        tools = [t for t in tools if t is not None]
        tools.sort(key=lambda x: (x.get("number") is None, x.get("number") or 0))
        snapshot = {
            "schema_version": TOOL_SCHEMA_VERSION,
            "session_id": self.session_id,
            "ts": time.time(),
            "source": source,
            "job_file": self._job_file,
            "axis_letters": list(self.axis_letters),
            "rrf_model_key": "tools",
            "rrf_flags": "v",
            "tools": tools,
            # Compact legacy array for older importers
            "tools_compact": list(self._tools_by_number.values()),
        }
        (self.run_dir / self.snapshot_filename).write_text(
            json.dumps(snapshot, indent=2),
            encoding="utf-8",
        )

    def _diff_tools_unlocked(
        self,
        tools: list[dict[str, Any]],
        *,
        file_position: int | None,
    ) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        previous_snaps = dict(self._last_snapshots)
        self._ingest_tools_unlocked(tools)
        # Detect offset changes for tools that existed before
        for num, after in self._last_snapshots.items():
            before = previous_snaps.get(num)
            if before is None:
                continue
            before_off = before.get("offsets_mm") or {}
            after_off = after.get("offsets_mm") or {}
            if offsets_equal(before_off, after_off):
                # Also treat offsetsProbed changes as noteworthy
                if before.get("offsets_probed") == after.get("offsets_probed"):
                    continue
            delta = compute_offset_delta(before_off, after_off)
            # Emit against the changed tool number; keep selection event separate
            payload = self._emit_unlocked(
                tool_number=num,
                previous_tool=self._last_tool,
                file_position=file_position,
                event="change",
                event_type="tool_offset_changed",
                tool_snapshot=after,
                previous_tool_snapshot=before,
                offset_before=before_off,
                offset_after=after_off,
                offsets_delta=delta,
                bump_seq=True,
                update_last_tool=False,
            )
            emitted.append(payload)
        return emitted

    def _emit_unlocked(
        self,
        *,
        tool_number: int | None,
        previous_tool: int | None,
        file_position: int | None,
        event: str,
        event_type: str,
        tool_snapshot: dict[str, Any] | None = None,
        previous_tool_snapshot: dict[str, Any] | None = None,
        offset_before: dict[str, float] | None = None,
        offset_after: dict[str, float] | None = None,
        offsets_delta: dict[str, float] | None = None,
        include_table: bool = False,
        force: bool = False,
        bump_seq: bool = False,
        update_last_tool: bool = True,
    ) -> dict[str, Any]:
        t_s = 0.0 if self._t0_mono is None else max(0.0, time.monotonic() - self._t0_mono)
        if bump_seq or (not force and event == "change"):
            self._seq += 1
            seq = self._seq
        elif event == "start" and event_type == "tool_table_snapshot":
            self._seq = 0
            seq = 0
        elif event == "start":
            seq = self._seq
        else:
            seq = self._seq

        name = None
        if isinstance(tool_number, int):
            name = self._tool_names.get(tool_number)
        if tool_snapshot is None and isinstance(tool_number, int):
            tool_snapshot = self._last_snapshots.get(tool_number)
        if previous_tool_snapshot is None and isinstance(previous_tool, int):
            previous_tool_snapshot = self._last_snapshots.get(previous_tool)

        event_id = f"tevt-{uuid.uuid4().hex[:12]}"
        payload: dict[str, Any] = {
            "schema_version": TOOL_SCHEMA_VERSION,
            "event_id": event_id,
            "session_id": self.session_id,
            "event": event,
            "event_type": event_type,
            "seq": seq,
            "t_s": t_s,
            "ts": time.time(),
            "time_basis": "recording_monotonic",
            "tool_number": tool_number,
            "previous_tool": previous_tool,
            "job_file": self._job_file,
            "file_position": file_position,
            "tool_name": name,
            "rrf_slot": tool_number,
            "tool_snapshot": tool_snapshot,
            "previous_tool_snapshot": previous_tool_snapshot,
        }
        if offset_before is not None or offset_after is not None:
            payload["offsets"] = {
                "kind": "rrf_tool_axis",
                "unit": "mm",
                "axis_letters": list(self.axis_letters),
                "before": offset_before or {},
                "after": offset_after or {},
                "delta": offsets_delta or {},
                "offsets_probed": (tool_snapshot or {}).get("offsets_probed"),
            }
        elif tool_snapshot and tool_snapshot.get("offsets_mm"):
            payload["offsets"] = {
                "kind": "rrf_tool_axis",
                "unit": "mm",
                "axis_letters": list(self.axis_letters),
                "axes": dict(tool_snapshot.get("offsets_mm") or {}),
                "offsets_probed": tool_snapshot.get("offsets_probed"),
            }
        if include_table:
            payload["tools"] = [
                normalize_tool_snapshot(t, axis_letters=self.axis_letters)
                for t in self._tools_by_number.values()
            ]

        if update_last_tool and event != "stop" and event_type != "tool_offset_changed":
            self._last_tool = tool_number
        elif event != "stop" and event_type in ("tool_selected", "tool_deselected"):
            self._last_tool = tool_number

        self._append_jsonl(payload)
        if self.mqtt is not None:
            try:
                extra = {
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "session_id",
                        "event",
                        "t_s",
                        "ts",
                        "tool_number",
                        "previous_tool",
                        "seq",
                        "job_file",
                        "file_position",
                        "tool_name",
                    }
                }
                self.mqtt.publish_tool_event(
                    t_s=t_s,
                    tool_number=tool_number,
                    previous_tool=previous_tool,
                    seq=seq,
                    job_file=self._job_file,
                    file_position=file_position,
                    tool_name=name,
                    event=event,
                    extra=extra,
                )
            except Exception as e:
                logger.debug("MQTT tool publish failed: %s", e)
        return payload

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        if self._events_path is None:
            return
        try:
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError as e:
            logger.warning("Failed to write tool event: %s", e)
