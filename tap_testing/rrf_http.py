"""
HTTP client for RepRapFirmware REST API (Duet boards), per Duet developer OpenAPI:

https://github.com/Duet3D/RepRapFirmware/blob/dev/Developer-documentation/OpenAPI.yaml

Uses stdlib only (urllib). Typical flow: ``rr_connect`` (session cookie), then ``rr_model``
for object-model keys (same idea as M409).

Live tests in ``tests/test_rrf_http.py`` use only ``rr_connect`` and ``rr_model`` (diagnostic
reads); they do not call ``rr_gcode`` or other endpoints that could move axes or the spindle.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from typing import Any

logger = logging.getLogger(__name__)

# Default hosts to try when discovering a Duet on the LAN (mDNS).
_DEFAULT_DISCOVERY_HOSTS = (
    "milo.local",
    "milo",
    "duet3.local",
    "duet2ethernet.local",
    "duet2wifi.local",
    "duet.local",
)

# RRF state.status values that mean a file job is still in play (see Duet object model docs).
_JOB_ACTIVE_STATUSES = frozenset(
    {
        "processing",
        "simulating",
        "paused",
        "pausing",
        "resuming",
        "cancelling",
    }
)


def rrf_default_poll_interval_s() -> float:
    return float(os.environ.get("TAP_RRF_POLL_S", "1.0"))


def rrf_default_base_url() -> str:
    return os.environ.get("TAP_RRF_BASE", "http://milo.local").strip()


def rrf_discovery_hosts() -> tuple[str, ...]:
    raw = os.environ.get("TAP_RRF_DISCOVER_HOSTS", "").strip()
    if not raw:
        return _DEFAULT_DISCOVERY_HOSTS
    return tuple(h.strip() for h in raw.split(",") if h.strip())


def infer_print_job_active(state: dict[str, Any] | None, job: dict[str, Any] | None) -> bool:
    """True if a print/simulate job is in progress (not idle / generic busy)."""
    if not state:
        return False
    st = state.get("status")
    if not isinstance(st, str):
        return False
    if st in _JOB_ACTIVE_STATUSES:
        return True
    # Brief transition into a file job; avoid treating config.g "starting" as a print.
    if st == "starting" and _job_has_file(job):
        return True
    return False


def is_sbc_mode(sbc: Any) -> bool:
    """
    True when RRF object model reports SBC / DSF mode.

    In Duet Web Control and the object model, ``sbc`` is non-null only when the
    board is paired with an SBC running Duet Software Framework (not standalone
    WiFi/Ethernet Duet HTTP).
    """
    return isinstance(sbc, dict) and bool(sbc)


def _job_has_file(job: dict[str, Any] | None) -> bool:
    if not job:
        return False
    f = job.get("file")
    if isinstance(f, dict):
        fn = f.get("fileName")
        if isinstance(fn, str) and fn.strip():
            return True
    fn2 = job.get("fileName")
    return isinstance(fn2, str) and fn2.strip() != ""


class RrfHttpError(Exception):
    pass


class RrfClient:
    """Minimal RRF HTTP client: session via rr_connect, queries via rr_model."""

    def __init__(
        self,
        base_url: str,
        *,
        password: str = "",
        timeout_s: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.timeout_s = timeout_s
        self._cookie_jar = CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._cookie_jar))

    def _request_json(self, path: str, query: dict[str, str]) -> Any:
        qs = urllib.parse.urlencode(query)
        url = f"{self.base_url}{path}?{qs}"
        req = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RrfHttpError(f"HTTP {e.code} for {path}") from e
        except urllib.error.URLError as e:
            raise RrfHttpError(str(e.reason if hasattr(e, "reason") else e)) from e
        except socket.timeout as e:
            raise RrfHttpError("timeout") from e
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RrfHttpError(f"invalid JSON from {path}") from e

    def connect(self) -> dict[str, Any]:
        """Call rr_connect; establishes session cookie when firmware requires it."""
        data = self._request_json("/rr_connect", {"password": self.password})
        if not isinstance(data, dict):
            raise RrfHttpError("rr_connect: expected JSON object")
        err = data.get("err")
        if err not in (0, None):
            if err == 1:
                raise RrfHttpError("rr_connect: invalid password")
            if err == 2:
                raise RrfHttpError("rr_connect: no more sessions")
            raise RrfHttpError(f"rr_connect: err={err}")
        return data

    def model(self, key: str, flags: str = "v") -> Any:
        """
        GET rr_model. Returns the ``result`` field (object-model subtree), or the whole
        payload if ``result`` is missing (defensive).
        """
        data = self._request_json("/rr_model", {"key": key, "flags": flags})
        if not isinstance(data, dict):
            raise RrfHttpError("rr_model: expected JSON object")
        if "result" in data:
            return data["result"]
        return data

    def fetch_state_and_job(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        state_raw = self.model("state", "v")
        job_raw = self.model("job", "v")
        state = state_raw if isinstance(state_raw, dict) else None
        job = job_raw if isinstance(job_raw, dict) else None
        return state, job

    def fetch_tools(self) -> list[dict[str, Any]]:
        """Return the ``tools`` object-model array (may be empty)."""
        raw = self.model("tools", "v")
        if isinstance(raw, list):
            return [t for t in raw if isinstance(t, dict)]
        return []

    def fetch_axis_letters(self) -> list[str]:
        """
        Return move.axes letter order (e.g. ``["X","Y","Z"]``).

        RRF ``tools[n].offsets[]`` is indexed in this same order.
        """
        raw = self.model("move.axes", "v")
        if not isinstance(raw, list):
            return []
        letters: list[str] = []
        for ax in raw:
            if not isinstance(ax, dict):
                continue
            letter = ax.get("letter")
            if isinstance(letter, str) and letter.strip():
                letters.append(letter.strip().upper())
            else:
                letters.append(f"A{len(letters)}")
        return letters

    def fetch_sbc(self) -> dict[str, Any] | None:
        """Return the ``sbc`` object-model subtree, or None when not in SBC mode / missing."""
        raw = self.model("sbc", "v")
        if isinstance(raw, dict):
            return raw
        return None

    def fetch_globals(self) -> dict[str, Any]:
        """Return user globals (``rr_model`` key ``global``) as a name → value dict."""
        raw = self.model("global", "v")
        return normalize_rrf_globals(raw)

    def fetch_spindles(self) -> list[dict[str, Any]]:
        """Return the ``spindles`` object-model array (may be empty)."""
        raw = self.model("spindles", "v")
        if isinstance(raw, list):
            return [s if isinstance(s, dict) else {} for s in raw]
        return []

    def send_gcode(self, line: str) -> dict[str, Any]:
        """Queue G/M/T-code via ``/rr_gcode`` (RRF OpenAPI). May move axes or run the spindle; use with care.

        On RRF, ``M115`` reports firmware info and ``M122`` runs diagnostics (neither moves axes).
        """
        data = self._request_json("/rr_gcode", {"gcode": line})
        if not isinstance(data, dict):
            raise RrfHttpError("rr_gcode: expected JSON object")
        return data

    def download_file(self, name: str, *, max_bytes: int = 8 * 1024 * 1024) -> bytes:
        """
        Download a file via ``GET /rr_download?name=…`` (read-only).

        Used to hash/summarize the running G-code for Jarvis CAM resolution.
        Bytes are not persisted by callers — compute digest/summary then discard.
        """
        if not name or not str(name).strip():
            raise RrfHttpError("rr_download: empty name")
        qs = urllib.parse.urlencode({"name": str(name)})
        url = f"{self.base_url}/rr_download?{qs}"
        req = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                chunks: list[bytes] = []
                total = 0
                while True:
                    block = resp.read(64 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        raise RrfHttpError(f"rr_download: file exceeds {max_bytes} bytes")
                    chunks.append(block)
                return b"".join(chunks)
        except urllib.error.HTTPError as e:
            raise RrfHttpError(f"HTTP {e.code} for /rr_download") from e
        except urllib.error.URLError as e:
            raise RrfHttpError(str(e.reason if hasattr(e, "reason") else e)) from e
        except socket.timeout as e:
            raise RrfHttpError("timeout") from e


def parse_current_tool(state: dict[str, Any] | None) -> int | None:
    """
    Selected tool number from ``state.currentTool``.

    RRF uses ``-1`` when no tool is selected. Returns ``None`` if missing/invalid.
    """
    if not state:
        return None
    raw = state.get("currentTool")
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n


def parse_job_file_name(job: dict[str, Any] | None) -> str | None:
    """Best-effort G-code filename from ``job.file.fileName`` or ``job.fileName``."""
    if not job:
        return None
    f = job.get("file")
    if isinstance(f, dict):
        fn = f.get("fileName")
        if isinstance(fn, str) and fn.strip():
            return fn.strip()
    fn2 = job.get("fileName")
    if isinstance(fn2, str) and fn2.strip():
        return fn2.strip()
    return None


def parse_job_file_position(job: dict[str, Any] | None) -> int | None:
    """Byte offset into the running G-code file (``job.filePosition``), if present."""
    if not job:
        return None
    raw = job.get("filePosition")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def summarize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Compact RRF tool row for snapshots / MQTT (Fusion GUIDs are not in RRF)."""
    out: dict[str, Any] = {}
    for key in (
        "number",
        "name",
        "state",
        "spindle",
        "spindleRpm",
        "offsets",
        "offsetsProbed",
        "axes",
        "extruders",
        "heaters",
        "fans",
        "active",
        "standby",
    ):
        if key in tool:
            out[key] = tool[key]
    return out


def summarize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize_tool(t) for t in tools]


def named_offsets_mm(
    offsets: Any,
    axis_letters: list[str] | None = None,
) -> dict[str, float]:
    """Map RRF offsets[] to axis-letter keys using move.axes order."""
    if not isinstance(offsets, list):
        return {}
    letters = list(axis_letters or [])
    out: dict[str, float] = {}
    for i, raw in enumerate(offsets):
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        key = letters[i] if i < len(letters) else ("XYZ"[i] if i < 3 else f"A{i}")
        out[str(key).upper()] = val
    return out


def normalize_tool_snapshot(
    tool: dict[str, Any] | None,
    *,
    axis_letters: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Normalize one RRF tool into a stable nested snapshot for MQTT / JSONL.

    Includes named axis offsets (mm). Fusion GUID is resolved later in Jarvis.
    """
    if not isinstance(tool, dict):
        return None
    compact = summarize_tool(tool)
    number = compact.get("number")
    try:
        number_i = int(number) if number is not None else None
    except (TypeError, ValueError):
        number_i = None
    offsets = compact.get("offsets")
    named = named_offsets_mm(offsets, axis_letters)
    offsets_probed = compact.get("offsetsProbed")
    try:
        offsets_probed_i = int(offsets_probed) if offsets_probed is not None else None
    except (TypeError, ValueError):
        offsets_probed_i = None
    name = compact.get("name")
    return {
        "number": number_i,
        "name": str(name).strip() if isinstance(name, str) and name.strip() else None,
        "state": compact.get("state"),
        "spindle": compact.get("spindle"),
        "spindle_rpm": compact.get("spindleRpm"),
        "axes": compact.get("axes"),
        "axis_letters": list(axis_letters or []),
        "offsets_mm": named,
        "offsets": list(offsets) if isinstance(offsets, list) else offsets,
        "offsets_probed": offsets_probed_i,
        "extruders": compact.get("extruders"),
        "heaters": compact.get("heaters"),
        "fans": compact.get("fans"),
        "active": compact.get("active"),
        "standby": compact.get("standby"),
        "rrf": {
            "model_key": "tools",
            "flags": "v",
            "raw_tool": compact,
        },
    }


def tool_snapshot_by_number(
    tools: list[dict[str, Any]] | None,
    number: int | None,
    *,
    axis_letters: list[str] | None = None,
) -> dict[str, Any] | None:
    if number is None or not tools:
        return None
    for t in tools:
        try:
            if int(t.get("number")) == int(number):
                return normalize_tool_snapshot(t, axis_letters=axis_letters)
        except (TypeError, ValueError):
            continue
    return None


def offsets_equal(a: dict[str, float] | None, b: dict[str, float] | None) -> bool:
    aa = a or {}
    bb = b or {}
    if set(aa) != set(bb):
        return False
    for k, v in aa.items():
        if abs(float(v) - float(bb[k])) > 1e-6:
            return False
    return True


def normalize_rrf_globals(raw: Any) -> dict[str, Any]:
    """
    Normalize ``rr_model`` ``global`` into a name → value dict.

    Handles a JSON object, or a list of ``{name, value}`` pairs used by some DWC builds.
    """
    if isinstance(raw, dict):
        if "name" in raw and "value" in raw and len(raw) <= 3:
            name = raw.get("name")
            return {str(name): raw.get("value")} if name else {}
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("key")
            if name is None:
                continue
            out[str(name)] = item.get("value") if "value" in item else item.get("val")
        return out
    return {}


def offset_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict[str, float]:
    aa = before or {}
    bb = after or {}
    keys = set(aa) | set(bb)
    out: dict[str, float] = {}
    for k in sorted(keys):
        d = float(bb.get(k, 0.0)) - float(aa.get(k, 0.0))
        if abs(d) > 1e-6:
            out[k] = d
    return out

def probe_rrf_base(base_url: str, *, password: str = "", timeout_s: float = 3.0) -> str:
    """
    Verify host speaks RRF HTTP API. Returns human-readable status line on success.
    Raises RrfHttpError on failure.
    """
    client = RrfClient(base_url, password=password, timeout_s=timeout_s)
    client.connect()
    state, _ = client.fetch_state_and_job()
    st = (state or {}).get("status", "?")
    return f"OK — state.status={st!r}"


def probe_rrf_sbc_mode(
    base_url: str | None = None,
    *,
    password: str | None = None,
    timeout_s: float = 3.0,
) -> tuple[bool, dict[str, Any] | None, str]:
    """
    Detect RRF SBC / DSF mode via ``rr_model?key=sbc``.

    Returns:
        (is_sbc, sbc_object_or_none, human_message)

    On HTTP/connect failure returns ``(False, None, reason)`` rather than raising,
    so MQTT gating can fail closed without aborting recording.
    """
    base = (base_url or rrf_default_base_url()).strip()
    pw = password if password is not None else os.environ.get("TAP_RRF_PASSWORD", "")
    try:
        client = RrfClient(base, password=pw, timeout_s=timeout_s)
        client.connect()
        sbc = client.fetch_sbc()
    except RrfHttpError as e:
        return False, None, f"RRF unreachable at {base}: {e}"
    except Exception as e:
        return False, None, f"RRF SBC probe failed at {base}: {e}"
    if is_sbc_mode(sbc):
        dsf = (sbc or {}).get("dsf") if isinstance(sbc, dict) else None
        dsf_ver = dsf.get("version") if isinstance(dsf, dict) else None
        distro = (sbc or {}).get("distribution") if isinstance(sbc, dict) else None
        bits = []
        if distro:
            bits.append(str(distro))
        if dsf_ver:
            bits.append(f"DSF {dsf_ver}")
        detail = ", ".join(bits) if bits else "sbc object present"
        return True, sbc, f"SBC mode active ({detail}) via {base}"
    return False, None, f"Standalone / non-SBC mode at {base} (rr_model.sbc is null)"


def discover_rrf_base(
    *,
    password: str = "",
    timeout_s: float = 2.0,
    hosts: tuple[str, ...] | None = None,
    scheme: str = "http",
    port: int | None = None,
) -> str | None:
    """
    Try common Duet hostnames (mDNS). Returns first base URL that responds to rr_connect
    + rr_model state, or None.
    """
    host_list = hosts if hosts is not None else rrf_discovery_hosts()
    for host in host_list:
        base = f"{scheme}://{host}"
        if port is not None and port not in (80, 443):
            base = f"{scheme}://{host}:{port}"
        try:
            probe_rrf_base(base, password=password, timeout_s=timeout_s)
            logger.info("RRF discovery: using %s", base)
            return base
        except RrfHttpError as e:
            logger.debug("RRF discovery skip %s: %s", base, e)
        except Exception as e:
            logger.debug("RRF discovery skip %s: %s", base, e)
    return None
