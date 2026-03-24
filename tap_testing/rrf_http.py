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

    def send_gcode(self, line: str) -> dict[str, Any]:
        """Queue G/M/T-code via ``/rr_gcode`` (RRF OpenAPI). May move axes or run the spindle; use with care.

        On RRF, ``M115`` reports firmware info and ``M122`` runs diagnostics (neither moves axes).
        """
        data = self._request_json("/rr_gcode", {"gcode": line})
        if not isinstance(data, dict):
            raise RrfHttpError("rr_gcode: expected JSON object")
        return data


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
