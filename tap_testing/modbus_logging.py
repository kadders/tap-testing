"""
Optional Modbus polling for live recordings (e.g. homing GUI).

Reads holding registers, input registers, and optionally discrete inputs and coils
in configurable address ranges (chunked) and writes wide CSV rows. Register meaning
and scaling are device-specific — values are logged as unsigned 16-bit unless the map
uses two-register floats (decode externally).

Requires: pip install pymodbus.

Default transport is **RTU** (RS-485 VFD); use ``TAP_MODBUS_TRANSPORT=tcp`` for Ethernet.
H100 preset: ``TAP_MODBUS_PROFILE=h100`` or ``apply_modbus_profile(cfg, "h100")`` — see docs/MODBUS_H100_VFD.md.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Modbus limits: many devices allow up to 125 registers per read; we default to 100.
_DEFAULT_CHUNK = 100


def modbus_logging_config_from_env() -> ModbusLoggingConfig:
    """Build ModbusLoggingConfig from environment (TAP_MODBUS_*)."""
    cfg = ModbusLoggingConfig()

    prof = os.environ.get("TAP_MODBUS_PROFILE", "").strip().lower()
    if prof == "h100":
        apply_modbus_profile(cfg, "h100")
    elif prof:
        logger.warning("Unknown TAP_MODBUS_PROFILE %r — ignoring (supported: h100)", prof)

    t = os.environ.get("TAP_MODBUS_TRANSPORT", "").strip().lower()
    if t in ("tcp", "rtu"):
        cfg.transport = t

    host = os.environ.get("TAP_MODBUS_HOST")
    if host:
        cfg.host = host.strip()

    port_e = os.environ.get("TAP_MODBUS_PORT")
    if port_e:
        try:
            cfg.port = int(port_e)
        except ValueError:
            pass

    unit_e = os.environ.get("TAP_MODBUS_UNIT")
    if unit_e:
        try:
            cfg.unit_id = int(unit_e)
        except ValueError:
            pass

    sp = os.environ.get("TAP_MODBUS_SERIAL_PORT")
    if sp:
        cfg.serial_port = sp.strip()

    br = os.environ.get("TAP_MODBUS_BAUDRATE")
    if br:
        try:
            cfg.baudrate = int(br)
        except ValueError:
            pass

    par = os.environ.get("TAP_MODBUS_PARITY")
    if par and len(par) == 1:
        cfg.parity = par.upper()

    sb = os.environ.get("TAP_MODBUS_STOPBITS")
    if sb:
        try:
            cfg.stopbits = int(sb)
        except ValueError:
            pass

    bs = os.environ.get("TAP_MODBUS_BYTESIZE")
    if bs:
        try:
            cfg.bytesize = int(bs)
        except ValueError:
            pass

    ph = os.environ.get("TAP_MODBUS_POLL_HZ")
    if ph:
        try:
            cfg.poll_hz = float(ph)
        except ValueError:
            pass

    mh = os.environ.get("TAP_MODBUS_MAX_HOLDING")
    if mh:
        try:
            cfg.max_holding = max(1, min(1000, int(mh)))
        except ValueError:
            pass

    mi = os.environ.get("TAP_MODBUS_MAX_INPUT")
    if mi:
        try:
            cfg.max_input = max(1, min(1000, int(mi)))
        except ValueError:
            pass

    ch = os.environ.get("TAP_MODBUS_CHUNK")
    if ch:
        try:
            cfg.chunk_size = max(1, min(125, int(ch)))
        except ValueError:
            pass

    md = os.environ.get("TAP_MODBUS_MAX_DISCRETE")
    if md:
        try:
            cfg.max_discrete = max(0, min(2000, int(md)))
        except ValueError:
            pass

    mc = os.environ.get("TAP_MODBUS_MAX_COILS")
    if mc:
        try:
            cfg.max_coils = max(0, min(2000, int(mc)))
        except ValueError:
            pass

    if os.environ.get("TAP_MODBUS_NO_DISCRETE", "").strip().lower() in ("1", "true", "yes"):
        cfg.include_discrete = False
    if os.environ.get("TAP_MODBUS_NO_COILS", "").strip().lower() in ("1", "true", "yes"):
        cfg.include_coils = False

    return cfg


def apply_modbus_profile(cfg: ModbusLoggingConfig, profile: str) -> None:
    """
    Apply a named device preset (register span, typical RTU baud).

    Profiles: ``h100`` — H100-class VFD on RS-485 (see docs/MODBUS_H100_VFD.md).
    """
    name = profile.strip().lower()
    if name == "h100":
        cfg.profile = "h100"
        cfg.transport = "rtu"
        cfg.baudrate = 19200
        cfg.bytesize = 8
        cfg.parity = "N"
        cfg.stopbits = 1
        # Holding 0x0201 (513): setpoint / command frequency; 5, 11: max/min; coils ~73–75: run/stop
        cfg.max_holding = max(cfg.max_holding, 520)
        cfg.max_input = max(cfg.max_input, 8)
        cfg.max_coils = max(cfg.max_coils, 96)
        return
    raise ValueError(f"Unknown Modbus profile: {profile!r} (supported: h100)")


@dataclass
class ModbusLoggingConfig:
    """Polling configuration (TCP or RTU)."""

    # Optional label written to modbus.csv and used by the homing GUI (e.g. h100).
    profile: str = ""

    # Default RTU: spindle VFD on USB–RS485 (set TAP_MODBUS_TRANSPORT=tcp for Ethernet gateways).
    transport: str = "rtu"  # "tcp" | "rtu"
    host: str = "127.0.0.1"
    port: int = 502
    unit_id: int = 1
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 19200
    parity: str = "N"  # N E O M S
    stopbits: int = 1
    bytesize: int = 8
    poll_hz: float = 10.0
    max_holding: int = 400
    max_input: int = 400
    max_discrete: int = 128
    max_coils: int = 128
    chunk_size: int = _DEFAULT_CHUNK
    include_discrete: bool = True
    include_coils: bool = True


def _chunks(start: int, end_exclusive: int, chunk: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = start
    while pos < end_exclusive:
        n = min(chunk, end_exclusive - pos)
        out.append((pos, n))
        pos += n
    return out


def build_fieldnames(cfg: ModbusLoggingConfig) -> list[str]:
    """CSV column order: t_s, hr_*, ir_*, di_*, co_*."""
    names = ["t_s"]
    names.extend(f"hr_{i}" for i in range(cfg.max_holding))
    names.extend(f"ir_{i}" for i in range(cfg.max_input))
    if cfg.include_discrete:
        names.extend(f"di_{i}" for i in range(cfg.max_discrete))
    if cfg.include_coils:
        names.extend(f"co_{i}" for i in range(cfg.max_coils))
    return names


def poll_registers_once(
    client: Any,
    cfg: ModbusLoggingConfig,
) -> dict[str, Any]:
    """
    Perform one poll cycle. Returns flat dict with t_s and register columns.
    Missing/failed reads leave keys absent (caller may treat as empty when writing CSV).
    """
    t_s = time.time()
    row: dict[str, Any] = {"t_s": t_s}

    for addr, count in _chunks(0, cfg.max_holding, cfg.chunk_size):
        try:
            resp = client.read_holding_registers(address=addr, count=count, slave=cfg.unit_id)
            if resp.isError():
                logger.debug("holding read error at %s count=%s: %s", addr, count, resp)
                continue
            for i, v in enumerate(resp.registers):
                row[f"hr_{addr + i}"] = int(v) & 0xFFFF
        except Exception as e:
            logger.debug("holding read exception at %s: %s", addr, e)

    for addr, count in _chunks(0, cfg.max_input, cfg.chunk_size):
        try:
            resp = client.read_input_registers(address=addr, count=count, slave=cfg.unit_id)
            if resp.isError():
                logger.debug("input register read error at %s count=%s: %s", addr, count, resp)
                continue
            for i, v in enumerate(resp.registers):
                row[f"ir_{addr + i}"] = int(v) & 0xFFFF
        except Exception as e:
            logger.debug("input register read exception at %s: %s", addr, e)

    if cfg.include_discrete and cfg.max_discrete > 0:
        for addr, count in _chunks(0, cfg.max_discrete, min(cfg.chunk_size, 128)):
            try:
                resp = client.read_discrete_inputs(address=addr, count=count, slave=cfg.unit_id)
                if resp.isError():
                    logger.debug("discrete read error at %s: %s", addr, resp)
                    continue
                bits = resp.bits
                for i in range(count):
                    if i < len(bits):
                        row[f"di_{addr + i}"] = 1 if bits[i] else 0
            except Exception as e:
                logger.debug("discrete read exception at %s: %s", addr, e)

    if cfg.include_coils and cfg.max_coils > 0:
        for addr, count in _chunks(0, cfg.max_coils, min(cfg.chunk_size, 128)):
            try:
                resp = client.read_coils(address=addr, count=count, slave=cfg.unit_id)
                if resp.isError():
                    logger.debug("coil read error at %s: %s", addr, resp)
                    continue
                bits = resp.bits
                for i in range(count):
                    if i < len(bits):
                        row[f"co_{addr + i}"] = 1 if bits[i] else 0
            except Exception as e:
                logger.debug("coil read exception at %s: %s", addr, e)

    return row


def _open_modbus_client(cfg: ModbusLoggingConfig) -> Any:
    """Return connected pymodbus client or raise."""
    t = (cfg.transport or "tcp").strip().lower()
    if t == "tcp":
        from pymodbus.client import ModbusTcpClient

        c = ModbusTcpClient(host=cfg.host, port=cfg.port)
    elif t == "rtu":
        from pymodbus.client import ModbusSerialClient

        c = ModbusSerialClient(
            port=cfg.serial_port,
            baudrate=cfg.baudrate,
            parity=cfg.parity,
            stopbits=cfg.stopbits,
            bytesize=cfg.bytesize,
        )
    else:
        raise ValueError(f"Unknown Modbus transport: {cfg.transport!r} (use tcp or rtu)")

    if not c.connect():
        if t == "tcp":
            raise ConnectionError(f"Modbus TCP connect failed ({cfg.host!r}:{cfg.port})")
        raise ConnectionError(f"Modbus RTU connect failed ({cfg.serial_port!r})")
    return c


def run_modbus_poll_loop(
    output_path: Path,
    stop_event: threading.Event,
    cfg: ModbusLoggingConfig,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    on_connect_fail: Callable[[str], None] | None = None,
) -> None:
    """
    Poll until stop_event is set; append one CSV row per successful cycle.

    on_row is called from this thread with the row dict (includes t_s and any keys read).
    On connect failure, writes a short error stub file and returns (no exception).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(cfg)
    interval = max(0.05, 1.0 / cfg.poll_hz) if cfg.poll_hz > 0 else 0.5
    client = None
    try:
        try:
            client = _open_modbus_client(cfg)
        except Exception as e:
            logger.exception("Modbus connect failed")
            err_path = path.with_suffix(path.suffix + ".error.txt")
            err_path.write_text(f"Modbus connect failed: {e}\n", encoding="utf-8")
            stub = path
            stub.write_text(
                f"# modbus connect failed — see {err_path.name}\n# error, {e!r}\n",
                encoding="utf-8",
            )
            if on_connect_fail is not None:
                on_connect_fail(str(e))
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            f.write(f"# poll_hz, {cfg.poll_hz}\n")
            f.write(f"# transport, {cfg.transport}\n")
            f.write(f"# unit_id, {cfg.unit_id}\n")
            if cfg.profile:
                f.write(f"# profile, {cfg.profile}\n")
            if cfg.transport.lower() == "tcp":
                f.write(f"# host, {cfg.host}\n")
                f.write(f"# port, {cfg.port}\n")
            else:
                f.write(f"# serial_port, {cfg.serial_port}\n")
                f.write(f"# baudrate, {cfg.baudrate}\n")
            f.flush()
            while not stop_event.is_set():
                loop_t0 = time.monotonic()
                row = poll_registers_once(client, cfg)
                out = {k: "" for k in fieldnames}
                out["t_s"] = f"{row['t_s']:.6f}"
                for k, v in row.items():
                    if k == "t_s":
                        continue
                    if k in out:
                        out[k] = v
                w.writerow(out)
                f.flush()
                if on_row is not None:
                    on_row(row)
                elapsed = time.monotonic() - loop_t0
                sleep_s = interval - elapsed
                if sleep_s > 0:
                    stop_event.wait(sleep_s)
    except Exception as e:
        logger.exception("Modbus polling failed")
        try:
            err_path = path.with_suffix(path.suffix + ".error.txt")
            err_path.write_text(f"Modbus polling stopped with error: {e}\n", encoding="utf-8")
        except OSError:
            pass
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

