"""Tests for Modbus logging helpers (no live Modbus device)."""
from __future__ import annotations

import pytest

from tap_testing.modbus_logging import (
    ModbusLoggingConfig,
    apply_modbus_profile,
    build_fieldnames,
    modbus_logging_config_from_env,
    poll_registers_once,
    _chunks,
)


def test_chunks() -> None:
    assert _chunks(0, 250, 100) == [(0, 100), (100, 100), (200, 50)]
    assert _chunks(0, 0, 100) == []


def test_build_fieldnames_counts() -> None:
    cfg = ModbusLoggingConfig(max_holding=2, max_input=3, max_discrete=4, max_coils=5)
    cfg.include_discrete = True
    cfg.include_coils = True
    names = build_fieldnames(cfg)
    assert names[0] == "t_s"
    assert "hr_0" in names and "hr_1" in names and "hr_2" not in names
    assert "ir_0" in names and "ir_2" in names
    assert "di_0" in names and "di_3" in names
    assert "co_4" in names


def test_modbus_logging_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAP_MODBUS_HOST", raising=False)
    monkeypatch.setenv("TAP_MODBUS_TRANSPORT", "rtu")
    monkeypatch.setenv("TAP_MODBUS_SERIAL_PORT", "/dev/ttyTEST")
    monkeypatch.setenv("TAP_MODBUS_BAUDRATE", "19200")
    monkeypatch.setenv("TAP_MODBUS_UNIT", "7")
    monkeypatch.setenv("TAP_MODBUS_POLL_HZ", "4")
    monkeypatch.setenv("TAP_MODBUS_MAX_HOLDING", "50")
    monkeypatch.setenv("TAP_MODBUS_NO_COILS", "1")
    cfg = modbus_logging_config_from_env()
    assert cfg.transport == "rtu"
    assert cfg.serial_port == "/dev/ttyTEST"
    assert cfg.baudrate == 19200
    assert cfg.unit_id == 7
    assert cfg.poll_hz == 4.0
    assert cfg.max_holding == 50
    assert cfg.include_coils is False


def test_poll_registers_once_merges_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def __init__(self, regs: list[int], err: bool = False) -> None:
            self.registers = regs
            self.bits = [True, False]
            self._err = err

        def isError(self) -> bool:
            return self._err

    class FakeClient:
        def read_holding_registers(self, address: int, count: int, slave: int = 1) -> FakeResp:
            return FakeResp([address, address + 1])

        def read_input_registers(self, address: int, count: int, slave: int = 1) -> FakeResp:
            return FakeResp([address + 100])

        def read_discrete_inputs(self, address: int, count: int, slave: int = 1) -> FakeResp:
            return FakeResp([], err=True)

        def read_coils(self, address: int, count: int, slave: int = 1) -> FakeResp:
            return FakeResp([], err=True)

    cfg = ModbusLoggingConfig(
        max_holding=2,
        max_input=1,
        max_discrete=0,
        max_coils=0,
        chunk_size=2,
        include_discrete=False,
        include_coils=False,
    )
    row = poll_registers_once(FakeClient(), cfg)
    assert "t_s" in row
    assert row["hr_0"] == 0
    assert row["hr_1"] == 1
    assert row["ir_0"] == 100


def test_apply_h100_profile() -> None:
    cfg = ModbusLoggingConfig(max_holding=100, max_input=4, max_coils=10)
    apply_modbus_profile(cfg, "h100")
    assert cfg.profile == "h100"
    assert cfg.transport == "rtu"
    assert cfg.baudrate == 19200
    assert cfg.max_holding == 520
    assert cfg.max_input == 8
    assert cfg.max_coils == 96


def test_h100_profile_from_env_then_baud_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAP_MODBUS_PROFILE", "h100")
    monkeypatch.setenv("TAP_MODBUS_BAUDRATE", "9600")
    cfg = modbus_logging_config_from_env()
    assert cfg.profile == "h100"
    assert cfg.baudrate == 9600
    assert cfg.max_holding == 520


def test_apply_modbus_profile_unknown_raises() -> None:
    cfg = ModbusLoggingConfig()
    with pytest.raises(ValueError, match="Unknown"):
        apply_modbus_profile(cfg, "no_such_vfd")
