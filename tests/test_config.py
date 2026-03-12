"""
Tests for tap_testing.config defaults and env overrides.
"""
import pytest

from tap_testing.config import TapTestConfig, get_config, rpm_from_spindle_frequency_hz


class TestGetConfig:
    def test_returns_tap_test_config(self):
        cfg = get_config()
        assert isinstance(cfg, TapTestConfig)

    def test_default_i2c_address(self):
        cfg = get_config()
        assert cfg.i2c_address == 0x53

    def test_default_sample_rate(self):
        cfg = get_config()
        assert cfg.sample_rate_hz == 800.0

    def test_default_record_duration(self):
        cfg = get_config()
        assert cfg.record_duration_s == 1.0

    def test_default_cycle_iterations_and_spacing(self):
        cfg = get_config()
        assert cfg.cycle_iterations == 3
        assert cfg.cycle_spacing_s == 15.0

    def test_default_status_led_gpio(self):
        cfg = get_config()
        assert cfg.status_led_gpio == 17

    def test_default_spi_first_byte_data(self):
        cfg = get_config()
        assert cfg.spi_first_byte_data is False

    def test_default_spi_bus_and_device(self):
        cfg = get_config()
        assert cfg.spi_bus == 0
        assert cfg.spi_device == 0

    def test_default_spindle_operating_frequency_hz(self):
        cfg = get_config()
        assert cfg.spindle_operating_frequency_hz == 400.0

    def test_rpm_from_spindle_frequency_rev_s_to_rpm(self):
        # 400 rev/s (Hz) → 24000 rpm; ref uses Ω in rev/s, we use rpm = freq_hz * 60
        assert rpm_from_spindle_frequency_hz(400.0) == pytest.approx(24000.0)
        assert rpm_from_spindle_frequency_hz(1.0) == pytest.approx(60.0)


class TestGetConfigEnvOverrides:
    """Env overrides (TAP_*) are read at get_config() time; use monkeypatch to avoid leaking."""

    def test_spi_first_byte_data_from_env(self, monkeypatch):
        monkeypatch.setenv("TAP_SPI_FIRST_BYTE_DATA", "1")
        cfg = get_config()
        assert cfg.spi_first_byte_data is True

    def test_spi_first_byte_data_env_true(self, monkeypatch):
        monkeypatch.setenv("TAP_SPI_FIRST_BYTE_DATA", "true")
        cfg = get_config()
        assert cfg.spi_first_byte_data is True

    def test_spi_first_byte_data_env_empty_stays_false(self, monkeypatch):
        monkeypatch.setenv("TAP_SPI_FIRST_BYTE_DATA", "")
        cfg = get_config()
        assert cfg.spi_first_byte_data is False
