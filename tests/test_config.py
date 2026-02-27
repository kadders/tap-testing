"""
Tests for tap_testing.config defaults.
"""
import pytest

from tap_testing.config import TapTestConfig, get_config


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
