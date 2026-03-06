"""
Tests for tap_testing.docs.excitation (force input types and tap-test relation).
"""
from tap_testing.config import DEFAULT_EXCITATION_TYPE, TapTestConfig, get_config
from tap_testing.docs.excitation import (
    DEFAULT_EXCITATION_TYPE as EXCITATION_DEFAULT,
    excitation_types_nutshell,
    force_input_summary,
    impact_excitation_nutshell,
    impact_hammer_hardware_nutshell,
    tap_test_excitation_nutshell,
)


class TestExcitationTypesNutshell:
    def test_mentions_sine_random_impulse(self):
        s = excitation_types_nutshell()
        assert "sine" in s.lower() or "sweep" in s.lower()
        assert "random" in s.lower()
        assert "impulse" in s.lower() or "impact" in s.lower()


class TestImpactExcitationNutshell:
    def test_mentions_hammer_and_frequencies(self):
        s = impact_excitation_nutshell()
        assert "hammer" in s.lower() or "impact" in s.lower()
        assert "frequencies" in s.lower() or "frequency" in s.lower()


class TestImpactHammerHardwareNutshell:
    def test_mentions_tip_and_bandwidth(self):
        s = impact_hammer_hardware_nutshell()
        assert "tip" in s.lower() or "hammer" in s.lower()
        assert "bandwidth" in s.lower() or "stiff" in s.lower()


class TestTapTestExcitationNutshell:
    def test_mentions_tap_and_impact(self):
        s = tap_test_excitation_nutshell()
        assert "tap" in s.lower()
        assert "impact" in s.lower() or "hammer" in s.lower()
        assert "fft" in s.lower() or "natural" in s.lower()


class TestForceInputSummary:
    def test_returns_all_keys(self):
        d = force_input_summary()
        assert set(d.keys()) == {"types", "impact", "impact_hammer", "tap_test"}
        assert "sine" in d["types"].lower() or "sweep" in d["types"].lower()
        assert "tap" in d["tap_test"].lower()


class TestDefaultExcitationIsImpulse:
    def test_config_default_is_impulse(self):
        assert DEFAULT_EXCITATION_TYPE == "impulse"
        cfg = get_config()
        assert cfg.excitation_type == "impulse"

    def test_excitation_module_exports_same_default(self):
        assert EXCITATION_DEFAULT == "impulse"
        assert EXCITATION_DEFAULT == DEFAULT_EXCITATION_TYPE
