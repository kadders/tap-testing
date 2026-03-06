"""
Tests for tap_testing.sdof (SDOF natural frequency ↔ stiffness/mass).
"""
import math

import pytest

from tap_testing.sdof import (
    forced_vibration_summary_nutshell,
    frequency_response_magnitude_phase_nutshell,
    is_near_resonance,
    linear_scale_display_nutshell,
    mass_from_natural_frequency,
    natural_frequency_hz,
    self_excited_vibration_summary_nutshell,
    stiffness_from_natural_frequency,
    sdof_summary_nutshell,
    vibration_theory_summary,
)


class TestNaturalFrequencyHz:
    def test_formula_f_n_equals_sqrt_k_over_m_over_2pi(self):
        # f_n = (1/(2π)) * sqrt(k/m). k=39478 N/m, m=0.1 kg => omega=628.3, f_n=100
        k = 0.1 * (2 * math.pi * 100) ** 2
        f = natural_frequency_hz(k, 0.1)
        assert abs(f - 100.0) < 0.01

    def test_high_stiffness_high_frequency(self):
        f_high = natural_frequency_hz(100_000, 0.1)
        f_low = natural_frequency_hz(10_000, 0.1)
        assert f_high > f_low

    def test_high_mass_low_frequency(self):
        f_light = natural_frequency_hz(40_000, 0.05)
        f_heavy = natural_frequency_hz(40_000, 0.2)
        assert f_light > f_heavy

    def test_invalid_mass_raises(self):
        with pytest.raises(ValueError):
            natural_frequency_hz(1000, 0)
        with pytest.raises(ValueError):
            natural_frequency_hz(1000, -0.1)

    def test_invalid_stiffness_raises(self):
        with pytest.raises(ValueError):
            natural_frequency_hz(0, 0.1)
        with pytest.raises(ValueError):
            natural_frequency_hz(-100, 0.1)


class TestStiffnessFromNaturalFrequency:
    def test_roundtrip_with_natural_frequency_hz(self):
        k0 = 50_000.0
        m = 0.08
        f = natural_frequency_hz(k0, m)
        k_out = stiffness_from_natural_frequency(f, m)
        assert abs(k_out - k0) < 0.01

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            stiffness_from_natural_frequency(100, 0)
        with pytest.raises(ValueError):
            stiffness_from_natural_frequency(0, 0.1)


class TestMassFromNaturalFrequency:
    def test_roundtrip_with_natural_frequency_hz(self):
        k = 40_000.0
        m0 = 0.1
        f = natural_frequency_hz(k, m0)
        m_out = mass_from_natural_frequency(f, k)
        assert abs(m_out - m0) < 1e-5

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            mass_from_natural_frequency(100, 0)
        with pytest.raises(ValueError):
            mass_from_natural_frequency(0, 1000)


class TestSdofSummaryNutshell:
    def test_contains_key_terms(self):
        s = sdof_summary_nutshell()
        assert "natural frequency" in s
        assert "k/m" in s or "stiffness" in s.lower()
        assert "mass" in s.lower()
        assert "tap test" in s.lower()


class TestForcedVibrationSummaryNutshell:
    def test_contains_resonance_and_forcing(self):
        s = forced_vibration_summary_nutshell()
        assert "resonance" in s.lower()
        assert "forcing" in s.lower() or "periodic" in s.lower()
        assert "tooth" in s.lower() or "natural" in s.lower()


class TestSelfExcitedVibrationSummaryNutshell:
    def test_contains_chatter_and_self_excited(self):
        s = self_excited_vibration_summary_nutshell()
        assert "chatter" in s.lower()
        assert "self" in s.lower() or "modulated" in s.lower()


class TestFrequencyResponseMagnitudePhaseNutshell:
    def test_contains_magnitude_phase_and_phase_angles(self):
        s = frequency_response_magnitude_phase_nutshell()
        assert "magnitude" in s.lower() or "motion" in s.lower()
        assert "phase" in s.lower()
        assert "90" in s or "180" in s or "0°" in s

    def test_contains_excitation_and_natural_frequency(self):
        s = frequency_response_magnitude_phase_nutshell()
        assert "excitation" in s.lower() or "ω" in s
        assert "natural" in s.lower()


class TestLinearScaleDisplayNutshell:
    def test_contains_linear_and_log(self):
        s = linear_scale_display_nutshell()
        assert "linear" in s.lower()
        assert "log" in s.lower()

    def test_mentions_intuitive_or_range(self):
        s = linear_scale_display_nutshell()
        assert "intuitive" in s.lower() or "range" in s.lower() or "graph" in s.lower()


class TestVibrationTheorySummary:
    def test_returns_all_five_keys(self):
        theory = vibration_theory_summary()
        assert set(theory.keys()) == {
            "free",
            "forced",
            "self_excited",
            "frf_magnitude_phase",
            "display_scale",
        }
        assert "natural" in theory["free"].lower()
        assert "resonance" in theory["forced"].lower()
        assert "chatter" in theory["self_excited"].lower()
        assert "phase" in theory["frf_magnitude_phase"].lower()
        assert "linear" in theory["display_scale"].lower()


class TestIsNearResonance:
    def test_at_resonance(self):
        assert is_near_resonance(100.0, 100.0) is True
        assert is_near_resonance(100.0, 100.0, tolerance_hz=0) is True

    def test_within_tolerance(self):
        assert is_near_resonance(98.0, 100.0, tolerance_hz=5.0) is True
        assert is_near_resonance(102.0, 100.0, tolerance_hz=5.0) is True

    def test_outside_tolerance(self):
        assert is_near_resonance(90.0, 100.0, tolerance_hz=5.0) is False
        assert is_near_resonance(110.0, 100.0, tolerance_hz=5.0) is False
