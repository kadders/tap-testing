"""
Tests for tap_testing.measurement_uncertainties (FFT resolution, FRF corrections, tap spread).
"""
import math

import pytest

from tap_testing.measurement_uncertainties import (
    correct_frf_mass_loading,
    correct_frf_misalignment_bias,
    fft_frequency_resolution_uncertainty_hz,
    natural_freq_uncertainty_from_tap_spread,
    combined_natural_freq_uncertainty_hz,
)


class TestFftFrequencyResolutionUncertainty:
    def test_half_bin_default(self):
        # 800 Hz, 400 samples => df = 2 Hz, half bin = 1 Hz
        u = fft_frequency_resolution_uncertainty_hz(800.0, 400)
        assert u == 1.0

    def test_full_bin(self):
        u = fft_frequency_resolution_uncertainty_hz(800.0, 400, half_bin=False)
        assert u == 2.0

    def test_zero_or_short_returns_zero(self):
        assert fft_frequency_resolution_uncertainty_hz(800.0, 1) == 0.0
        assert fft_frequency_resolution_uncertainty_hz(0.0, 100) == 0.0


class TestCorrectFrfMisalignmentBias:
    def test_small_angle_reduces_magnitude(self):
        frf = 1.0 + 0.5j
        u_rad = math.radians(3)  # ~3 degrees
        corrected = correct_frf_misalignment_bias(frf, u_rad)
        factor = 1.0 - 0.5 * (u_rad ** 2)
        assert abs(corrected - frf * factor) < 1e-10
        assert abs(corrected) < abs(frf)

    def test_zero_uncertainty_unchanged(self):
        frf = 2.0 - 1.0j
        assert correct_frf_misalignment_bias(frf, 0.0) == frf


class TestCorrectFrfMassLoading:
    def test_formula(self):
        # FRF_t = FRF_m / (1 + m_a * omega^2 * FRF_m)
        frf_m = 1e-6 + 0j  # 1e-6 m/N
        omega = 2 * math.pi * 100  # 100 Hz
        m_a = 0.01  # 10 g
        denom = 1.0 + m_a * (omega ** 2) * frf_m
        expected = frf_m / denom
        corrected = correct_frf_mass_loading(frf_m, omega, m_a)
        assert abs(corrected - expected) < 1e-20

    def test_zero_mass_unchanged(self):
        frf = 1e-5 * (1 + 1j)
        assert correct_frf_mass_loading(frf, 2 * math.pi * 50, 0.0) == frf


class TestNaturalFreqUncertaintyFromTapSpread:
    def test_two_values(self):
        std = natural_freq_uncertainty_from_tap_spread([100.0, 102.0])
        assert 1.0 <= std <= 2.0  # sample std of [100, 102] is 1.41...

    def test_identical_returns_zero(self):
        assert natural_freq_uncertainty_from_tap_spread([100.0, 100.0, 100.0]) == 0.0

    def test_single_or_empty_returns_zero(self):
        assert natural_freq_uncertainty_from_tap_spread([100.0]) == 0.0
        assert natural_freq_uncertainty_from_tap_spread([]) == 0.0


class TestCombinedNaturalFreqUncertainty:
    def test_fft_only(self):
        u = combined_natural_freq_uncertainty_hz(1.0, None)
        assert u == 1.0

    def test_combined_quadrature(self):
        u = combined_natural_freq_uncertainty_hz(1.0, 1.0)
        assert abs(u - math.sqrt(2.0)) < 1e-10

    def test_zero_spread_returns_fft(self):
        assert combined_natural_freq_uncertainty_hz(0.5, 0.0) == 0.5
        assert combined_natural_freq_uncertainty_hz(0.5, None) == 0.5
