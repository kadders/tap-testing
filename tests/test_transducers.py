"""
Tests for tap_testing.transducers (inertance/receptance, phase correction).
"""
import numpy as np
import pytest

from tap_testing.transducers import (
    DEFAULT_ACCELEROMETER,
    correct_frf_real_imag_from_phase_slope,
    correct_phase_linear,
    inertance_to_receptance,
    inertance_receptance_nutshell,
    phase_error_from_time_delay,
    phase_error_time_delay_nutshell,
    receptance_to_inertance,
    slope_deg_per_hz_from_time_delay,
    transducers_nutshell,
)


class TestInertanceToReceptance:
    def test_formula(self):
        omega = 100.0 * 2 * np.pi  # 100 Hz
        H_inertance = 1e-4 + 0.5e-4j
        H_receptance = inertance_to_receptance(omega, H_inertance)
        expected = H_inertance / (omega * omega)
        assert np.isclose(H_receptance.real, expected.real)
        assert np.isclose(H_receptance.imag, expected.imag)

    def test_roundtrip(self):
        omega = 200.0 * 2 * np.pi
        H_rec = 1e-6 - 2e-6j
        H_inert = receptance_to_inertance(omega, H_rec)
        H_rec_back = inertance_to_receptance(omega, H_inert)
        assert np.isclose(H_rec_back.real, H_rec.real)
        assert np.isclose(H_rec_back.imag, H_rec.imag)

    def test_zero_omega_raises(self):
        with pytest.raises(ValueError):
            inertance_to_receptance(0.0, 1.0 + 0.0j)


class TestPhaseErrorFromTimeDelay:
    def test_50_us_at_5kHz(self):
        # 50 μs → slope -18 deg/kHz → at 5 kHz: -90 deg
        err = phase_error_from_time_delay(50e-6, 5000.0)
        assert abs(err - (-90.0)) < 0.1

    def test_linear_in_frequency(self):
        err_1k = phase_error_from_time_delay(50e-6, 1000.0)
        err_2k = phase_error_from_time_delay(50e-6, 2000.0)
        assert np.isclose(err_2k, 2.0 * err_1k)


class TestSlopeDegPerHzFromTimeDelay:
    def test_50_us_gives_18_deg_per_khz(self):
        S = slope_deg_per_hz_from_time_delay(50e-6)
        # -360 * 50e-6 = -0.018 deg/Hz = -18 deg/kHz
        assert abs(S - (-0.018)) < 1e-6


class TestCorrectPhaseLinear:
    def test_subtracts_linear_term(self):
        phi_c = correct_phase_linear(1000.0, -90.0, -0.018)
        # Δφ = -0.018 * 1000 = -18 deg, φ_c = φ_m - Δφ = -90 - (-18) = -72
        assert abs(phi_c - (-72.0)) < 0.01


class TestCorrectFrfRealImagFromPhaseSlope:
    def test_magnitude_unchanged_phase_corrected(self):
        mag = 1e-5
        phi_m_deg = -90.0
        f_hz = 1000.0
        S = -0.018
        Re_c, Im_c = correct_frf_real_imag_from_phase_slope(mag, phi_m_deg, f_hz, S)
        phi_c = -90.0 - (-0.018 * 1000)  # -72 deg
        expected_Re = mag * np.cos(np.deg2rad(phi_c))
        expected_Im = mag * np.sin(np.deg2rad(phi_c))
        assert np.isclose(Re_c, expected_Re)
        assert np.isclose(Im_c, expected_Im)


class TestTransducersNutshell:
    def test_mentions_accelerometer_and_noncontact(self):
        s = transducers_nutshell()
        assert "accelerometer" in s.lower() or "contact" in s.lower()
        assert "tool" in s.lower() or "mass" in s.lower()

    def test_mentions_adxl345(self):
        s = transducers_nutshell()
        assert "ADXL345" in s


class TestDefaultAccelerometer:
    def test_is_adxl345(self):
        assert DEFAULT_ACCELEROMETER == "ADXL345"


class TestInertanceReceptanceNutshell:
    def test_mentions_omega_squared(self):
        s = inertance_receptance_nutshell()
        assert "ω" in s or "omega" in s.lower() or "inertance" in s.lower()
        assert "receptance" in s.lower()


class TestPhaseErrorTimeDelayNutshell:
    def test_mentions_phase_and_correction(self):
        s = phase_error_time_delay_nutshell()
        assert "phase" in s.lower()
        assert "correct" in s.lower() or "φ" in s