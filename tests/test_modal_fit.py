"""
Tests for tap_testing.modal_fit (peak-picking system identification).
"""
import math

import numpy as np
import pytest

from tap_testing.modal_fit import (
    ModalFitResult2DOF,
    chain_params_from_local_matrices,
    find_bandwidth_zero_crossings,
    find_two_minima_imaginary,
    local_matrices_from_modal,
    measured_frf_all_modes_nutshell,
    modal_fit_error_sources_nutshell,
    modal_matrix_from_cross_direct_peaks,
    modal_params_from_peak_pick,
    model_definition_from_fit_and_cross_peaks,
    model_definition_nutshell,
    peak_pick_direct_frf_2dof,
    system_identification_nutshell,
)


class TestModalParamsFromPeakPick:
    def test_formulas_recover_known_params(self):
        # Use Example 2.4–style values: ωn1≈350.43, ωn2≈988.53, ζq1≈0.035, ζq2≈0.099
        omega_n1, omega_n2 = 350.43, 988.53
        zeta1, zeta2 = 0.035, 0.099
        # Bandwidth: 2*zeta*omega_n = omega4 - omega3
        omega3 = omega_n1 - zeta1 * omega_n1
        omega4 = omega_n1 + zeta1 * omega_n1
        omega5 = omega_n2 - zeta2 * omega_n2
        omega6 = omega_n2 + zeta2 * omega_n2
        kq1, kq2 = 2.782e5, 1.75e6
        A = 1.0 / (2.0 * zeta1 * kq1)
        B = 1.0 / (2.0 * zeta2 * kq2)
        result = modal_params_from_peak_pick(
            omega_n1, omega_n2, omega3, omega4, omega5, omega6, A, B
        )
        assert abs(result.zeta_q1 - zeta1) < 0.001
        assert abs(result.zeta_q2 - zeta2) < 0.001
        assert abs(result.kq1 - kq1) < 1e3
        assert abs(result.kq2 - kq2) < 1e4
        assert abs(result.omega_n1 - omega_n1) < 0.01
        assert abs(result.omega_n2 - omega_n2) < 0.01
        assert result.Mq.shape == (2, 2)
        assert result.Cq.shape == (2, 2)
        assert result.Kq.shape == (2, 2)
        assert abs(result.Mq[0, 1]) < 1e-10 and abs(result.Cq[0, 1]) < 1e-10

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            modal_params_from_peak_pick(0, 100, 90, 110, 980, 1000, 1e-5, 1e-6)
        with pytest.raises(ValueError):
            modal_params_from_peak_pick(100, 200, 95, 105, 195, 205, 0, 1e-6)
        with pytest.raises(ValueError):
            modal_params_from_peak_pick(100, 200, 105, 95, 195, 205, 1e-5, 1e-6)


class TestFindTwoMinimaImaginary:
    def test_identifies_two_peaks(self):
        # Standard receptance form so Im(H) has two local minima (negative peaks) near 350 and 1000
        omega = np.linspace(200, 1500, 500)
        H = 1.0 / (1 - (omega / 350) ** 2 + 0.07j * (omega / 350))
        H += 0.5 / (1 - (omega / 1000) ** 2 + 0.1j * (omega / 1000))
        (o1, a1), (o2, a2) = find_two_minima_imaginary(omega, H)
        assert o1 < o2
        assert 200 < o1 < 500
        assert 800 < o2 < 1200
        assert a1 > 0 and a2 > 0

    def test_returns_positive_magnitudes(self):
        omega = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
        # Im(H) has local minima at 3 and 10 (negative peaks)
        H = np.array(
            [-1.0, -0.5, -3.0 - 1.0j, -0.5, -1.0, -2.0 - 0.5j, -1.0], dtype=complex
        )
        (_, a1), (_, a2) = find_two_minima_imaginary(omega, H)
        assert a1 > 0 and a2 > 0


class TestFindBandwidthZeroCrossings:
    def test_finds_bracket_around_omega_n(self):
        omega = np.linspace(300, 400, 200)
        # Real part that crosses zero twice around 350
        re_part = (omega - 340) * (360 - omega) * 1e-6
        out = find_bandwidth_zero_crossings(omega, re_part, 350.0)
        assert out is not None
        omega_lo, omega_hi = out
        assert omega_lo < 350 < omega_hi
        assert 330 < omega_lo < 350 and 350 < omega_hi < 370


class TestPeakPickDirectFrf2dof:
    def test_with_synthetic_frf_from_twodof(self):
        from tap_testing.twodof import frf_direct_modal_chain

        m1, m2, k1, k2, c1, c2 = 2.0, 1.0, 4e5, 6e5, 80.0, 120.0
        omega = np.linspace(50, 1200, 400)
        H = np.array(
            [frf_direct_modal_chain(w, m1, m2, k1, k2, c1, c2) for w in omega],
            dtype=complex,
        )
        result = peak_pick_direct_frf_2dof(omega, H, bandwidth_from_zero_crossings=True)
        if result is not None:
            o1, o2 = result.omega_n1, result.omega_n2
            assert o1 < o2
            # Known natural frequencies from Example 2.4 approx 350 and 988 rad/s
            assert 300 < o1 < 400
            assert 900 < o2 < 1100

    def test_fallback_without_zero_crossings(self):
        omega = np.linspace(1, 1000, 100)
        H = 1.0 / (1 - (omega / 100) ** 2 + 0.1j * (omega / 100))
        H += 0.5 / (1 - (omega / 400) ** 2 + 0.05j * (omega / 400))
        result = peak_pick_direct_frf_2dof(
            omega, H, bandwidth_from_zero_crossings=False
        )
        assert result is not None
        assert result.omega_n1 < result.omega_n2


class TestSystemIdentificationNutshell:
    def test_contains_peak_and_modal(self):
        s = system_identification_nutshell()
        assert "peak" in s.lower() or "FRF" in s
        assert "modal" in s.lower()
        assert "Mq" in s or "stiffness" in s.lower()


class TestModalFitErrorSourcesNutshell:
    def test_mentions_resolution_and_simplifications(self):
        s = modal_fit_error_sources_nutshell()
        assert "resolution" in s.lower()
        assert "simplification" in s.lower() or "fitting" in s.lower()
        assert "error" in s.lower() or "true" in s.lower()

    def test_mentions_out_of_band_modes(self):
        s = modal_fit_error_sources_nutshell()
        assert "outside" in s.lower() or "out-of-band" in s.lower() or "range" in s.lower()


class TestMeasuredFrfAllModesNutshell:
    def test_mentions_all_modes_and_influence(self):
        s = measured_frf_all_modes_nutshell()
        assert "all modes" in s.lower() or "modes" in s.lower()
        assert "influence" in s.lower() or "affect" in s.lower() or "outside" in s.lower()


class TestModalMatrixFromCrossDirectPeaks:
    def test_p1_p2_ratios(self):
        A, B = 1e-5, 0.5e-5
        C, D = 0.7953e-5, 0.6287e-5  # Example 2.4 style: p1≈0.7953, p2≈-0.6287 for cross
        P = modal_matrix_from_cross_direct_peaks(A, B, C, D)
        assert P.shape == (2, 2)
        assert np.allclose(P[1, :], [1.0, 1.0])
        assert abs(P[0, 0] - C / A) < 1e-10
        assert abs(P[0, 1] - D / B) < 1e-10

    def test_zero_direct_peak_raises(self):
        with pytest.raises(ValueError):
            modal_matrix_from_cross_direct_peaks(0.0, 1.0, 0.5, 0.5)


class TestLocalMatricesFromModal:
    def test_roundtrip_with_twodof(self):
        from tap_testing.twodof import (
            matrices_chain,
            modal_matrix_chain,
            modal_mass_damping_stiffness_chain,
        )

        m1, m2, k1, k2, c1, c2 = 2.0, 1.0, 4e5, 6e5, 80.0, 120.0
        M, C, K = matrices_chain(m1, m2, k1, k2, c1, c2)
        Mq, Cq, Kq = modal_mass_damping_stiffness_chain(m1, m2, k1, k2, c1, c2, normalize_to="x2")
        P = modal_matrix_chain(m1, m2, k1, k2, normalize_to="x2")
        M_loc, C_loc, K_loc = local_matrices_from_modal(Mq, Cq, Kq, P)
        assert np.allclose(M_loc, M, atol=1e-6)
        assert np.allclose(C_loc, C, atol=1e-6)
        assert np.allclose(K_loc, K, atol=1e-6)


class TestChainParamsFromLocalMatrices:
    def test_recovers_chain_params(self):
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        K = np.array([[1e6, -6e5], [-6e5, 6e5]])
        C = np.array([[200, -120], [-120, 120]])
        m1, m2, k1, k2, c1, c2 = chain_params_from_local_matrices(M, C, K)
        assert m1 == 2.0 and m2 == 1.0
        assert abs(k1 - 4e5) < 1 and abs(k2 - 6e5) < 1
        assert abs(c1 - 80) < 1 and abs(c2 - 120) < 1


class TestModelDefinitionFromFitAndCrossPeaks:
    def test_returns_P_M_C_K_and_chain(self):
        fit = modal_params_from_peak_pick(
            350.0, 988.0,
            340.0, 360.0, 978.0, 998.0,
            1e-5, 0.5e-5,
        )
        P, M, C, K, chain = model_definition_from_fit_and_cross_peaks(fit, 0.8e-5, 0.3e-5)
        assert P.shape == (2, 2)
        assert M.shape == (2, 2)
        assert len(chain) == 6
        m1, m2, k1, k2, c1, c2 = chain
        assert m1 > 0 and m2 > 0 and k1 > 0 and k2 > 0


class TestModelDefinitionNutshell:
    def test_contains_P_and_chain(self):
        s = model_definition_nutshell()
        assert "P" in s or "modal" in s.lower()
        assert "chain" in s.lower() or "local" in s.lower()
        assert "p1" in s or "C/A" in s or "cross" in s.lower()
