"""
Tests for tap_testing.twodof (2DOF chain-type spring-mass-damper).
"""
import math

import numpy as np
import pytest

from tap_testing.twodof import (
    damping_matrix_chain,
    dynamic_stiffness_chain,
    forced_response_complex_chain,
    free_response_modal,
    free_response_modal_single,
    frequency_response_matrix_chain,
    frf_cross_modal_chain,
    frf_direct_modal_chain,
    initial_conditions_modal,
    local_response_from_modal,
    mass_matrix_chain,
    matrices_chain,
    modal_damping_ratio,
    modal_force_transform,
    modal_mass_damping_stiffness_chain,
    modal_mass_stiffness_chain,
    modal_matrix_chain,
    mode_shape_ratio_x1_over_x2,
    mode_shape_ratio_x2_over_x1,
    mode_shapes_chain,
    mode_shapes_chain_normalized_to_x1,
    mode_shapes_chain_normalized_to_x2,
    natural_frequencies_hz,
    natural_frequencies_rad_s,
    proportional_damping_chain,
    sdof_receptance_modal,
    stiffness_matrix_chain,
    twodof_forced_vibration_nutshell,
    twodof_free_vibration_nutshell,
    twodof_modal_analysis_nutshell,
    twodof_modal_forced_frf_nutshell,
    twodof_mode_shapes_nutshell,
)


class TestMassMatrixChain:
    def test_diagonal(self):
        M = mass_matrix_chain(1.0, 2.0)
        assert M.shape == (2, 2)
        assert M[0, 0] == 1.0
        assert M[1, 1] == 2.0
        assert M[0, 1] == 0.0
        assert M[1, 0] == 0.0


class TestStiffnessMatrixChain:
    def test_structure(self):
        K = stiffness_matrix_chain(100.0, 50.0)
        assert K.shape == (2, 2)
        assert K[0, 0] == 150.0  # k1 + k2
        assert K[1, 1] == 50.0   # k2
        assert K[0, 1] == -50.0
        assert K[1, 0] == -50.0
        assert np.allclose(K, K.T)


class TestDampingMatrixChain:
    def test_structure(self):
        C = damping_matrix_chain(1.0, 2.0)
        assert C.shape == (2, 2)
        assert C[0, 0] == 3.0  # c1 + c2
        assert C[1, 1] == 2.0  # c2
        assert C[0, 1] == -2.0
        assert C[1, 0] == -2.0


class TestMatricesChain:
    def test_returns_M_C_K(self):
        M, C, K = matrices_chain(1.0, 1.0, 10.0, 10.0, 0.5, 0.5)
        assert M.shape == (2, 2)
        assert C.shape == (2, 2)
        assert K.shape == (2, 2)
        assert np.allclose(M, mass_matrix_chain(1.0, 1.0))
        assert np.allclose(K, stiffness_matrix_chain(10.0, 10.0))
        assert np.allclose(C, damping_matrix_chain(0.5, 0.5))

    def test_zero_damping_default(self):
        M, C, K = matrices_chain(1.0, 1.0, 10.0, 10.0)
        # c1=c2=0 (default) => C = [[0,0],[0,0]]
        assert np.allclose(C, np.zeros((2, 2)))


class TestNaturalFrequenciesRadS:
    def test_identical_masses_identical_springs(self):
        # m1=m2=1, k1=k2=1 => characteristic eq and check ordering
        o1, o2 = natural_frequencies_rad_s(1.0, 1.0, 1.0, 1.0)
        assert o1 > 0 and o2 > 0
        assert o1 <= o2

    def test_convert_to_hz(self):
        o1, o2 = natural_frequencies_rad_s(1.0, 1.0, 100.0, 100.0)
        f1 = o1 / (2 * math.pi)
        f2 = o2 / (2 * math.pi)
        assert f1 < f2
        f1_out, f2_out = natural_frequencies_hz(1.0, 1.0, 100.0, 100.0)
        assert abs(f1_out - f1) < 1e-10
        assert abs(f2_out - f2) < 1e-10

    def test_invalid_mass_raises(self):
        with pytest.raises(ValueError):
            natural_frequencies_rad_s(0.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            natural_frequencies_rad_s(1.0, -0.1, 1.0, 1.0)

    def test_invalid_stiffness_raises(self):
        with pytest.raises(ValueError):
            natural_frequencies_rad_s(1.0, 1.0, -1.0, 1.0)
        with pytest.raises(ValueError):
            natural_frequencies_rad_s(1.0, 1.0, 1.0, 0.0)


class TestNaturalFrequenciesHz:
    def test_ordering(self):
        f1, f2 = natural_frequencies_hz(1.0, 2.0, 100.0, 50.0)
        assert f1 <= f2
        assert f1 > 0 and f2 > 0


class TestModeShapeRatioX2OverX1:
    def test_matches_top_row_equation(self):
        m1, k1, k2 = 1.0, 10.0, 5.0
        o1, o2 = natural_frequencies_rad_s(1.0, 1.0, k1, k2)
        r1 = mode_shape_ratio_x2_over_x1(o1 * o1, m1, k1, k2)
        r2 = mode_shape_ratio_x2_over_x1(o2 * o2, m1, k1, k2)
        # (k1+k2 - ω²·m1)·1 - k2·r = 0
        assert abs((k1 + k2 - o1 * o1 * m1) - k2 * r1) < 1e-10
        assert abs((k1 + k2 - o2 * o2 * m1) - k2 * r2) < 1e-10

    def test_first_mode_in_phase_second_out_of_phase(self):
        # First mode (ω₁) should have r1 > 0, second (ω₂) r2 < 0 for typical chain
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        r1 = mode_shape_ratio_x2_over_x1(o1 * o1, m1, k1, k2)
        r2 = mode_shape_ratio_x2_over_x1(o2 * o2, m1, k1, k2)
        assert r1 > 0
        assert r2 < 0


class TestModeShapesChain:
    def test_shape_and_normalize_first(self):
        modes = mode_shapes_chain(1.0, 1.0, 10.0, 10.0, normalize="first")
        assert modes.shape == (2, 2)
        assert np.allclose(modes[0, :], [1.0, 1.0])  # first component = 1 for each column

    def test_satisfy_eigenvalue_equation(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        M, _, K = matrices_chain(m1, m2, k1, k2)
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        modes = mode_shapes_chain(m1, m2, k1, k2, normalize="first")
        # (K - ω² M) @ v ≈ 0
        v1 = modes[:, 0]
        v2 = modes[:, 1]
        r1 = (K - o1**2 * M) @ v1
        r2 = (K - o2**2 * M) @ v2
        assert np.linalg.norm(r1) < 1e-10
        assert np.linalg.norm(r2) < 1e-10

    def test_norm_normalize(self):
        modes = mode_shapes_chain(1.0, 2.0, 50.0, 30.0, normalize="norm")
        assert modes.shape == (2, 2)
        assert np.allclose(np.linalg.norm(modes[:, 0]), 1.0)
        assert np.allclose(np.linalg.norm(modes[:, 1]), 1.0)

    def test_first_mode_in_phase_second_out_of_phase(self):
        modes = mode_shapes_chain(1.0, 0.5, 100.0, 50.0, normalize="first")
        assert modes[1, 0] > 0  # first mode: in phase
        assert modes[1, 1] < 0  # second mode: out of phase


class TestModeShapesChainNormalizedToX1:
    def test_matches_explicit_ratio_formula(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        explicit = mode_shapes_chain_normalized_to_x1(m1, m2, k1, k2)
        assert explicit.shape == (2, 2)
        assert np.allclose(explicit[0, :], [1.0, 1.0])
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        assert abs(explicit[1, 0] - mode_shape_ratio_x2_over_x1(o1 * o1, m1, k1, k2)) < 1e-10
        assert abs(explicit[1, 1] - mode_shape_ratio_x2_over_x1(o2 * o2, m1, k1, k2)) < 1e-10

    def test_equals_mode_shapes_chain_first(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        explicit = mode_shapes_chain_normalized_to_x1(m1, m2, k1, k2)
        from_eig = mode_shapes_chain(m1, m2, k1, k2, normalize="first")
        # Columns may differ by sign; ratios X2/X1 must match
        for j in range(2):
            assert abs(explicit[1, j] - from_eig[1, j]) < 1e-10 or abs(
                explicit[1, j] + from_eig[1, j]
            ) < 1e-10


class TestTwodofModeShapesNutshell:
    def test_contains_in_phase_and_out_of_phase(self):
        s = twodof_mode_shapes_nutshell()
        assert "in phase" in s.lower()
        assert "out of phase" in s.lower()
        assert "X2/X1" in s or "x2" in s.lower()


class TestModeShapeRatioX1OverX2:
    def test_from_bottom_row(self):
        m2, k2 = 0.5, 50.0
        o1, o2 = natural_frequencies_rad_s(1.0, m2, 100.0, k2)
        r1 = mode_shape_ratio_x1_over_x2(o1 * o1, m2, k2)
        r2 = mode_shape_ratio_x1_over_x2(o2 * o2, m2, k2)
        # Bottom row: -k2*X1 + (k2 - ω²*m2)*X2 = 0 => X1/X2 = (k2 - ω²*m2)/k2
        assert abs((k2 - o1 * o1 * m2) / k2 - r1) < 1e-10
        assert abs((k2 - o2 * o2 * m2) / k2 - r2) < 1e-10


class TestModeShapesChainNormalizedToX2:
    def test_second_row_is_ones(self):
        P = mode_shapes_chain_normalized_to_x2(1.0, 0.5, 100.0, 50.0)
        assert P.shape == (2, 2)
        assert np.allclose(P[1, :], [1.0, 1.0])

    def test_satisfy_eigenvalue_equation(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        M, _, K = matrices_chain(m1, m2, k1, k2)
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        P = mode_shapes_chain_normalized_to_x2(m1, m2, k1, k2)
        for j, o in enumerate([o1, o2]):
            v = P[:, j]
            assert np.linalg.norm((K - o * o * M) @ v) < 1e-9


class TestModalMatrixChain:
    def test_x1_equals_normalized_to_x1(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        P = modal_matrix_chain(m1, m2, k1, k2, normalize_to="x1")
        expected = mode_shapes_chain_normalized_to_x1(m1, m2, k1, k2)
        assert np.allclose(P, expected)

    def test_x2_second_row_ones(self):
        P = modal_matrix_chain(1.0, 0.5, 100.0, 50.0, normalize_to="x2")
        assert np.allclose(P[1, :], [1.0, 1.0])

    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError):
            modal_matrix_chain(1.0, 1.0, 1.0, 1.0, normalize_to="x3")


class TestModalMassStiffnessChain:
    def test_diagonal_dominant(self):
        Mq, Kq = modal_mass_stiffness_chain(1.0, 0.5, 100.0, 50.0, normalize_to="x2")
        assert Mq.shape == (2, 2) and Kq.shape == (2, 2)
        assert abs(Mq[0, 1]) < 1e-10 and abs(Mq[1, 0]) < 1e-10
        assert abs(Kq[0, 1]) < 1e-8 and abs(Kq[1, 0]) < 1e-8

    def test_modal_frequencies_match(self):
        m1, m2, k1, k2 = 1.0, 0.5, 100.0, 50.0
        Mq, Kq = modal_mass_stiffness_chain(m1, m2, k1, k2, normalize_to="x1")
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        assert abs(math.sqrt(Kq[0, 0] / Mq[0, 0]) - o1) < 1e-10
        assert abs(math.sqrt(Kq[1, 1] / Mq[1, 1]) - o2) < 1e-10


class TestInitialConditionsModal:
    def test_roundtrip(self):
        P = modal_matrix_chain(1.0, 0.5, 100.0, 50.0, normalize_to="x2")
        x0 = np.array([1.0, -1.0])
        dx0 = np.array([0.0, 0.0])
        q0, dq0 = initial_conditions_modal(x0, dx0, P)
        x_back = P @ q0
        assert np.allclose(x_back, x0)
        assert np.allclose(P @ dq0, dx0)


class TestFreeResponseModalSingle:
    def test_zero_velocity_cosine_only(self):
        t = np.linspace(0, 0.01, 100)
        q = free_response_modal_single(2505.0, 0.2833, 0.0, t)
        assert np.allclose(q, 0.2833 * np.cos(2505.0 * t))

    def test_zero_displacement_sine_only(self):
        t = np.linspace(0, 0.001, 50)
        q = free_response_modal_single(100.0, 0.0, 10.0, t)
        assert np.allclose(q, (10.0 / 100.0) * np.sin(100.0 * t))


class TestFreeResponseModal:
    def test_shape_and_initial_values(self):
        t = np.array([0.0, 0.001, 0.002])
        q0 = np.array([0.2833, -1.283])
        dq0 = np.array([0.0, 0.0])
        q = free_response_modal(t, q0, dq0, 2505.0, 7983.0)
        assert q.shape == (2, 3)
        assert np.allclose(q[:, 0], q0)


class TestLocalResponseFromModal:
    def test_x_equals_P_times_q(self):
        P = modal_matrix_chain(1.0, 0.5, 100.0, 50.0, normalize_to="x2")
        q = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2) two time steps
        x = local_response_from_modal(q, P)
        assert np.allclose(x, P @ q)
        assert x.shape == (2, 2)


class TestTwodofModalAnalysisNutshell:
    def test_contains_diagonal_and_uncoupled(self):
        s = twodof_modal_analysis_nutshell()
        assert "P" in s or "modal" in s.lower()
        assert "uncouple" in s.lower() or "diagonal" in s.lower()
        assert "x2" in s or "q1" in s


class TestDynamicStiffnessChain:
    def test_zero_omega_is_K(self):
        Z = dynamic_stiffness_chain(0.0, 1.0, 0.5, 100.0, 50.0)
        K = stiffness_matrix_chain(100.0, 50.0)
        assert np.allclose(Z.real, K)
        assert np.allclose(Z.imag, 0.0)

    def test_with_damping_has_imaginary_part(self):
        Z = dynamic_stiffness_chain(100.0, 1.0, 0.5, 100.0, 50.0, c1=1.0, c2=0.5)
        assert np.any(np.abs(Z.imag) > 1e-10)


class TestForcedResponseComplexChain:
    def test_Z_times_X_equals_F(self):
        omega = 500.0
        m1, m2, k1, k2 = 1.0, 0.5, 1e5, 5e4
        F = np.array([0.0, 1.0])
        X = forced_response_complex_chain(omega, m1, m2, k1, k2, F)
        Z = dynamic_stiffness_chain(omega, m1, m2, k1, k2)
        assert np.allclose(Z @ X, F)

    def test_force_at_x2(self):
        omega = 100.0
        F = np.array([0.0, 1.0])
        X = forced_response_complex_chain(omega, 1.0, 1.0, 1000.0, 500.0, F)
        assert X.shape == (2,)
        assert np.iscomplexobj(X)

    def test_F_wrong_length_raises(self):
        with pytest.raises(ValueError):
            forced_response_complex_chain(100.0, 1.0, 1.0, 1.0, 1.0, np.array([1.0]))


class TestFrequencyResponseMatrixChain:
    def test_X_equals_H_times_F(self):
        omega = 200.0
        H = frequency_response_matrix_chain(omega, 1.0, 0.5, 1e4, 5e3)
        F = np.array([0.0, 1.0])
        X_direct = forced_response_complex_chain(omega, 1.0, 0.5, 1e4, 5e3, F)
        X_via_H = H @ F
        assert np.allclose(X_direct, X_via_H)

    def test_second_column_is_response_to_unit_force_at_x2(self):
        omega = 100.0
        H = frequency_response_matrix_chain(omega, 1.0, 1.0, 1000.0, 500.0)
        assert H.shape == (2, 2)
        X = H[:, 1]
        Z = dynamic_stiffness_chain(omega, 1.0, 1.0, 1000.0, 500.0)
        assert np.allclose(Z @ X, np.array([0.0, 1.0]))


class TestTwodofForcedVibrationNutshell:
    def test_contains_forced_and_complex(self):
        s = twodof_forced_vibration_nutshell()
        assert "forced" in s.lower()
        assert "e^(i" in s or "Re(" in s
        assert "Z" in s or "matrix" in s.lower() or "inversion" in s.lower()


class TestProportionalDampingChain:
    def test_example_2_4_proportional(self):
        # Example 2.4: C = β*K with β = 1/5000
        m1, m2, k1, k2 = 2.0, 1.0, 4e5, 6e5
        c1, c2 = 80.0, 120.0
        result = proportional_damping_chain(m1, m2, k1, k2, c1, c2)
        assert result is not None
        alpha, beta = result
        assert abs(alpha) < 1e-6
        assert abs(beta - 1.0 / 5000.0) < 1e-6

    def test_zero_damping_is_proportional(self):
        result = proportional_damping_chain(1.0, 1.0, 100.0, 50.0, 0.0, 0.0)
        assert result is not None
        assert result[0] == 0.0 and result[1] == 0.0


class TestModalMassDampingStiffnessChain:
    def test_diagonal_and_natural_freq_match(self):
        m1, m2, k1, k2, c1, c2 = 2.0, 1.0, 4e5, 6e5, 80.0, 120.0
        Mq, Cq, Kq = modal_mass_damping_stiffness_chain(m1, m2, k1, k2, c1, c2, normalize_to="x2")
        o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
        assert abs(math.sqrt(Kq[0, 0] / Mq[0, 0]) - o1) < 1.0
        assert abs(math.sqrt(Kq[1, 1] / Mq[1, 1]) - o2) < 1.0
        assert abs(Mq[0, 1]) < 1e-8 and abs(Cq[0, 1]) < 1e-6


class TestModalForceTransform:
    def test_force_at_x2_gives_R1_eq_R2(self):
        P = mode_shapes_chain_normalized_to_x2(1.0, 0.5, 100.0, 50.0)
        F = np.array([0.0, 100.0])
        R = modal_force_transform(F, P)
        assert np.allclose(R, [100.0, 100.0])


class TestModalDampingRatio:
    def test_formula(self):
        z = modal_damping_ratio(55.63, 2.782e5, 2.265)
        assert 0.03 < z < 0.04


class TestSdofReceptanceModal:
    def test_low_frequency_near_static(self):
        # ω << ωn => Q/R ≈ 1/k
        H = sdof_receptance_modal(10.0, 350.0, 2.782e5, 0.035)
        assert abs(H.real - 1.0 / 2.782e5) < 5e-9
        assert abs(H.imag) < 1e-8


class TestFrfDirectModalChain:
    def test_matches_complex_inversion_at_low_omega(self):
        # At low frequency, both methods should agree
        m1, m2, k1, k2, c1, c2 = 2.0, 1.0, 4e5, 6e5, 80.0, 120.0
        omega = 50.0
        H_direct = frf_direct_modal_chain(omega, m1, m2, k1, k2, c1, c2)
        F = np.array([0.0, 1.0])
        X = forced_response_complex_chain(omega, m1, m2, k1, k2, F, c1, c2)
        assert abs(X[1] - H_direct) < 1e-10


class TestFrfCrossModalChain:
    def test_matches_complex_inversion_at_low_omega(self):
        m1, m2, k1, k2, c1, c2 = 2.0, 1.0, 4e5, 6e5, 80.0, 120.0
        omega = 50.0
        H_cross = frf_cross_modal_chain(omega, m1, m2, k1, k2, c1, c2)
        F = np.array([0.0, 1.0])
        X = forced_response_complex_chain(omega, m1, m2, k1, k2, F, c1, c2)
        assert abs(X[0] - H_cross) < 1e-10


class TestTwodofModalForcedFrfNutshell:
    def test_contains_direct_and_cross(self):
        s = twodof_modal_forced_frf_nutshell()
        assert "direct" in s.lower()
        assert "cross" in s.lower()
        assert "Q1" in s or "R1" in s


class TestTwodofFreeVibrationNutshell:
    def test_contains_2dof_and_equations(self):
        s = twodof_free_vibration_nutshell()
        assert "2DOF" in s or "two" in s.lower()
        assert "m1" in s and "m2" in s
        assert "natural" in s.lower()
