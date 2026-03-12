"""
Tests for tap_testing.milling_dynamics (nutshell summaries, angles, radial immersion).
"""
import math
import pytest

from tap_testing.milling_dynamics import (
    angle_increment_deg,
    angle_to_time_sec,
    average_teeth_in_cut,
    average_tooth_angle_deg,
    axial_force_with_edge_N,
    ball_chip_width_db_mm,
    coefficient_of_determination_r2,
    cutting_coefficients_from_slotting_regression,
    cutting_speed_m_per_s,
    directional_factors_down_milling,
    directional_factors_up_milling,
    ball_forces_xyz_N,
    ball_kappa_prime_rad,
    ball_slice_width_dz_mm,
    ball_theta0_rad,
    chip_thickness_ball_mm,
    chip_thickness_regenerative_mm,
    constant_force_axial_depth_mm,
    cutting_forces_xy_N,
    exit_angle_up_milling_deg,
    helical_lag_angle_deg,
    linear_regression_slope_intercept,
    mean_force_per_rev_slotting_N,
    milling_ball_endmill_simulation_nutshell,
    milling_chatter_to_stable_strategy_nutshell,
    milling_example_6_2_competing_lobes_nutshell,
    milling_experimental_cutting_coefficients_nutshell,
    milling_experimental_techniques_nutshell,
    milling_process_damping_nutshell,
    milling_process_damping_stability_nutshell,
    milling_linear_regression_coefficients_nutshell,
    milling_nonlinear_optimization_coefficients_nutshell,
    milling_updated_force_model_nutshell,
    milling_cutting_force_geometry_nutshell,
    milling_cutting_forces_example_4_2_nutshell,
    milling_dynamics_nutshell,
    milling_dynamics_summary,
    milling_helical_teeth_nutshell,
    milling_straight_vs_helical_forces_nutshell,
    milling_start_exit_angles_nutshell,
    milling_multiple_teeth_engagement_nutshell,
    milling_stability_lobes_nutshell,
    milling_time_domain_simulation_nutshell,
    milling_surf_update_rule_nutshell,
    normal_force_N,
    normal_force_with_edge_N,
    normal_force_with_process_damping_N,
    normal_surface_vibration_mm,
    normal_velocity_from_xy_mm_s,
    process_damping_cnew_x_up_milling,
    process_damping_cnew_y_up_milling,
    process_damping_force_N,
    radial_immersion_percent,
    simulation_steps_rev,
    simulation_time_step_s,
    stability_lobe_best_spindle_speed_rpm,
    stability_lobe_phase_epsilon_rad,
    start_angle_down_milling_deg,
    tangential_force_N,
    tangential_force_with_edge_N,
    tooth_angle_deg,
    tooth_period_s,
    tooth_passing_frequency_hz,
    tooth_passing_frequency_nutshell,
)


class TestMillingDynamicsNutshell:
    def test_nutshell_non_empty(self):
        s = milling_dynamics_nutshell()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_nutshell_mentions_turning_and_milling(self):
        s = milling_dynamics_nutshell()
        assert "turning" in s.lower()
        assert "milling" in s.lower()

    def test_nutshell_mentions_chip_thickness(self):
        s = milling_dynamics_nutshell()
        assert "chip thickness" in s.lower()

    def test_nutshell_mentions_slotting_pocket_sculptured(self):
        s = milling_dynamics_nutshell()
        assert "slotting" in s.lower()
        assert "pocket" in s.lower()
        assert "sculptured" in s.lower()

    def test_nutshell_mentions_engagement_or_engagement_complexity(self):
        s = milling_dynamics_nutshell()
        assert "engagement" in s.lower() or "complicated" in s.lower()


class TestMillingDynamicsSummary:
    def test_summary_contains_nutshell(self):
        d = milling_dynamics_summary()
        assert "nutshell" in d
        assert d["nutshell"] == milling_dynamics_nutshell()

    def test_summary_contains_all_nutshell_keys(self):
        d = milling_dynamics_summary()
        assert "start_exit_angles" in d
        assert "cutting_force_geometry" in d
        assert "cutting_forces_example_4_2" in d
        assert "multiple_teeth_engagement" in d
        assert "tooth_passing_frequency" in d
        assert "stability_lobes" in d
        assert "time_domain_simulation" in d
        assert "surf_update_rule" in d
        assert "helical_teeth" in d
        assert "straight_vs_helical_forces" in d
        assert "ball_endmill_simulation" in d
        assert "experimental_cutting_coefficients" in d
        assert "updated_force_model" in d
        assert "linear_regression_coefficients" in d
        assert "nonlinear_optimization_coefficients" in d
        assert "experimental_techniques" in d
        assert "process_damping" in d
        assert "process_damping_stability" in d
        assert "chatter_to_stable_strategy" in d
        assert "example_6_2_competing_lobes" in d


class TestHelicalLagAndConstantForceDepth:
    """Example 4.14: lag angle χ = 2b·tan(γ)/d; constant-force b = d·φp/(2·tan(γ))."""

    def test_lag_angle_example_4_14(self):
        # b=5 mm, γ=45°, d=19 mm → χ ≈ 30.2°
        chi = helical_lag_angle_deg(45.0, 5.0, 19.0)
        assert chi == pytest.approx(30.2, abs=0.5)

    def test_constant_force_depth_example_4_14(self):
        # 19 mm, pitch 90°, 45° helix → b ≈ 14.9 mm
        b = constant_force_axial_depth_mm(19.0, 90.0, 45.0)
        assert b == pytest.approx(14.9, abs=0.2)

    def test_zero_diameter_lag_returns_zero(self):
        assert helical_lag_angle_deg(45.0, 5.0, 0.0) == 0.0

    def test_zero_helix_constant_force_depth_returns_zero(self):
        assert constant_force_axial_depth_mm(19.0, 90.0, 0.0) == 0.0


class TestEngagementPercentAtDepthHelix:
    """Engagement % = 100 * min(1, lag/tooth_pitch) at given depth and helix."""

    def test_at_constant_force_depth_100_percent(self):
        from tap_testing.milling_dynamics import engagement_percent_at_depth_helix
        b_const = constant_force_axial_depth_mm(19.0, 90.0, 45.0)
        assert engagement_percent_at_depth_helix(b_const, 19.0, 4, 45.0) == pytest.approx(100.0)

    def test_below_constant_force_depth_less_than_100(self):
        from tap_testing.milling_dynamics import engagement_percent_at_depth_helix
        b_const = constant_force_axial_depth_mm(19.0, 90.0, 45.0)
        b_half = b_const * 0.5
        assert engagement_percent_at_depth_helix(b_half, 19.0, 4, 45.0) == pytest.approx(50.0, abs=2.0)

    def test_zero_helix_returns_100(self):
        from tap_testing.milling_dynamics import engagement_percent_at_depth_helix
        assert engagement_percent_at_depth_helix(5.0, 19.0, 4, 0.0) == 100.0


class TestEngagementVsLobeIndexData:
    def test_returns_lobe_index_rpm_engagement_tuples(self):
        from tap_testing.milling_dynamics import engagement_vs_lobe_index_data
        data = engagement_vs_lobe_index_data(500.0, 4, 19.0, 10.0, 45.0, n_lobes=3)
        assert len(data) == 3
        for n, (idx, rpm, eng_pct) in enumerate(data):
            assert idx == n
            assert rpm > 0
            assert 0 <= eng_pct <= 100


class TestMillingHelicalTeethNutshell:
    def test_mentions_helical_and_simulation(self):
        s = milling_helical_teeth_nutshell()
        assert "helical" in s.lower()
        assert "simulation" in s.lower()

    def test_mentions_analytical_or_tradeoff(self):
        s = milling_helical_teeth_nutshell()
        assert "analytical" in s.lower() or "trade-off" in s.lower() or "computationally" in s.lower()


class TestMillingStraightVsHelicalForcesNutshell:
    def test_mentions_example_4_14_and_lag(self):
        s = milling_straight_vs_helical_forces_nutshell()
        assert "4.14" in s or "example" in s.lower()
        assert "lag" in s.lower() or "saw" in s.lower() or "constant" in s.lower()

    def test_mentions_straight_and_helical(self):
        s = milling_straight_vs_helical_forces_nutshell()
        assert "straight" in s.lower() and "helical" in s.lower()


class TestBallEndmillGeometry:
    """Ball endmill slice geometry (Sect. 4.6, Eqs. 4.70–4.73)."""

    def test_ball_slice_width_dz(self):
        # d=19 mm, dφ = 2π/1200 rad, γ=45° → dz = 19*(2π/1200)/(2*tan(45°)) = 19*π/1200 ≈ 0.0498
        d_rad = 2.0 * math.pi / 1200.0
        dz = ball_slice_width_dz_mm(19.0, d_rad, 45.0)
        assert dz == pytest.approx(19.0 * math.pi / 1200.0, rel=1e-5)

    def test_ball_kappa_prime_first_slice(self):
        # j=1: κ′ = cos⁻¹(1 - 2*dz/d); small dz gives κ′ near 0
        dz = 0.05
        kappa = ball_kappa_prime_rad(1, dz, 19.0)
        assert kappa == pytest.approx(math.acos(1.0 - 2.0 * dz / 19.0), rel=1e-6)

    def test_ball_kappa_prime_90_at_apex(self):
        # When 2*j*dz/d >= 1, κ′ = π/2
        kappa = ball_kappa_prime_rad(10, 1.0, 19.0)  # 2*10*1/19 > 1
        assert kappa == pytest.approx(math.pi / 2.0)

    def test_ball_theta0_first_slice_zero(self):
        assert ball_theta0_rad(1, 0.1, 19.0) == 0.0

    def test_ball_theta0_second_slice(self):
        theta0 = ball_theta0_rad(2, 0.5, 19.0)
        expected = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * 0.5 * 1.0 / 19.0)))
        assert theta0 == pytest.approx(expected)

    def test_ball_chip_width_db(self):
        # db = r*(κ′ - θ0); r=9.5, κ′=0.5, θ0=0.2 → db = 9.5*0.3 = 2.85
        db = ball_chip_width_db_mm(9.5, 0.5, 0.2)
        assert db == pytest.approx(9.5 * 0.3)

    def test_ball_chip_width_db_zero_when_theta_negative(self):
        assert ball_chip_width_db_mm(9.5, 0.1, 0.2) == 0.0


class TestChipThicknessBall:
    def test_chip_thickness_ball_projection(self):
        # h = h_xy * sin(κ′); κ′=90° → h = h_xy
        h = chip_thickness_ball_mm(0.1, math.pi / 2.0)
        assert h == pytest.approx(0.1)

    def test_chip_thickness_ball_zero_kappa(self):
        assert chip_thickness_ball_mm(0.1, 0.0) == 0.0

    def test_chip_thickness_ball_negative_h_xy_returns_zero(self):
        assert chip_thickness_ball_mm(-0.01, 0.5) == 0.0


class TestBallForcesXyz:
    """Ball endmill force projection Eq. 4.77 and Eq. 4.78 at κ′=90°."""

    def test_ball_forces_xyz_at_90_deg_same_as_square_xy(self):
        # At κ′=90°: Fx = cos(φ)*Ft + sin(φ)*Fn, Fy = sin(φ)*Ft - cos(φ)*Fn, Fz = -Fa (or +Fa by convention)
        phi = math.radians(30.0)
        Ft, Fn, Fa = 100.0, 50.0, 25.0
        Fx, Fy, Fz = ball_forces_xyz_N(Ft, Fn, Fa, phi, math.pi / 2.0)
        c, s = math.cos(phi), math.sin(phi)
        assert Fx == pytest.approx(c * Ft + s * Fn)
        assert Fy == pytest.approx(s * Ft - c * Fn)
        assert Fz == pytest.approx(-Fa)  # Eq. 4.77 with cos(90°)=0, sin(90°)=1 → Fz = -Fa

    def test_ball_forces_xyz_zero_phi(self):
        # At φ=0: Fx = Ft, Fy = -cos(κ′)*Fa - sin(κ′)*Fn, Fz = cos(κ′)*Fn - sin(κ′)*Fa
        Fx, Fy, Fz = ball_forces_xyz_N(10.0, 5.0, 2.0, 0.0, 0.5)
        ck, sk = math.cos(0.5), math.sin(0.5)
        assert Fx == pytest.approx(10.0)  # Ft only in x at φ=0
        assert Fy == pytest.approx(-ck * 2.0 - sk * 5.0)
        assert Fz == pytest.approx(ck * 5.0 - sk * 2.0)


class TestMillingBallEndmillSimulationNutshell:
    def test_mentions_ball_and_helical(self):
        s = milling_ball_endmill_simulation_nutshell()
        assert "ball" in s.lower()
        assert "4.6" in s or "κ" in s or "kappa" in s.lower()

    def test_mentions_example_4_15(self):
        s = milling_ball_endmill_simulation_nutshell()
        assert "4.15" in s or "example" in s.lower()


class TestMillingExperimentalCuttingCoefficientsNutshell:
    def test_mentions_dynamometer_and_fixed_frame(self):
        s = milling_experimental_cutting_coefficients_nutshell()
        assert "dynamometer" in s.lower() or "fixed" in s.lower()
        assert "x" in s and ("y" in s or "z" in s)

    def test_mentions_kt_kn_and_square_endmill(self):
        s = milling_experimental_cutting_coefficients_nutshell()
        assert "kt" in s or "kn" in s or "coefficient" in s.lower()
        assert "square" in s.lower() or "endmill" in s.lower()


class TestForceWithEdge:
    """Updated force model with edge effect (Eqs. 4.79–4.81)."""

    def test_tangential_force_with_edge(self):
        # Ft = kt*b*h + kte*b; h=0 -> Ft = kte*b
        Ft = tangential_force_with_edge_N(100.0, 2.0, 5.0, 0.0)
        assert Ft == pytest.approx(2.0 * 5.0)  # 10 N
        Ft = tangential_force_with_edge_N(100.0, 2.0, 5.0, 0.1)
        assert Ft == pytest.approx(100.0 * 5.0 * 0.1 + 2.0 * 5.0)  # 50 + 10 = 60 N

    def test_normal_force_with_edge(self):
        Fn = normal_force_with_edge_N(50.0, 1.0, 5.0, 0.1)
        assert Fn == pytest.approx(50.0 * 5.0 * 0.1 + 1.0 * 5.0)  # 25 + 5 = 30 N

    def test_axial_force_with_edge(self):
        Fa = axial_force_with_edge_N(30.0, 0.5, 5.0, 0.1)
        assert Fa == pytest.approx(30.0 * 5.0 * 0.1 + 0.5 * 5.0)  # 15 + 2.5 = 17.5 N


class TestMeanForcePerRevSlotting:
    """Mean cutting force per revolution for slotting (Eqs. 4.96–4.98)."""

    def test_mean_force_slotting_zero_teeth_returns_zeros(self):
        Fx, Fy, Fz = mean_force_per_rev_slotting_N(
            0, 5.0, 0.1, 100.0, 50.0, 30.0, 2.0, 1.0, 0.5
        )
        assert Fx == Fy == Fz == 0.0

    def test_mean_force_slotting_formula(self):
        # Fx = Nt*b*(kn/4*ft + kne/π), Fy = Nt*b*(kt/4*ft + kte/π), Fz = -Nt*b*(ka/π*ft + kae/2)
        Nt, b, ft = 4, 5.0, 0.1
        kt, kn, ka = 100.0, 50.0, 30.0
        kte, kne, kae = 2.0, 1.0, 0.5
        Fx, Fy, Fz = mean_force_per_rev_slotting_N(Nt, b, ft, kt, kn, ka, kte, kne, kae)
        assert Fx == pytest.approx(Nt * b * (kn / 4.0 * ft + kne / math.pi))
        assert Fy == pytest.approx(Nt * b * (kt / 4.0 * ft + kte / math.pi))
        assert Fz == pytest.approx(-Nt * b * (ka / math.pi * ft + kae / 2.0))


class TestLinearRegressionSlopeIntercept:
    """Least-squares slope and intercept (Eqs. 4.99–4.100)."""

    def test_perfect_line(self):
        ft = [0.05, 0.1, 0.15, 0.2, 0.25]
        F = [10.0 + 100.0 * x for x in ft]  # F = 10 + 100*ft
        a1, a0 = linear_regression_slope_intercept(ft, F)
        assert a1 == pytest.approx(100.0)
        assert a0 == pytest.approx(10.0)

    def test_single_point_returns_intercept_only(self):
        a1, a0 = linear_regression_slope_intercept([1.0], [5.0])
        assert a1 == 0.0
        assert a0 == 5.0


class TestCoefficientOfDeterminationR2:
    def test_perfect_fit_r2_one(self):
        ft = [0.05, 0.1, 0.15]
        F = [15.0 + 200.0 * x for x in ft]
        a1, a0 = linear_regression_slope_intercept(ft, F)
        r2 = coefficient_of_determination_r2(ft, F, a1, a0)
        assert r2 == pytest.approx(1.0)

    def test_poor_fit_r2_low(self):
        ft = [1.0, 2.0, 3.0]
        F = [1.0, 100.0, 1.0]  # not linear
        a1, a0 = linear_regression_slope_intercept(ft, F)
        r2 = coefficient_of_determination_r2(ft, F, a1, a0)
        assert r2 < 0.5


class TestCuttingCoefficientsFromSlottingRegression:
    """Recover six coefficients from slotting regression (Eqs. 4.102–4.104)."""

    def test_inverse_of_mean_force_formulas(self):
        # Use known coefficients to get mean forces at one ft, then slopes/intercepts
        # from two points (ft1, F1), (ft2, F2) -> a1 = (F2-F1)/(ft2-ft1), a0 = F1 - a1*ft1
        Nt, b = 2, 5.0
        kn, kne, kt, kte, ka, kae = 190.0, 5.0, 710.0, 4.0, 95.0, 2.0
        ft1, ft2 = 0.1, 0.2
        Fx1, Fy1, Fz1 = mean_force_per_rev_slotting_N(Nt, b, ft1, kt, kn, ka, kte, kne, kae)
        Fx2, Fy2, Fz2 = mean_force_per_rev_slotting_N(Nt, b, ft2, kt, kn, ka, kte, kne, kae)
        a1x = (Fx2 - Fx1) / (ft2 - ft1)
        a0x = Fx1 - a1x * ft1
        a1y = (Fy2 - Fy1) / (ft2 - ft1)
        a0y = Fy1 - a1y * ft1
        a1z = (Fz2 - Fz1) / (ft2 - ft1)
        a0z = Fz1 - a1z * ft1
        kn_out, kne_out, kt_out, kte_out, ka_out, kae_out = (
            cutting_coefficients_from_slotting_regression(
                Nt, b, a1x, a0x, a1y, a0y, a1z, a0z
            )
        )
        assert kn_out == pytest.approx(kn)
        assert kne_out == pytest.approx(kne)
        assert kt_out == pytest.approx(kt)
        assert kte_out == pytest.approx(kte)
        assert ka_out == pytest.approx(ka)
        assert kae_out == pytest.approx(kae)


class TestExample4_16:
    """Example 4.16: linear regression to recover coefficients from slotting data."""

    def test_recover_six_coefficients_from_simulated_means(self):
        # Coefficients from Example 4.16: kn=190, kne=5, kt=710, kte=4, ka=95, kae=2; b=5, Nt=2
        Nt, b = 2, 5.0
        kn, kne, kt, kte, ka, kae = 190.0, 5.0, 710.0, 4.0, 95.0, 2.0
        ft_list = [0.05, 0.1, 0.15, 0.2, 0.25]
        Fx_list = []
        Fy_list = []
        Fz_list = []
        for ft in ft_list:
            Fx, Fy, Fz = mean_force_per_rev_slotting_N(
                Nt, b, ft, kt, kn, ka, kte, kne, kae
            )
            Fx_list.append(Fx)
            Fy_list.append(Fy)
            Fz_list.append(Fz)
        a1x, a0x = linear_regression_slope_intercept(ft_list, Fx_list)
        a1y, a0y = linear_regression_slope_intercept(ft_list, Fy_list)
        a1z, a0z = linear_regression_slope_intercept(ft_list, Fz_list)
        kn_out, kne_out, kt_out, kte_out, ka_out, kae_out = (
            cutting_coefficients_from_slotting_regression(
                Nt, b, a1x, a0x, a1y, a0y, a1z, a0z
            )
        )
        assert kn_out == pytest.approx(kn, rel=1e-10)
        assert kne_out == pytest.approx(kne, rel=1e-10)
        assert kt_out == pytest.approx(kt, rel=1e-10)
        assert kte_out == pytest.approx(kte, rel=1e-10)
        assert ka_out == pytest.approx(ka, rel=1e-10)
        assert kae_out == pytest.approx(kae, rel=1e-10)

    def test_r2_near_one_for_noiseless_data(self):
        Nt, b = 2, 5.0
        kt, kn, ka, kte, kne, kae = 710.0, 190.0, 95.0, 4.0, 5.0, 2.0
        ft_list = [0.05, 0.1, 0.15, 0.2, 0.25]
        Fx_list = [
            mean_force_per_rev_slotting_N(Nt, b, ft, kt, kn, ka, kte, kne, kae)[0]
            for ft in ft_list
        ]
        a1x, a0x = linear_regression_slope_intercept(ft_list, Fx_list)
        r2 = coefficient_of_determination_r2(ft_list, Fx_list, a1x, a0x)
        assert r2 == pytest.approx(1.0)


class TestMillingUpdatedForceModelNutshell:
    def test_mentions_edge_and_six_coefficients(self):
        s = milling_updated_force_model_nutshell()
        assert "edge" in s.lower() or "rubbing" in s.lower()
        assert "4.79" in s or "4.81" in s or "six" in s.lower()

    def test_mentions_slotting_and_linear_regression(self):
        s = milling_updated_force_model_nutshell()
        assert "slotting" in s.lower() or "4.96" in s or "4.98" in s
        assert "regression" in s.lower() or "stability" in s.lower()


class TestMillingLinearRegressionCoefficientsNutshell:
    def test_mentions_slope_intercept_and_r2(self):
        s = milling_linear_regression_coefficients_nutshell()
        assert "slope" in s.lower() or "intercept" in s.lower()
        assert "r²" in s or "4.99" in s or "4.101" in s or "4.102" in s

    def test_mentions_example_4_16(self):
        s = milling_linear_regression_coefficients_nutshell()
        assert "4.16" in s or "example" in s.lower()


class TestMillingNonlinearOptimizationCoefficientsNutshell:
    def test_mentions_instantaneous_and_single_cut(self):
        s = milling_nonlinear_optimization_coefficients_nutshell()
        assert "instantaneous" in s.lower() or "nonlinear" in s.lower()
        assert "single" in s.lower() or "one cut" in s.lower() or "one cut" in s


class TestMillingExperimentalTechniquesNutshell:
    def test_mentions_dynamometer_and_bandwidth(self):
        s = milling_experimental_techniques_nutshell()
        assert "dynamometer" in s.lower()
        assert "bandwidth" in s.lower() or "frequency" in s.lower() or "tooth passing" in s.lower()

    def test_mentions_stable_or_stability(self):
        s = milling_experimental_techniques_nutshell()
        assert "stable" in s.lower() or "stability" in s.lower()


class TestProcessDampingForce:
    """Process damping force Eq. 4.108: Fd = C·b·ṅ/v."""

    def test_process_damping_force_formula(self):
        # Fd = C * b * n_dot / v; C=1, b=5, v=100, n_dot=2 -> Fd = 0.1 N
        Fd = process_damping_force_N(1.0, 5.0, 100.0, 2.0)
        assert Fd == pytest.approx(0.1)

    def test_process_damping_zero_velocity(self):
        assert process_damping_force_N(1.0, 5.0, 100.0, 0.0) == 0.0

    def test_process_damping_zero_cutting_speed_returns_zero(self):
        assert process_damping_force_N(1.0, 5.0, 0.0, 2.0) == 0.0


class TestStabilityLobePhaseEpsilon:
    """Phase ε = 2π − 2·atan(Im/Re) for Eq. 4.110."""

    def test_epsilon_negative_Re_positive_Im(self):
        # Re < 0, Im > 0: atan2(Im, Re) in (pi/2, pi) -> epsilon in (0, pi)
        eps = stability_lobe_phase_epsilon_rad(-1.0, 1.0)
        assert eps == pytest.approx(2.0 * math.pi - 2.0 * math.atan2(1.0, -1.0))

    def test_epsilon_Re_zero(self):
        eps = stability_lobe_phase_epsilon_rad(0.0, 1.0)
        assert eps == pytest.approx(2.0 * math.pi - 2.0 * (math.pi / 2.0))


class TestCuttingSpeedMPerS:
    def test_cutting_speed_formula(self):
        # v = π*d*Ω/60; d=0.019 m, Ω=60 rpm -> v = π*0.019*60/60 = 0.019π m/s
        v = cutting_speed_m_per_s(0.019, 60.0)
        assert v == pytest.approx(math.pi * 0.019)

    def test_zero_rpm_returns_zero(self):
        assert cutting_speed_m_per_s(0.019, 0.0) == 0.0


class TestProcessDampingCnewUpMilling:
    """New damping cnew,x and cnew,y (Eqs. 4.113–4.116)."""

    def test_cnew_x_at_phi_ave_45(self):
        # cos²(90−45)=cos²(45)=0.5; cnew = cx + (C*b/v)*0.5
        cnew = process_damping_cnew_x_up_milling(100.0, 2e4, 0.005, 1.0, 45.0)
        add = (2e4 * 0.005 / 1.0) * 0.5
        assert cnew == pytest.approx(100.0 + add)

    def test_cnew_y_at_phi_ave_45(self):
        # cos²(180−45)=cos²(135)=0.5
        cnew = process_damping_cnew_y_up_milling(100.0, 2e4, 0.005, 1.0, 45.0)
        add = (2e4 * 0.005 / 1.0) * 0.5
        assert cnew == pytest.approx(100.0 + add)

    def test_zero_cutting_speed_returns_original_damping(self):
        assert process_damping_cnew_x_up_milling(50.0, 2e4, 0.005, 0.0, 45.0) == 50.0


class TestDirectionalFactorsUpMilling:
    def test_returns_two_factors(self):
        mu_x, mu_y = directional_factors_up_milling(70.0, 45.0)
        assert isinstance(mu_x, float) and isinstance(mu_y, float)

    def test_phi_ave_30_force_angle_70(self):
        # Example Fig. 4.74: φave=30°, β=70°
        mu_x, mu_y = directional_factors_up_milling(70.0, 30.0)
        # μx = cos(60° - 70°)*cos(60°) = cos(-10°)*cos(60°); μy = cos(150° - 70°)*cos(150°)
        a_rad = math.radians(60.0)
        expected_mu_x = math.cos(a_rad - math.radians(70.0)) * math.cos(a_rad)
        assert mu_x == pytest.approx(expected_mu_x)


class TestDirectionalFactorsDownMilling:
    """Ref Fig. 4.24, Example 4.4: 50% radial immersion down milling, φave=135°."""

    def test_returns_two_factors(self):
        mu_x, mu_y = directional_factors_down_milling(70.0, 135.0)
        assert isinstance(mu_x, float) and isinstance(mu_y, float)

    def test_50_percent_phi_ave_135_force_angle_70(self):
        # μx = cos(45+β)*cos(45), μy = cos(β−45)*cos(45) (ref Example 4.4)
        mu_x, mu_y = directional_factors_down_milling(70.0, 135.0)
        a_rad = math.radians(45.0)
        beta_rad = math.radians(70.0)
        expected_mu_x = math.cos(beta_rad + a_rad) * math.cos(a_rad)
        expected_mu_y = math.cos(beta_rad - a_rad) * math.cos(a_rad)
        assert mu_x == pytest.approx(expected_mu_x)
        assert mu_y == pytest.approx(expected_mu_y)


class TestNormalVelocityFromXy:
    def test_same_projection_as_displacement(self):
        # ṅ = ẋ*sin(φ) − ẏ*cos(φ); φ=0 -> ṅ = −ẏ
        n_dot = normal_velocity_from_xy_mm_s(10.0, 5.0, 0.0)
        assert n_dot == pytest.approx(-5.0)


class TestNormalForceWithProcessDamping:
    def test_reduces_to_cutting_only_when_v_zero(self):
        Fn = normal_force_with_process_damping_N(50.0, 5.0, 0.1, 1.0, 0.0, 2.0)
        assert Fn == pytest.approx(50.0 * 5.0 * 0.1)

    def test_process_damping_subtracts(self):
        # Fn = kn*b*h - C*b/v*n_dot; with n_dot>0, damping term positive so Fn reduced
        Fn = normal_force_with_process_damping_N(50.0, 5.0, 0.1, 1.0, 100.0, 2.0)
        cutting = 50.0 * 5.0 * 0.1
        damping = 1.0 * 5.0 * 2.0 / 100.0
        assert Fn == pytest.approx(cutting - damping)


class TestMillingChatterToStableStrategyNutshell:
    def test_mentions_chatter_and_tooth_passing(self):
        s = milling_chatter_to_stable_strategy_nutshell()
        assert "chatter" in s.lower()
        assert "tooth" in s.lower() or "teeth" in s.lower()

    def test_mentions_record_signal_and_spindle_speed(self):
        s = milling_chatter_to_stable_strategy_nutshell()
        assert "record" in s.lower() or "signal" in s.lower()
        assert "spindle" in s.lower() or "speed" in s.lower()


class TestMillingExample6_2CompetingLobesNutshell:
    def test_mentions_919_804_and_regulation_speeds(self):
        s = milling_example_6_2_competing_lobes_nutshell()
        assert "919" in s or "804" in s
        assert "13,785" in s or "13785" in s
        assert "12,060" in s or "12060" in s

    def test_mentions_competing_lobes_and_two_regulations(self):
        s = milling_example_6_2_competing_lobes_nutshell()
        assert "competing" in s.lower()
        assert "regulation" in s.lower() or "stable" in s.lower()


class TestMillingProcessDampingNutshell:
    def test_mentions_clearance_and_viscous(self):
        s = milling_process_damping_nutshell()
        assert "clearance" in s.lower() or "viscous" in s.lower()
        assert "4.108" in s or "Fd" in s


class TestMillingProcessDampingStabilityNutshell:
    def test_mentions_blim_and_fc_Omega(self):
        s = milling_process_damping_stability_nutshell()
        assert "blim" in s or "4.109" in s
        assert "4.110" in s or "ε" in s or "phase" in s.lower()


class TestRadialImmersionPercent:
    def test_example_10_percent(self):
        # 1.9 mm radial depth, 19 mm diameter → 10%
        assert radial_immersion_percent(1.9, 19.0) == pytest.approx(10.0)

    def test_slotting_100_percent(self):
        assert radial_immersion_percent(19.0, 19.0) == pytest.approx(100.0)

    def test_zero_diameter_returns_zero(self):
        assert radial_immersion_percent(1.9, 0.0) == 0.0


class TestExitAngleUpMilling:
    def test_example_4_1(self):
        # 19 mm diameter → r = 9.5 mm, a = 1.9 mm → cos⁻¹(0.8) ≈ 36.87°
        phi_e = exit_angle_up_milling_deg(9.5, 1.9)
        assert 36 <= phi_e <= 38

    def test_full_slot_exit_90_deg(self):
        # a = r → (r-a)/r = 0 → cos⁻¹(0) = 90°
        assert exit_angle_up_milling_deg(10.0, 10.0) == pytest.approx(90.0)

    def test_very_shallow_cut_small_exit_angle(self):
        # a small ⇒ (r-a)/r close to 1 ⇒ φe small (e.g. a=0.01, r=10 → ~2.6°)
        phi_e = exit_angle_up_milling_deg(10.0, 0.01)
        assert 0 <= phi_e <= 5

    def test_zero_radius_returns_zero(self):
        assert exit_angle_up_milling_deg(0.0, 1.0) == 0.0


class TestStartAngleDownMilling:
    def test_example_4_2_25_percent_immersion(self):
        # a = r/2 → (r - a)/r = 0.5 → cos⁻¹(0.5) = 60° → φs = 180° - 60° = 120°
        phi_s = start_angle_down_milling_deg(10.0, 5.0)
        assert 119 <= phi_s <= 121

    def test_full_slot_start_zero(self):
        # a = r → (r-a)/r = 0 → cos⁻¹(0) = 90° → φs = 90°
        assert start_angle_down_milling_deg(10.0, 10.0) == pytest.approx(90.0)

    def test_zero_radius_returns_180(self):
        assert start_angle_down_milling_deg(0.0, 1.0) == 180.0


class TestStartExitAnglesNutshell:
    def test_mentions_phi_s_phi_e_and_37_deg(self):
        s = milling_start_exit_angles_nutshell()
        assert "φs" in s or "start" in s.lower()
        assert "φe" in s or "exit" in s.lower()
        assert "37" in s
        assert "10%" in s or "10 %" in s


class TestCuttingForceGeometryNutshell:
    def test_mentions_Fx_Fy_and_angle(self):
        s = milling_cutting_force_geometry_nutshell()
        assert "Fx" in s or "x" in s
        assert "Fy" in s or "y" in s
        assert "φ" in s or "angle" in s.lower()
        assert "Ft" in s or "Fn" in s


class TestCuttingForcesExample42Nutshell:
    def test_mentions_25_percent_and_60_deg(self):
        s = milling_cutting_forces_example_4_2_nutshell()
        assert "25%" in s
        assert "60" in s

    def test_mentions_up_and_down_milling_and_finishing(self):
        s = milling_cutting_forces_example_4_2_nutshell()
        assert "up milling" in s or "Up milling" in s
        assert "down milling" in s or "Down milling" in s
        assert "finishing" in s

    def test_mentions_typical_params_kt_kn_ft(self):
        s = milling_cutting_forces_example_4_2_nutshell()
        assert "750" in s or "kt" in s
        assert "0.1" in s or "ft" in s


class TestAngleToTimeSec:
    def test_example_4_2_90_deg_at_7500_rpm(self):
        # 90° at 7500 rpm → 0.002 s (tooth period for 4 teeth)
        t = angle_to_time_sec(90.0, 7500.0)
        assert abs(t - 0.002) < 1e-6

    def test_zero_rpm_returns_zero(self):
        assert angle_to_time_sec(360.0, 0.0) == 0.0


class TestToothPassingFrequencyHz:
    def test_example_7500_rpm_4_teeth(self):
        # 7500 * 4 / 60 = 500 Hz
        assert tooth_passing_frequency_hz(7500.0, 4) == pytest.approx(500.0)

    def test_matches_avoid_rpm_relation(self):
        # At avoid RPM, f_tooth * k = natural_freq. So f_tooth = 100 Hz, k=1 => RPM = 60*100/4 = 1500
        from tap_testing.analyze import rpm_to_avoid
        avoid = rpm_to_avoid(100.0, 4, harmonic_orders=1, rpm_min=1, rpm_max=20000)
        assert 1500 in avoid
        assert tooth_passing_frequency_hz(1500.0, 4) == pytest.approx(100.0)


class TestToothPassingFrequencyNutshell:
    def test_mentions_f_tooth_and_tap_test(self):
        s = tooth_passing_frequency_nutshell()
        assert "f_tooth" in s or "tooth passing" in s.lower()
        assert "tap" in s.lower() or "natural frequency" in s.lower()

    def test_mentions_avoid_rpm_or_resonance(self):
        s = tooth_passing_frequency_nutshell()
        assert "avoid" in s.lower() or "resonance" in s.lower()


class TestToothAngleDeg:
    def test_four_teeth_tooth_1_at_40(self):
        # Tooth 1 at 40° → 2 at 310°, 3 at 220°, 4 at 130°
        assert tooth_angle_deg(40.0, 1, 4) == pytest.approx(40.0)
        assert tooth_angle_deg(40.0, 2, 4) == pytest.approx(310.0)
        assert tooth_angle_deg(40.0, 3, 4) == pytest.approx(220.0)
        assert tooth_angle_deg(40.0, 4, 4) == pytest.approx(130.0)

    def test_pitch_90_for_four_teeth(self):
        assert tooth_angle_deg(0.0, 2, 4) == pytest.approx(270.0)
        assert tooth_angle_deg(90.0, 2, 4) == pytest.approx(0.0)


class TestMultipleTeethEngagementNutshell:
    def test_mentions_75_percent_and_engagement(self):
        s = milling_multiple_teeth_engagement_nutshell()
        assert "75%" in s
        assert "tooth" in s.lower() or "teeth" in s.lower()

    def test_mentions_slotting_and_constant_force(self):
        s = milling_multiple_teeth_engagement_nutshell()
        assert "slotting" in s.lower()
        assert "constant" in s.lower()

    def test_mentions_tap_test_or_flute_count(self):
        s = milling_multiple_teeth_engagement_nutshell()
        assert "tap" in s.lower() or "flute" in s.lower() or "Nt" in s


class TestAverageTeethInCut:
    def test_slotting_four_teeth(self):
        # φs=0, φe=180° → Nt* = (180/360)*4 = 2
        assert average_teeth_in_cut(0.0, 180.0, 4) == pytest.approx(2.0)

    def test_25_percent_up_four_teeth(self):
        # φs=0, φe=60° → Nt* = (60/360)*4 ≈ 0.667
        assert average_teeth_in_cut(0.0, 60.0, 4) == pytest.approx(60.0 / 360.0 * 4)

    def test_zero_teeth_returns_zero(self):
        assert average_teeth_in_cut(0.0, 180.0, 0) == 0.0


class TestAverageToothAngleDeg:
    def test_slotting_90_deg(self):
        assert average_tooth_angle_deg(0.0, 180.0) == pytest.approx(90.0)

    def test_25_percent_up_30_deg(self):
        assert average_tooth_angle_deg(0.0, 60.0) == pytest.approx(30.0)


class TestStabilityLobeBestSpindleSpeedRpm:
    def test_example_4_6_n0_500_hz_4_teeth(self):
        # fn=500 Hz, Nt=4, N=0 → Ω = 500*60/4 = 7500 rpm
        assert stability_lobe_best_spindle_speed_rpm(500.0, 4, 0) == pytest.approx(7500.0)

    def test_n1_half_of_n0(self):
        # N=1 → (N+1)*Nt = 8 → 500*60/8 = 3750
        assert stability_lobe_best_spindle_speed_rpm(500.0, 4, 1) == pytest.approx(3750.0)

    def test_uses_tap_test_natural_freq(self):
        # Our tap test might give 100 Hz; best N=0 for 3 teeth = 100*60/3 = 2000 rpm
        assert stability_lobe_best_spindle_speed_rpm(100.0, 3, 0) == pytest.approx(2000.0)

    def test_lobe_speed_satisfies_f_tooth_equals_fn_over_n_plus_one(self):
        # At lobe N: f_tooth = Ω·Nt/60 must equal fn/(N+1). Cross-check of the formula.
        fn = 500.0
        n_teeth = 4
        for n in range(4):
            rpm = stability_lobe_best_spindle_speed_rpm(fn, n_teeth, lobe_index_n=n)
            f_tooth = tooth_passing_frequency_hz(rpm, n_teeth)
            assert f_tooth == pytest.approx(fn / (n + 1)), (
                f"Lobe N={n}: expected f_tooth={fn/(n+1):.2f} Hz, got {f_tooth:.2f} Hz"
            )


class TestMillingStabilityLobesNutshell:
    def test_mentions_blim_and_oriented_frf(self):
        s = milling_stability_lobes_nutshell()
        assert "blim" in s or "limiting" in s.lower()
        assert "FRF" in s or "oriented" in s.lower()

    def test_mentions_tap_test_and_best_speed(self):
        s = milling_stability_lobes_nutshell()
        assert "tap" in s.lower() or "fn" in s
        assert "best" in s.lower() or "Ω" in s or "speed" in s.lower()


class TestToothPeriodS:
    def test_10000_rpm_4_teeth(self):
        # τ = 60/(10000*4) = 0.0015 s
        assert tooth_period_s(10000.0, 4) == pytest.approx(0.0015)

    def test_consistent_with_tooth_passing_frequency(self):
        # f_tooth = 1/τ
        tau = tooth_period_s(7500.0, 4)
        assert tau > 0
        assert 1.0 / tau == pytest.approx(tooth_passing_frequency_hz(7500.0, 4))


class TestNormalSurfaceVibrationMm:
    def test_at_90_deg(self):
        # φ = π/2 → n = x*1 - y*0 = x
        import math
        assert normal_surface_vibration_mm(0.01, 0.02, math.pi / 2) == pytest.approx(0.01)

    def test_at_zero_deg(self):
        # φ = 0 → n = x*0 - y*1 = -y
        import math
        assert normal_surface_vibration_mm(0.01, 0.02, 0.0) == pytest.approx(-0.02)


class TestChipThicknessRegenerativeMm:
    def test_nominal_only(self):
        import math
        # h = ft*sin(φ) when n_prev = n_curr = 0
        h = chip_thickness_regenerative_mm(0.1, math.radians(30), 0.0, 0.0)
        assert h == pytest.approx(0.1 * 0.5)

    def test_regenerative_term(self):
        # n_prev - n_curr adds to h
        h = chip_thickness_regenerative_mm(0.0, 0.0, 0.01, 0.02)
        assert h == pytest.approx(0.01)


class TestAngleIncrementDeg:
    def test_652_steps(self):
        assert angle_increment_deg(652) == pytest.approx(360.0 / 652)

    def test_zero_steps_returns_zero(self):
        assert angle_increment_deg(0) == 0.0


class TestSimulationTimeStepS:
    def test_example_4_10(self):
        # 652 steps, 10000 rpm → dt = 60/(652*10000) ≈ 9.2025e-6 s
        dt = simulation_time_step_s(652, 10000.0)
        assert 9e-6 <= dt <= 10e-6


class TestSimulationStepsRev:
    def test_example_4_10(self):
        # 650 steps, 4 teeth → 650/4 = 162.5 → round 163 → 652
        assert simulation_steps_rev(650, 4) == 652

    def test_already_integer(self):
        assert simulation_steps_rev(652, 4) == 652


class TestMillingTimeDomainSimulationNutshell:
    def test_mentions_regenerative_and_chip_thickness(self):
        s = milling_time_domain_simulation_nutshell()
        assert "regenerative" in s.lower()
        assert "chip" in s.lower() or "h =" in s

    def test_mentions_steps_rev_and_surf(self):
        s = milling_time_domain_simulation_nutshell()
        assert "steps" in s.lower() or "surf" in s.lower()

    def test_mentions_tap_test_or_modal(self):
        s = milling_time_domain_simulation_nutshell()
        assert "tap" in s.lower() or "modal" in s.lower() or "natural" in s.lower()


class TestMillingSurfUpdateRuleNutshell:
    def test_mentions_surf_and_in_cut(self):
        s = milling_surf_update_rule_nutshell()
        assert "surf" in s.lower()
        assert "cut" in s.lower()

    def test_summary_includes_surf_update_rule(self):
        d = milling_dynamics_summary()
        assert "surf_update_rule" in d
        assert "ft" in d["surf_update_rule"] or "sin" in d["surf_update_rule"]


class TestMillingCuttingForces:
    """Tangential/normal forces (Eq. 4.8) and projection to x, y (Eqs. 4.11, 4.12)."""

    def test_tangential_force_zero_when_no_cut(self):
        assert tangential_force_N(750.0, 5.0, 0.0) == 0.0
        assert tangential_force_N(750.0, 5.0, -0.01) == 0.0

    def test_tangential_force_positive(self):
        # Ft = kt * b * h = 750 * 5 * 0.1 = 375 N
        assert tangential_force_N(750.0, 5.0, 0.1) == pytest.approx(375.0)

    def test_normal_force_zero_when_no_cut(self):
        assert normal_force_N(250.0, 5.0, 0.0) == 0.0
        assert normal_force_N(250.0, 5.0, -0.01) == 0.0

    def test_normal_force_positive(self):
        # Fn = kn * b * h = 250 * 5 * 0.1 = 125 N
        assert normal_force_N(250.0, 5.0, 0.1) == pytest.approx(125.0)

    def test_cutting_forces_xy_at_phi_zero(self):
        # φ = 0: Fx = Ft*1 + Fn*0 = Ft, Fy = Ft*0 - Fn*1 = -Fn
        Ft, Fn = 100.0, 50.0
        Fx, Fy = cutting_forces_xy_N(Ft, Fn, 0.0)
        assert Fx == pytest.approx(100.0)
        assert Fy == pytest.approx(-50.0)

    def test_cutting_forces_xy_at_phi_90_deg(self):
        # φ = π/2: Fx = Ft*0 + Fn*1 = Fn, Fy = Ft*1 - Fn*0 = Ft
        Ft, Fn = 100.0, 50.0
        Fx, Fy = cutting_forces_xy_N(Ft, Fn, math.pi / 2)
        assert Fx == pytest.approx(50.0)
        assert Fy == pytest.approx(100.0)

    def test_cutting_forces_xy_resultant_magnitude(self):
        # |F| = sqrt(Fx^2 + Fy^2) = sqrt(Ft^2 + Fn^2) for any φ
        Ft, Fn = 80.0, 60.0
        Fx, Fy = cutting_forces_xy_N(Ft, Fn, math.radians(45))
        assert (Fx**2 + Fy**2) ** 0.5 == pytest.approx((Ft**2 + Fn**2) ** 0.5)
