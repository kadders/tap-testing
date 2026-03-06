"""
Modal / system identification: nutshell summaries for documentation or UI.

Peak-picking, error sources, measured FRF, model definition. See tap_testing.modal_fit
for computation (modal_params_from_peak_pick, peak_pick_direct_frf_2dof, etc.).
"""
from __future__ import annotations

__all__ = [
    "system_identification_nutshell",
    "modal_fit_error_sources_nutshell",
    "measured_frf_all_modes_nutshell",
    "model_definition_nutshell",
]


def system_identification_nutshell() -> str:
    """Short summary of system identification (modal fitting) for documentation or UI."""
    return (
        "System identification: measure FRFs on the assembly; mass/damping/stiffness are unknown. "
        "Peak-picking uses the direct FRF: natural frequencies from min Im(H), damping from real-part "
        "bandwidth (ω4−ω3)/(2ωn1), modal stiffness from peak A = 1/(2·ζ·kq). Then mq = kq/ωn², cq = 2ζ√(kq·mq). "
        "Assumes proportional damping; modal matrices Mq, Cq, Kq suffice for time-domain simulations."
    )


def modal_fit_error_sources_nutshell() -> str:
    """
    Short note on why fitted modal parameters can differ from the true system.

    For documentation or UI when explaining fit quality or advising on measurement setup.
    """
    return (
        "Errors between true and fit parameters arise from (1) limited resolution in the "
        "measurement (frequency and magnitude), (2) the simplifications used to create "
        "the fitting rules (e.g. single-mode peak formulas, proportional damping assumption), "
        "and (3) the influence of modes outside the measured frequency range (the measured FRF "
        "includes all modes; out-of-band modes still affect the in-band data)."
    )


def measured_frf_all_modes_nutshell() -> str:
    """
    Short note that a measured FRF includes all modes; out-of-band modes still influence in-band data.

    For documentation or UI when interpreting fits or choosing bandwidth.
    """
    return (
        "A measured FRF includes the effect of all modes. Some modes may be outside the "
        "measured frequency range; they still influence the data in the measured range. "
        "Fitting a limited number of modes (e.g. 2DOF) can therefore be biased by out-of-band modes."
    )


def model_definition_nutshell() -> str:
    """
    Short summary of model definition from direct and cross FRFs after peak-picking.

    Mode shapes from cross/direct peak ratios; P then gives local M, C, K and chain params.
    """
    return (
        "Model definition: after peak-picking (Mq, Cq, Kq), use direct and cross FRF peaks. "
        "Normalize to x2: P = [[p1, p2], [1, 1]] with p1 = C/A, p2 = D/B (cross/direct peak per mode). "
        "Local matrices M = Pᵀ Mq P⁻¹, C = Pᵀ Cq P⁻¹, K = Pᵀ Kq P⁻¹; then read off chain m1, m2, k1, k2, c1, c2. "
        "For three modes, P is 3×3 and at least two cross FRFs are needed."
    )
