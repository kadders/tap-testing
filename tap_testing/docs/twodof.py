"""
2DOF vibration (chain model): nutshell summaries for documentation or UI.

Free vibration, mode shapes, modal analysis, forced vibration, modal FRF.
See tap_testing.twodof for computation (frf_direct_modal_chain, etc.).
"""
from __future__ import annotations

__all__ = [
    "twodof_free_vibration_nutshell",
    "twodof_mode_shapes_nutshell",
    "twodof_modal_analysis_nutshell",
    "twodof_forced_vibration_nutshell",
    "twodof_modal_forced_frf_nutshell",
]


def twodof_free_vibration_nutshell() -> str:
    """Short summary of 2DOF free vibration (chain model) for documentation or UI."""
    return (
        "2DOF free vibration: chain of two masses (m1, m2) with springs (k1, k2) and optional "
        "dampers (c1, c2). Equations: m1·ẍ1+(c1+c2)·ẋ1+(k1+k2)·x1−c2·ẋ2−k2·x2=0 and "
        "m2·ẍ2−c2·ẋ1−k2·x1+c2·ẋ2+k2·x2=0. Two natural frequencies ω₁, ω₂ and two mode shapes."
    )


def twodof_mode_shapes_nutshell() -> str:
    """
    Short summary of 2DOF mode shapes: ratio X2/X1 and in-phase / out-of-phase.

    Mode shapes are found by substituting ω₁² and ω₂² into the top row of
    (K − ω²M)·X = 0; normalizing to x1 gives ψ = [1, (k1+k2−ω²·m1)/k2]. First
    mode: in phase (second component > 0). Second mode: out of phase (second < 0).
    """
    return (
        "2DOF mode shapes: from (K−ω²M)·X=0, X2/X1 = (k1+k2−ω²·m1)/k2; normalize to x1. "
        "First mode (ω₁): [1, a] with a > 0 (masses in phase). Second mode (ω₂): [1, a] "
        "with a < 0 (masses out of phase). Often normalize to tool point for machining."
    )


def twodof_modal_analysis_nutshell() -> str:
    """
    Short summary of 2DOF modal analysis: diagonalization and uncoupled equations.

    Modal matrix P (columns = mode shapes) diagonalizes M and K: Mq = Pᵀ M P,
    Kq = Pᵀ K P. Equations uncouple to two SDOF systems. q0 = P⁻¹·x0, q̇0 = P⁻¹·ẋ0.
    Response q_j(t) = (q̇_j0/ω_j)·sin(ω_j·t) + q_j0·cos(ω_j·t); x = P·q. Normalizing
    to x2 gives x2 = q1 + q2 (sum of modal contributions—useful for FRF fitting).
    """
    return (
        "2DOF modal analysis: P = [ψ₁ ψ₂]. Mq = Pᵀ M P, Kq = Pᵀ K P (diagonal); "
        "uncoupled Mq·q̈ + Kq·q = 0. q0 = P⁻¹·x0; q_j(t) = (q̇_j0/ω_j)·sin(ω_j·t) + q_j0·cos(ω_j·t); "
        "x = P·q. Normalize to x2 ⇒ x2 = q1 + q2 (for modal FRF fitting)."
    )


def twodof_forced_vibration_nutshell() -> str:
    """
    Short summary of 2DOF forced vibration (harmonic excitation).

    M·ẍ + C·ẋ + K·x = F·e^(iωt). Assume x = X·e^(iωt) ⇒ (K − ω²M + iωC)·X = F.
    Two methods: (1) modal analysis (proportional damping); (2) complex matrix
    inversion X = Z⁻¹·F, Z = K − ω²M + iωC (no restriction on damping).
    """
    return (
        "2DOF forced vibration: M·ẍ+C·ẋ+K·x = F·e^(iωt); assume x = X·e^(iωt) ⇒ "
        "(K−ω²M+iωC)·X = F. Modal analysis (proportional damping) or complex "
        "matrix inversion X = Z⁻¹·F (any damping). x(t) = Re(X·e^(iωt))."
    )


def twodof_modal_forced_frf_nutshell() -> str:
    """
    Short summary of direct and cross FRFs by modal analysis.

    With proportional damping and P normalized to force location (x2): R = Pᵀ·F = [f2, f2]ᵀ.
    Modal receptance Qj/Rj = (1/kqj)·((1−rj²)−i·2ζqj·rj)/((1−rj²)²+(2ζqj·rj)²). Direct FRF
    X2/F2 = Q1/R1 + Q2/R2 (sum of modal contributions). Cross FRF X1/F2 = p1·Q1/R1 + p2·Q2/R2.
    """
    return (
        "2DOF modal FRF: R = Pᵀ·F (e.g. [f2, f2]ᵀ for force at x2). Direct FRF X2/F2 = Q1/R1 + Q2/R2; "
        "cross FRF X1/F2 = p1·Q1/R1 + p2·Q2/R2. Modal receptance SDOF form with r = ω/ωn, ζq = cq/(2√(kq·mq))."
    )
