"""
Two degree of freedom (2DOF) free vibration: chain-type spring-mass-damper model.

The system has two masses in a vertical chain. Top mass m1 is connected to ground
by k1, c1 and to m2 by k2, c2. Bottom mass m2 is connected only to m1 (k2, c2).

Equations of motion (free body diagrams; sum of forces = 0):

  Top mass (m1):
    m1·ẍ1 + (c1 + c2)·ẋ1 + (k1 + k2)·x1 − c2·ẋ2 − k2·x2 = 0

  Bottom mass (m2):
    m2·ẍ2 − c2·ẋ1 − k2·x1 + c2·ẋ2 + k2·x2 = 0

Matrix form:  M·ẍ + C·ẋ + K·x = 0  with  x = [x1, x2]ᵀ

  M = [[m1,  0 ],     K = [[k1+k2, −k2],     C = [[c1+c2, −c2],
       [0,   m2]]          [−k2,   k2 ]]          [−c2,   c2 ]]

For undamped free vibration (c1 = c2 = 0), natural frequencies ω₁, ω₂ (rad/s) are
the square roots of the eigenvalues of M⁻¹K (or the roots of det(K − λM) = 0).

Mode shapes (eigenvectors):
  Substitute ω₁² and ω₂² into the top row of (K − ω²M)·X = 0 (either row gives the
  same solution once the determinant is zero). Top row:
    (k1 + k2 − ω²·m1)·X1 − k2·X2 = 0  ⇒  X2/X1 = (k1 + k2 − ω²·m1) / k2.
  Normalizing to coordinate x1: X1/X1 = 1 and X2/X1 as above. So
    ψ₁ = [1, (k1+k2 − ω₁²·m1)/k2]ᵀ,   ψ₂ = [1, (k1+k2 − ω₂²·m1)/k2]ᵀ.
  With ω₁ < ω₂ and normalization to x1, the first mode has the form [1, a] with
  a > 0 (masses vibrate in phase). The second mode has [1, a] with a < 0 (masses
  vibrate out of phase). The system vibrates in a linear combination of both
  mode shapes depending on initial conditions. For machining dynamics, normalization
  is often chosen at the tool point (e.g. x1 or x2).

Modal analysis:
  The modal matrix [P] has columns equal to the mode shapes. With normalization to
  x2, the second row of P is [1, 1], so x2 = q1 + q2 (local response at x2 is the sum
  of modal contributions—useful for modal fitting of FRFs). Orthogonality of
  eigenvectors diagonalizes M and K in modal coordinates: Mq = Pᵀ M P, Kq = Pᵀ K P,
  giving uncoupled equations Mq·q̈ + Kq·q = 0 (two separate SDOF systems). Initial
  conditions in modal coordinates: {q} = P⁻¹{x}, so q0 = P⁻¹·x0 and q̇0 = P⁻¹·ẋ0.
  Free response form: q_j(t) = (q̇_j0/ω_j)·sin(ω_j·t) + q_j0·cos(ω_j·t). Transform back
  to local coordinates: x(t) = P·q(t).

2DOF Forced vibration (harmonic excitation):
  With external harmonic forces at x1 and x2, by linearity (superposition) consider
  each force separately and sum. Example: F = [0, f2]ᵀ with f2·e^(iωt) at x2.
  Equations: M·ẍ + C·ẋ + K·x = F·e^(iωt). Assuming steady-state x = X·e^(iωt):
    (−ω²M + iωC + K)·X·e^(iωt) = F·e^(iωt)  ⇒  (K − ω²M + iωC)·X = F.
  Two methods for steady-state response: (1) modal analysis (requires proportional
  damping); (2) complex matrix inversion: solve (K − ω²M + iωC)·X = F for X (no
  restriction on damping). The dynamic stiffness is Z(ω) = K − ω²M + iωC; then X = Z⁻¹·F.

  Modal analysis (proportional damping C = α·M + β·K):
  Eigensolution from undamped system; P = [ψ₁ ψ₂] normalized to force location (e.g. x2).
  Mq = Pᵀ M P, Cq = Pᵀ C P, Kq = Pᵀ K P (diagonal). Modal force R = Pᵀ F; for F = [0, f2]ᵀ
  and P with second row [1, 1], R = [f2, f2]ᵀ. Uncoupled modal equations: mqj·q̈j + cqj·q̇j + kqj·qj = Rj.
  Modal FRF (SDOF): Qj/Rj = (1/kqj)·((1−rj²) − i·2ζqj·rj) / ((1−rj²)² + (2ζqj·rj)²), rj = ω/ωnj,
  ζqj = cqj/(2√(kqj·mqj)). Transform to local: X = P·Q. Direct FRF (measure at force): X2/F2 = Q1/R1 + Q2/R2.
  Cross FRF: X1/F2 = p1·Q1/R1 + p2·Q2/R2 (p1, p2 = first row of P when normalized to x2).
"""
from __future__ import annotations

import math

import numpy as np


def mass_matrix_chain(m1: float, m2: float) -> np.ndarray:
    """Mass matrix M for the chain (2×2). M = diag(m1, m2)."""
    return np.array([[m1, 0.0], [0.0, m2]], dtype=float)


def stiffness_matrix_chain(k1: float, k2: float) -> np.ndarray:
    """Stiffness matrix K for the chain (2×2)."""
    return np.array(
        [
            [k1 + k2, -k2],
            [-k2, k2],
        ],
        dtype=float,
    )


def damping_matrix_chain(c1: float, c2: float) -> np.ndarray:
    """Damping matrix C for the chain (2×2)."""
    return np.array(
        [
            [c1 + c2, -c2],
            [-c2, c2],
        ],
        dtype=float,
    )


def matrices_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float = 0.0,
    c2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mass, damping, and stiffness matrices for the chain-type 2DOF system.

    Args:
        m1, m2: Masses (kg).
        k1, k2: Spring stiffnesses (N/m).
        c1, c2: Damping coefficients (N·s/m); default 0.

    Returns:
        (M, C, K) each 2×2.
    """
    M = mass_matrix_chain(m1, m2)
    C = damping_matrix_chain(c1, c2)
    K = stiffness_matrix_chain(k1, k2)
    return M, C, K


def natural_frequencies_rad_s(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
) -> tuple[float, float]:
    """
    Undamped natural frequencies ω₁, ω₂ (rad/s) for the chain 2DOF system.

    Solves det(K − λM) = 0 for λ = ω²; returns (ω₁, ω₂) with ω₁ ≤ ω₂.

    Args:
        m1, m2: Masses in kg.
        k1, k2: Stiffnesses in N/m.

    Returns:
        (omega1, omega2) in rad/s.
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("masses must be positive")
    if k1 < 0 or k2 <= 0:
        raise ValueError("stiffnesses must be positive (k2 > 0)")
    M, _, K = matrices_chain(m1, m2, k1, k2, 0.0, 0.0)
    # Generalized eigenvalue problem K·v = λ·M·v  =>  M⁻¹K·v = λ·v.
    # M⁻¹K is not symmetric (unless M ∝ I), so use eig not eigvalsh.
    MinvK = np.linalg.solve(M, K)
    eigvals = np.linalg.eigvals(MinvK)
    eigvals = np.real(eigvals)
    if eigvals[0] < 0 or eigvals[1] < 0:
        raise ValueError("system has negative eigenvalue (unstable or invalid parameters)")
    omega1 = math.sqrt(float(np.min(eigvals)))
    omega2 = math.sqrt(float(np.max(eigvals)))
    return (omega1, omega2)


def natural_frequencies_hz(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
) -> tuple[float, float]:
    """
    Undamped natural frequencies f₁, f₂ (Hz) for the chain 2DOF system.

    Returns (f₁, f₂) with f₁ ≤ f₂, where f = ω/(2π).
    """
    o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
    return (o1 / (2.0 * math.pi), o2 / (2.0 * math.pi))


def mode_shape_ratio_x2_over_x1(
    omega_sq: float,
    m1: float,
    k1: float,
    k2: float,
) -> float:
    """
    Ratio X2/X1 for one mode from the top-row equation (K − ω²M)·X = 0.

    (k1 + k2 − ω²·m1)·X1 − k2·X2 = 0  ⇒  X2/X1 = (k1 + k2 − ω²·m1) / k2.

    Args:
        omega_sq: ω² in (rad/s)² for this mode.
        m1, k1, k2: Chain parameters (kg, N/m).

    Returns:
        X2/X1 for the mode at that natural frequency.
    """
    return (k1 + k2 - omega_sq * m1) / k2


def mode_shape_ratio_x1_over_x2(
    omega_sq: float,
    m2: float,
    k2: float,
) -> float:
    """
    Ratio X1/X2 for one mode from the bottom-row equation (K − ω²M)·X = 0.

    −k2·X1 + (k2 − ω²·m2)·X2 = 0  ⇒  X1/X2 = (k2 − ω²·m2) / k2.

    Used when normalizing mode shapes to coordinate x2 (e.g. tool point).

    Args:
        omega_sq: ω² in (rad/s)² for this mode.
        m2, k2: Chain parameters (kg, N/m).

    Returns:
        X1/X2 for the mode at that natural frequency.
    """
    return (k2 - omega_sq * m2) / k2


def mode_shapes_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    normalize: str = "first",
) -> np.ndarray:
    """
    Mode shapes (eigenvectors) for the undamped chain 2DOF system.

    Columns are the two mode shapes corresponding to ω₁ and ω₂. Each eigenvector
    satisfies (K − ω²M)·v = 0. When normalize="first", mode shapes are normalized
    to coordinate x1 (X1/X1 = 1, X2/X1 from the top-row equation). First mode (ω₁):
    second component > 0 (masses in phase). Second mode (ω₂): second component < 0
    (masses out of phase).

    Args:
        m1, m2: Masses in kg.
        k1, k2: Stiffnesses in N/m.
        normalize: "first" = normalize to x1 (first coordinate); "norm" = unit norm.

    Returns:
        Shape (2, 2): column 0 = mode 1 (ω₁), column 1 = mode 2 (ω₂).
    """
    M, _, K = matrices_chain(m1, m2, k1, k2, 0.0, 0.0)
    MinvK = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(MinvK)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    if eigvals[0] < 0 or eigvals[1] < 0:
        raise ValueError("system has negative eigenvalue")
    # Sort by eigenvalue ascending (ω₁ ≤ ω₂)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if normalize == "first":
        for j in range(2):
            scale = eigvecs[0, j]
            if abs(scale) < 1e-12:
                scale = 1.0
            eigvecs[:, j] = eigvecs[:, j] / scale
    elif normalize == "norm":
        for j in range(2):
            n = np.linalg.norm(eigvecs[:, j])
            if n > 1e-12:
                eigvecs[:, j] = eigvecs[:, j] / n
    return eigvecs.astype(float)


def mode_shapes_chain_normalized_to_x1(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """
    Mode shapes normalized to coordinate x1 using the explicit ratio X2/X1.

    ψ₁ = [1, (k1+k2 − ω₁²·m1)/k2]ᵀ,  ψ₂ = [1, (k1+k2 − ω₂²·m1)/k2]ᵀ.
    First mode: second component > 0 (in phase). Second mode: second component < 0
    (out of phase). Useful when the coordinate of interest or force application
    is x1 (e.g. tool point in machining dynamics).

    Returns:
        Shape (2, 2): column 0 = mode 1 (ω₁), column 1 = mode 2 (ω₂).
    """
    o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
    r1 = mode_shape_ratio_x2_over_x1(o1 * o1, m1, k1, k2)
    r2 = mode_shape_ratio_x2_over_x1(o2 * o2, m1, k1, k2)
    return np.array([[1.0, 1.0], [r1, r2]], dtype=float)


def mode_shapes_chain_normalized_to_x2(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """
    Mode shapes normalized to coordinate x2 (second coordinate = 1).

    ψ₁ = [(k2−ω₁²·m2)/k2, 1]ᵀ,  ψ₂ = [(k2−ω₂²·m2)/k2, 1]ᵀ.
    The modal matrix P then has second row [1, 1], so x2 = q1 + q2 (local x2
    is the sum of modal contributions—useful for modal fitting of FRFs).

    Returns:
        Shape (2, 2): column 0 = mode 1 (ω₁), column 1 = mode 2 (ω₂).
    """
    o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
    r1 = mode_shape_ratio_x1_over_x2(o1 * o1, m2, k2)
    r2 = mode_shape_ratio_x1_over_x2(o2 * o2, m2, k2)
    return np.array([[r1, r2], [1.0, 1.0]], dtype=float)


def modal_matrix_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    normalize_to: str = "x1",
) -> np.ndarray:
    """
    Modal matrix [P] whose columns are the mode shapes.

    P = [ψ₁ ψ₂]. Choice of normalization: "x1" (first coordinate = 1) or "x2"
    (second coordinate = 1). With "x2", the second row of P is [1, 1], so
    x2 = q1 + q2.

    Returns:
        P shape (2, 2).
    """
    if normalize_to == "x1":
        return mode_shapes_chain_normalized_to_x1(m1, m2, k1, k2)
    if normalize_to == "x2":
        return mode_shapes_chain_normalized_to_x2(m1, m2, k1, k2)
    raise ValueError("normalize_to must be 'x1' or 'x2'")


def modal_mass_stiffness_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    normalize_to: str = "x1",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modal mass and stiffness matrices Mq, Kq (diagonalized via P).

    Mq = Pᵀ M P,  Kq = Pᵀ K P. Off-diagonals are zero in exact arithmetic; small
    residuals may remain due to round-off. The two equations of motion in modal
    coordinates are uncoupled: Mq·q̈ + Kq·q = 0 (two separate SDOF systems).

    Returns:
        (Mq, Kq) each 2×2 (diagonal).
    """
    M, _, K = matrices_chain(m1, m2, k1, k2, 0.0, 0.0)
    P = modal_matrix_chain(m1, m2, k1, k2, normalize_to=normalize_to)
    Mq = P.T @ M @ P
    Kq = P.T @ K @ P
    return (Mq.astype(float), Kq.astype(float))


def modal_mass_damping_stiffness_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float,
    c2: float,
    normalize_to: str = "x2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modal mass, damping, and stiffness Mq, Cq, Kq (diagonal) for forced-vibration modal analysis.

    Mq = Pᵀ M P, Cq = Pᵀ C P, Kq = Pᵀ K P. With proportional damping (C = α·M + β·K),
    Cq is diagonal and the modal equations mqj·q̈j + cqj·q̇j + kqj·qj = Rj are uncoupled.

    Returns:
        (Mq, Cq, Kq) each 2×2 (diagonal).
    """
    M, C, K = matrices_chain(m1, m2, k1, k2, c1, c2)
    P = modal_matrix_chain(m1, m2, k1, k2, normalize_to=normalize_to)
    Mq = P.T @ M @ P
    Cq = P.T @ C @ P
    Kq = P.T @ K @ P
    return (Mq.astype(float), Cq.astype(float), Kq.astype(float))


def proportional_damping_chain(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float,
    c2: float,
) -> tuple[float, float] | None:
    """
    If C = α·M + β·K for some α, β, return (α, β); otherwise None.

    Proportional damping allows the undamped eigensolution and diagonal Cq in modal coordinates.
    """
    M, C, K = matrices_chain(m1, m2, k1, k2, c1, c2)
    # C = α*M + β*K gives 4 equations in 2 unknowns; use first two (e.g. [0,0] and [1,0])
    # C[0,0] = α*M[0,0] + β*K[0,0], C[0,1] = α*M[0,1] + β*K[0,1]. M[0,1]=0 so C[0,1] = β*K[0,1].
    if abs(K[0, 1]) < 1e-20:
        return None
    beta = C[0, 1] / K[0, 1]
    alpha = (C[0, 0] - beta * K[0, 0]) / M[0, 0] if M[0, 0] != 0 else 0.0
    C_check = alpha * M + beta * K
    if np.allclose(C, C_check):
        return (alpha, beta)
    return None


def modal_force_transform(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Modal force vector R = Pᵀ·F. For F = [0, f2]ᵀ and P with second row [1, 1], R = [f2, f2]ᵀ."""
    return (P.T @ np.asarray(F, dtype=float).ravel()).astype(float)


def modal_damping_ratio(cq: float, kq: float, mq: float) -> float:
    """Modal damping ratio ζq = cq / (2·√(kq·mq))."""
    if mq <= 0 or kq <= 0:
        raise ValueError("mq and kq must be positive")
    return cq / (2.0 * math.sqrt(kq * mq))


def sdof_receptance_modal(
    omega: float,
    omega_n: float,
    kq: float,
    zeta_q: float,
) -> complex:
    """
    SDOF receptance (displacement/force) Q/R for one modal coordinate under harmonic force R·e^(iωt).

    Q/R = (1/kq)·((1−r²) − i·2ζ·r) / ((1−r²)² + (2ζ·r)²),  r = ω/ωn.

    Returns:
        Complex receptance (m/N or similar).
    """
    r = omega / omega_n if omega_n > 0 else 0.0
    den = (1.0 - r * r) ** 2 + (2.0 * zeta_q * r) ** 2
    if den < 1e-30:
        return complex(0.0, 0.0)  # at resonance, limit depends on damping
    numer = (1.0 - r * r) - 1j * (2.0 * zeta_q * r)
    return (numer / den) / kq


def frf_direct_modal_chain(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float,
    c2: float,
    force_at: str = "x2",
) -> complex:
    """
    Direct FRF (measurement at force location) via modal analysis: X2/F2 = Q1/R1 + Q2/R2.

    Requires proportional damping. With P normalized to x2, R1 = R2 = F2, so direct FRF
    is the sum of the modal receptances. Returns complex X_meas/F (e.g. m/N).

    Args:
        omega: Excitation frequency (rad/s).
        force_at: "x2" = force at coordinate 2 (only supported case for now).

    Returns:
        Complex direct FRF (displacement at force location / force magnitude).
    """
    if force_at != "x2":
        raise ValueError("force_at must be 'x2'")
    Mq, Cq, Kq = modal_mass_damping_stiffness_chain(m1, m2, k1, k2, c1, c2, normalize_to="x2")
    o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
    z1 = modal_damping_ratio(float(Cq[0, 0]), float(Kq[0, 0]), float(Mq[0, 0]))
    z2 = modal_damping_ratio(float(Cq[1, 1]), float(Kq[1, 1]), float(Mq[1, 1]))
    H1 = sdof_receptance_modal(omega, o1, float(Kq[0, 0]), z1)
    H2 = sdof_receptance_modal(omega, o2, float(Kq[1, 1]), z2)
    return H1 + H2


def frf_cross_modal_chain(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float,
    c2: float,
    force_at: str = "x2",
) -> complex:
    """
    Cross FRF (measurement away from force) via modal analysis: X1/F2 = p1·Q1/R1 + p2·Q2/R2.

    p1, p2 are the first row of P (mode shape values at x1 when normalized to x2).
    Returns complex X1/F2 (m/N).

    Args:
        force_at: "x2" = force at coordinate 2.

    Returns:
        Complex cross FRF.
    """
    if force_at != "x2":
        raise ValueError("force_at must be 'x2'")
    P = modal_matrix_chain(m1, m2, k1, k2, normalize_to="x2")
    p1, p2 = float(P[0, 0]), float(P[0, 1])
    Mq, Cq, Kq = modal_mass_damping_stiffness_chain(m1, m2, k1, k2, c1, c2, normalize_to="x2")
    o1, o2 = natural_frequencies_rad_s(m1, m2, k1, k2)
    z1 = modal_damping_ratio(float(Cq[0, 0]), float(Kq[0, 0]), float(Mq[0, 0]))
    z2 = modal_damping_ratio(float(Cq[1, 1]), float(Kq[1, 1]), float(Mq[1, 1]))
    H1 = sdof_receptance_modal(omega, o1, float(Kq[0, 0]), z1)
    H2 = sdof_receptance_modal(omega, o2, float(Kq[1, 1]), z2)
    return p1 * H1 + p2 * H2


def initial_conditions_modal(
    x0: np.ndarray,
    dx0: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform initial conditions from local to modal coordinates.

    {q} = P⁻¹{x}, so q0 = P⁻¹·x0 and q̇0 = P⁻¹·ẋ0.

    Args:
        x0: Initial displacement (2,) in local coordinates.
        dx0: Initial velocity (2,) in local coordinates.
        P: Modal matrix (2×2).

    Returns:
        (q0, dq0) each shape (2,).
    """
    Pinv = np.linalg.inv(P)
    q0 = Pinv @ np.asarray(x0, dtype=float).ravel()
    dq0 = Pinv @ np.asarray(dx0, dtype=float).ravel()
    return (q0, dq0)


def free_response_modal_single(
    omega_n: float,
    q0_j: float,
    dq0_j: float,
    t: np.ndarray,
) -> np.ndarray:
    """
    Undamped free response for one modal coordinate: q_j(t) = (q̇_j0/ω_n)·sin(ω_n·t) + q_j0·cos(ω_n·t).

    Args:
        omega_n: Natural frequency for this mode (rad/s).
        q0_j, dq0_j: Initial displacement and velocity for this mode.
        t: Time vector (s).

    Returns:
        q_j(t) shape (len(t),).
    """
    t = np.asarray(t, dtype=float)
    return (dq0_j / omega_n) * np.sin(omega_n * t) + q0_j * np.cos(omega_n * t)


def free_response_modal(
    t: np.ndarray,
    q0: np.ndarray,
    dq0: np.ndarray,
    omega1: float,
    omega2: float,
) -> np.ndarray:
    """
    Undamped free response in modal coordinates q(t).

    q1(t) = (q̇10/ω1)·sin(ω1·t) + q10·cos(ω1·t), and similarly for q2.
    Then x(t) = P·q(t) for local coordinates.

    Args:
        t: Time vector (s), shape (N,).
        q0, dq0: Initial modal displacement and velocity (2,).
        omega1, omega2: Natural frequencies (rad/s).

    Returns:
        q shape (2, N): row 0 = q1(t), row 1 = q2(t).
    """
    t = np.asarray(t, dtype=float)
    q1 = free_response_modal_single(omega1, q0[0], dq0[0], t)
    q2 = free_response_modal_single(omega2, q0[1], dq0[1], t)
    return np.stack([q1, q2], axis=0)


def dynamic_stiffness_chain(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float = 0.0,
    c2: float = 0.0,
) -> np.ndarray:
    """
    Dynamic stiffness matrix Z(ω) = K − ω²M + iωC for harmonic forcing e^(iωt).

    Steady-state response satisfies Z·X = F, so X = Z⁻¹·F.

    Args:
        omega: Excitation frequency (rad/s).
        m1, m2, k1, k2, c1, c2: Chain parameters.

    Returns:
        Z shape (2, 2), complex.
    """
    M, C, K = matrices_chain(m1, m2, k1, k2, c1, c2)
    Z = K - (omega * omega) * M + 1j * omega * C
    return Z.astype(complex)


def forced_response_complex_chain(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    F: np.ndarray,
    c1: float = 0.0,
    c2: float = 0.0,
) -> np.ndarray:
    """
    Steady-state complex displacement X for harmonic forcing F·e^(iωt) (complex matrix inversion).

    (K − ω²M + iωC)·X = F  ⇒  X = Z⁻¹·F. No restriction on damping (unlike modal
    analysis, which requires proportional damping). Physical displacement is
    x(t) = Re(X·e^(iωt)).

    Args:
        omega: Excitation frequency (rad/s).
        m1, m2, k1, k2: Chain parameters (kg, N/m).
        F: Force vector (2,) complex or real; e.g. [0, f2] for force at x2.
        c1, c2: Damping (N·s/m); default 0.

    Returns:
        X shape (2,) complex; |X| = amplitude, arg(X) = phase (rad).
    """
    Z = dynamic_stiffness_chain(omega, m1, m2, k1, k2, c1, c2)
    F = np.asarray(F, dtype=complex).ravel()
    if F.shape != (2,):
        raise ValueError("F must have length 2")
    X = np.linalg.solve(Z, F)
    return X


def frequency_response_matrix_chain(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    c1: float = 0.0,
    c2: float = 0.0,
) -> np.ndarray:
    """
    Frequency response matrix H(ω) = Z(ω)⁻¹ such that X = H·F.

    Column j of H is the complex displacement response per unit force at coordinate j
    (e.g. H[:, 1] gives [X1, X2]ᵀ when F = [0, 1]ᵀ at x2).

    Returns:
        H shape (2, 2), complex.
    """
    Z = dynamic_stiffness_chain(omega, m1, m2, k1, k2, c1, c2)
    return np.linalg.inv(Z)


def local_response_from_modal(q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Transform modal coordinates to local coordinates: x = P·q.

    Args:
        q: Modal displacements (2, N) or (2,).
        P: Modal matrix (2×2).

    Returns:
        x same shape as q (2, N) or (2,).
    """
    return (P @ np.asarray(q, dtype=float)).astype(float)


# Nutshell text for docs or UI (re-exported from tap_testing.docs.twodof)
from tap_testing.docs.twodof import (
    twodof_forced_vibration_nutshell,
    twodof_free_vibration_nutshell,
    twodof_modal_analysis_nutshell,
    twodof_modal_forced_frf_nutshell,
    twodof_mode_shapes_nutshell,
)
