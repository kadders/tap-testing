"""
System identification: modal fitting (peak-picking) from measured FRFs.

We typically measure the direct and cross FRFs on a tool-holder assembly; the mass,
damping, and stiffness matrices are not known. A "peak-picking" method uses the real
and imaginary parts of the direct FRF to identify modal parameters and populate the
2×2 modal matrices Mq, Cq, Kq. This approach assumes proportional damping and works
well when modes are not closely spaced (reasonable fits are still possible for
moderately close modes).

Peak-picking steps (direct FRF, two modes in bandwidth):
  1. Natural frequencies ωn1, ωn2: from the minima of the imaginary part (frequencies
     at the two negative peaks).
  2. Modal damping ratios: ζq1 = (ω4 − ω3) / (2·ωn1), ζq2 = (ω6 − ω5) / (2·ωn2), where
     ω3, ω4 bracket the first mode and ω5, ω6 bracket the second on the real part
     (e.g. zero-crossing frequencies or half-power bandwidth).
  3. Modal stiffness: A = |min Im(H)| at mode 1 ⇒ kq1 = 1/(2·ζq1·A). Similarly
     kq2 = 1/(2·ζq2·B) from peak B at mode 2.
  4. Modal mass: mq1 = kq1/ωn1², mq2 = kq2/ωn2².
  5. Modal damping: cq1 = 2·ζq1·√(kq1·mq1), cq2 = 2·ζq2·√(kq2·mq2).

With Mq, Cq, Kq we can run time-domain simulations or reconstruct FRFs in local
coordinates if the modal matrix P (mode shapes) is also identified (e.g. from cross FRF).

Model definition (from direct and cross FRFs):
  After peak-picking we have Mq, Cq, Kq. Next: use the measured direct and cross FRFs
  to find the mode shapes and build P. For a 2DOF model we normalize to one coordinate
  (e.g. x2); then P = [[p1, p2], [1, 1]] and we need only one cross FRF to get p1, p2.
  Denote A, B = (magnitudes of) the negative imaginary peaks of the direct FRF at ωn1, ωn2,
  and C, D = the corresponding peak magnitudes of the cross FRF (e.g. X1/F2). Then
  p1 = C/A and p2 = D/B (ratio of cross to direct peak in each mode). Once P is defined,
  local matrices are obtained by transforming from modal to local coordinates:
  M = Pᵀ Mq P⁻¹,  C = Pᵀ Cq P⁻¹,  K = Pᵀ Kq P⁻¹.
  For the preselected chain-type lumped parameter model, M = diag(m1, m2) and K, C have
  the standard chain form; we can read off m1, m2, k1, k2, c1, c2. If the direct FRF has
  three modes to model, P is 3×3 and we need at least two cross FRFs to get the two
  ratios for each 3×1 mode shape.

All modes in the measured FRF:
  A measured FRF includes the effect of all modes. Some modes may lie outside the
  measured frequency range; they still influence the data within that range (e.g. as
  a residual stiffness or mass line). When we fit a limited number of modes (e.g. 2DOF),
  the fitted parameters can be biased by these out-of-band modes.

Error sources:
  The errors between the true and fit parameters arise from (1) limited resolution
  in the measurement (frequency and magnitude), (2) the simplifications used to create
  the fitting rules (e.g. single-mode peak formulas, proportional damping, half-power
  or zero-crossing bandwidth), and (3) the influence of modes outside the measured
  frequency range. Finer frequency resolution and more sophisticated fitting (e.g. curve
  fitting, polyreference, residual terms) can reduce these errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ModalFitResult2DOF:
    """Result of peak-picking modal fit from a direct FRF (two modes)."""

    omega_n1: float  # rad/s
    omega_n2: float  # rad/s
    zeta_q1: float
    zeta_q2: float
    kq1: float  # N/m
    kq2: float  # N/m
    mq1: float  # kg
    mq2: float  # kg
    cq1: float  # N·s/m
    cq2: float  # N·s/m
    Mq: np.ndarray  # (2,2) diagonal
    Cq: np.ndarray  # (2,2) diagonal
    Kq: np.ndarray  # (2,2) diagonal


def modal_params_from_peak_pick(
    omega_n1: float,
    omega_n2: float,
    omega3: float,
    omega4: float,
    omega5: float,
    omega6: float,
    A: float,
    B: float,
) -> ModalFitResult2DOF:
    """
    Compute modal matrices from peak-picking frequencies and peak values.

    Natural frequencies ωn1, ωn2 come from the minima of Im(direct FRF). The
    bandwidths (ω4−ω3) and (ω6−ω5) from the real part give damping ratios.
    Peak magnitudes A, B (positive, from the negative imaginary peaks) give modal stiffness.

    Formulas:
      ζq1 = (ω4 − ω3) / (2·ωn1),  ζq2 = (ω6 − ω5) / (2·ωn2)
      kq1 = 1 / (2·ζq1·A),         kq2 = 1 / (2·ζq2·B)
      mq1 = kq1 / ωn1²,            mq2 = kq2 / ωn2²
      cq1 = 2·ζq1·√(kq1·mq1),     cq2 = 2·ζq2·√(kq2·mq2)

    Args:
        omega_n1, omega_n2: Natural frequencies (rad/s), from min Im(FRF).
        omega3, omega4: Frequencies bracketing mode 1 on real part (ω3 < ωn1 < ω4).
        omega5, omega6: Frequencies bracketing mode 2 on real part (ω5 < ωn2 < ω6).
        A: Magnitude of the (negative) imaginary peak at mode 1 (|min Im(H)|).
        B: Magnitude of the (negative) imaginary peak at mode 2.

    Returns:
        ModalFitResult2DOF with Mq, Cq, Kq and scalar modal parameters.
        Fitted values may differ from the true system due to measurement resolution
        and simplifications in the fitting rules (see modal_fit_error_sources_nutshell).
    """
    if omega_n1 <= 0 or omega_n2 <= 0:
        raise ValueError("omega_n1 and omega_n2 must be positive")
    if A <= 0 or B <= 0:
        raise ValueError("Peak magnitudes A and B must be positive")
    if omega4 <= omega3 or omega6 <= omega5:
        raise ValueError("Bandwidth frequencies must satisfy omega4 > omega3, omega6 > omega5")

    zeta_q1 = (omega4 - omega3) / (2.0 * omega_n1)
    zeta_q2 = (omega6 - omega5) / (2.0 * omega_n2)
    zeta_q1 = max(1e-6, min(zeta_q1, 1.0))
    zeta_q2 = max(1e-6, min(zeta_q2, 1.0))

    kq1 = 1.0 / (2.0 * zeta_q1 * A)
    kq2 = 1.0 / (2.0 * zeta_q2 * B)

    mq1 = kq1 / (omega_n1 * omega_n1)
    mq2 = kq2 / (omega_n2 * omega_n2)

    cq1 = 2.0 * zeta_q1 * math.sqrt(kq1 * mq1)
    cq2 = 2.0 * zeta_q2 * math.sqrt(kq2 * mq2)

    Mq = np.array([[mq1, 0.0], [0.0, mq2]], dtype=float)
    Cq = np.array([[cq1, 0.0], [0.0, cq2]], dtype=float)
    Kq = np.array([[kq1, 0.0], [0.0, kq2]], dtype=float)

    return ModalFitResult2DOF(
        omega_n1=omega_n1,
        omega_n2=omega_n2,
        zeta_q1=zeta_q1,
        zeta_q2=zeta_q2,
        kq1=kq1,
        kq2=kq2,
        mq1=mq1,
        mq2=mq2,
        cq1=cq1,
        cq2=cq2,
        Mq=Mq,
        Cq=Cq,
        Kq=Kq,
    )


def find_two_minima_imaginary(
    omega: np.ndarray,
    H: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Find the two lowest (most negative) minima in the imaginary part of H(ω).

    Returns ((omega_n1, A1), (omega_n2, A2)) where A1, A2 are the magnitudes
    |Im(H)| at the minima (positive). Assumes two distinct modes; the first minimum
    (lower frequency) is mode 1, the second is mode 2.

    Args:
        omega: Frequency vector (rad/s), shape (N,).
        H: Complex FRF, shape (N,).

    Returns:
        ((omega_n1, A1), (omega_n2, A2)) with omega in rad/s, A in same units as |H|.
    """
    omega = np.asarray(omega).ravel()
    H = np.asarray(H).ravel()
    im = np.imag(H)
    if omega.shape[0] != im.shape[0]:
        raise ValueError("omega and H must have the same length")
    # Find local minima: im[i] < im[i-1] and im[i] < im[i+1]
    n = len(im)
    minima = []
    for i in range(1, n - 1):
        if im[i] < im[i - 1] and im[i] < im[i + 1]:
            o = float(omega[i])
            mag = max(0.0, float(-im[i]))
            minima.append((o, mag))
    if len(minima) < 2:
        # Fallback: take two global minima by sorting
        idx = np.argsort(im)
        i1, i2 = idx[0], idx[1]
        if len(idx) > 2 and abs(omega[i1] - omega[i2]) < 1e-6:
            i2 = idx[2]
        o1, o2 = float(omega[i1]), float(omega[i2])
        a1 = max(0.0, float(-im[i1]))
        a2 = max(0.0, float(-im[i2]))
        if o1 > o2:
            o1, o2 = o2, o1
            a1, a2 = a2, a1
        return ((o1, a1), (o2, a2))
    # Prefer interior: drop minima within 5% of frequency range ends to avoid edge artifacts
    o_min, o_max = float(omega[0]), float(omega[-1])
    margin = 0.05 * (o_max - o_min) if o_max > o_min else 0.0
    interior = [(o, a) for o, a in minima if o_min + margin <= o <= o_max - margin]
    candidates = interior if len(interior) >= 2 else minima
    # Take the two with largest magnitude (dominant peaks), then sort by frequency
    candidates.sort(key=lambda x: -x[1])
    (o1, a1), (o2, a2) = candidates[0], candidates[1]
    if o1 > o2:
        o1, o2 = o2, o1
        a1, a2 = a2, a1
    return ((o1, a1), (o2, a2))


def find_bandwidth_zero_crossings(
    omega: np.ndarray,
    re_part: np.ndarray,
    omega_n: float,
    window_frac: float = 0.3,
) -> tuple[float, float] | None:
    """
    Find two frequencies where the real part crosses zero around omega_n (for damping bandwidth).

    Returns (omega_lo, omega_hi) such that omega_lo < omega_n < omega_hi and Re(H) crosses
    zero between them, giving bandwidth ≈ 2·ζ·ωn. Returns None if not found.

    Args:
        omega: Frequency vector (rad/s).
        re_part: Real part of FRF.
        omega_n: Natural frequency (center).
        window_frac: Search within [omega_n*(1-window_frac), omega_n*(1+window_frac)].
    """
    omega = np.asarray(omega).ravel()
    re_part = np.asarray(re_part).ravel()
    lo = omega_n * (1.0 - window_frac)
    hi = omega_n * (1.0 + window_frac)
    mask = (omega >= lo) & (omega <= hi)
    if np.sum(mask) < 3:
        return None
    o = omega[mask]
    r = re_part[mask]
    sign_changes_lo = np.where(np.diff(np.sign(r)) != 0)[0]
    if len(sign_changes_lo) < 2:
        return None
    # Take first and last zero-crossing in the window
    i_lo = sign_changes_lo[0]
    i_hi = sign_changes_lo[-1]
    omega_lo = (o[i_lo] + o[i_lo + 1]) / 2.0
    omega_hi = (o[i_hi] + o[i_hi + 1]) / 2.0
    if omega_lo >= omega_n or omega_hi <= omega_n:
        return None
    return (float(omega_lo), float(omega_hi))


def peak_pick_direct_frf_2dof(
    omega_rad_s: np.ndarray,
    H_direct: np.ndarray,
    bandwidth_from_zero_crossings: bool = True,
) -> ModalFitResult2DOF | None:
    """
    Perform peak-picking on a direct FRF to obtain 2DOF modal parameters.

    Identifies two minima in Im(H) for ωn1, ωn2 and peak magnitudes A, B. If
    bandwidth_from_zero_crossings is True, finds zero-crossings of Re(H) around
    each natural frequency to get (ω3, ω4) and (ω5, ω6) for damping. If not
    possible, returns None (caller can use modal_params_from_peak_pick with
    manually chosen bandwidth frequencies).

    Args:
        omega_rad_s: Frequency vector in rad/s.
        H_direct: Complex direct FRF (e.g. X2/F2), same length as omega.

    Returns:
        ModalFitResult2DOF or None if bandwidth cannot be determined.
    """
    omega = np.asarray(omega_rad_s).ravel()
    H = np.asarray(H_direct).ravel()
    (omega_n1, A), (omega_n2, B) = find_two_minima_imaginary(omega, H)
    if A < 1e-20 or B < 1e-20:
        return None
    re_part = np.real(H)
    if bandwidth_from_zero_crossings:
        bw1 = find_bandwidth_zero_crossings(omega, re_part, omega_n1)
        bw2 = find_bandwidth_zero_crossings(omega, re_part, omega_n2)
        if bw1 is None or bw2 is None:
            return None
        omega3, omega4 = bw1
        omega5, omega6 = bw2
    else:
        # Default: assume 2% damping to get a bandwidth
        z_guess = 0.02
        omega3 = omega_n1 * (1.0 - z_guess)
        omega4 = omega_n1 * (1.0 + z_guess)
        omega5 = omega_n2 * (1.0 - z_guess)
        omega6 = omega_n2 * (1.0 + z_guess)
    return modal_params_from_peak_pick(
        omega_n1, omega_n2, omega3, omega4, omega5, omega6, A, B
    )


# Nutshell text for docs or UI (re-exported from tap_testing.docs.modal_fit)
from tap_testing.docs.modal_fit import (
    measured_frf_all_modes_nutshell,
    modal_fit_error_sources_nutshell,
    model_definition_nutshell,
    system_identification_nutshell,
)


def modal_matrix_from_cross_direct_peaks(
    A: float,
    B: float,
    C: float,
    D: float,
    normalize_to: str = "x2",
) -> np.ndarray:
    """
    Build the 2×2 modal matrix P from direct and cross FRF peak magnitudes.

    For normalization to x2, P = [[p1, p2], [1, 1]] with p1 = C/A and p2 = D/B,
    where A, B are the (positive) magnitudes of the negative imaginary peaks of the
    direct FRF at ωn1 and ωn2, and C, D are the corresponding peak magnitudes of the
    cross FRF (e.g. X1/F2). The ratio cross/direct in each mode gives the mode shape
    entry (Eq. 2.64).

    Args:
        A, B: Direct FRF peak magnitudes at mode 1 and 2 (|min Im(H_direct)|).
        C, D: Cross FRF peak magnitudes at mode 1 and 2 (|min Im(H_cross)|).
        normalize_to: "x2" gives second row [1, 1]; only "x2" is implemented.

    Returns:
        P shape (2, 2).
    """
    if normalize_to != "x2":
        raise ValueError("normalize_to must be 'x2'")
    if abs(A) < 1e-30 or abs(B) < 1e-30:
        raise ValueError("Direct FRF peaks A and B must be non-zero")
    p1 = C / A
    p2 = D / B
    return np.array([[p1, p2], [1.0, 1.0]], dtype=float)


def local_matrices_from_modal(
    Mq: np.ndarray,
    Cq: np.ndarray,
    Kq: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform from modal to local coordinates so that Mq = Pᵀ M P (and similarly C, K).

    We have M = P⁻ᵀ Mq P⁻¹, C = P⁻ᵀ Cq P⁻¹, K = P⁻ᵀ Kq P⁻¹. The resulting M, C, K
    correspond to the physical coordinates (e.g. x1, x2). For a chain-type model they
    have the standard form (Eqs. 2.65–2.67).

    Args:
        Mq, Cq, Kq: Modal mass, damping, stiffness (2×2 diagonal).
        P: Modal matrix (2×2), e.g. from modal_matrix_from_cross_direct_peaks.

    Returns:
        (M, C, K) each 2×2 in local coordinates.
    """
    P = np.asarray(P, dtype=float)
    Pinv = np.linalg.inv(P)
    PinvT = np.linalg.inv(P.T)
    M = (PinvT @ Mq @ Pinv).astype(float)
    C = (PinvT @ Cq @ Pinv).astype(float)
    K = (PinvT @ Kq @ Pinv).astype(float)
    return (M, C, K)


def chain_params_from_local_matrices(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Read off chain-type lumped parameters from local M, C, K.

    Assumes M = diag(m1, m2), K = [[k1+k2, -k2], [-k2, k2]], C = [[c1+c2, -c2], [-c2, c2]].
    Returns (m1, m2, k1, k2, c1, c2).

    Args:
        M, C, K: Local mass, damping, stiffness (2×2) from local_matrices_from_modal.

    Returns:
        (m1, m2, k1, k2, c1, c2) in kg, N/m, N·s/m.
    """
    M = np.asarray(M)
    C = np.asarray(C)
    K = np.asarray(K)
    m1 = float(M[0, 0])
    m2 = float(M[1, 1])
    k2 = float(-K[0, 1])
    k1 = float(K[0, 0]) - k2
    c2 = float(-C[0, 1])
    c1 = float(C[0, 0]) - c2
    return (m1, m2, k1, k2, c1, c2)


def model_definition_from_fit_and_cross_peaks(
    fit: ModalFitResult2DOF,
    C_peak: float,
    D_peak: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float, float]]:
    """
    Complete model definition: modal matrix P and local M, C, K and chain params.

    Uses direct FRF peaks A, B from the fit (via 1/(2·ζ·kq) at each mode) and the
    cross FRF peak magnitudes C_peak, D_peak to build P, then transforms to local
    coordinates and extracts chain parameters.

    Args:
        fit: Result of peak-picking (modal_params_from_peak_pick or peak_pick_direct_frf_2dof).
        C_peak: Magnitude |min Im(cross FRF)| at ωn1.
        D_peak: Magnitude |min Im(cross FRF)| at ωn2.

    Returns:
        (P, M, C, K, (m1, m2, k1, k2, c1, c2)).
    """
    A = 1.0 / (2.0 * fit.zeta_q1 * fit.kq1)
    B = 1.0 / (2.0 * fit.zeta_q2 * fit.kq2)
    P = modal_matrix_from_cross_direct_peaks(A, B, C_peak, D_peak)
    M, C_loc, K = local_matrices_from_modal(fit.Mq, fit.Cq, fit.Kq, P)
    chain = chain_params_from_local_matrices(M, C_loc, K)
    return (P, M, C_loc, K, chain)
