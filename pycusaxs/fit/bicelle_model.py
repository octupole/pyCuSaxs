"""Core-shell bicelle model implementation.

This module implements the SasView core_shell_bicelle model
for fitting SAXS data from circular bicelle structures.

Reference:
    https://www.sasview.org/docs/user/models/core_shell_bicelle.html
"""

import numpy as np
from typing import Tuple
from scipy.special import j1  # Bessel function of first kind, order 1
from scipy.integrate import dblquad


def _besinc(x: np.ndarray) -> np.ndarray:
    """Bessel function sinc: 2*J1(x)/x, with proper limit at x=0."""
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-6
    result[mask] = 2.0 * j1(x[mask]) / x[mask]
    result[~mask] = 1.0  # limit as x->0 is 1
    return result


def _sinc(x: np.ndarray) -> np.ndarray:
    """Sinc function: sin(x)/x with proper limit at x=0."""
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-6
    result[mask] = np.sin(x[mask]) / x[mask]
    result[~mask] = 1.0
    return result


def bicelle_form_factor_1d(
    q: np.ndarray,
    radius: float,
    thick_rim: float,
    thick_face: float,
    length: float,
    sld_core: float,
    sld_face: float,
    sld_rim: float,
    sld_solvent: float,
) -> np.ndarray:
    """
    Compute the orientationally-averaged form factor for core-shell bicelle.

    This performs numerical integration over orientation angle alpha.

    Parameters
    ----------
    q : np.ndarray
        Scattering vector magnitude (1/Å)
    radius : float
        Core cylinder radius (Å)
    thick_rim : float
        Rim shell thickness (Å)
    thick_face : float
        Face shell thickness (Å)
    length : float
        Core cylinder length (Å)
    sld_core : float
        Core SLD (1e-6/Å²)
    sld_face : float
        Face shell SLD (1e-6/Å²)
    sld_rim : float
        Rim shell SLD (1e-6/Å²)
    sld_solvent : float
        Solvent SLD (1e-6/Å²)

    Returns
    -------
    np.ndarray
        Form factor squared <F²(q)> averaged over all orientations
    """
    q = np.asarray(q, dtype=float)
    R = float(max(radius, 0.1))
    tr = float(max(thick_rim, 0.0))
    tf = float(max(thick_face, 0.0))
    L = float(max(length, 1.0))

    drho_c = float(sld_core - sld_face)
    drho_f = float(sld_face - sld_rim)
    drho_r = float(sld_rim - sld_solvent)

    # Volumes
    V_core = np.pi * R * R * L
    V_core_face = np.pi * R * R * (L + 2.0 * tf)
    V_total = np.pi * (R + tr) * (R + tr) * (L + 2.0 * tf)

    result = np.zeros_like(q)

    # Integration over alpha (0 to pi/2)
    n_alpha = 30
    alpha_vals = np.linspace(0, np.pi / 2, n_alpha)

    for alpha in alpha_vals:
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

        # Q projections
        q_perp = q * sin_alpha  # perpendicular to cylinder axis
        q_par = q * cos_alpha  # parallel to cylinder axis

        # Bessel and sinc terms for three regions
        # Core contribution
        bes_c = _besinc(q_perp * R)
        sinc_c = _sinc(q_par * L / 2.0)
        F_c = drho_c * V_core * bes_c * sinc_c

        # Core+face contribution
        sinc_cf = _sinc(q_par * (L / 2.0 + tf))
        F_cf = drho_f * V_core_face * bes_c * sinc_cf

        # Total (rim) contribution
        bes_t = _besinc(q_perp * (R + tr))
        sinc_t = _sinc(q_par * (L / 2.0 + tf))
        F_t = drho_r * V_total * bes_t * sinc_t

        # Sum contributions
        F_total = F_c + F_cf + F_t

        # Accumulate |F|² weighted by sin(alpha) for spherical integration
        result += (F_total * F_total) * sin_alpha

    # Normalize by integration weights
    norm = n_alpha
    result /= norm

    return result


def bicelle_intensity(
    q: np.ndarray,
    radius: float,
    thick_rim: float,
    thick_face: float,
    length: float,
    sld_core: float,
    sld_face: float,
    sld_rim: float,
    sld_solvent: float,
    scale: float = 1.0,
    background: float = 0.0,
) -> np.ndarray:
    """
    Compute I(q) for core-shell bicelle model.

    I(q) = (scale / V_total) * <F²(q)> + background

    Parameters
    ----------
    q : np.ndarray
        Scattering vector magnitude (1/Å)
    radius : float
        Core cylinder radius (Å)
    thick_rim : float
        Rim shell thickness (Å)
    thick_face : float
        Face shell thickness (Å)
    length : float
        Core cylinder length (Å)
    sld_core : float
        Core SLD (1e-6/Å²)
    sld_face : float
        Face shell SLD (1e-6/Å²)
    sld_rim : float
        Rim shell SLD (1e-6/Å²)
    sld_solvent : float
        Solvent SLD (1e-6/Å²)
    scale : float
        Scale factor / volume fraction
    background : float
        Flat background

    Returns
    -------
    np.ndarray
        Intensity I(q)
    """
    q = np.asarray(q, dtype=float)
    R = float(max(radius, 0.1))
    tr = float(max(thick_rim, 0.0))
    tf = float(max(thick_face, 0.0))
    L = float(max(length, 1.0))

    # Total volume
    V_total = np.pi * (R + tr) * (R + tr) * (L + 2.0 * tf)

    # Form factor
    F2_avg = bicelle_form_factor_1d(
        q, radius, thick_rim, thick_face, length,
        sld_core, sld_face, sld_rim, sld_solvent
    )

    # Intensity
    I = (scale / V_total) * F2_avg + background

    return I


def estimate_bicelle_size(q: np.ndarray, i: np.ndarray) -> Tuple[float, float]:
    """
    Estimate rough bicelle dimensions from I(q) data.

    Returns
    -------
    radius_est : float
        Estimated minor radius (Å)
    length_est : float
        Estimated length (Å)
    """
    from scipy.signal import find_peaks

    q = np.asarray(q)
    i = np.asarray(i)

    # Try to find oscillation period
    # Smooth data
    from scipy.ndimage import gaussian_filter1d
    y = gaussian_filter1d(i, sigma=2)

    # Find peaks
    peaks, _ = find_peaks(y)
    if peaks.size >= 2:
        dq = np.median(np.diff(q[peaks]))
        if dq > 1e-6:
            # First zero of 2*J1(qR)/qR occurs at qR ≈ 3.83
            # Rough estimate: R ≈ 3.83/q_first_min
            radius_est = 3.83 / (2.0 * dq)
        else:
            radius_est = 30.0
    else:
        radius_est = 30.0

    # Estimate length from low-q behavior
    # For cylinders, Guinier radius relates to length
    # Very rough: L ~ radius
    length_est = 1.5 * radius_est

    return float(np.clip(radius_est, 10.0, 200.0)), float(np.clip(length_est, 10.0, 300.0))
