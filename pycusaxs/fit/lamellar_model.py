import numpy as np
from typing import Tuple


def lamellar_hg_intensity(q: np.ndarray,
                          length_head: float,
                          length_tail: float,
                          dr_head: float,
                          dr_tail: float,
                          scale: float = 1.0,
                          background: float = 0.0) -> np.ndarray:
    """
    Lamellar_HG intensity (SasView model) with flexible scale/background.

    I(q) = scale * [ 4/q^4 * ( drh*(sin(q*(H+T)) - sin(q*T)) + drt*sin(q*T) )^2
                      / (2*(H+T)) ] + background

    This matches the algebra in sasmodels/models/lamellar_hg.c, but replaces the
    fixed 2e-4*pi factor with a free overall ``scale`` to adapt to arbitrary
    data normalization.

    Parameters
    ----------
    q : array-like
        Scattering vector magnitude (1/Ang).
    length_head : float
        Head group thickness H (Ang), >= 0.
    length_tail : float
        Tail thickness T (Ang), >= 0.
    dr_head : float
        SLD contrast of head vs solvent (1e-6/Ang^2), i.e. (sld_head - sld_solvent).
    dr_tail : float
        SLD contrast of tail vs solvent (1e-6/Ang^2), i.e. (sld_tail - sld_solvent).
    scale : float
        Overall scale factor (free), defaults to 1.0.
    background : float
        Flat background, defaults to 0.0.

    Returns
    -------
    np.ndarray
        Intensity I(q) with same shape as ``q``.
    """
    q = np.asarray(q, dtype=float)
    H = float(max(0.0, length_head))
    T = float(max(0.0, length_tail))
    drh = float(dr_head)
    drt = float(dr_tail)
    s = float(max(0.0, scale))
    b = float(background)

    q2 = np.maximum(q*q, 1e-24)
    sin_qT = np.sin(q*T)
    sin_qHT = np.sin(q*(H+T))
    S = drh * (sin_qHT - sin_qT) + drt * sin_qT
    # 4/q^2 * S^2, then divide by q^2 and by 2*(H+T)
    pref = 4.0 / q2
    Pq = pref * (S*S)
    denom = np.maximum(2.0 * (H + T), 1e-12)
    I = s * (Pq / q2) / denom + b
    return I


def ht_from_total_fraction(t_total: float, frac_head: float) -> Tuple[float, float]:
    """
    Convert thickness and head fraction to (H, T).

    Note: Historically this function interpreted ``t_total`` as H+T.
    For bilayers there are two heads and two tails; the physical bilayer
    thickness is D = 2*(H+T). Prefer using ``ht_from_bilayer_thickness``
    for clarity. This function now treats ``t_total`` as a bilayer
    thickness D to avoid confusion with the normalization used in the
    scattering expression.
    """
    D = float(max(0.0, t_total))
    f = float(np.clip(frac_head, 1e-6, 1.0 - 1e-6))
    # H = (D/2)*f ; T = (D/2)*(1-f)
    half = 0.5 * D
    H = half * f
    T = half * (1.0 - f)
    return H, T


def ht_from_bilayer_thickness(bilayer_total: float, frac_head: float) -> Tuple[float, float]:
    """Explicit helper: use bilayer thickness D = 2*(H+T) to compute (H, T)."""
    return ht_from_total_fraction(bilayer_total, frac_head)


def estimate_periodic_delta_q(q: np.ndarray, i: np.ndarray) -> float:
    """
    Estimate the oscillation period Δq from local extrema.

    Returns median spacing between successive extrema; falls back to robust FFT
    estimate if not enough extrema are found.
    """
    q = np.asarray(q)
    i = np.asarray(i)
    # Smooth slightly to reduce noise
    from scipy.ndimage import gaussian_filter1d
    y = gaussian_filter1d(i, sigma=2)

    # Find extrema
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y)
    troughs, _ = find_peaks(-y)
    idx = np.sort(np.concatenate([peaks, troughs]))
    if idx.size >= 3:
        dq = np.diff(q[idx])
        dq = dq[(dq > 1e-6)]
        if dq.size:
            return float(np.median(dq))

    # FFT-based fallback: pick dominant frequency on q grid
    # Interpolate onto uniform q grid for FFT
    q_uniform = np.linspace(q.min(), q.max(), num=min(2048, max(64, q.size)))
    y_uniform = np.interp(q_uniform, q, y)
    y_uniform -= y_uniform.mean()
    spec = np.fft.rfft(y_uniform)
    freqs = np.fft.rfftfreq(q_uniform.size, d=(q_uniform[1]-q_uniform[0]))
    k = np.argmax(np.abs(spec[1:])) + 1  # skip DC
    freq = max(freqs[k], 1e-12)
    return 1.0/freq
