#!/usr/bin/env python3
# bicelle_fit.py
"""Robust fitter for the SasView core_shell_bicelle_elliptical model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import j1
from numba import jit

# --- Merged bicelle_model.py content ---


def _besinc(x: np.ndarray) -> np.ndarray:
    """Bessel function sinc: 2*J1(x)/x, with proper limit at x=0."""
    x = np.asarray(x)
    result = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-6
    # Numba cannot JIT scipy.special functions in nopython mode, so this part remains in pure Python
    # and is pre-calculated before entering the JIT'd kernel.
    result[mask] = 2.0 * j1(x[mask]) / x[mask]
    return result


@jit(nopython=True, fastmath=True)
def _sinc_jitted(x: np.ndarray) -> np.ndarray:
    """JIT-compiled Sinc function: sin(x)/x with proper limit at x=0."""
    # This pattern is efficiently compiled by Numba.
    result = np.ones_like(x)
    mask = np.abs(x) > 1e-6
    result[mask] = np.sin(x[mask]) / x[mask]
    return result


@jit(nopython=True, fastmath=True)
def _form_factor_kernel(
    q: np.ndarray,
    alpha_vals: np.ndarray,
    bes_c_all: np.ndarray,
    bes_t_all: np.ndarray,
    L: float, tf: float,
    drho_c: float, drho_f: float, drho_r: float,
    V_core: float, V_core_face: float, V_total: float
) -> np.ndarray:
    """JIT-compiled kernel for calculating the form factor."""
    result = np.zeros_like(q)
    n_alpha = len(alpha_vals)
    for i in range(n_alpha):
        alpha = alpha_vals[i]
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

        q_par = q * cos_alpha

        # Use pre-calculated Bessel function values
        bes_c = bes_c_all[:, i]
        sinc_c = _sinc_jitted(q_par * L / 2.0)
        F_c = drho_c * V_core * bes_c * sinc_c

        sinc_cf = _sinc_jitted(q_par * (L / 2.0 + tf))
        F_cf = drho_f * V_core_face * bes_c * sinc_cf

        bes_t = bes_t_all[:, i]
        sinc_t = _sinc_jitted(q_par * (L / 2.0 + tf))
        F_t = drho_r * V_total * bes_t * sinc_t

        F_total = F_c + F_cf + F_t
        result += (F_total * F_total) * sin_alpha

    return result


def bicelle_form_factor_1d(
    q: np.ndarray, radius: float, thick_rim: float, thick_face: float,
    length: float, sld_core: float, sld_face: float, sld_rim: float,
    sld_solvent: float
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    R = float(max(radius, 0.1))
    tr = float(max(thick_rim, 0.0))
    tf = float(max(thick_face, 0.0))
    L = float(max(length, 1.0))
    drho_c = float(sld_core - sld_face)
    drho_f = float(sld_face - sld_rim)
    drho_r = float(sld_rim - sld_solvent)
    V_core = np.pi * R * R * L
    V_core_face = np.pi * R * R * (L + 2.0 * tf)
    V_total = np.pi * (R + tr) * (R + tr) * (L + 2.0 * tf)

    n_alpha = 30
    alpha_vals = np.linspace(0, np.pi / 2, n_alpha)

    # Pre-computation of values involving the non-jittable _besinc function
    # This creates 2D arrays of shape (len(q), n_alpha)
    q_mesh, alpha_mesh = np.meshgrid(q, alpha_vals, indexing='ij')
    q_perp_mesh = q_mesh * np.sin(alpha_mesh)

    bes_c_all = _besinc(q_perp_mesh * R)
    bes_t_all = _besinc(q_perp_mesh * (R + tr))

    # Call the fast, JIT-compiled kernel for the main computation loop
    result = _form_factor_kernel(
        q, alpha_vals, bes_c_all, bes_t_all, L, tf, drho_c, drho_f, drho_r,
        V_core, V_core_face, V_total
    )

    result /= n_alpha
    return result


def bicelle_intensity(
    q: np.ndarray, radius: float, thick_rim: float, thick_face: float,
    length: float, sld_core: float, sld_face: float, sld_rim: float,
    sld_solvent: float, scale: float = 1.0, background: float = 0.0
) -> np.ndarray:
    R = float(max(radius, 0.1))
    tr = float(max(thick_rim, 0.0))
    tf = float(max(thick_face, 0.0))
    L = float(max(length, 1.0))
    V_total = np.pi * (R + tr) * (R + tr) * (L + 2.0 * tf)
    F2_avg = bicelle_form_factor_1d(
        q, radius, thick_rim, thick_face, length,
        sld_core, sld_face, sld_rim, sld_solvent
    )
    return (scale / V_total) * F2_avg + background


def estimate_bicelle_size(q: np.ndarray, i: np.ndarray) -> Tuple[float, float]:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    q, i = np.asarray(q), np.asarray(i)
    y = gaussian_filter1d(i, sigma=2)
    peaks, _ = find_peaks(y)
    if peaks.size >= 2:
        dq = np.median(np.diff(q[peaks]))
        radius_est = 3.83 / (2.0 * dq) if dq > 1e-6 else 30.0
    else:
        radius_est = 30.0
    length_est = 1.5 * radius_est
    return float(np.clip(radius_est, 10.0, 200.0)), float(np.clip(length_est, 10.0, 300.0))

# --- Main bicelle_fit.py content ---


@dataclass
class BicelleBounds:
    """Parameter bounds for core-shell bicelle fitting."""
    radius: Tuple[float, float] = (10.0, 200.0)
    thick_rim: Tuple[float, float] = (1.0, 40.0)
    thick_face: Tuple[float, float] = (1.0, 50.0)
    length: Tuple[float, float] = (10.0, 50.0)
    sld_core: Tuple[float, float] = (-10.0, 10.0)
    sld_face: Tuple[float, float] = (-10.0, 10.0)
    sld_rim: Tuple[float, float] = (-10.0, 10.0)
    sld_solvent: Tuple[float, float] = (-10.0, 10.0)
    scale: Tuple[float, float] = (1e-10, 1e6)
    background: Tuple[float, float] = (0.0, np.inf)


def _ls_scale_background(M: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> Tuple[float, float]:
    M, y = np.asarray(M, dtype=float), np.asarray(y, dtype=float)
    if w is None:
        Aw = np.c_[M, np.ones_like(M)]
        lhs, rhs = Aw.T @ Aw, Aw.T @ y
    else:
        w = np.asarray(w, dtype=float)
        Aw = np.c_[M, np.ones_like(M)] * w[:, None]
        yw = y * w
        lhs, rhs = Aw.T @ Aw, Aw.T @ yw
    try:
        s, b = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        s, b = 1.0, float(np.clip(np.median(y[-10:]), 0.0, None))
    return float(np.clip(s, 0.0, np.inf)), float(np.clip(b, 0.0, np.inf))


def _weights_from_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    eps = 1e-3 * max(1.0, np.nanmax(y))
    return 1.0 / (y + eps)


def _objective_stage1(x: np.ndarray, q: np.ndarray, y: np.ndarray) -> float:
    radius, thick_rim, thick_face, length, sld_c, sld_f, sld_r, sld_s = x
    M = bicelle_intensity(q, radius, thick_rim, thick_face,
                          length, sld_c, sld_f, sld_r, sld_s, 1.0, 0.0)
    w = _weights_from_y(y)
    s, b = _ls_scale_background(M, y, w)
    r = (s * M + b) - y
    return float(np.sum((w * r) ** 2))


def _residuals_stage2(theta: np.ndarray, q: np.ndarray, y: np.ndarray, logspace: bool) -> np.ndarray:
    radius, thick_rim, thick_face, length, sld_c, sld_f, sld_r, sld_s, s, b = theta
    yhat = bicelle_intensity(
        q, radius, thick_rim, thick_face, length, sld_c, sld_f, sld_r, sld_s, s, b)
    if logspace:
        eps = 1e-12
        return np.log(yhat + eps) - np.log(y + eps)
    w = _weights_from_y(y)
    return (yhat - y) * w


def fit_bicelle(
    q: np.ndarray, y: np.ndarray, use_logspace: bool = False,
    bounds: Optional[BicelleBounds] = None
):
    bounds = bounds or BicelleBounds()
    try:
        radius_est, length_est = estimate_bicelle_size(q, y)
    except Exception:
        radius_est, length_est = 50.0, 15.0
    init_guess = (
        np.clip(radius_est, *bounds.radius), 10.0, 10.0,
        np.clip(length_est, *bounds.length), 1.0, 4.0, 4.0, 1.0
    )
    gl_bounds = [
        bounds.radius, bounds.thick_rim, bounds.thick_face, bounds.length,
        bounds.sld_core, bounds.sld_face, bounds.sld_rim, bounds.sld_solvent
    ]
    print("Starting global optimization xxxx...", flush=True)
    result_de = differential_evolution(
        lambda x: _objective_stage1(x, q, y),
        bounds=gl_bounds, maxiter=300, popsize=15, tol=1e-6, polish=False,
        # OPTIMIZATION: Use all available CPU cores
        seed=42, updating='deferred', workers=-1
    )
    radius_opt, tr_opt, tf_opt, length_opt, sld_c_opt, sld_f_opt, sld_r_opt, sld_s_opt = result_de.x
    M0 = bicelle_intensity(q, radius_opt, tr_opt, tf_opt, length_opt,
                           sld_c_opt, sld_f_opt, sld_r_opt, sld_s_opt, 1.0, 0.0)
    s0, b0 = _ls_scale_background(M0, y, _weights_from_y(y))
    x0 = np.array([radius_opt, tr_opt, tf_opt, length_opt, sld_c_opt,
                  sld_f_opt, sld_r_opt, sld_s_opt, s0, b0], dtype=float)
    lower = np.array([b[0] for b in gl_bounds] +
                     [bounds.scale[0], bounds.background[0]])
    upper = np.array([b[1] for b in gl_bounds] +
                     [bounds.scale[1], bounds.background[1]])
    upper[~np.isfinite(upper)] = 1e12
    print("Starting local refinement...", flush=True)
    res_ls = least_squares(
        _residuals_stage2, x0, bounds=(lower, upper),
        args=(q, y, use_logspace), method='trf', loss='soft_l1',
        f_scale=0.1, x_scale='jac', max_nfev=5000
    )
    return res_ls, result_de

# --- CLI main function (example usage) ---


def main():
    # This is an example of how the script could be run.
    # In a real scenario, you would parse command-line arguments.
    print("This is a library, not an executable script.")
    print("Example: generate dummy data and fit it.")

    # Generate some dummy data
    q_dummy = np.logspace(-2, 0, 100)
    params_true = {
        'radius': 75.0, 'thick_rim': 10.0, 'thick_face': 15.0,
        'length': 30.0, 'sld_core': 0.5, 'sld_face': 4.5, 'sld_rim': 4.0,
        'sld_solvent': 6.3, 'scale': 1e-3, 'background': 0.01
    }
    y_true = bicelle_intensity(q_dummy, **params_true)
    y_noisy = y_true + np.random.normal(0, 0.1 * np.sqrt(y_true))

    # Fit the data
    res_ls, res_de = fit_bicelle(q_dummy, y_noisy)

    # Print results
    p = dict(zip(['radius', 'thick_rim', 'thick_face', 'length', 'sld_core',
                  'sld_face', 'sld_rim', 'sld_solvent', 'scale', 'background'], res_ls.x))

    print('\n' + '='*60)
    print('Fit success:', bool(res_ls.success))
    print('Message:', res_ls.message)
    print('='*60)
    for name, val in p.items():
        print(f'  {name:20s}: {val:.3f}')
    print('='*60)


if __name__ == '__main__':
    main()
