#!/usr/bin/env python3
"""Robust fitter for the SasView core_shell_bicelle_elliptical model.

Features
 - Physically motivated parameterization for elliptical bicelle structures
 - Automatic initialization from data features
 - Global search (differential evolution) followed by local refinement
 - Weighted least-squares by default; optional log-residual objective
 - Optional plotting and export of the fitted curve

Usage
-----
python -m pycusaxs.fit.bicelle_fit -f data.dat -p -o fit.dat
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .bicelle_model import (
    bicelle_intensity,
    estimate_bicelle_size,
)


@dataclass
class BicelleBounds:
    """Parameter bounds for core-shell bicelle fitting."""
    radius: Tuple[float, float] = (10.0, 200.0)           # Å
    thick_rim: Tuple[float, float] = (1.0, 40.0)          # Å
    thick_face: Tuple[float, float] = (1.0, 50.0)         # Å
    length: Tuple[float, float] = (10.0, 50.0)           # Å
    sld_core: Tuple[float, float] = (-10.0, 10.0)         # 1e-6/Å²
    sld_face: Tuple[float, float] = (-10.0, 10.0)         # 1e-6/Å²
    sld_rim: Tuple[float, float] = (-10.0, 10.0)          # 1e-6/Å²
    sld_solvent: Tuple[float, float] = (-10.0, 10.0)      # 1e-6/Å²
    scale: Tuple[float, float] = (1e-10, 1e6)             # arbitrary
    background: Tuple[float, float] = (0.0, np.inf)       # arbitrary


def _ls_scale_background(M: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> Tuple[float, float]:
    """Solve for scale and background by (weighted) least squares.

    Minimizes || w*(s*M + b - y) ||_2.
    Returns (s, b), with s>=0 clipped and b>=0 clipped.
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        W = np.eye(len(y))
        Aw = np.c_[M, np.ones_like(M)]
        lhs = Aw.T @ Aw
        rhs = Aw.T @ y
    else:
        w = np.asarray(w, dtype=float)
        Aw = np.c_[M, np.ones_like(M)] * w[:, None]
        yw = y * w
        lhs = Aw.T @ Aw
        rhs = Aw.T @ yw
    try:
        s, b = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        s, b = 1.0, float(np.clip(np.median(y[-10:]), 0.0, None))
    s = float(np.clip(s, 0.0, np.inf))
    b = float(np.clip(b, 0.0, np.inf))
    return s, b


def _weights_from_y(y: np.ndarray) -> np.ndarray:
    """Default weights balancing dynamic range: w = 1/(y + eps)."""
    y = np.asarray(y, dtype=float)
    eps = 1e-3 * max(1.0, np.nanmax(y))
    return 1.0 / (y + eps)


def _objective_stage1(x: np.ndarray, q: np.ndarray, y: np.ndarray, bnds: BicelleBounds) -> float:
    """Global search objective with analytic (s, b)."""
    radius, thick_rim, thick_face, length, sld_c, sld_f, sld_r, sld_s = x

    # Model shape with unit scale/background
    M = bicelle_intensity(
        q, radius, thick_rim, thick_face, length,
        sld_c, sld_f, sld_r, sld_s,
        scale=1.0, background=0.0
    )

    w = _weights_from_y(y)
    s, b = _ls_scale_background(M, y, w)
    r = (s * M + b) - y
    return float(np.sum((w * r) ** 2))


def _residuals_stage2(theta: np.ndarray, q: np.ndarray, y: np.ndarray, logspace: bool) -> np.ndarray:
    """Residuals for local refinement."""
    radius, thick_rim, thick_face, length, sld_c, sld_f, sld_r, sld_s, s, b = theta

    yhat = bicelle_intensity(
        q, radius, thick_rim, thick_face, length,
        sld_c, sld_f, sld_r, sld_s,
        scale=s, background=b
    )

    if logspace:
        eps = 1e-12
        return np.log(yhat + eps) - np.log(y + eps)

    w = _weights_from_y(y)
    return (yhat - y) * w


def fit_bicelle(
    q: np.ndarray,
    y: np.ndarray,
    use_logspace: bool = False,
    init_guess: Optional[Tuple] = None,
    bounds: Optional[BicelleBounds] = None,
    mask: Optional[np.ndarray] = None,
):
    """
    Fit core_shell_bicelle_elliptical model to (q, y).

    Returns a dict with parameters and diagnostics.
    """
    if bounds is None:
        bounds = BicelleBounds()

    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        q = q[mask]
        y = y[mask]

    # Initial guess
    if init_guess is None:
        try:
            radius_est, length_est = estimate_bicelle_size(q, y)
        except Exception:
            radius_est = 50.0
            length_est = 15.0

        radius0 = np.clip(radius_est, *bounds.radius)
        length0 = np.clip(length_est, *bounds.length)
        thick_rim0 = 10.0
        thick_face0 = 10.0
        sld_core0 = 1.0
        sld_face0 = 4.0
        sld_rim0 = 4.0
        sld_solvent0 = 1.0

        init_guess = (radius0, thick_rim0, thick_face0, length0,
                      sld_core0, sld_face0, sld_rim0, sld_solvent0)

    # Stage 1: global search
    gl_bounds = [
        bounds.radius,
        bounds.thick_rim,
        bounds.thick_face,
        bounds.length,
        bounds.sld_core,
        bounds.sld_face,
        bounds.sld_rim,
        bounds.sld_solvent,
    ]

    def fun_de(x):
        return _objective_stage1(x, q, y, bounds)

    print("Starting global optimization (differential evolution)...", flush=True)
    result_de = differential_evolution(
        fun_de,
        bounds=gl_bounds,
        init='latinhypercube',
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=1e-6,
        polish=False,
        seed=42,
        updating='deferred',
        workers=1,
    )

    radius_opt, tr_opt, tf_opt, length_opt, sld_c_opt, sld_f_opt, sld_r_opt, sld_s_opt = result_de.x

    # Compute scale and background
    M0 = bicelle_intensity(
        q, radius_opt, tr_opt, tf_opt, length_opt,
        sld_c_opt, sld_f_opt, sld_r_opt, sld_s_opt,
        scale=1.0, background=0.0
    )
    s0, b0 = _ls_scale_background(M0, y, _weights_from_y(y))

    # Stage 2: local refinement
    x0 = np.array([radius_opt, tr_opt, tf_opt, length_opt,
                   sld_c_opt, sld_f_opt, sld_r_opt, sld_s_opt, s0, b0], dtype=float)

    lower = np.array([
        bounds.radius[0], bounds.thick_rim[0], bounds.thick_face[0],
        bounds.length[0], bounds.sld_core[0], bounds.sld_face[0], bounds.sld_rim[0],
        bounds.sld_solvent[0], bounds.scale[0], bounds.background[0]
    ])
    upper = np.array([
        bounds.radius[1], bounds.thick_rim[1], bounds.thick_face[1],
        bounds.length[1], bounds.sld_core[1], bounds.sld_face[1], bounds.sld_rim[1],
        bounds.sld_solvent[1], bounds.scale[1],
        bounds.background[1] if np.isfinite(bounds.background[1]) else 1e12
    ])

    print("Starting local refinement (least squares)...", flush=True)
    res_ls = least_squares(
        _residuals_stage2,
        x0,
        bounds=(lower, upper),
        args=(q, y, use_logspace),
        method='trf',
        loss='soft_l1',
        f_scale=0.1,
        x_scale='jac',
        max_nfev=5000,
        verbose=0,
    )

    radius, tr, tf, length, sld_c, sld_f, sld_r, sld_s, s, b = res_ls.x

    yfit = bicelle_intensity(
        q, radius, tr, tf, length,
        sld_c, sld_f, sld_r, sld_s, s, b
    )

    return {
        'success': bool(res_ls.success),
        'message': res_ls.message,
        'params': {
            'radius': float(radius),
            'thick_rim': float(tr),
            'thick_face': float(tf),
            'length': float(length),
            'sld_core': float(sld_c),
            'sld_face': float(sld_f),
            'sld_rim': float(sld_r),
            'sld_solvent': float(sld_s),
            'scale': float(s),
            'background': float(b),
        },
        'yfit': yfit,
        'cost': float(res_ls.cost),
        'nfev': int(res_ls.nfev),
        'global_cost': float(result_de.fun),
    }


def _load_xy(path: str):
    """Load q, I(q) from two-column file."""
    data = np.loadtxt(path)
    q = np.asarray(data[:, 0], dtype=float)
    i = np.asarray(data[:, 1], dtype=float)
    # Clean up: filter invalid values and nonpositive q
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q = q[m]
    i = i[m]
    return q, i


def _parse_exclude(arg_list: Sequence[str]) -> List[Tuple[float, float]]:
    """Parse exclude ranges from command line."""
    ranges: List[Tuple[float, float]] = []
    for s in arg_list:
        if ':' in s:
            a, b = s.split(':', 1)
            try:
                lo = float(a)
                hi = float(b)
            except ValueError:
                continue
        else:
            try:
                lo = float(s)
                hi = np.inf
            except ValueError:
                continue
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def _apply_q_window(q: np.ndarray,
                    y: np.ndarray,
                    qmin: Optional[float],
                    qmax: Optional[float],
                    exclude: Optional[List[Tuple[float, float]]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply q-range filtering."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones_like(q, dtype=bool)
    if qmin is not None:
        mask &= (q >= qmin)
    if qmax is not None and np.isfinite(qmax):
        mask &= (q <= qmax)
    if exclude:
        for lo, hi in exclude:
            mask &= ~((q >= lo) & (q <= hi))
    return q[mask], y[mask], mask


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        description='Fit core_shell_bicelle model to I(q) data')
    ap.add_argument('-f', '--file', required=True,
                    help='Two-column file: q I(q)')
    ap.add_argument('-p', '--plot', action='store_true',
                    help='Show data and fit')
    ap.add_argument('-o', '--output', help='Write fitted I(q) to this file')
    ap.add_argument('--log', action='store_true',
                    help='Use log-residual objective for local refinement')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--qmin', type=float, help='Minimum q to include in fit')
    ap.add_argument('--qmax', type=float, help='Maximum q to include in fit')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Exclude q range(s) as lo:hi; repeatable')

    # Bounds tuning
    ap.add_argument('--radius-min', type=float,
                    help='Lower bound for radius [Å]')
    ap.add_argument('--radius-max', type=float,
                    help='Upper bound for radius [Å]')
    ap.add_argument('--length-min', type=float,
                    help='Lower bound for length [Å]')
    ap.add_argument('--length-max', type=float,
                    help='Upper bound for length [Å]')

    args = ap.parse_args(argv)

    # Load data
    q, y = _load_xy(args.file)

    # Apply q-window/mask
    exclude_ranges = _parse_exclude(args.exclude)
    q_fit, y_fit, mask = _apply_q_window(
        q, y, args.qmin, args.qmax, exclude_ranges)
    if q_fit.size < 10:
        print('Warning: very few points in the selected q-window', flush=True)

    # Build bounds
    b = BicelleBounds()
    if args.radius_min is not None:
        b.radius = (float(args.radius_min), b.radius[1])
    if args.radius_max is not None:
        b.radius = (b.radius[0], float(args.radius_max))
    if args.length_min is not None:
        b.length = (float(args.length_min), b.length[1])
    if args.length_max is not None:
        b.length = (b.length[0], float(args.length_max))

    # Fit
    res = fit_bicelle(q_fit, y_fit, use_logspace=args.log, bounds=b)

    p = res['params']

    # Calculate total bicelle thickness
    total_thickness = 2.0 * p['thick_face'] + p['length']

    print('\n' + '='*60)
    print('Fit success:', res['success'])
    print('Message:', res['message'])
    print('='*60)
    print('Parameters:')
    print('  radius              : {:.3f} Å'.format(p['radius']))
    print('  thick_rim           : {:.3f} Å'.format(p['thick_rim']))
    print('  thick_face          : {:.3f} Å'.format(p['thick_face']))
    print('  length              : {:.3f} Å'.format(p['length']))
    print(
        '  total_thickness     : {:.3f} Å  (2*thick_face + length)'.format(total_thickness))
    print('  sld_core            : {:.3f} (1e-6/Å²)'.format(p['sld_core']))
    print('  sld_face            : {:.3f} (1e-6/Å²)'.format(p['sld_face']))
    print('  sld_rim             : {:.3f} (1e-6/Å²)'.format(p['sld_rim']))
    print('  sld_solvent         : {:.3f} (1e-6/Å²)'.format(p['sld_solvent']))
    print('  scale               : {:.6g}'.format(p['scale']))
    print('  background          : {:.6g}'.format(p['background']))
    print('Cost (local)  :', res['cost'])
    print('Cost (global) :', res['global_cost'])
    print('='*60)

    if args.output:
        out = np.column_stack([q_fit, res['yfit']])
        np.savetxt(args.output, out, fmt=['%.6f', '%.8g'],
                   header='q\tI_fit', comments='')
        print('Wrote fit to', args.output)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.loglog(q, y, 'o', ms=3, alpha=0.3, label='data (all)')
        plt.loglog(q_fit, y_fit, 'o', ms=4, label='data (fit window)')
        plt.loglog(q_fit, res['yfit'], '-', lw=2, label='fit')
        plt.xlabel('q (1/Å)')
        plt.ylabel('I(q)')
        plt.legend()
        plt.title('Core-shell bicelle fit')
        plt.tight_layout()
        plt.show()

    return 0 if res['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
