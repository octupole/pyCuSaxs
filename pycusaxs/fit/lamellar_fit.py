#!/usr/bin/env python3
"""Robust fitter for the SasView lamellar_hg model.

Features
 - Physically motivated parameterization using total thickness + head fraction
 - Automatic initialization from oscillation period Δq (t_total ≈ π/Δq)
 - Global search (differential evolution) followed by local refinement
 - Weighted least-squares by default; optional log-residual objective
 - Optional plotting and export of the fitted curve

Usage
-----
python -m pycusaxs.fit.lamellar_fit -f pycusaxs/fit/data/c18-1-00-100.dat -p -o fit.dat
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .lamellar_model import (
    lamellar_hg_intensity,
    ht_from_total_fraction,
    estimate_periodic_delta_q,
)


@dataclass
class Bounds:
    # Bilayer thickness D = 2*(H+T)
    t_total: Tuple[float, float] = (30.0, 300.0)        # Angstroms
    frac_head: Tuple[float, float] = (0.05, 0.95)       # H/(H+T)
    dr_head: Tuple[float, float] = (-10.0, 10.0)        # 1e-6 / Ang^2
    dr_tail: Tuple[float, float] = (-10.0, 10.0)        # 1e-6 / Ang^2
    scale: Tuple[float, float] = (1e-10, 1e6)           # arbitrary
    background: Tuple[float, float] = (0.0, np.inf)     # arbitrary


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


def _objective_stage1(x: np.ndarray, q: np.ndarray, y: np.ndarray, bnds: Bounds) -> float:
    """Global search objective with analytic (s, b)."""
    t_total, frac_head, drh, drt = x
    H, T = ht_from_total_fraction(t_total, frac_head)
    # Model shape with unit scale/background
    M = lamellar_hg_intensity(q, H, T, drh, drt, scale=1.0, background=0.0)
    w = _weights_from_y(y)
    s, b = _ls_scale_background(M, y, w)
    r = (s * M + b) - y
    return float(np.sum((w * r) ** 2))


def _residuals_stage2(theta: np.ndarray, q: np.ndarray, y: np.ndarray, logspace: bool) -> np.ndarray:
    t_total, frac_head, drh, drt, s, b = theta
    H, T = ht_from_total_fraction(t_total, frac_head)
    yhat = lamellar_hg_intensity(q, H, T, drh, drt, scale=s, background=b)
    if logspace:
        eps = 1e-12
        return np.log(yhat + eps) - np.log(y + eps)
    w = _weights_from_y(y)
    return (yhat - y) * w


def fit_lamellar(
    q: np.ndarray,
    y: np.ndarray,
    use_logspace: bool = False,
    init_guess: Tuple[float, float, float, float] | None = None,
    bounds: Bounds | None = None,
    mask: Optional[np.ndarray] = None,
):
    """Fit lamellar_hg to (q, y).

    Returns a dict with parameters and diagnostics.
    """
    if bounds is None:
        bounds = Bounds()

    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        q = q[mask]
        y = y[mask]

    # Initial guess: for lamellar_hg the oscillation is driven by (H+T),
    # so bilayer thickness D ≈ 2 * π / Δq
    if init_guess is None:
        try:
            dq = estimate_periodic_delta_q(q, y)
            t_total0 = np.clip(2.0 * np.pi / max(dq, 1e-6), *bounds.t_total)
        except Exception:
            t_total0 = np.mean(bounds.t_total)
        frac_head0 = 0.25
        drh0, drt0 = -3.0, -5.6  # water-ish defaults
        init_guess = (t_total0, frac_head0, drh0, drt0)

    # Stage 1: global search over [t_total, frac_head, drh, drt]
    gl_bounds = [bounds.t_total, bounds.frac_head, bounds.dr_head, bounds.dr_tail]

    def fun_de(x):
        return _objective_stage1(x, q, y, bounds)

    result_de = differential_evolution(
        fun_de,
        bounds=gl_bounds,
        init='latinhypercube',
        strategy='best1bin',
        maxiter=200,
        popsize=10,
        tol=1e-6,
        polish=False,
        seed=42,
        updating='deferred',
        workers=1,
    )

    tt_opt, fh_opt, drh_opt, drt_opt = result_de.x
    H0, T0 = ht_from_total_fraction(tt_opt, fh_opt)
    M0 = lamellar_hg_intensity(q, H0, T0, drh_opt, drt_opt, scale=1.0, background=0.0)
    s0, b0 = _ls_scale_background(M0, y, _weights_from_y(y))

    # Stage 2: local refine on full parameter vector [t_total, frac_head, drh, drt, s, b]
    x0 = np.array([tt_opt, fh_opt, drh_opt, drt_opt, s0, b0], dtype=float)
    lower = np.array([
        bounds.t_total[0], bounds.frac_head[0], bounds.dr_head[0], bounds.dr_tail[0],
        bounds.scale[0], bounds.background[0]
    ])
    upper = np.array([
        bounds.t_total[1], bounds.frac_head[1], bounds.dr_head[1], bounds.dr_tail[1],
        bounds.scale[1], bounds.background[1] if np.isfinite(bounds.background[1]) else 1e12
    ])

    res_ls = least_squares(
        _residuals_stage2,
        x0,
        bounds=(lower, upper),
        args=(q, y, use_logspace),
        method='trf',
        loss='soft_l1',
        f_scale=0.1,
        x_scale='jac',
        max_nfev=4000,
        verbose=0,
    )

    tt, fh, drh, drt, s, b = res_ls.x
    H, T = ht_from_total_fraction(tt, fh)
    yfit = lamellar_hg_intensity(q, H, T, drh, drt, s, b)

    return {
        'success': bool(res_ls.success),
        'message': res_ls.message,
        'params': {
            'bilayer_thickness': float(tt),
            'length_head': float(H),
            'length_tail': float(T),
            'frac_head': float(fh),
            'dr_head': float(drh),
            'dr_tail': float(drt),
            'scale': float(s),
            'background': float(b),
        },
        'yfit': yfit,
        'cost': float(res_ls.cost),
        'nfev': int(res_ls.nfev),
        'global_cost': float(result_de.fun),
    }


def _load_xy(path: str):
    data = np.loadtxt(path)
    q = np.asarray(data[:, 0], dtype=float)
    i = np.asarray(data[:, 1], dtype=float)
    # Clean up: filter invalid values and nonpositive q
    m = np.isfinite(q) & np.isfinite(i) & (q > 0)
    q = q[m]
    i = i[m]
    return q, i


def _parse_exclude(arg_list: Sequence[str]) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for s in arg_list:
        if ':' in s:
            a, b = s.split(':', 1)
            try:
                lo = float(a); hi = float(b)
            except ValueError:
                continue
        else:
            # Single number = exclude below this q (lo-only)
            try:
                lo = float(s); hi = np.inf
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
    ap = argparse.ArgumentParser(description='Fit lamellar_hg model to I(q) data')
    ap.add_argument('-f', '--file', required=True, help='Two-column file: q I(q)')
    ap.add_argument('-p', '--plot', action='store_true', help='Show data and fit')
    ap.add_argument('-o', '--output', help='Write fitted I(q) to this file')
    ap.add_argument('--log', action='store_true', help='Use log-residual objective for local refinement')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    ap.add_argument('--qmin', type=float, help='Minimum q to include in fit')
    ap.add_argument('--qmax', type=float, help='Maximum q to include in fit')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Exclude q range(s) as lo:hi; repeatable. Example: --exclude 0.02:0.05 --exclude 1.6:inf')
    # Bounds tuning
    ap.add_argument('--tt-min', type=float, default=None, help='Lower bound for bilayer thickness 2(H+T) [A]')
    ap.add_argument('--tt-max', type=float, default=None, help='Upper bound for bilayer thickness 2(H+T) [A]')
    ap.add_argument('--frac-min', type=float, default=None, help='Lower bound for head fraction H/(H+T)')
    ap.add_argument('--frac-max', type=float, default=None, help='Upper bound for head fraction H/(H+T)')
    ap.add_argument('--drh-min', type=float, default=None, help='Lower bound for head contrast (1e-6/A^2)')
    ap.add_argument('--drh-max', type=float, default=None, help='Upper bound for head contrast (1e-6/A^2)')
    ap.add_argument('--drt-min', type=float, default=None, help='Lower bound for tail contrast (1e-6/A^2)')
    ap.add_argument('--drt-max', type=float, default=None, help='Upper bound for tail contrast (1e-6/A^2)')
    args = ap.parse_args(argv)

    # Load data
    q, y = _load_xy(args.file)

    # Apply q-window/mask as requested
    exclude_ranges = _parse_exclude(args.exclude)
    q_fit, y_fit, mask = _apply_q_window(q, y, args.qmin, args.qmax, exclude_ranges)
    if q_fit.size < 10:
        print('Warning: very few points in the selected q-window; consider widening it.', flush=True)

    # Build bounds from CLI overrides
    b = Bounds()
    if args.tt_min is not None:
        b.t_total = (float(args.tt_min), b.t_total[1])
    if args.tt_max is not None:
        b.t_total = (b.t_total[0], float(args.tt_max))
    if args.frac_min is not None:
        b.frac_head = (float(args.frac_min), b.frac_head[1])
    if args.frac_max is not None:
        b.frac_head = (b.frac_head[0], float(args.frac_max))
    if args.drh_min is not None:
        b.dr_head = (float(args.drh_min), b.dr_head[1])
    if args.drh_max is not None:
        b.dr_head = (b.dr_head[0], float(args.drh_max))
    if args.drt_min is not None:
        b.dr_tail = (float(args.drt_min), b.dr_tail[1])
    if args.drt_max is not None:
        b.dr_tail = (b.dr_tail[0], float(args.drt_max))

    # Clamp and sort bounds if needed
    def _sorted_pair(p):
        a, c = p
        return (min(a, c), max(a, c))
    b.t_total = _sorted_pair(b.t_total)
    b.frac_head = (max(1e-6, b.frac_head[0]), min(1.0-1e-6, b.frac_head[1]))
    b.frac_head = _sorted_pair(b.frac_head)

    # Fit
    res = fit_lamellar(q_fit, y_fit, use_logspace=args.log, bounds=b)

    p = res['params']
    print('Fit success:', res['success'])
    print('Message:', res['message'])
    print('Parameters:')
    print('  bilayer thickness 2(H+T): {:.3f} A'.format(p['bilayer_thickness']))
    print('  head thickness            : {:.3f} A'.format(p['length_head']))
    print('  tail thickness            : {:.3f} A'.format(p['length_tail']))
    print('  head fraction  H/(H+T)    : {:.3f}'.format(p['frac_head']))
    print('  head contrast (dr_head)   : {:.3f} (1e-6/A^2)'.format(p['dr_head']))
    print('  tail contrast (dr_tail)   : {:.3f} (1e-6/A^2)'.format(p['dr_tail']))
    print('  scale                     : {:.6g}'.format(p['scale']))
    print('  background                : {:.6g}'.format(p['background']))
    print('Cost (local)  :', res['cost'])
    print('Cost (global) :', res['global_cost'])

    if args.output:
        # For output, evaluate on the filtered q-grid so it overlays the used window
        out = np.column_stack([q_fit, res['yfit']])
        np.savetxt(args.output, out, fmt=['%.6f', '%.8g'], header='q	I_fit', comments='')
        print('Wrote fit to', args.output)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        # show all data in light color
        plt.loglog(q, y, 'o', ms=3, alpha=0.3, label='data (all)')
        # highlight used window
        plt.loglog(q_fit, y_fit, 'o', ms=4, label='data (fit window)')
        plt.loglog(q_fit, res['yfit'], '-', lw=2, label='fit')
        plt.xlabel('q (1/Å)')
        plt.ylabel('I(q)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return 0 if res['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
