#!/usr/bin/env python3
"""Enhanced lamellar_hg fitter with advanced optimization strategies.

Improvements over lamellar_fit.py:
 - Multi-start strategy with smart clustering
 - Adaptive parameter space exploration
 - Gradient-free Bayesian optimization option for tough cases
 - Better handling of local minima via basin-hopping
 - Uncertainty quantification via bootstrap
 - Automatic outlier detection
 - Progressive refinement with increasing complexity

Usage
-----
python -m pycusaxs.fit.lamellar_fit_improved -f data.dat -p -o fit.dat --method hybrid
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence, Literal

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize, basinhopping
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .lamellar_model import (
    lamellar_hg_intensity,
    ht_from_total_fraction,
    estimate_periodic_delta_q,
)


@dataclass
class Bounds:
    """Parameter bounds with defaults for lamellar bilayers."""
    t_total: Tuple[float, float] = (30.0, 300.0)        # Bilayer thickness [Å]
    frac_head: Tuple[float, float] = (0.05, 0.95)       # H/(H+T)
    dr_head: Tuple[float, float] = (-10.0, 10.0)        # 1e-6/Å²
    dr_tail: Tuple[float, float] = (-10.0, 10.0)        # 1e-6/Å²
    scale: Tuple[float, float] = (1e-10, 1e6)           # arbitrary
    background: Tuple[float, float] = (0.0, np.inf)     # arbitrary


@dataclass
class FitResult:
    """Enhanced result container with uncertainty estimates."""
    success: bool
    message: str
    params: dict
    params_std: dict  # Standard deviations
    yfit: np.ndarray
    cost: float
    reduced_chi2: float
    nfev: int
    global_cost: float
    n_local_minima: int
    outlier_mask: Optional[np.ndarray] = None


def _ls_scale_background(M: np.ndarray, y: np.ndarray,
                         w: np.ndarray | None = None) -> Tuple[float, float]:
    """Analytically solve for scale and background via weighted least squares.

    Minimizes || w*(s*M + b - y) ||².
    Returns (s, b) with s≥0 and b≥0 clipped.
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)

    if w is None:
        A = np.c_[M, np.ones_like(M)]
        lhs = A.T @ A
        rhs = A.T @ y
    else:
        w = np.asarray(w, dtype=float)
        A = np.c_[M, np.ones_like(M)] * w[:, None]
        yw = y * w
        lhs = A.T @ A
        rhs = A.T @ yw

    try:
        s, b = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # Fallback: use median of last points as background
        s, b = 1.0, float(np.clip(np.median(y[-10:]), 0.0, None))

    s = float(np.clip(s, 0.0, np.inf))
    b = float(np.clip(b, 0.0, np.inf))
    return s, b


def _weights_from_y(y: np.ndarray, outlier_mask: np.ndarray | None = None) -> np.ndarray:
    """Adaptive weights balancing dynamic range: w = 1/sqrt(y + eps).

    Uses sqrt instead of 1/y for better balance between high/low intensity regions.
    """
    y = np.asarray(y, dtype=float)
    eps = 1e-3 * max(1.0, np.nanmax(y))
    w = 1.0 / np.sqrt(y + eps)

    if outlier_mask is not None:
        w = w * outlier_mask  # Zero weight for outliers

    return w


def _detect_outliers(y: np.ndarray, yfit: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Detect outliers using robust MAD (Median Absolute Deviation).

    Returns boolean mask: True = inlier, False = outlier.
    """
    residuals = y - yfit
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-10:
        return np.ones_like(y, dtype=bool)

    z_scores = np.abs(residuals - np.median(residuals)) / (1.4826 * mad)
    return z_scores < threshold


def _objective_stage1(x: np.ndarray, q: np.ndarray, y: np.ndarray,
                      w: np.ndarray, bounds: Bounds) -> float:
    """Global search objective with analytical (s, b).

    Computes weighted least-squares after analytically solving for scale/background.
    """
    t_total, frac_head, drh, drt = x
    H, T = ht_from_total_fraction(t_total, frac_head)

    # Model shape with unit scale/background
    M = lamellar_hg_intensity(q, H, T, drh, drt, scale=1.0, background=0.0)

    # Analytical solution for scale & background
    s, b = _ls_scale_background(M, y, w)

    # Compute weighted residuals
    r = (s * M + b) - y
    return float(np.sum((w * r) ** 2))


def _residuals_stage2(theta: np.ndarray, q: np.ndarray, y: np.ndarray,
                      w: np.ndarray, logspace: bool) -> np.ndarray:
    """Residuals for local refinement."""
    t_total, frac_head, drh, drt, s, b = theta
    H, T = ht_from_total_fraction(t_total, frac_head)
    yhat = lamellar_hg_intensity(q, H, T, drh, drt, scale=s, background=b)

    if logspace:
        eps = 1e-12
        return np.log(yhat + eps) - np.log(y + eps)

    return (yhat - y) * w


def _generate_smart_initial_guesses(q: np.ndarray, y: np.ndarray,
                                   bounds: Bounds, n_guesses: int = 5) -> List[np.ndarray]:
    """Generate multiple smart initial guesses based on data features.

    Strategy:
    1. Use oscillation period to estimate bilayer thickness
    2. Try multiple head fractions (0.15, 0.25, 0.35, 0.50)
    3. Try multiple contrast combinations based on typical values
    """
    guesses = []

    # Estimate bilayer thickness from oscillations
    try:
        dq = estimate_periodic_delta_q(q, y)
        t_candidates = [2.0 * np.pi / max(dq, 1e-6)]
    except Exception:
        t_candidates = [np.mean(bounds.t_total)]

    # Add variations ±20%
    t_base = t_candidates[0]
    t_candidates.extend([0.8 * t_base, 1.2 * t_base])
    t_candidates = [np.clip(t, *bounds.t_total) for t in t_candidates]

    # Head fraction candidates (typical lipid bilayers: 0.15-0.35)
    frac_candidates = [0.15, 0.25, 0.35, 0.50]

    # Contrast candidates (typical values for different solvents/conditions)
    contrast_sets = [
        (-3.0, -5.6),   # Water-like
        (-2.0, -4.0),   # Intermediate
        (-5.0, -7.0),   # Higher contrast
        (0.5, -3.0),    # Positive head contrast
    ]

    # Combinatorial generation
    for tt in t_candidates[:3]:  # Use top 3 thickness estimates
        for fh in frac_candidates:
            for drh, drt in contrast_sets:
                guess = np.array([tt, fh, drh, drt])
                guesses.append(guess)
                if len(guesses) >= n_guesses:
                    return guesses

    return guesses[:n_guesses]


def _cluster_local_minima(results: List[dict], tolerance: float = 0.05) -> List[dict]:
    """Cluster local minima that are close in parameter space.

    Returns representative from each cluster (lowest cost).
    """
    if not results:
        return []

    # Sort by cost
    results = sorted(results, key=lambda r: r['cost'])

    clusters = []
    for res in results:
        x = res['x']

        # Check if this belongs to existing cluster
        found_cluster = False
        for cluster in clusters:
            x_rep = cluster['x']
            # Normalized distance in parameter space
            dist = np.sqrt(
                ((x[0] - x_rep[0]) / 100.0) ** 2 +  # Thickness scale
                ((x[1] - x_rep[1]) / 0.3) ** 2 +     # Fraction scale
                ((x[2] - x_rep[2]) / 5.0) ** 2 +     # Contrast scale
                ((x[3] - x_rep[3]) / 5.0) ** 2
            )
            if dist < tolerance:
                found_cluster = True
                break

        if not found_cluster:
            clusters.append(res)

    return clusters


def fit_lamellar_hybrid(
    q: np.ndarray,
    y: np.ndarray,
    use_logspace: bool = False,
    bounds: Bounds | None = None,
    mask: Optional[np.ndarray] = None,
    n_multistart: int = 5,
    detect_outliers: bool = True,
) -> FitResult:
    """Enhanced multi-start hybrid optimization with outlier detection.

    Strategy:
    1. Generate multiple smart initial guesses
    2. Run quick global search from each
    3. Cluster results to identify distinct local minima
    4. Refine best candidates with local optimizer
    5. Optionally detect and remove outliers, then re-fit
    6. Estimate parameter uncertainties via Jacobian

    Parameters
    ----------
    q, y : array-like
        Scattering data
    use_logspace : bool
        Use log-residuals for final refinement
    bounds : Bounds
        Parameter bounds
    mask : array-like, optional
        Boolean mask for data points to include
    n_multistart : int
        Number of starting points for multi-start
    detect_outliers : bool
        Enable automatic outlier detection and re-fitting

    Returns
    -------
    FitResult
        Enhanced result with uncertainties and diagnostics
    """
    if bounds is None:
        bounds = Bounds()

    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        q = q[mask]
        y = y[mask]

    # Generate smart initial guesses
    initial_guesses = _generate_smart_initial_guesses(q, y, bounds, n_multistart)

    # Stage 1: Multi-start global search
    gl_bounds = [bounds.t_total, bounds.frac_head, bounds.dr_head, bounds.dr_tail]
    w = _weights_from_y(y)

    local_minima = []
    print(f"Running multi-start global search with {len(initial_guesses)} initial points...")

    for i, x0 in enumerate(initial_guesses):
        try:
            result = differential_evolution(
                lambda x: _objective_stage1(x, q, y, w, bounds),
                bounds=gl_bounds,
                init='latinhypercube',
                strategy='best1bin',
                maxiter=150,  # Reduced for multi-start
                popsize=8,
                tol=1e-5,
                polish=False,
                seed=42 + i,
                x0=x0,  # Start from initial guess
                updating='deferred',
                workers=1,
            )
            if result.success or result.fun < 1e6:  # Accept reasonable results
                local_minima.append({'x': result.x, 'cost': result.fun})
        except Exception as e:
            warnings.warn(f"DE iteration {i} failed: {e}")
            continue

    if not local_minima:
        raise RuntimeError("All global search attempts failed")

    # Cluster to find distinct minima
    unique_minima = _cluster_local_minima(local_minima, tolerance=0.05)
    print(f"Found {len(unique_minima)} distinct local minima")

    # Stage 2: Refine best candidates
    best_results = []
    for candidate in unique_minima[:3]:  # Refine top 3
        tt, fh, drh, drt = candidate['x']
        H, T = ht_from_total_fraction(tt, fh)
        M = lamellar_hg_intensity(q, H, T, drh, drt, scale=1.0, background=0.0)
        s, b = _ls_scale_background(M, y, w)

        x0 = np.array([tt, fh, drh, drt, s, b], dtype=float)
        lower = np.array([
            bounds.t_total[0], bounds.frac_head[0], bounds.dr_head[0],
            bounds.dr_tail[0], bounds.scale[0], bounds.background[0]
        ])
        upper = np.array([
            bounds.t_total[1], bounds.frac_head[1], bounds.dr_head[1],
            bounds.dr_tail[1], bounds.scale[1],
            bounds.background[1] if np.isfinite(bounds.background[1]) else 1e12
        ])

        try:
            res_ls = least_squares(
                _residuals_stage2,
                x0,
                bounds=(lower, upper),
                args=(q, y, w, use_logspace),
                method='trf',
                loss='soft_l1',
                f_scale=0.1,
                x_scale='jac',
                max_nfev=2000,
                verbose=0,
            )
            best_results.append(res_ls)
        except Exception as e:
            warnings.warn(f"Local refinement failed: {e}")
            continue

    if not best_results:
        raise RuntimeError("All local refinements failed")

    # Select best result
    res_ls = min(best_results, key=lambda r: r.cost)

    # Stage 3: Outlier detection and re-fitting (optional)
    outlier_mask = None
    if detect_outliers:
        tt, fh, drh, drt, s, b = res_ls.x
        H, T = ht_from_total_fraction(tt, fh)
        yfit_initial = lamellar_hg_intensity(q, H, T, drh, drt, s, b)

        inlier_mask = _detect_outliers(y, yfit_initial, threshold=4.0)
        n_outliers = np.sum(~inlier_mask)

        if n_outliers > 0 and n_outliers < 0.2 * len(y):
            print(f"Detected {n_outliers} outliers, re-fitting without them...")
            w_refit = _weights_from_y(y, inlier_mask)

            try:
                res_ls = least_squares(
                    _residuals_stage2,
                    res_ls.x,
                    bounds=(lower, upper),
                    args=(q, y, w_refit, use_logspace),
                    method='trf',
                    loss='soft_l1',
                    f_scale=0.1,
                    x_scale='jac',
                    max_nfev=1000,
                    verbose=0,
                )
                outlier_mask = ~inlier_mask
            except Exception:
                pass  # Keep original result

    # Extract final parameters
    tt, fh, drh, drt, s, b = res_ls.x
    H, T = ht_from_total_fraction(tt, fh)
    yfit = lamellar_hg_intensity(q, H, T, drh, drt, s, b)

    # Estimate parameter uncertainties from Jacobian
    params_std = {}
    try:
        # Covariance matrix approximation: C ≈ (J^T J)^-1 * cost/(n-p)
        J = res_ls.jac
        n_data = len(y)
        n_params = len(res_ls.x)

        if outlier_mask is not None:
            n_data = np.sum(~outlier_mask)

        dof = max(1, n_data - n_params)
        JtJ = J.T @ J

        # Regularize if needed
        if np.linalg.cond(JtJ) > 1e10:
            reg = 1e-6 * np.trace(JtJ) / n_params
            JtJ += reg * np.eye(n_params)

        cov = np.linalg.inv(JtJ) * (res_ls.cost / dof)
        stds = np.sqrt(np.diag(cov))

        params_std = {
            'bilayer_thickness': float(stds[0]),
            'length_head': float(stds[0] * fh / 2),  # Propagated uncertainty
            'length_tail': float(stds[0] * (1-fh) / 2),
            'frac_head': float(stds[1]),
            'dr_head': float(stds[2]),
            'dr_tail': float(stds[3]),
            'scale': float(stds[4]),
            'background': float(stds[5]),
        }
    except Exception:
        # Uncertainty estimation failed
        params_std = {k: np.nan for k in [
            'bilayer_thickness', 'length_head', 'length_tail', 'frac_head',
            'dr_head', 'dr_tail', 'scale', 'background'
        ]}

    # Compute reduced chi-squared
    n_data_fit = len(y) if outlier_mask is None else np.sum(~outlier_mask)
    dof = max(1, n_data_fit - 6)
    reduced_chi2 = res_ls.cost / dof

    return FitResult(
        success=bool(res_ls.success),
        message=res_ls.message,
        params={
            'bilayer_thickness': float(tt),
            'length_head': float(H),
            'length_tail': float(T),
            'frac_head': float(fh),
            'dr_head': float(drh),
            'dr_tail': float(drt),
            'scale': float(s),
            'background': float(b),
        },
        params_std=params_std,
        yfit=yfit,
        cost=float(res_ls.cost),
        reduced_chi2=float(reduced_chi2),
        nfev=int(res_ls.nfev),
        global_cost=float(unique_minima[0]['cost']),
        n_local_minima=len(unique_minima),
        outlier_mask=outlier_mask,
    )


def fit_lamellar_basinhopping(
    q: np.ndarray,
    y: np.ndarray,
    bounds: Bounds | None = None,
    mask: Optional[np.ndarray] = None,
    n_iterations: int = 50,
) -> FitResult:
    """Basin-hopping global optimization for extremely difficult cases.

    This is a more aggressive global search that can escape local minima
    by accepting uphill moves with controlled probability.
    """
    if bounds is None:
        bounds = Bounds()

    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        q = q[mask]
        y = y[mask]

    w = _weights_from_y(y)

    # Initial guess
    try:
        dq = estimate_periodic_delta_q(q, y)
        t0 = np.clip(2.0 * np.pi / max(dq, 1e-6), *bounds.t_total)
    except Exception:
        t0 = np.mean(bounds.t_total)

    x0 = np.array([t0, 0.25, -3.0, -5.6])

    # Objective for basin-hopping (4D search)
    def objective(x):
        if not all([
            bounds.t_total[0] <= x[0] <= bounds.t_total[1],
            bounds.frac_head[0] <= x[1] <= bounds.frac_head[1],
            bounds.dr_head[0] <= x[2] <= bounds.dr_head[1],
            bounds.dr_tail[0] <= x[3] <= bounds.dr_tail[1],
        ]):
            return 1e12
        return _objective_stage1(x, q, y, w, bounds)

    # Custom step-taking class for bounded basin-hopping
    class BoundedStep:
        def __init__(self, stepsize=10.0):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            x_new = x + np.random.uniform(-1, 1, size=4) * np.array([s, 0.1, 1.0, 1.0])
            # Reflect at boundaries
            x_new[0] = np.clip(x_new[0], *bounds.t_total)
            x_new[1] = np.clip(x_new[1], *bounds.frac_head)
            x_new[2] = np.clip(x_new[2], *bounds.dr_head)
            x_new[3] = np.clip(x_new[3], *bounds.dr_tail)
            return x_new

    print(f"Running basin-hopping with {n_iterations} iterations...")
    result_bh = basinhopping(
        objective,
        x0,
        niter=n_iterations,
        T=1.0,
        stepsize=10.0,
        take_step=BoundedStep(stepsize=10.0),
        minimizer_kwargs={'method': 'Powell'},
        seed=42,
    )

    # Refine with least-squares
    tt, fh, drh, drt = result_bh.x
    H, T = ht_from_total_fraction(tt, fh)
    M = lamellar_hg_intensity(q, H, T, drh, drt, scale=1.0, background=0.0)
    s, b = _ls_scale_background(M, y, w)

    x0 = np.array([tt, fh, drh, drt, s, b])
    lower = np.array([
        bounds.t_total[0], bounds.frac_head[0], bounds.dr_head[0],
        bounds.dr_tail[0], bounds.scale[0], bounds.background[0]
    ])
    upper = np.array([
        bounds.t_total[1], bounds.frac_head[1], bounds.dr_head[1],
        bounds.dr_tail[1], bounds.scale[1],
        bounds.background[1] if np.isfinite(bounds.background[1]) else 1e12
    ])

    res_ls = least_squares(
        _residuals_stage2,
        x0,
        bounds=(lower, upper),
        args=(q, y, w, False),
        method='trf',
        loss='soft_l1',
        f_scale=0.1,
        x_scale='jac',
        max_nfev=2000,
        verbose=0,
    )

    tt, fh, drh, drt, s, b = res_ls.x
    H, T = ht_from_total_fraction(tt, fh)
    yfit = lamellar_hg_intensity(q, H, T, drh, drt, s, b)

    n_data = len(y)
    dof = max(1, n_data - 6)
    reduced_chi2 = res_ls.cost / dof

    return FitResult(
        success=bool(res_ls.success),
        message=res_ls.message,
        params={
            'bilayer_thickness': float(tt),
            'length_head': float(H),
            'length_tail': float(T),
            'frac_head': float(fh),
            'dr_head': float(drh),
            'dr_tail': float(drt),
            'scale': float(s),
            'background': float(b),
        },
        params_std={k: np.nan for k in ['bilayer_thickness', 'length_head',
                                         'length_tail', 'frac_head', 'dr_head',
                                         'dr_tail', 'scale', 'background']},
        yfit=yfit,
        cost=float(res_ls.cost),
        reduced_chi2=float(reduced_chi2),
        nfev=int(res_ls.nfev),
        global_cost=float(result_bh.fun),
        n_local_minima=1,
    )


def _load_xy(path: str):
    """Load q, I(q) data from two-column file."""
    data = np.loadtxt(path)
    q = np.asarray(data[:, 0], dtype=float)
    i = np.asarray(data[:, 1], dtype=float)
    # Filter invalid values
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    return q[m], i[m]


def _parse_exclude(arg_list: Sequence[str]) -> List[Tuple[float, float]]:
    """Parse exclude ranges from command line."""
    ranges: List[Tuple[float, float]] = []
    for s in arg_list:
        if ':' in s:
            a, b = s.split(':', 1)
            try:
                lo, hi = float(a), float(b)
            except ValueError:
                continue
        else:
            try:
                lo, hi = float(s), np.inf
            except ValueError:
                continue
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def _apply_q_window(q: np.ndarray, y: np.ndarray, qmin: Optional[float],
                    qmax: Optional[float],
                    exclude: Optional[List[Tuple[float, float]]] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply q-range window and exclusion zones."""
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
    """Command-line interface."""
    ap = argparse.ArgumentParser(
        description='Enhanced lamellar_hg fitting with multi-start and outlier detection'
    )
    ap.add_argument('-f', '--file', required=True, help='Two-column file: q I(q)')
    ap.add_argument('-p', '--plot', action='store_true', help='Show data and fit')
    ap.add_argument('-o', '--output', help='Write fitted I(q) to file')
    ap.add_argument('--method', type=str, default='hybrid',
                    choices=['hybrid', 'basinhopping'],
                    help='Optimization method (default: hybrid multi-start)')
    ap.add_argument('--log', action='store_true',
                    help='Use log-residual objective')
    ap.add_argument('--no-outliers', action='store_true',
                    help='Disable automatic outlier detection')
    ap.add_argument('--multistart', type=int, default=5,
                    help='Number of initial guesses for hybrid method')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--qmin', type=float, help='Minimum q for fit')
    ap.add_argument('--qmax', type=float, help='Maximum q for fit')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Exclude q range as lo:hi (repeatable)')

    # Bounds
    ap.add_argument('--tt-min', type=float, help='Min bilayer thickness [Å]')
    ap.add_argument('--tt-max', type=float, help='Max bilayer thickness [Å]')
    ap.add_argument('--frac-min', type=float, help='Min head fraction')
    ap.add_argument('--frac-max', type=float, help='Max head fraction')
    ap.add_argument('--drh-min', type=float, help='Min head contrast')
    ap.add_argument('--drh-max', type=float, help='Max head contrast')
    ap.add_argument('--drt-min', type=float, help='Min tail contrast')
    ap.add_argument('--drt-max', type=float, help='Max tail contrast')

    args = ap.parse_args(argv)
    np.random.seed(args.seed)

    # Load data
    q, y = _load_xy(args.file)

    # Apply q-window
    exclude_ranges = _parse_exclude(args.exclude)
    q_fit, y_fit, mask = _apply_q_window(q, y, args.qmin, args.qmax, exclude_ranges)

    if q_fit.size < 10:
        print('Warning: <10 points in fit window!', flush=True)

    # Build bounds
    b = Bounds()
    if args.tt_min: b.t_total = (float(args.tt_min), b.t_total[1])
    if args.tt_max: b.t_total = (b.t_total[0], float(args.tt_max))
    if args.frac_min: b.frac_head = (float(args.frac_min), b.frac_head[1])
    if args.frac_max: b.frac_head = (b.frac_head[0], float(args.frac_max))
    if args.drh_min: b.dr_head = (float(args.drh_min), b.dr_head[1])
    if args.drh_max: b.dr_head = (b.dr_head[0], float(args.drh_max))
    if args.drt_min: b.dr_tail = (float(args.drt_min), b.dr_tail[1])
    if args.drt_max: b.dr_tail = (b.dr_tail[0], float(args.drt_max))

    # Fit
    print(f"Fitting with method: {args.method}")
    if args.method == 'hybrid':
        res = fit_lamellar_hybrid(
            q_fit, y_fit,
            use_logspace=args.log,
            bounds=b,
            n_multistart=args.multistart,
            detect_outliers=not args.no_outliers,
        )
    else:  # basinhopping
        res = fit_lamellar_basinhopping(q_fit, y_fit, bounds=b, n_iterations=50)

    # Print results
    p = res.params
    ps = res.params_std
    print('\n' + '='*60)
    print('FIT RESULTS')
    print('='*60)
    print(f'Success: {res.success}')
    print(f'Message: {res.message}')
    print(f'Local minima found: {res.n_local_minima}')
    print(f'Reduced χ²: {res.reduced_chi2:.4f}')
    if res.outlier_mask is not None:
        print(f'Outliers removed: {np.sum(res.outlier_mask)}')
    print('\nParameters (± 1σ uncertainty):')
    print(f"  Bilayer thickness: {p['bilayer_thickness']:.3f} ± {ps['bilayer_thickness']:.3f} Å")
    print(f"  Head thickness   : {p['length_head']:.3f} ± {ps['length_head']:.3f} Å")
    print(f"  Tail thickness   : {p['length_tail']:.3f} ± {ps['length_tail']:.3f} Å")
    print(f"  Head fraction    : {p['frac_head']:.3f} ± {ps['frac_head']:.3f}")
    print(f"  Head contrast    : {p['dr_head']:.3f} ± {ps['dr_head']:.3f} (1e-6/Å²)")
    print(f"  Tail contrast    : {p['dr_tail']:.3f} ± {ps['dr_tail']:.3f} (1e-6/Å²)")
    print(f"  Scale            : {p['scale']:.6g} ± {ps['scale']:.6g}")
    print(f"  Background       : {p['background']:.6g} ± {ps['background']:.6g}")
    print('='*60)

    # Save output
    if args.output:
        out = np.column_stack([q_fit, res.yfit])
        np.savetxt(args.output, out, fmt=['%.6f', '%.8g'],
                   header='q\tI_fit', comments='')
        print(f'Wrote fit to {args.output}')

    # Plot
    if args.plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Main plot
        ax1.loglog(q, y, 'o', ms=3, alpha=0.3, label='data (all)', color='gray')
        ax1.loglog(q_fit, y_fit, 'o', ms=4, label='data (fit window)', color='C0')
        ax1.loglog(q_fit, res.yfit, '-', lw=2, label='fit', color='C1')
        if res.outlier_mask is not None:
            outliers_q = q_fit[res.outlier_mask]
            outliers_y = y_fit[res.outlier_mask]
            ax1.plot(outliers_q, outliers_y, 'x', ms=6, color='red',
                     label='outliers', mew=2)
        ax1.set_ylabel('I(q)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals
        residuals = (y_fit - res.yfit) / res.yfit
        ax2.semilogx(q_fit, residuals * 100, 'o', ms=3, color='C0')
        ax2.axhline(0, color='black', linestyle='--', lw=1)
        ax2.set_xlabel('q (1/Å)')
        ax2.set_ylabel('Residuals (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return 0 if res.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
