# CMA-ES Implementation for Bicelle Fitting

## Overview

This directory contains two implementations of the bicelle fitter:

1. **`bicelle_fit.py`** - Uses Differential Evolution (DE) for global optimization
2. **`bicelle_fit_cma.py`** - Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Both scripts perform two-stage optimization:
- **Stage 1**: Global search (DE or CMA-ES) with analytical scale/background optimization
- **Stage 2**: Local refinement using Trust Region Least Squares with Soft L1 loss

## Installation

The CMA-ES version requires the `cma` package:
```bash
pip install cma
```

## Usage

### Differential Evolution (Default)
```bash
bicelle-fit -f data.dat --qmin 0.02 --qmax 0.6 --radius-min 90 --radius-max 140
```

### CMA-ES Version
```bash
python -m pycusaxs.fit.bicelle_fit_cma -f data.dat --qmin 0.02 --qmax 0.6 --radius-min 90 --radius-max 140
```

## CMA-ES Specific Options

- `--sigma0 FLOAT` - Initial step size (default: 0.3)
  - Fraction of the search space width
  - Larger values = more global exploration
  - Smaller values = faster convergence but higher risk of local minima

- `--maxfevals INT` - Maximum function evaluations (default: 2000)
  - DE default is maxiter=300 × popsize=15 = 4500 evaluations
  - CMA-ES often converges in fewer evaluations

- `--verbose` - Show detailed CMA-ES progress

## Performance Comparison

On `test_data.dat` (q-range 0.02-0.6, radius 90-140 Å):

| Method | Cost (final) | Function Evaluations | Time | Parameters Found |
|--------|-------------|---------------------|------|------------------|
| **DE** | 0.902 | ~4500 | ~5s | radius=140.0 Å, thick_face=6.9 Å |
| **CMA-ES** | 0.988 | 2010 | ~4s | radius=99.4 Å, thick_face=7.8 Å |

### Key Observations

1. **DE found a better minimum** in this test case (lower cost)
2. **CMA-ES used fewer function evaluations** (~45% of DE)
3. **Both converged to different local minima** (non-convex optimization landscape)
4. **Similar computational time** because bicelle model evaluation is expensive

## When to Use CMA-ES

CMA-ES may be preferable when:

- **Function evaluations are expensive** - CMA-ES typically needs fewer evaluations
- **Smooth fitness landscape** - CMA-ES excels on continuous, differentiable problems
- **Good initial guess available** - CMA-ES can exploit local structure better
- **Medium dimensionality (5-15 parameters)** - CMA-ES scaling advantage

## When to Use Differential Evolution

DE may be preferable when:

- **Highly multimodal landscape** - DE's population diversity helps escape local minima
- **Robustness is critical** - DE is less sensitive to hyperparameter choices
- **No dependencies beyond scipy** - DE is built into scipy.optimize
- **Larger populations needed** - DE can easily scale population size

## Algorithm Details

### CMA-ES Adaptation

The key difference from DE is that CMA-ES:
1. Maintains a full covariance matrix of the search distribution
2. Adapts the covariance based on successful steps
3. Uses rank-based selection (not fitness values directly)
4. Automatically adjusts step size σ

This allows CMA-ES to:
- Learn correlations between parameters
- Orient the search ellipsoid along valleys
- Take larger steps when improving, smaller when stagnating

### Parameter Space

Both methods optimize 8 structural parameters:
- `radius` (10-200 Å)
- `thick_rim` (1-40 Å)
- `thick_face` (1-50 Å)
- `length` (10-50 Å)
- `sld_core` (-10 to +10 × 10⁻⁶/Ų)
- `sld_face` (-10 to +10 × 10⁻⁶/Ų)
- `sld_rim` (-10 to +10 × 10⁻⁶/Ų)
- `sld_solvent` (-10 to +10 × 10⁻⁶/Ų)

The `scale` and `background` parameters are solved analytically in Stage 1.

## Recommendations

For routine fitting, **use DE (bicelle-fit)** because:
- More robust across different datasets
- No additional dependencies
- Better documented and tested

For research or when DE is too slow:
- **Try CMA-ES with default settings first**
- If CMA-ES gets stuck, increase `--sigma0` (try 0.5)
- If CMA-ES doesn't converge, increase `--maxfevals` (try 4000)
- Always verify fit quality visually with `--plot`

## Multiple Runs Strategy

Because both methods can find different local minima, consider:

```bash
# Run both and compare
bicelle-fit -f data.dat --qmin 0.02 --qmax 0.6 -o fit_de.dat
python -m pycusaxs.fit.bicelle_fit_cma -f data.dat --qmin 0.02 --qmax 0.6 -o fit_cma.dat

# Compare visually
python compare_fits.py
```

Choose the fit with:
1. Lower final cost
2. More physically reasonable parameters
3. Better visual agreement in critical q-ranges

## References

- CMA-ES: Hansen & Ostermeier (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies"
- Differential Evolution: Storn & Price (1997). "Differential Evolution – A Simple and Efficient Heuristic"
- Python CMA package: https://github.com/CMA-ES/pycma

## Author Notes

This implementation was created to evaluate whether CMA-ES offers advantages over DE for SAXS model fitting. Initial tests show comparable performance, with each method having situational advantages. The choice of optimizer appears less important than proper data quality, q-range selection, and parameter bounds.
