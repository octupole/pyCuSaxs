# Lamellar Fitting Scripts Installation Guide

## Overview

Two lamellar SAXS fitting scripts are now available as executable commands after installation:

1. **`lamellar-fit`** - Original fitting script with differential evolution + local refinement
2. **`lamellar-fit-improved`** - Enhanced version with multi-start, outlier detection, and uncertainty quantification

## Installation

After installing the package, both scripts will be available as commands:

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

This will create executable scripts in your environment's `bin/` directory:
- `lamellar-fit`
- `lamellar-fit-improved`

## Usage

### Original Fitting Script

```bash
# Basic usage
lamellar-fit -f data.dat -p -o fit.dat

# With q-range selection
lamellar-fit -f data.dat -p --qmin 0.01 --qmax 0.5

# With parameter bounds
lamellar-fit -f data.dat -p \
    --tt-min 40 --tt-max 80 \
    --frac-min 0.2 --frac-max 0.4

# Exclude specific q-ranges (e.g., artifacts, noise)
lamellar-fit -f data.dat -p \
    --exclude 0.02:0.05 \
    --exclude 1.6:inf
```

### Improved Fitting Script (Recommended)

```bash
# Hybrid multi-start method (default, recommended)
lamellar-fit-improved -f data.dat -p -o fit.dat

# With more starting points for difficult fits
lamellar-fit-improved -f data.dat -p --multistart 10

# Basin-hopping for extremely difficult cases
lamellar-fit-improved -f data.dat -p --method basinhopping

# Disable automatic outlier detection
lamellar-fit-improved -f data.dat -p --no-outliers

# With parameter constraints
lamellar-fit-improved -f data.dat -p \
    --tt-min 40 --tt-max 80 \
    --frac-min 0.2 --frac-max 0.4 \
    --drh-min -8 --drh-max 0 \
    --drt-min -10 --drt-max -2
```

## Command-Line Options

### Common Options (Both Scripts)

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --file` | Input file (q, I(q) two-column) | Required |
| `-p, --plot` | Show plot of data and fit | Off |
| `-o, --output` | Save fitted I(q) to file | None |
| `--qmin` | Minimum q for fitting | None |
| `--qmax` | Maximum q for fitting | None |
| `--exclude` | Exclude q range as `lo:hi` (repeatable) | None |
| `--tt-min/max` | Bilayer thickness bounds [Å] | 30-300 |
| `--frac-min/max` | Head fraction H/(H+T) bounds | 0.05-0.95 |
| `--drh-min/max` | Head contrast bounds [10⁻⁶/Å²] | -10 to 10 |
| `--drt-min/max` | Tail contrast bounds [10⁻⁶/Å²] | -10 to 10 |

### Improved Script Additional Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method` | Optimization method: `hybrid` or `basinhopping` | `hybrid` |
| `--multistart` | Number of initial guesses for hybrid | 5 |
| `--no-outliers` | Disable automatic outlier detection | Off |
| `--seed` | Random seed for reproducibility | 42 |

## Output

### Standard Output

Both scripts print:
- Fit success status
- Final parameter values with physical units
- Fitting diagnostics (cost, number of function evaluations)

### Improved Script Additional Output

- **Parameter uncertainties** (± 1σ standard deviations)
- **Reduced χ²** statistic (goodness-of-fit)
- **Number of local minima** found
- **Outlier count** (if detected and removed)

Example:
```
============================================================
FIT RESULTS
============================================================
Success: True
Message: Optimization terminated successfully.
Local minima found: 3
Reduced χ²: 1.2345
Outliers removed: 2

Parameters (± 1σ uncertainty):
  Bilayer thickness: 56.234 ± 0.123 Å
  Head thickness   : 8.435 ± 0.087 Å
  Tail thickness   : 19.682 ± 0.095 Å
  Head fraction    : 0.300 ± 0.004
  Head contrast    : -3.456 ± 0.234 (1e-6/Å²)
  Tail contrast    : -5.678 ± 0.189 (1e-6/Å²)
  Scale            : 1.234e-03 ± 2.345e-05
  Background       : 0.012 ± 0.001
============================================================
```

## Key Improvements in `lamellar-fit-improved`

1. **Multi-Start Strategy** - Tests multiple smart initial guesses
2. **Clustering** - Identifies distinct local minima
3. **Outlier Detection** - Automatic MAD-based outlier removal and re-fitting
4. **Uncertainty Quantification** - Parameter standard deviations from Jacobian
5. **Basin-Hopping** - Alternative method for very difficult cases
6. **Better Weighting** - Uses 1/√y instead of 1/y for better balance
7. **Enhanced Diagnostics** - Reduced χ², residual plots, outlier flagging

## When to Use Which Script

### Use `lamellar-fit` (Original) When:
- Quick fits on well-behaved data
- You have good initial guesses
- Computational time is critical
- You don't need uncertainty estimates

### Use `lamellar-fit-improved` (Recommended) When:
- Data has local minima issues
- Fitting is challenging or unstable
- You need parameter uncertainties
- Data may contain outliers
- You want to explore multiple solutions

## Python Module Access

Both scripts can also be used as Python modules:

```python
# Original
from pycusaxs.fit.lamellar_fit import fit_lamellar

# Improved
from pycusaxs.fit.lamellar_fit_improved import fit_lamellar_hybrid, fit_lamellar_basinhopping
```

## Troubleshooting

### Script not found after installation
```bash
# Check if scripts are installed
which lamellar-fit
which lamellar-fit-improved

# If not found, ensure your PATH includes the environment's bin directory
echo $PATH

# Or use full path
~/.local/bin/lamellar-fit -f data.dat -p
```

### Import errors
```bash
# Ensure all dependencies are installed
pip install scipy matplotlib numpy

# Or reinstall package
pip install --force-reinstall -e .
```

### Fitting fails to converge
```bash
# Try improved script with more starting points
lamellar-fit-improved -f data.dat -p --multistart 10

# Or use basin-hopping
lamellar-fit-improved -f data.dat -p --method basinhopping

# Adjust parameter bounds if you have prior knowledge
lamellar-fit-improved -f data.dat -p \
    --tt-min 50 --tt-max 70 \
    --frac-min 0.25 --frac-max 0.35
```

## References

- **SasView Lamellar_HG Model**: https://www.sasview.org/docs/user/models/lamellar_hg.html
- **Model Paper**: F. Nallet et al., J. Phys. II France, 3 (1993) 487-502

## Version History

- **v0.1.0** - Initial release with both fitting scripts
  - Original differential evolution + local refinement
  - Enhanced multi-start with outlier detection and uncertainties
