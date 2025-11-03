# Core-Shell Elliptical Bicelle Fitting

This module provides fitting capabilities for the SasView `core_shell_bicelle_elliptical` model, following the same structure as `lamellar_fit.py`.

## Model Description

The core-shell elliptical bicelle model describes an elliptical cylinder with a core-shell scattering length density profile. It's useful for fitting SAXS data from:
- Disk-like micelles
- Bicelles
- Elliptical vesicles
- Other elliptical core-shell structures

## Files

- `bicelle_model.py` - Implementation of the core-shell elliptical bicelle intensity formula
- `bicelle_fit.py` - Fitting script with global + local optimization

## Model Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| radius | Minor radius of elliptical core (R_minor) | Å | 10-200 |
| x_core | Axial ratio (R_major/R_minor) | - | 1-10 |
| thick_rim | Rim shell thickness | Å | 1-50 |
| thick_face | Face shell thickness | Å | 1-50 |
| length | Core cylinder length | Å | 10-300 |
| sld_core | Core SLD | 10⁻⁶Å⁻² | -10 to 10 |
| sld_face | Face shell SLD | 10⁻⁶Å⁻² | -10 to 10 |
| sld_rim | Rim shell SLD | 10⁻⁶Å⁻² | -10 to 10 |
| sld_solvent | Solvent SLD | 10⁻⁶Å⁻² | -10 to 10 |
| scale | Scale factor / volume fraction | - | auto-fitted |
| background | Flat background | - | auto-fitted |

## Usage

### Basic Fitting

```bash
python -m pycusaxs.fit.bicelle_fit -f data.dat -p -o fit.dat
```

### With Custom Q-Range

```bash
python -m pycusaxs.fit.bicelle_fit -f data.dat \
    --qmin 0.01 \
    --qmax 0.5 \
    -p -o fit.dat
```

### With Parameter Bounds

```bash
python -m pycusaxs.fit.bicelle_fit -f data.dat \
    --radius-min 20.0 \
    --radius-max 50.0 \
    --x-core-min 2.0 \
    --x-core-max 5.0 \
    --length-min 30.0 \
    --length-max 100.0 \
    -p -o fit.dat
```

### Exclude Regions

```bash
python -m pycusaxs.fit.bicelle_fit -f data.dat \
    --exclude 0.0:0.02 \
    --exclude 0.3:inf \
    -p -o fit.dat
```

## Options

- `-f, --file` - Input data file (two columns: q I(q))
- `-p, --plot` - Show matplotlib plot of data and fit
- `-o, --output` - Save fitted I(q) to file
- `--log` - Use log-residual objective for local refinement
- `--qmin` - Minimum q value to include in fit
- `--qmax` - Maximum q value to include in fit
- `--exclude` - Exclude q-ranges (repeatable)
- `--radius-min/max` - Bounds for radius parameter
- `--x-core-min/max` - Bounds for axial ratio
- `--length-min/max` - Bounds for length parameter

## Python API

```python
from pycusaxs.fit.bicelle_fit import fit_bicelle
from pycusaxs.fit.bicelle_model import bicelle_intensity
import numpy as np

# Load data
q = np.loadtxt('data.dat')[:, 0]
y = np.loadtxt('data.dat')[:, 1]

# Fit
result = fit_bicelle(q, y)

print(result['params'])
print('Fit success:', result['success'])

# Access fitted curve
y_fit = result['yfit']
```

## Algorithm

The fitting uses a two-stage approach:

1. **Global Search** - Differential Evolution to explore parameter space
   - Finds approximate optimal parameters
   - Scale and background fitted analytically at each iteration

2. **Local Refinement** - Trust Region Reflective least-squares
   - Polishes the solution from global search
   - Uses soft_l1 loss for robustness
   - Weighted residuals to balance dynamic range

## Comparison with Lamellar Fit

Both modules follow the same structure:

| Feature | lamellar_fit.py | bicelle_fit.py |
|---------|----------------|----------------|
| Model | Lamellar head-tail | Core-shell elliptical bicelle |
| Geometry | Flat bilayer | Elliptical cylinder |
| Key params | H, T (head/tail thickness) | radius, x_core, length |
| Optimization | 2-stage (DE + LSQ) | 2-stage (DE + LSQ) |
| SLD regions | 2 (head, tail) | 3 (core, face, rim) |

## Reference

SasView core_shell_bicelle_elliptical model:
https://www.sasview.org/docs/user/models/core_shell_bicelle_elliptical.html

## Notes

- The model uses numerical integration over orientation angles (alpha, psi)
- Computation may be slower than analytical models due to integration
- For faster fitting, reduce the integration resolution in `bicelle_model.py` (n_alpha, n_psi)
- Default SLD values are water-ish contrasts; adjust bounds for your system
