# Quick Start Guide

This guide will help you get started with pyCuSaxs for SAXS profile calculations from MD trajectories.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.0 or newer
- Python 3.9+
- At least 4GB GPU memory (recommended)

## Installation

### Using Conda (Recommended)

```bash
# Create environment with dependencies
conda create -n pycusaxs python=3.11 cmake fmt pybind11 numpy
conda activate pycusaxs

# Install Python packages
pip install PySide6 MDAnalysis networkx

# Build and install pyCuSaxs
cd /path/to/pyCuSaxs
pip install .
```

### Verify Installation

```bash
# Check CUDA backend
python -c "import pycusaxs_cuda; print('CUDA backend OK')"

# Check package
python -c "import pycusaxs; print('pycusaxs OK')"

# Test commands
pycusaxs --help
saxs-widget
```

## Basic Examples

### Command-Line Interface

#### Simple Calculation

```bash
pycusaxs \
    -s protein.tpr \
    -x trajectory.xtc \
    -g 64 \
    -b 0 \
    -e 999 \
    --dt 10 \
    -o saxs_profile.dat
```

This will:
- Process frames 0-999 with stride 10 (every 10th frame)
- Use a 64×64×64 density grid
- Output results to `saxs_profile.dat`

#### With Explicit Solvent

```bash
pycusaxs \
    -s solvated.tpr \
    -x md.xtc \
    -g 128 \
    --water tip3p \
    --na 150 \
    --cl 150 \
    -o saxs_solvated.dat
```

#### High-Resolution Calculation

```bash
pycusaxs \
    -s system.tpr \
    -x traj.xtc \
    -g 128 \
    --gridS 256 \
    --order 6 \
    --bin 0.01 \
    --qcut 0.5 \
    -o high_res_saxs.dat
```

### Graphical Interface

Launch the GUI with no arguments:

```bash
pycusaxs
# or
saxs-widget
```

The GUI provides:
- File browsers for topology/trajectory selection
- Parameter configuration dialogs
- Real-time output display
- Settings persistence across sessions

### Python API

```python
from pycusaxs.topology import Topology
from pycusaxs.core import run_saxs_calculation

# Load and analyze topology
topo = Topology("system.tpr", "trajectory.xtc")
print(f"System: {topo.n_atoms} atoms, {topo.n_frames} frames")

# Count molecules
total, proteins, waters, ions, others = topo.count_molecules()
print(f"Proteins: {proteins}, Waters: {waters}, Ions: {ions}")

# Run SAXS calculation
required = {
    "topology": "system.tpr",
    "trajectory": "trajectory.xtc",
    "grid_size": (128, 128, 128),
    "initial_frame": 0,
    "last_frame": 100
}

advanced = {
    "dt": 10,
    "order": 4,
    "bin_size": 0.01,
    "qcut": 0.5,
    "out": "saxs.dat"
}

results = run_saxs_calculation(required, advanced)
```

## Understanding Output

### SAXS Profile File

The output file (`saxs.dat` by default) contains:

```
# Column 1: q (Å⁻¹)
# Column 2: I(q) (arbitrary units)
   0.00100    1234.5678
   0.00200    1123.4567
   0.00300    1012.3456
   ...
```

### Console Output

```
*************************************************
*            Running CuSAXS                     *
* Cell Grid               128 *  128 *  128     *
* Supercell Grid          256 *  256 *  256     *
* Order    4          Sigma      2.000          *
* Bin Size 0.010      Q Cutoff   0.500          *
* Padding             avg Border                *
*************************************************

--> Frame:    0      Time Step: 0.00 fs
--> Frame:   10      Time Step: 100.00 fs
...

Done 100 Steps
Results written to saxs.dat

=========================================================
=                    CuSAXS Timing                     =
=           CUDA Time:     25.43 ms/per step           =
=           Read Time:     5.12 ms/per step            =
=           Total Time:    30.55 ms/per step           =
=========================================================
```

## Common Parameters

### Grid Size (`-g`, `--grid`)

- Larger grids = better resolution but more memory/time
- Recommended: 64-128 for most systems
- Can specify as single value (64) or three values (64,64,128)

### Frame Range

- `-b, --begin`: Starting frame (default: 0)
- `-e, --end`: Last frame (default: same as begin)
- `--dt`: Frame stride (default: 1)

### Advanced Options

- `--order`: B-spline interpolation order, 1-8 (default: 4)
- `--gridS`: Scaled grid size for supersampling
- `--Scale`: Grid scaling factor (default: 1.0)
- `--bin`: Histogram bin width (Å⁻¹)
- `--qcut`: Q-space cutoff (Å⁻¹)

## Next Steps

- Read the [Tutorial](tutorial.md) for detailed workflow examples
- Check [CLI Reference](cli_reference.md) for all command-line options
- See [Python API](api/python.rst) for programmatic access
- Join [GitHub Discussions](https://github.com/yourusername/pyCuSaxs/discussions) for help

## Troubleshooting

### CUDA Backend Not Found

```bash
# Set library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Rebuild
pip install --force-reinstall .
```

### GPU Memory Errors

- Reduce grid size (e.g., 64 instead of 128)
- Increase frame stride (process fewer frames)
- Close other GPU applications

### Build Failures

```bash
# Check CUDA version
nvcc --version

# Check for GPU
nvidia-smi

# Ensure compatible compiler (GCC 9-11 for CUDA 11.x)
gcc --version
```
