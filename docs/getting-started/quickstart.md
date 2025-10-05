# Quick Start

Get started with pyCuSAXS in just a few minutes! This guide shows you how to run your first SAXS calculation.

## Basic Example

The simplest way to compute a SAXS profile from a trajectory:

```bash
# Compute SAXS profile from trajectory
python -m pycusaxs.main \
    --topology system.tpr \
    --trajectory trajectory.xtc \
    --grid 128,128,128 \
    --begin 0 \
    --end 100 \
    --dt 10 \
    --out saxs_profile.dat
```

This command will:

1. Load the topology file (`system.tpr`) and trajectory (`trajectory.xtc`)
2. Process frames 0-100, sampling every 10th frame
3. Use a 128³ density grid for calculations
4. Save the SAXS profile to `saxs_profile.dat`

!!! tip "Grid Size Selection"
    Start with 64³ or 128³ grids for testing. Larger grids (256³) provide better resolution but require more GPU memory and computation time.

## GUI Mode

For interactive use, launch the graphical interface:

```bash
# Launch graphical interface
saxs-widget

# or alternatively
python -m pycusaxs.main gui
```

The GUI provides:

- File browser for selecting topology and trajectory files
- Interactive parameter configuration
- Real-time progress feedback
- Configuration summary display

!!! success "GUI Benefits"
    The GUI is perfect for exploring parameters and validating configurations before running large-scale calculations.

## Understanding the Output

### Output File Format

The output file (`saxs_profile.dat`) contains two columns:

```
# Column 1: q (Å⁻¹)
# Column 2: I(q) (arbitrary units)
   0.00100    1234.5678
   0.00200    1123.4567
   0.00300    1012.3456
   ...
```

- **q**: Scattering vector magnitude (Å⁻¹)
- **I(q)**: Scattering intensity (arbitrary units)

### Console Output

During execution, you'll see:

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
--> Frame:   10      Time Step: 10.00 fs
...

Done 100 Steps
Results written to saxs_profile.dat

=========================================================
=                                                       =
=                    CuSAXS Timing                     =
=                                                       =
=           CUDA Time:     25.43 ms/per step           =
=           Read Time:     5.12 ms/per step            =
=           Total Time:    30.55 ms/per step           =
=                                                       =
=========================================================
```

This shows:

- Configuration summary (grid sizes, parameters)
- Frame processing progress
- Performance timing statistics

## Common Use Cases

### Case 1: Quick Test Run

Test on a small number of frames first:

```bash
python -m pycusaxs.main \
    -s protein.tpr \
    -x trajectory.xtc \
    -g 64 \
    -b 0 -e 10 \
    -o test_saxs.dat
```

### Case 2: Production Run

For production analysis with better statistics:

```bash
python -m pycusaxs.main \
    -s protein.tpr \
    -x md_trajectory.xtc \
    -g 128 \
    -b 0 -e 999 --dt 10 \
    --order 6 \
    --bin 0.01 --qcut 0.5 \
    -o production_saxs.dat
```

### Case 3: With Explicit Solvent

For systems with explicit solvent:

```bash
python -m pycusaxs.main \
    -s solvated.tpr \
    -x md.xtc \
    -g 100 \
    --water tip3p \
    --na 150 --cl 150 \
    --simulation nvt \
    -o saxs_solvated.dat
```

## Visualizing Results

You can plot the SAXS profile using Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load SAXS data
data = np.loadtxt('saxs_profile.dat')
q = data[:, 0]
I = data[:, 1]

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(q, I, 'b-', linewidth=2)
plt.xlabel('q (Å⁻¹)', fontsize=12)
plt.ylabel('I(q) (a.u.)', fontsize=12)
plt.title('SAXS Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('saxs_plot.png', dpi=300)
plt.show()
```

## What's Next?

Now that you've run your first calculation, explore more features:

=== "Learn the CLI"
    [Command Line Interface](../user-guide/cli.md) - Master all CLI options and parameters

=== "Use Python API"
    [Python API](../user-guide/python-api.md) - Integrate SAXS calculations into your scripts

=== "Configure Advanced Options"
    [Configuration](configuration.md) - Fine-tune grid sizing, binning, and solvent models

=== "Understand the Algorithm"
    [Algorithm Overview](../algorithm/overview.md) - Learn how SAXS calculations work

## Tips for Best Results

!!! tip "Frame Sampling"
    - Use `--dt` to skip frames and reduce computation time
    - Process 50-100 frames for good statistics
    - More frames improve signal-to-noise but increase runtime

!!! tip "Grid Resolution"
    - 64³: Fast, suitable for preliminary analysis
    - 128³: Good balance of speed and accuracy (recommended)
    - 256³: High resolution, requires more memory

!!! tip "Performance"
    - Start with small frame ranges for testing
    - Monitor GPU memory usage with `nvidia-smi`
    - Use frame stride (`--dt`) to process large trajectories efficiently

!!! warning "Common Pitfalls"
    - Ensure topology and trajectory are compatible
    - Check that frame indices are within trajectory bounds
    - Verify sufficient GPU memory for your grid size
