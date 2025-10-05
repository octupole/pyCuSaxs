# Command Line Interface

The pyCuSAXS command-line interface provides full control over SAXS calculations through terminal commands.

## Basic Usage

```bash
python -m pycusaxs.main [OPTIONS]
```

!!! tip "Alternative Invocation"
    You can also use the installed command:
    ```bash
    saxs-widget [OPTIONS]  # GUI mode if no options
    ```

## Required Parameters

These parameters must be specified for every calculation:

### Topology File (`-s`, `--topology`)

GROMACS topology file containing system structure information.

```bash
-s system.tpr
--topology system.tpr
```

**Supported Formats:**
- `.tpr` - GROMACS portable binary run input file (recommended)

### Trajectory File (`-x`, `--trajectory`)

MD trajectory file containing atomic coordinates over time.

```bash
-x trajectory.xtc
--trajectory trajectory.xtc
```

**Supported Formats:**
- `.xtc` - GROMACS compressed trajectory (recommended)
- `.trr` - GROMACS full-precision trajectory
- Other formats supported by MDAnalysis

### Grid Size (`-g`, `--grid`)

Density grid dimensions for SAXS calculations.

```bash
# Cubic grid (same in all dimensions)
--grid 128

# Non-cubic grid
--grid 128,128,256
```

**Common Values:**
- `64` - Fast, lower resolution
- `128` - Standard (recommended)
- `256` - High resolution (requires more memory)

### Frame Range

Define which frames to process from the trajectory.

#### Begin Frame (`-b`, `--begin`)

Starting frame index (0-based).

```bash
--begin 0  # Start from first frame
-b 100     # Start from frame 100
```

#### End Frame (`-e`, `--end`)

Ending frame index (inclusive).

```bash
--end 999  # Process up to frame 999
-e 1000    # Process 1000 frames
```

!!! warning "Frame Indices"
    Frame indices are 0-based. To process frames 1-1000 from your trajectory, use `--begin 0 --end 999`.

## Optional Parameters

### Output File (`-o`, `--out`)

Specify output file path for the SAXS profile.

```bash
--out results/saxs_profile.dat
-o my_saxs.dat
```

**Default:** `saxs.dat` in the current directory

### Frame Stride (`--dt`)

Step size for frame sampling (process every Nth frame).

```bash
--dt 10  # Process every 10th frame
```

**Default:** 1 (process every frame)

**Usage Guide:**
- `dt=1`: Maximum statistics, slowest
- `dt=5-10`: Good balance (recommended)
- `dt=20+`: Quick analysis

### B-spline Order (`--order`)

Interpolation order for density assignment (1-8).

```bash
--order 6  # Higher accuracy
```

**Default:** 4

**Guidelines:**
- **4**: Good balance (default)
- **5-6**: Better accuracy for publication
- **7-8**: Maximum accuracy (rarely needed)

### Grid Scaling

#### Scale Factor (`--Scale`)

Grid supersampling ratio (σ).

```bash
--Scale 2.0  # 2× oversampling
```

**Default:** 1.0

#### Explicit Scaled Grid (`--gridS`)

Manually specify the supersampled grid dimensions.

```bash
--gridS 256,256,256
```

**Default:** Auto-calculated from `--Scale`

!!! info "Grid Scaling"
    The scaled grid is used for FFT calculations. Either specify `--Scale` (automatic) or `--gridS` (manual), but typically not both.

### Histogram Parameters

#### Bin Size (`--bin`, `--Dq`)

Histogram bin width in reciprocal space (Å⁻¹).

```bash
--bin 0.01  # 0.01 Å⁻¹ bins
--Dq 0.005  # Finer binning
```

**Common Values:**
- `0.005` - Fine resolution
- `0.01` - Standard (recommended)
- `0.02` - Coarse, faster

#### Q Cutoff (`--qcut`, `-q`)

Maximum q value in output (Å⁻¹).

```bash
--qcut 0.5  # Limit to 0.5 Å⁻¹
-q 1.0      # Extended range
```

**Typical Ranges:**
- `0.3` - Low-angle, overall shape
- `0.5` - Standard SAXS (recommended)
- `1.0` - Wide-angle

### Solvent Parameters

#### Water Model (`--water`)

Specify water model for explicit solvation.

```bash
--water tip3p
--water spce
```

**Supported Models:**
- `tip3p`, `tip4p`, `tip5p`
- `spc`, `spce`
- Others as defined in the scattering database

#### Ion Counts

Number of ions in the simulation box.

```bash
--na 150   # 150 sodium ions
--cl 150   # 150 chloride ions
```

**Default:** 0 for both

!!! tip "Getting Ion Counts"
    Use the Python API to analyze your topology and get accurate ion counts:
    ```python
    from pycusaxs.topology import Topology
    topo = Topology("system.tpr", "trajectory.xtc")
    total, protein, water, ions, other = topo.count_molecules()
    print(f"Ions: {ions}")
    ```

#### Simulation Type (`--simulation`)

Simulation ensemble type.

```bash
--simulation nvt  # Constant volume
--simulation npt  # Constant pressure
```

**Default:** `nvt`

**Options:**
- `nvt` - Constant number, volume, temperature
- `npt` - Constant number, pressure, temperature

## Complete Examples

### Example 1: Basic SAXS Calculation

Minimal configuration for a quick SAXS profile:

```bash
python -m pycusaxs.main \
    -s protein.tpr \
    -x md_trajectory.xtc \
    -g 64,64,64 \
    -b 0 -e 999 --dt 10 \
    -o results/saxs.dat
```

This will:

- Process frames 0-999, every 10th frame (100 frames total)
- Use a 64³ grid for fast computation
- Save results to `results/saxs.dat`

### Example 2: High-Resolution with Custom Grid Scaling

Production-quality calculation with fine control:

```bash
python -m pycusaxs.main \
    -s system.tpr \
    -x traj.xtc \
    -g 128 \
    --gridS 256,256,256 \
    --order 6 \
    --Scale 2.0 \
    --bin 0.01 --qcut 0.5 \
    -o high_res_saxs.dat
```

Features:

- 128³ primary grid, 256³ scaled grid
- Order 6 B-spline interpolation
- Fine histogram binning (0.01 Å⁻¹)
- Output limited to q ≤ 0.5 Å⁻¹

### Example 3: With Explicit Solvent Model

For systems with explicit water and ions:

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

Features:

- TIP3P water model
- 150 mM NaCl (150 Na⁺, 150 Cl⁻)
- NVT ensemble

### Example 4: Quick Test Run

Fast test on a few frames:

```bash
python -m pycusaxs.main \
    -s system.tpr \
    -x trajectory.xtc \
    -g 64 \
    -b 0 -e 10 \
    -o test.dat
```

Features:

- Small grid (64³)
- Only 10 frames
- Fast validation before full run

### Example 5: Full Production Run

Complete configuration for publication-quality results:

```bash
python -m pycusaxs.main \
    --topology protein.tpr \
    --trajectory production.xtc \
    --grid 128 \
    --order 6 \
    --Scale 2.0 \
    --begin 0 \
    --end 999 \
    --dt 5 \
    --bin 0.005 \
    --qcut 0.5 \
    --water tip3p \
    --na 150 \
    --cl 150 \
    --simulation nvt \
    --out publication_saxs.dat
```

Features:

- All parameters explicitly specified
- High accuracy settings
- Explicit solvent model
- Fine histogram binning
- 200 frames (every 5th from 1000)

## Parameter Combinations

### For Speed

Optimized for fast calculations:

```bash
python -m pycusaxs.main \
    -s system.tpr -x trajectory.xtc \
    -g 64 --order 4 --dt 20 \
    -b 0 -e 100 --bin 0.02 \
    -o quick_saxs.dat
```

### For Accuracy

Optimized for high-quality results:

```bash
python -m pycusaxs.main \
    -s system.tpr -x trajectory.xtc \
    -g 128 --gridS 256,256,256 \
    --order 6 --Scale 2.0 \
    -b 0 -e 999 --dt 5 \
    --bin 0.005 --qcut 0.5 \
    -o accurate_saxs.dat
```

## Shell Script Automation

For repeated calculations, create a shell script:

```bash
#!/bin/bash
# run_saxs.sh - Automated SAXS calculation

# Configuration
TOPOLOGY="system.tpr"
TRAJECTORY="trajectory.xtc"
OUTPUT_DIR="results"
GRID="128"
ORDER=6
DT=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run SAXS calculation
python -m pycusaxs.main \
    -s "$TOPOLOGY" \
    -x "$TRAJECTORY" \
    -g "$GRID" \
    --order "$ORDER" \
    --dt "$DT" \
    -b 0 -e 999 \
    --bin 0.01 \
    --qcut 0.5 \
    -o "$OUTPUT_DIR/saxs_profile.dat"

echo "SAXS calculation complete!"
echo "Results saved to $OUTPUT_DIR/saxs_profile.dat"
```

Make executable and run:

```bash
chmod +x run_saxs.sh
./run_saxs.sh
```

## Batch Processing

Process multiple trajectories:

```bash
#!/bin/bash
# batch_saxs.sh - Process multiple trajectories

for i in {1..10}; do
    echo "Processing trajectory $i..."
    python -m pycusaxs.main \
        -s system.tpr \
        -x trajectory_${i}.xtc \
        -g 128 -b 0 -e 999 --dt 10 \
        -o results/saxs_${i}.dat
done
```

## Getting Help

Display CLI help message:

```bash
python -m pycusaxs.main --help
```

## Exit Codes

The CLI returns standard exit codes:

- **0**: Success
- **1**: Error (invalid parameters, file not found, calculation failure)

Use in scripts:

```bash
if python -m pycusaxs.main -s system.tpr -x traj.xtc -g 128 -b 0 -e 100; then
    echo "SAXS calculation successful"
else
    echo "SAXS calculation failed"
    exit 1
fi
```

## See Also

- [Configuration](../getting-started/configuration.md) - Detailed parameter descriptions
- [Python API](python-api.md) - Programmatic interface
- [GUI Guide](gui.md) - Graphical interface
- [Examples](../getting-started/quickstart.md) - Quick start examples
