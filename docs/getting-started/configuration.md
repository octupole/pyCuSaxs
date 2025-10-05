# Configuration

This guide covers all configuration parameters available in pyCuSAXS and how to optimize them for your calculations.

## Parameter Categories

pyCuSAXS parameters are organized into several categories:

- **Grid Parameters**: Control density grid resolution
- **Histogram Parameters**: Define binning in reciprocal space
- **Solvent Parameters**: Configure solvent models
- **Performance Tuning**: Optimize computation speed

## Grid Parameters

Grid parameters control the spatial resolution of the SAXS calculation.

### Primary Grid Size (`--grid`)

The primary density grid where atomic positions are mapped.

| Parameter | Type | Description | Recommended Values |
|-----------|------|-------------|-------------------|
| **grid** | int or 3×int | Primary density grid size | 64-128 for most systems |

**Usage:**

```bash
# Cubic grid (same size in all dimensions)
--grid 128

# Non-cubic grid
--grid 128,128,256
```

!!! tip "Choosing Grid Size"
    - **64³**: Fast, suitable for quick tests and small molecules
    - **128³**: Good balance for most protein systems (recommended)
    - **256³**: High resolution for detailed analysis (requires 8GB+ GPU memory)

### Scaled Grid Size (`--gridS`)

The supersampled grid used for FFT calculations. If not specified, it's automatically calculated based on the scale factor.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| **gridS** | 3×int | Scaled (supersampled) grid | Auto-calculated or 2× primary |

**Usage:**

```bash
# Explicit scaled grid
--gridS 256,256,256

# Auto-calculated based on Scale factor
--Scale 2.0  # Creates gridS = 2 × grid
```

### Scale Factor (`--Scale`)

Controls the grid supersampling ratio (σ).

| Parameter | Type | Description | Range | Default |
|-----------|------|-------------|-------|---------|
| **Scale** | float | Grid scaling factor (σ) | 1.0-3.0 | 1.0 |

**Usage:**

```bash
--Scale 2.0  # 2× oversampling
```

!!! info "Grid Resolution"
    The reciprocal space resolution is given by:

    $$\Delta q = \frac{2\pi}{N \cdot \sigma \cdot L}$$

    where N is grid dimension, σ is the scaling factor, and L is box length.

### B-spline Interpolation Order (`--order`)

Order of B-spline interpolation for density assignment.

| Parameter | Type | Description | Range | Default |
|-----------|------|-------------|-------|---------|
| **order** | int | B-spline interpolation order | 1-8 | 4 |

**Usage:**

```bash
--order 6  # Higher accuracy, slower
```

**Order Selection Guide:**

- **1-3**: Fast but less accurate (not recommended)
- **4**: Good balance (default)
- **5-6**: Better accuracy for high-resolution work
- **7-8**: Maximum accuracy (slower, rarely needed)

## Histogram Parameters

Control binning and range of the output SAXS profile.

### Bin Size (`--bin`, `--Dq`)

Width of histogram bins in reciprocal space.

| Parameter | Type | Description | Typical Range | Default |
|-----------|------|-------------|---------------|---------|
| **bin_size (Dq)** | float | Histogram bin width (Å⁻¹) | 0.001-0.05 | Auto |

**Usage:**

```bash
--bin 0.01  # 0.01 Å⁻¹ bins
```

!!! tip "Bin Size Selection"
    - **0.001-0.005 Å⁻¹**: Very fine, good for detailed features
    - **0.01 Å⁻¹**: Standard resolution (recommended)
    - **0.02-0.05 Å⁻¹**: Coarse, faster binning

### Q Cutoff (`--qcut`, `-q`)

Maximum q value in the output profile.

| Parameter | Type | Description | Typical Range | Default |
|-----------|------|-------------|---------------|---------|
| **qcut** | float | Reciprocal space cutoff (Å⁻¹) | 0.3-1.0 | Auto |

**Usage:**

```bash
--qcut 0.5  # Limit to q ≤ 0.5 Å⁻¹
```

**Cutoff Selection:**

- **0.3 Å⁻¹**: Low-angle scattering, overall shape
- **0.5 Å⁻¹**: Standard SAXS range (recommended)
- **1.0 Å⁻¹**: Wide-angle, requires finer grids

## Solvent Parameters

Configure explicit solvent modeling for accurate background subtraction.

### Water Model (`--water`)

Specify the water model used in your simulation.

| Parameter | Type | Description | Common Values |
|-----------|------|-------------|---------------|
| **water** | string | Water model identifier | `tip3p`, `tip4p`, `spc`, `spce` |

**Usage:**

```bash
--water tip3p
```

!!! warning "Solvent Padding Mode"
    When `--water` is specified, padding mode automatically switches to **explicit** (given densities). Otherwise, **average** mode is used (computed from border).

### Ion Counts

Number of ions in the simulation box.

| Parameter | Type | Description | Usage |
|-----------|------|-------------|-------|
| **sodium** (`--na`) | int | Na⁺ ion count | From topology |
| **chlorine** (`--cl`) | int | Cl⁻ ion count | From topology |

**Usage:**

```bash
--na 150 --cl 150  # 150 mM NaCl
```

### Simulation Type (`--simulation`)

Ensemble type of your simulation.

| Parameter | Type | Description | Values |
|-----------|------|-------------|--------|
| **simulation** | string | Simulation ensemble | `nvt`, `npt` |

**Usage:**

```bash
--simulation nvt  # Constant volume
```

- **NVT**: Constant number, volume, temperature
- **NPT**: Constant number, pressure, temperature

## Performance Tuning

Optimize calculation speed and resource usage.

### Frame Stride (`--dt`)

Step size for frame sampling.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| **dt** | int | Frame stride/step | 1 |

**Usage:**

```bash
--dt 10  # Process every 10th frame
```

!!! tip "Sampling Strategy"
    - **dt=1**: All frames (best statistics, slowest)
    - **dt=5-10**: Good balance (recommended)
    - **dt=20+**: Quick analysis (fewer statistics)

### Frame Range (`--begin`, `--end`)

Define which frames to process.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| **begin** (`-b`) | int | Starting frame index | 0 |
| **end** (`-e`) | int | Ending frame index | same as begin |

**Usage:**

```bash
--begin 0 --end 999  # Process frames 0-999
```

## Parameter Optimization

### For Speed

Optimize for fast calculations (testing, exploration):

```bash
python -m pycusaxs.main \
    -s topology.tpr -x trajectory.xtc \
    --grid 64 \
    --order 4 \
    --dt 20 \
    --begin 0 --end 100 \
    --bin 0.02
```

### For Accuracy

Optimize for high-quality results (publication):

```bash
python -m pycusaxs.main \
    -s topology.tpr -x trajectory.xtc \
    --grid 128 \
    --gridS 256,256,256 \
    --order 6 \
    --Scale 2.0 \
    --dt 5 \
    --begin 0 --end 999 \
    --bin 0.005 \
    --qcut 0.5
```

### For Large Systems

Optimize for systems with many atoms:

```bash
python -m pycusaxs.main \
    -s large_system.tpr -x trajectory.xtc \
    --grid 128 \  # Don't go larger unless needed
    --order 4 \
    --dt 10 \
    --bin 0.01
```

## Configuration File Support

Currently, pyCuSAXS accepts parameters via command line. For complex workflows, consider using shell scripts:

```bash
#!/bin/bash
# saxs_config.sh

TOPOLOGY="system.tpr"
TRAJECTORY="trajectory.xtc"
GRID="128,128,128"
ORDER=6
SCALE=2.0
BIN=0.01
QCUT=0.5

python -m pycusaxs.main \
    -s "$TOPOLOGY" \
    -x "$TRAJECTORY" \
    --grid "$GRID" \
    --order "$ORDER" \
    --Scale "$SCALE" \
    --bin "$BIN" \
    --qcut "$QCUT" \
    -o "saxs_profile.dat"
```

## Best Practices

!!! success "Grid Sizing"
    - Use powers of 2 for optimal FFT performance (64, 128, 256)
    - Match grid size to system size (1-2 Å per grid point)
    - Consider GPU memory limits

!!! success "Histogram Binning"
    - Choose bin size based on desired q resolution
    - Smaller bins = more detail but more noise
    - Use qcut to limit output range

!!! success "Solvent Modeling"
    - Always specify water model if using explicit solvent
    - Count ions from topology output
    - Match simulation type to your MD protocol

!!! success "Performance"
    - Start with small grids and frame ranges for testing
    - Increase stride for long trajectories
    - Monitor GPU memory with `nvidia-smi`

## Parameter Summary Table

| Category | Parameter | CLI Flag | Type | Default | Range |
|----------|-----------|----------|------|---------|-------|
| **Grid** | Primary grid | `--grid`, `-g` | int/3×int | Required | 16-512 |
| | Scaled grid | `--gridS` | 3×int | Auto | - |
| | Scale factor | `--Scale` | float | 1.0 | 1.0-3.0 |
| | Spline order | `--order` | int | 4 | 1-8 |
| **Histogram** | Bin size | `--bin`, `--Dq` | float | Auto | 0.001+ |
| | Q cutoff | `--qcut`, `-q` | float | Auto | 0.1+ |
| **Solvent** | Water model | `--water` | string | None | - |
| | Sodium count | `--na` | int | 0 | 0+ |
| | Chlorine count | `--cl` | int | 0 | 0+ |
| | Simulation type | `--simulation` | string | nvt | nvt/npt |
| **I/O** | Topology | `-s`, `--topology` | path | Required | - |
| | Trajectory | `-x`, `--trajectory` | path | Required | - |
| | Output | `-o`, `--out` | path | saxs.dat | - |
| **Frames** | Begin frame | `-b`, `--begin` | int | 0 | 0+ |
| | End frame | `-e`, `--end` | int | same as begin | 0+ |
| | Frame stride | `--dt` | int | 1 | 1+ |

## See Also

- [Command Line Interface](../user-guide/cli.md) - Complete CLI reference
- [Algorithm Overview](../algorithm/overview.md) - How parameters affect calculations
- [Performance](../algorithm/performance.md) - Performance optimization guide
