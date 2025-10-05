# Graphical User Interface

The pyCuSAXS GUI provides an intuitive interface for configuring and running SAXS calculations without using the command line.

## Launching the GUI

There are two ways to launch the graphical interface:

```bash
# Method 1: Using the installed command
saxs-widget

# Method 2: Using the Python module
python -m pycusaxs.main gui
```

!!! tip "No Arguments = GUI Mode"
    Running `python -m pycusaxs.main` without any arguments automatically launches the GUI.

## Interface Overview

The GUI is organized into several sections:

1. **Required Parameters Panel** - Essential input fields
2. **Advanced Parameters Button** - Opens additional options dialog
3. **Run Button** - Execute the SAXS calculation
4. **Output Display** - Shows results and configuration summary

## Required Parameters Panel

### File Selection

#### Topology File

Select your GROMACS topology file (`.tpr`):

- Click **Browse** next to "Topology File"
- Navigate to your `.tpr` file
- Select and open

#### Trajectory File

Select your MD trajectory file:

- Click **Browse** next to "Trajectory File"
- Navigate to your trajectory (`.xtc`, `.trr`, etc.)
- Select and open

!!! info "File Compatibility"
    The topology and trajectory must be compatible (same system). The GUI will display an error if files are incompatible.

### Grid Size

Specify the density grid dimensions:

**Single Value (Cubic Grid):**
```
128
```
Creates a 128×128×128 grid.

**Three Values (Non-Cubic Grid):**
```
128,128,256
```
Creates a 128×128×256 grid.

!!! tip "Grid Size Hints"
    - Start with 64 for testing
    - Use 128 for production (recommended)
    - Use 256 for high resolution (requires more GPU memory)

### Frame Range

#### Begin Frame

Starting frame index (0-based):
```
0  (first frame)
100  (frame 101)
```

#### End Frame

Ending frame index (inclusive):
```
999  (process 1000 frames if begin=0)
```

!!! warning "Frame Validation"
    The GUI will validate that:

    - Begin < End
    - Both values are within trajectory bounds
    - Values are non-negative integers

## Advanced Parameters Dialog

Click the **Advanced Parameters** button to open a dialog with additional options.

### Output Settings

#### Output File Path

Specify where to save the SAXS profile:

```
results/saxs_profile.dat
/path/to/output/saxs.dat
```

**Default:** `saxs.dat` in current directory

### Grid Settings

#### Frame Stride (dt)

Process every Nth frame:

```
1  - Every frame
10 - Every 10th frame
```

Increase to reduce computation time while maintaining statistical sampling.

#### B-spline Order

Interpolation order (1-8):

```
4  - Standard (default)
6  - Higher accuracy
```

Higher values provide better accuracy but slower computation.

#### Scaled Grid Size

Manually specify supersampled grid dimensions:

```
256,256,256
```

Leave empty to use automatic calculation based on Scale Factor.

#### Scale Factor

Grid supersampling ratio (σ):

```
1.0  - No oversampling (default)
2.0  - 2× oversampling
```

### Histogram Settings

#### Bin Size

Histogram bin width (Å⁻¹):

```
0.01  - Standard resolution
0.005 - Fine resolution
```

#### Q Cutoff

Maximum q value (Å⁻¹):

```
0.5  - Standard range
1.0  - Extended range
```

### Solvent Settings

#### Water Model

Select water model from dropdown:

- None (use average padding)
- tip3p
- tip4p
- spc
- spce
- Other models

#### Sodium Count

Number of Na⁺ ions:
```
150  (for 150 mM NaCl)
```

#### Chlorine Count

Number of Cl⁻ ions:
```
150  (for 150 mM NaCl)
```

#### Simulation Type

Select ensemble:

- **NVT** - Constant volume
- **NPT** - Constant pressure

## Running a Calculation

### Step-by-Step Workflow

1. **Load Files**
   - Browse and select topology file
   - Browse and select trajectory file

2. **Set Grid Parameters**
   - Enter grid size (e.g., `128`)
   - Set frame range (begin/end)

3. **Configure Advanced Options** (Optional)
   - Click "Advanced Parameters"
   - Adjust settings as needed
   - Click "OK" to save

4. **Execute**
   - Click "Run" button
   - Wait for calculation to complete

5. **View Results**
   - Configuration summary appears in output panel
   - Timing statistics displayed
   - Output file saved to specified location

### Progress Feedback

During execution, the GUI shows:

- Configuration summary (grid sizes, parameters)
- Frame processing messages
- Timing statistics
- Success/error messages

!!! success "Completion Message"
    When complete, you'll see:
    ```
    Done N Steps
    Results written to [output_file]

    ========================================
    =          CuSAXS Timing              =
    ========================================
    =   CUDA Time: XX.XX ms/per step      =
    =   Read Time: XX.XX ms/per step      =
    =   Total Time: XX.XX ms/per step     =
    ========================================
    ```

## Example Workflows

### Workflow 1: Quick Test

For quick validation:

1. Select files: `system.tpr`, `trajectory.xtc`
2. Grid size: `64`
3. Frames: Begin=`0`, End=`10`
4. Click **Run**

**Result:** Fast test run on 10 frames with low-resolution grid.

### Workflow 2: Standard Production

For typical production run:

1. Select files: `protein.tpr`, `production.xtc`
2. Grid size: `128`
3. Frames: Begin=`0`, End=`999`
4. Advanced Parameters:
   - Frame stride: `10`
   - Output: `results/saxs_profile.dat`
   - Bin size: `0.01`
   - Q cutoff: `0.5`
5. Click **Run**

**Result:** Process 100 frames (every 10th) with standard settings.

### Workflow 3: High-Accuracy Calculation

For publication-quality results:

1. Select files: `system.tpr`, `trajectory.xtc`
2. Grid size: `128`
3. Frames: Begin=`0`, End=`999`
4. Advanced Parameters:
   - Frame stride: `5`
   - Order: `6`
   - Scale factor: `2.0`
   - Bin size: `0.005`
   - Q cutoff: `0.5`
   - Output: `publication_saxs.dat`
5. Click **Run**

**Result:** High-accuracy calculation with fine binning.

### Workflow 4: Explicit Solvent

For solvated systems:

1. Select files: `solvated.tpr`, `md.xtc`
2. Grid size: `100`
3. Frames: Begin=`0`, End=`999`
4. Advanced Parameters:
   - Water model: `tip3p`
   - Sodium: `150`
   - Chlorine: `150`
   - Simulation: `nvt`
   - Frame stride: `10`
5. Click **Run**

**Result:** SAXS calculation with explicit water model and ions.

## Output Display

The output panel shows:

### Configuration Summary

```
*************************************************
*            Running CuSAXS                     *
* Cell Grid               128 *  128 *  128     *
* Supercell Grid          256 *  256 *  256     *
* Order    4          Sigma      2.000          *
* Bin Size 0.010      Q Cutoff   0.500          *
* Padding             avg Border                *
*************************************************
```

### System Information

```
Total molecules: 50234
Proteins: 1
Waters: 50000
Ions: 233
Others: 0
```

### Frame Progress

```
--> Frame:    0      Time Step: 0.00 fs
--> Frame:   10      Time Step: 100.00 fs
--> Frame:   20      Time Step: 200.00 fs
...
```

### Timing Statistics

```
=========================================================
=                    CuSAXS Timing                     =
=========================================================
=           CUDA Time:     25.43 ms/per step           =
=           Read Time:     5.12 ms/per step            =
=           Total Time:    30.55 ms/per step           =
=========================================================
```

## Error Handling

The GUI provides informative error messages for common issues:

### File Not Found
```
Error: Topology file not found: /path/to/file.tpr
Please check the file path and try again.
```

**Solution:** Verify file exists and path is correct.

### Invalid Frame Range
```
Error: Begin frame (1000) must be less than end frame (999)
```

**Solution:** Ensure begin < end.

### Grid Size Error
```
Error: Grid size must be 1 or 3 positive integers
```

**Solution:** Enter either `128` or `128,128,128`.

### Calculation Error
```
CUDA Error: Out of memory
```

**Solution:** Reduce grid size or close other GPU applications.

## Tips and Best Practices

!!! tip "Parameter Testing"
    Use the GUI to test different parameter combinations quickly. Once you find optimal settings, you can use the same parameters in the CLI for batch processing.

!!! tip "Save Output"
    The output display shows the exact configuration used. Copy this for reproducibility or use it to construct CLI commands.

!!! tip "Resource Monitoring"
    Keep `nvidia-smi` running in a terminal to monitor GPU memory usage during calculations.

!!! warning "Long Calculations"
    For very long calculations (many frames, large grids), the GUI may appear frozen. This is normal - the calculation is still running. Check the output file for progress.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open topology file |
| Ctrl+T | Open trajectory file |
| Ctrl+R | Run calculation |
| Ctrl+Q | Quit application |

## Integration with CLI

After using the GUI to determine optimal parameters, you can replicate the calculation using the CLI:

### GUI Settings:
- Topology: `system.tpr`
- Trajectory: `trajectory.xtc`
- Grid: `128`
- Begin: `0`, End: `999`
- Frame stride: `10`
- Order: `6`
- Bin size: `0.01`

### Equivalent CLI Command:
```bash
python -m pycusaxs.main \
    -s system.tpr \
    -x trajectory.xtc \
    -g 128 \
    -b 0 -e 999 \
    --dt 10 \
    --order 6 \
    --bin 0.01 \
    -o saxs_profile.dat
```

## See Also

- [Command Line Interface](cli.md) - CLI reference for batch processing
- [Configuration](../getting-started/configuration.md) - Detailed parameter descriptions
- [Quick Start](../getting-started/quickstart.md) - Getting started guide
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
