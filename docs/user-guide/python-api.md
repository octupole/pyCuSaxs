# Python API

The pyCuSAXS Python API allows you to integrate SAXS calculations into your Python scripts and workflows.

## Overview

The Python API provides:

- **Trajectory Analysis**: Load and analyze MD trajectories using MDAnalysis
- **System Information**: Extract molecular composition and structure
- **SAXS Calculations**: Run GPU-accelerated SAXS calculations programmatically
- **Frame Streaming**: Memory-efficient iteration over large trajectories

## Core Modules

### pycusaxs.topology

The `Topology` class provides trajectory handling and system analysis.

### pycusaxs.main

The `cuda_connect` function orchestrates SAXS calculations.

## Getting Started

### Basic Example

```python
from pycusaxs.topology import Topology
from pycusaxs.main import cuda_connect

# Load topology and trajectory
topo = Topology("system.tpr", "trajectory.xtc")

# Print system information
print(f"Atoms: {topo.n_atoms}")
print(f"Frames: {topo.n_frames}")

# Get molecular composition
total, protein, water, ions, other = topo.count_molecules()
print(f"Proteins: {protein}, Waters: {water}, Ions: {ions}")

# Configure SAXS calculation
required = {
    "topology": "system.tpr",
    "trajectory": "trajectory.xtc",
    "grid_size": [128, 128, 128],
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

# Run SAXS calculation
results = cuda_connect(required, advanced)
for line in results:
    print(line)
```

## Topology Class

### Constructor

```python
from pycusaxs.topology import Topology

topo = Topology(tpr_file: str, xtc_file: str)
```

**Parameters:**

- `tpr_file` - Path to GROMACS topology file (`.tpr`)
- `xtc_file` - Path to trajectory file (`.xtc`, `.trr`, etc.)

**Example:**

```python
topo = Topology("system.tpr", "trajectory.xtc")
```

### Properties

#### n_atoms

Total number of atoms in the system.

```python
num_atoms = topo.n_atoms
print(f"System has {num_atoms} atoms")
```

#### n_frames

Total number of frames in the trajectory.

```python
num_frames = topo.n_frames
print(f"Trajectory has {num_frames} frames")
```

### Methods

#### count_molecules()

Count molecules by category.

```python
total, protein, water, ions, other = topo.count_molecules()
```

**Returns:** Tuple of (total, protein, water, ions, other)

**Example:**

```python
total, protein, water, ions, other = topo.count_molecules()
print(f"Total molecules: {total}")
print(f"Proteins: {protein}")
print(f"Waters: {water}")
print(f"Ions: {ions}")
print(f"Others: {other}")
```

#### get_atom_index()

Get atom indices grouped by element symbol.

```python
atom_index = topo.get_atom_index()
```

**Returns:** Dictionary mapping element symbols to lists of atom indices

**Example:**

```python
atom_index = topo.get_atom_index()
for element, indices in atom_index.items():
    print(f"{element}: {len(indices)} atoms")

# Output:
# C: 1234 atoms
# N: 456 atoms
# O: 789 atoms
# H: 2345 atoms
```

#### iter_frames_stream()

Stream frames lazily for memory-efficient processing.

```python
for frame_data in topo.iter_frames_stream(start, stop, step=1):
    # Process frame
    pass
```

**Parameters:**

- `start` - Starting frame index (0-based)
- `stop` - Ending frame index (exclusive)
- `step` - Frame stride (default: 1)

**Returns:** Iterator yielding dictionaries with:

- `frame` - Frame number
- `time` - Simulation time (ps)
- `positions` - Atomic positions (NumPy array, Angstroms)
- `box` - Box dimensions (Angstroms)

**Example:**

```python
# Stream every 10th frame
for frame_data in topo.iter_frames_stream(0, 100, step=10):
    frame_num = frame_data['frame']
    time = frame_data['time']
    positions = frame_data['positions']
    box = frame_data['box']

    print(f"Frame {frame_num} at {time:.2f} ps")
    print(f"  Box: {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} Å")
    print(f"  Atoms: {len(positions)}")
```

#### read_frame()

Load a specific frame.

```python
timestep = topo.read_frame(frame_number)
```

**Parameters:**

- `frame_number` - Frame index to load (0-based)

**Returns:** MDAnalysis Timestep object

**Example:**

```python
# Load frame 100
ts = topo.read_frame(100)

# Access frame data through getter methods
coords = topo.get_coordinates()
box = topo.get_box()
time = topo.get_time()
```

#### Getter Methods

After calling `read_frame()`, use these methods to access frame data:

```python
# Get coordinates (NumPy array, Angstroms)
coords = topo.get_coordinates()

# Get box dimensions (Angstroms)
box = topo.get_box()

# Get simulation time (ps)
time = topo.get_time()

# Get time step (ps)
step = topo.get_step()
```

!!! warning "Read Frame First"
    Getter methods raise `RuntimeError` if `read_frame()` hasn't been called.

## SAXS Calculation

### cuda_connect()

Run SAXS calculation with specified parameters.

```python
from pycusaxs.main import cuda_connect

results = cuda_connect(required_params, advanced_params)
```

**Parameters:**

- `required_params` - Dictionary of required parameters
- `advanced_params` - Dictionary of optional parameters

**Returns:** Iterable of output strings (configuration summary and results)

### Required Parameters

```python
required = {
    "topology": str,          # Path to topology file
    "trajectory": str,        # Path to trajectory file
    "grid_size": [int, int, int],  # Grid dimensions
    "initial_frame": int,     # Starting frame (0-based)
    "last_frame": int        # Ending frame (inclusive)
}
```

**Example:**

```python
required = {
    "topology": "/path/to/system.tpr",
    "trajectory": "/path/to/trajectory.xtc",
    "grid_size": [128, 128, 128],
    "initial_frame": 0,
    "last_frame": 999
}
```

### Advanced Parameters

```python
advanced = {
    "dt": int,               # Frame stride (default: 1)
    "order": int,            # B-spline order (default: 4)
    "bin_size": float,       # Histogram bin width (Å⁻¹)
    "qcut": float,           # Q cutoff (Å⁻¹)
    "scale_factor": float,   # Grid scaling factor (default: 1.0)
    "scaled_grid": [int, int, int],  # Explicit scaled grid
    "water_model": str,      # Water model (e.g., "tip3p")
    "sodium": int,           # Number of Na⁺ ions
    "chlorine": int,         # Number of Cl⁻ ions
    "simulation": str,       # "nvt" or "npt"
    "out": str              # Output file path
}
```

**Example:**

```python
advanced = {
    "dt": 10,
    "order": 6,
    "bin_size": 0.01,
    "qcut": 0.5,
    "scale_factor": 2.0,
    "out": "saxs_profile.dat",
    "simulation": "nvt"
}
```

## Complete Examples

### Example 1: System Analysis

Analyze trajectory without running SAXS:

```python
from pycusaxs.topology import Topology

# Load system
topo = Topology("system.tpr", "trajectory.xtc")

# System information
print(f"System: {topo.n_atoms} atoms, {topo.n_frames} frames")

# Molecular composition
total, protein, water, ions, other = topo.count_molecules()
print(f"\nMolecular Composition:")
print(f"  Proteins: {protein}")
print(f"  Waters: {water}")
print(f"  Ions: {ions}")
print(f"  Others: {other}")

# Element distribution
atom_index = topo.get_atom_index()
print(f"\nElement Distribution:")
for element in sorted(atom_index.keys()):
    count = len(atom_index[element])
    print(f"  {element}: {count} atoms")
```

### Example 2: Frame Analysis

Process individual frames:

```python
from pycusaxs.topology import Topology
import numpy as np

topo = Topology("system.tpr", "trajectory.xtc")

# Analyze specific frames
for frame_num in [0, 100, 200, 300]:
    topo.read_frame(frame_num)

    coords = topo.get_coordinates()
    box = topo.get_box()
    time = topo.get_time()

    # Calculate center of mass
    com = np.mean(coords, axis=0)

    print(f"Frame {frame_num} (t={time:.1f} ps):")
    print(f"  Box: {box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} Å")
    print(f"  COM: {com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f} Å")
```

### Example 3: Streaming Large Trajectories

Memory-efficient processing:

```python
from pycusaxs.topology import Topology
import numpy as np

topo = Topology("system.tpr", "large_trajectory.xtc")

# Stream frames without loading entire trajectory
box_volumes = []

for frame_data in topo.iter_frames_stream(0, topo.n_frames, step=10):
    box = frame_data['box']
    volume = np.prod(box[:3])  # Box volume
    box_volumes.append(volume)

    if frame_data['frame'] % 100 == 0:
        print(f"Processed frame {frame_data['frame']}")

# Statistics
mean_vol = np.mean(box_volumes)
std_vol = np.std(box_volumes)
print(f"\nBox Volume: {mean_vol:.1f} ± {std_vol:.1f} Å³")
```

### Example 4: Batch SAXS Calculations

Run multiple calculations:

```python
from pycusaxs.main import cuda_connect

# Common parameters
base_required = {
    "topology": "system.tpr",
    "trajectory": "trajectory.xtc",
    "grid_size": [128, 128, 128]
}

base_advanced = {
    "dt": 10,
    "order": 6,
    "bin_size": 0.01,
    "qcut": 0.5
}

# Process different frame ranges
frame_ranges = [
    (0, 249, "saxs_0-249.dat"),
    (250, 499, "saxs_250-499.dat"),
    (500, 749, "saxs_500-749.dat"),
    (750, 999, "saxs_750-999.dat")
]

for start, end, output in frame_ranges:
    print(f"\nProcessing frames {start}-{end}...")

    required = base_required.copy()
    required["initial_frame"] = start
    required["last_frame"] = end

    advanced = base_advanced.copy()
    advanced["out"] = output

    try:
        results = cuda_connect(required, advanced)
        for line in results:
            print(line)
        print(f"Saved to {output}")
    except Exception as e:
        print(f"Error: {e}")
```

### Example 5: Parameter Scan

Scan grid sizes for optimal parameters:

```python
from pycusaxs.main import cuda_connect
import time

# Test different grid sizes
grid_sizes = [64, 96, 128, 160, 192]

results_summary = []

for grid_size in grid_sizes:
    print(f"\nTesting grid size: {grid_size}³")

    required = {
        "topology": "system.tpr",
        "trajectory": "trajectory.xtc",
        "grid_size": [grid_size, grid_size, grid_size],
        "initial_frame": 0,
        "last_frame": 10  # Small test
    }

    advanced = {
        "dt": 1,
        "out": f"test_grid_{grid_size}.dat"
    }

    start_time = time.time()

    try:
        results = list(cuda_connect(required, advanced))
        elapsed = time.time() - start_time

        results_summary.append({
            'grid': grid_size,
            'time': elapsed,
            'success': True
        })

        print(f"  Time: {elapsed:.2f} s")
        print(f"  Time per frame: {elapsed/11*1000:.2f} ms")

    except Exception as e:
        print(f"  Error: {e}")
        results_summary.append({
            'grid': grid_size,
            'time': None,
            'success': False
        })

# Print summary
print("\n" + "="*50)
print("Grid Size Performance Summary")
print("="*50)
for result in results_summary:
    if result['success']:
        print(f"{result['grid']:3d}³: {result['time']:6.2f} s "
              f"({result['time']/11*1000:6.2f} ms/frame)")
    else:
        print(f"{result['grid']:3d}³: FAILED")
```

## Error Handling

### Exception Types

```python
try:
    results = cuda_connect(required, advanced)
    for line in results:
        print(line)
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except RuntimeError as e:
    print(f"Calculation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation Example

```python
from pathlib import Path
from pycusaxs.main import cuda_connect

def safe_saxs_calculation(tpr_path, xtc_path, output_path):
    """Run SAXS with comprehensive error handling."""

    # Validate input files
    tpr = Path(tpr_path)
    xtc = Path(xtc_path)

    if not tpr.exists():
        raise FileNotFoundError(f"Topology not found: {tpr}")
    if not xtc.exists():
        raise FileNotFoundError(f"Trajectory not found: {xtc}")

    # Configure calculation
    required = {
        "topology": str(tpr.resolve()),
        "trajectory": str(xtc.resolve()),
        "grid_size": [128, 128, 128],
        "initial_frame": 0,
        "last_frame": 100
    }

    advanced = {
        "dt": 10,
        "out": output_path
    }

    # Run with error handling
    try:
        results = cuda_connect(required, advanced)
        return list(results)
    except ValueError as e:
        print(f"Parameter error: {e}")
        raise
    except RuntimeError as e:
        print(f"Backend error: {e}")
        raise

# Usage
try:
    output = safe_saxs_calculation(
        "system.tpr",
        "trajectory.xtc",
        "output/saxs.dat"
    )
    print("\n".join(output))
except Exception as e:
    print(f"Calculation failed: {e}")
    exit(1)
```

## Integration with Workflows

### Jupyter Notebook Example

```python
# Cell 1: Import and setup
from pycusaxs.topology import Topology
from pycusaxs.main import cuda_connect
import matplotlib.pyplot as plt
import numpy as np

# Cell 2: Load and analyze topology
topo = Topology("system.tpr", "trajectory.xtc")
total, protein, water, ions, other = topo.count_molecules()

print(f"System: {topo.n_atoms} atoms, {topo.n_frames} frames")
print(f"Composition: {protein} proteins, {water} waters, {ions} ions")

# Cell 3: Run SAXS
required = {
    "topology": "system.tpr",
    "trajectory": "trajectory.xtc",
    "grid_size": [128, 128, 128],
    "initial_frame": 0,
    "last_frame": 100
}

advanced = {
    "dt": 10,
    "order": 6,
    "bin_size": 0.01,
    "qcut": 0.5,
    "out": "saxs.dat"
}

results = cuda_connect(required, advanced)
for line in results:
    print(line)

# Cell 4: Plot results
data = np.loadtxt('saxs.dat')
q = data[:, 0]
I = data[:, 1]

plt.figure(figsize=(10, 6))
plt.loglog(q, I, 'b-', linewidth=2)
plt.xlabel('q (Å⁻¹)', fontsize=14)
plt.ylabel('I(q) (a.u.)', fontsize=14)
plt.title('SAXS Profile', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## See Also

- [Backend API](../api/backend.md) - C++/CUDA backend reference
- [Python API Reference](../api/python.md) - Complete Python API documentation
- [CLI Guide](cli.md) - Command-line interface
- [Examples](../getting-started/quickstart.md) - Quick start examples
