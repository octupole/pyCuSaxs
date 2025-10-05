# Python APIs

The Python package (`pycusaxs`) bridges trajectory handling and the CUDA backend while providing CLI and GUI front-ends. This reference documents the key classes and functions.

**Recent Updates (v0.1.0):**
- Added comprehensive path sanitization to prevent directory traversal attacks
- Enhanced input validation for all user-provided parameters
- Improved error messages with context information
- Fixed security vulnerabilities in file path handling

## Modules Overview

| Module | Path | Purpose |
|--------|------|---------|
| `pycusaxs.topology` | `pycusaxs/topology.py` | Trajectory/topology loading via MDAnalysis |
| `pycusaxs.main` | `pycusaxs/main.py` | CLI/GUI entry points and backend orchestration |
| `pycusaxs.saxs_widget` | `pycusaxs/saxs_widget.py` | PySide6 widgets for the GUI |
| Root launcher | `main.py` | Thin wrapper calling `pycusaxs.main:main` |

## Classes

### `Topology`
- **Module**: `pycusaxs.topology`
- **Constructor**: `Topology(tpr_file: str, xtc_file: str)`
- **Responsibilities**:
  - Loads a GROMACS topology/trajectory into an `MDAnalysis.Universe`.
  - Builds a connectivity graph (`networkx.Graph`) to classify molecules (proteins, water, ions, others).
  - Stores convenient subsets (`protein_atoms`, molecule lists) for downstream analysis.
- **Key Methods**:
  - `iter_frames_stream(start: int, stop: int, step: int = 1) -> Iterator[Dict]`
    - Streams frames lazily; yields dicts with `frame`, `time`, `positions`, `box`.
    - Positions and boxes are returned in Angstroms (nm-to-Å conversion applied).
  - `get_atom_index() -> Dict[str, List[int]]`
    - Groups atom indices by element symbol; used to map scatterers inside CUDA.
  - `read_frame(frame_number: int) -> Timestep`
    - Loads a specific frame; updates `self.ts` for access through getters.
  - `get_box()`, `get_step()`, `get_time()`, `get_coordinates()`
    - Accessors for the currently loaded frame (raise `RuntimeError` if `read_frame` was not invoked).
  - `count_molecules() -> Tuple[int, int, int, int, int]`
    - Returns totals for all molecules and per-class breakdown.
- **Properties**: `n_frames`, `n_atoms`
- **Notes**: Suppresses certain MDAnalysis warnings; expects topology and trajectory to be compatible.

### `SaxsParametersWindow`
- **Module**: `pycusaxs.saxs_widget`
- **Base Class**: `QWidget`
- **Purpose**: High-level GUI container combining required and advanced parameter forms, plus output display.
- **Key Methods**:
  - `execute()` (default implementation)
    - Collects parameters, echoes them in the output pane. Overridden by `SaxsMainWindow` to run the backend.
  - `show_advanced_dialog()`
    - Opens the modal dialog containing advanced options.
  - `_init_ui()`
    - Builds the widget layout; not intended for external use but useful when customizing the GUI.

### `RequiredParametersWidget`
- **Module**: `pycusaxs.saxs_widget`
- **Purpose**: Form for required CLI-equivalent fields (`topology`, `trajectory`, `grid`, `begin`, `end`).
- **Method**: `parameters() -> Dict[str, Any]`
  - Validates grid input (1 or 3 integers) and returns an ordered dictionary of values.

### `AdvancedParametersWidget`
- **Module**: `pycusaxs.saxs_widget`
- **Purpose**: Form for optional fields (output path, dt, order, scaled grid, scale factor, bin/qcut, solvent details, help flag).
- **Method**: `parameters() -> Dict[str, Any]`
  - Converts widget values into an ordered dictionary consumed by the GUI runner.

### `SaxsMainWindow`
- **Module**: `pycusaxs.main`
- **Inherits**: `SaxsParametersWindow`
- **Override**: `execute()`
  - Validates required parameters, merges advanced ones, invokes `cuda_connect`, displays results or GUI errors.
- **Error Display**: Shows error dialogs for validation failures or backend errors
- **Result Display**: Renders backend summary and molecule counts in the output panel

## Functions

### `build_output_paths(base: str, frame_range: range) -> List[Path]`
- **Module**: `pycusaxs.main`
- **Purpose**: Expand a user-provided output base (file or directory) into per-frame PDB paths used for frame export.
- **Behavior**: Handles single-frame runs, directories, and default naming (`trajectory_frame_{frame:05d}.pdb`).
- **Security (v0.1.0)**:
  - Sanitizes paths using `resolve()` to prevent symlink attacks
  - Validates paths are within current working directory or home directory
  - Rejects paths outside allowed directories with `ValueError`
  - Handles `OSError` and `RuntimeError` during path resolution

### `_invoke_cuda_backend(required_params: Dict[str, Any], advanced_params: Dict[str, Any], topology: Topology) -> List[str]`
- **Module**: `pycusaxs.main`
- **Purpose**: Imports `pycusaxs_cuda`, assembles keyword arguments, and calls the wrapped C++ backend.
- **Return**: List containing the formatted configuration summary (empty if backend unavailable or raises an exception).

### `cuda_connect(required_params: Dict[str, Any], advanced_params: Dict[str, Any]) -> Iterable[str]`
- **Module**: `pycusaxs.main`
- **Responsibilities**:
  - Validates filesystem inputs, frame bounds, and stride.
  - Instantiates `Topology` and gathers molecule statistics.
  - Invokes the CUDA backend and returns the combined textual output (counts + backend summary).
- **Security & Validation (v0.1.0)**:
  - **Path Sanitization**:
    - Resolves topology and trajectory paths to absolute paths
    - Validates paths are within CWD or home directory
    - Raises `ValueError` for paths outside allowed directories
  - **Input Validation**:
    - Checks files exist before processing
    - Validates frame range (begin < end)
    - Validates all numeric parameters are within acceptable ranges

### `_run_cli(namespace: argparse.Namespace) -> int`
- **Module**: `pycusaxs.main`
- **Purpose**: Executes the non-GUI code path; handles argument parsing errors, backend failures, and exit codes (0 success / 1 failure).

### `_run_gui() -> int`
- **Module**: `pycusaxs.main`
- **Purpose**: Launches the PySide6 application and returns its exit code.

### `main(argv: Optional[List[str]] = None) -> int`
- **Module**: `pycusaxs.main`
- **Purpose**: Entry point for CLI/GUI selection. Without arguments, launches the GUI; otherwise parses CLI arguments and calls `_run_cli`.
- **Integration**: Exposed both as console script `saxs-widget` and via `python -m pycusaxs.main`.
- **Return**: Exit code (0 for success, 1 for failure)

### `_parse_grid_values(value: str) -> tuple[int, int, int]`
- **Module**: `pycusaxs.main`
- **Purpose**: Parse grid size from command-line string (e.g., "128" or "128,128,128")
- **Validation**: Ensures 1 or 3 values, all positive integers
- **Return**: Tuple of (nx, ny, nz)

### `_build_cli_parser() -> argparse.ArgumentParser`
- **Module**: `pycusaxs.main`
- **Purpose**: Constructs the argument parser with all CLI options
- **Special Handling**:
  - Detects `gui` as first argument to launch GUI mode
  - Grid size accepts single or triple values
  - Type conversions for all numeric parameters

### GUI Widget Helpers
- `RequiredParametersWidget._parse_grid_size()` ensures the grid input contains either one or three integers.
- Advanced widget uses `QSpinBox`/`QDoubleSpinBox` to enforce numeric ranges (e.g., stride ≥ 1).
- `AdvancedParametersWidget` provides tooltips and help text for all advanced options.

## Root Launcher

### `main()` in `main.py`
- Calls `pycusaxs.main.main()` and exits with its status code, enabling `python main.py` as a shorthand.

## Error Handling

- CLI and GUI propagate user-facing errors via `ValueError`, `FileNotFoundError`, and GUI dialogs (`QMessageBox`).
- Backend failures bubble up as generic `Exception` objects, which are rendered as strings in the GUI or printed to stderr in the CLI.

### Exception Hierarchy (v0.1.0)

**Python → C++ → Python Flow:**
1. **Python Input Validation**: `ValueError` for invalid parameters, `FileNotFoundError` for missing files
2. **C++ Backend Validation**: `std::invalid_argument` for configuration errors, `std::runtime_error` for execution failures
3. **Exception Translation**: C++ exceptions properly translated to Python via pybind11:
   - `std::invalid_argument` → `ValueError`
   - `std::runtime_error` → `RuntimeError`
   - Generic exceptions → `RuntimeError` with "CuSAXS error:" prefix

**Security Exceptions:**
- Path outside allowed directories → `ValueError`
- Invalid file permissions → `ValueError` or `FileNotFoundError`
- File I/O failures → `RuntimeError` from C++ backend

## Extending the Python Layer

- To expose new backend options, update:
  1. CLI argument parser (`_build_cli_parser`).
  2. GUI widgets (add fields in `AdvancedParametersWidget` or `RequiredParametersWidget`).
  3. Parameter mapping in `cuda_connect` and `_invoke_cuda_backend`.
- Additional MDAnalysis-based analyses can be slotted into `Topology` or separate helper modules before passing frame data to the backend.

### Best Practices (v0.1.0)

**Security:**
- Always validate and sanitize user-provided file paths
- Use `Path.resolve()` to get absolute paths and detect symlink traversal
- Validate paths are within expected directories (CWD or home)
- Check file existence and permissions before processing

**Error Handling:**
- Provide specific, actionable error messages
- Validate inputs early (fail fast)
- Use appropriate exception types for different error categories
- Include context in error messages (file names, parameter values)

**Performance:**
- Use streaming iteration for large trajectories
- Release GIL during CPU-intensive operations
- Validate parameters before expensive operations

## Usage Examples

### Basic Topology Analysis

```python
from pycusaxs.topology import Topology

# Load trajectory
topo = Topology("system.tpr", "trajectory.xtc")

# Print system information
print(f"Total atoms: {topo.n_atoms}")
print(f"Total frames: {topo.n_frames}")

# Molecular composition
total, protein, water, ions, other = topo.count_molecules()
print(f"Proteins: {protein}")
print(f"Waters: {water}")
print(f"Ions: {ions}")

# Atom grouping by element
atom_index = topo.get_atom_index()
for element, indices in atom_index.items():
    print(f"{element}: {len(indices)} atoms")
```

### Streaming Frames

```python
from pycusaxs.topology import Topology

topo = Topology("system.tpr", "trajectory.xtc")

# Stream frames efficiently (memory-friendly for large trajectories)
for frame_data in topo.iter_frames_stream(start=0, stop=100, step=10):
    frame_num = frame_data['frame']
    time = frame_data['time']
    positions = frame_data['positions']  # NumPy array in Angstroms
    box = frame_data['box']  # Box dimensions

    print(f"Frame {frame_num} at {time:.2f} ps")
    print(f"  Box: {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} Å")
```

### Complete SAXS Workflow

```python
from pycusaxs.main import cuda_connect

# Configure calculation
required = {
    "topology": "/path/to/system.tpr",
    "trajectory": "/path/to/traj.xtc",
    "grid_size": [128, 128, 128],
    "initial_frame": 0,
    "last_frame": 999
}

advanced = {
    "dt": 10,              # Process every 10th frame
    "order": 4,            # B-spline order
    "bin_size": 0.01,      # 0.01 Å⁻¹ histogram bins
    "qcut": 0.5,           # 0.5 Å⁻¹ cutoff
    "scale_factor": 2.0,   # Grid scaling
    "out": "saxs.dat",     # Output file
    "simulation": "nvt"    # NVT ensemble
}

# Run SAXS calculation
try:
    results = cuda_connect(required, advanced)
    for line in results:
        print(line)
except ValueError as e:
    print(f"Validation error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Backend error: {e}")
```

### GUI Customization

```python
from PySide6.QtWidgets import QApplication
from pycusaxs.main import SaxsMainWindow
import sys

app = QApplication(sys.argv)

# Create custom window
window = SaxsMainWindow()
window.setWindowTitle("Custom SAXS Analysis")
window.resize(800, 600)

# Pre-fill some parameters (optional)
# Access widgets via window attributes and set values

window.show()
sys.exit(app.exec())
```

### Error Handling Best Practices

```python
from pathlib import Path
from pycusaxs.main import cuda_connect

def safe_saxs_calculation(tpr_path: str, xtc_path: str):
    """Safe SAXS calculation with comprehensive error handling."""

    # Validate files exist
    tpr = Path(tpr_path)
    xtc = Path(xtc_path)

    if not tpr.exists():
        raise FileNotFoundError(f"Topology not found: {tpr}")
    if not xtc.exists():
        raise FileNotFoundError(f"Trajectory not found: {xtc}")

    # Configure with validated paths
    required = {
        "topology": str(tpr.resolve()),
        "trajectory": str(xtc.resolve()),
        "grid_size": [64, 64, 64],
        "initial_frame": 0,
        "last_frame": 100
    }

    advanced = {
        "dt": 1,
        "out": "output/saxs.dat"
    }

    try:
        # Run calculation
        results = cuda_connect(required, advanced)
        return results
    except ValueError as e:
        print(f"Invalid parameter: {e}")
        raise
    except RuntimeError as e:
        print(f"Calculation failed: {e}")
        raise

# Usage
try:
    output = safe_saxs_calculation("system.tpr", "traj.xtc")
    for line in output:
        print(line)
except Exception as e:
    print(f"Error: {e}")
    exit(1)
```

Refer to the module source files for exact signatures and implementation details.
