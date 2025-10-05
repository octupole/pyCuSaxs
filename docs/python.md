# Python APIs

The Python package (`pycusaxs`) bridges trajectory handling and the CUDA backend while providing CLI and GUI front-ends. This reference documents the key classes and functions.

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

## Functions

### `build_output_paths(base: str, frame_range: range) -> List[Path]`
- **Module**: `pycusaxs.main`
- **Purpose**: Expand a user-provided output base (file or directory) into per-frame PDB paths used for frame export.
- **Behavior**: Handles single-frame runs, directories, and default naming (`trajectory_frame_{frame:05d}.pdb`).

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

### GUI Widget Helpers
- `RequiredParametersWidget._parse_grid_size()` ensures the grid input contains either one or three integers.
- Advanced widget uses `QSpinBox`/`QDoubleSpinBox` to enforce numeric ranges (e.g., stride ≥ 1).

## Root Launcher

### `main()` in `main.py`
- Calls `pycusaxs.main.main()` and exits with its status code, enabling `python main.py` as a shorthand.

## Error Handling

- CLI and GUI propagate user-facing errors via `ValueError`, `FileNotFoundError`, and GUI dialogs (`QMessageBox`).
- Backend failures bubble up as generic `Exception` objects, which are rendered as strings in the GUI or printed to stderr in the CLI.

## Extending the Python Layer

- To expose new backend options, update:
  1. CLI argument parser (`_build_cli_parser`).
  2. GUI widgets (add fields in `AdvancedParametersWidget` or `RequiredParametersWidget`).
  3. Parameter mapping in `cuda_connect` and `_invoke_cuda_backend`.
- Additional MDAnalysis-based analyses can be slotted into `Topology` or separate helper modules before passing frame data to the backend.

Refer to the module source files for exact signatures and implementation details.
