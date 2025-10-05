# CuSAXS Documentation

CuSAXS is a CUDA-accelerated pipeline for generating small-angle X-ray scattering (SAXS) profiles from large molecular dynamics trajectories. The project combines a high-performance C++/CUDA backend with Python tooling for trajectory loading, configuration, and GUI/CLI orchestration.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Concepts](#core-concepts)
3. [C++ / CUDA Backend](#c--cuda-backend)
4. [Python Tooling](#python-tooling)
5. [Building and Installing](#building-and-installing)
6. [Running the Pipeline](#running-the-pipeline)
7. [Configuration Reference](#configuration-reference)
8. [Output Files](#output-files)
9. [Developer Guide](#developer-guide)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

## Project Structure

```
.
├── CMakeLists.txt          # Top-level build configuration (scikit-build + CUDA)
├── cpp-src/                # CUDA/C++ implementation and pybind11 bridge
│   ├── CMakeLists.txt
│   ├── Exec/               # Entry-point logic (RunSaxs, Options)
│   ├── Saxs/               # GPU kernels, spline helpers, histogram logic
│   ├── System/             # Cell transformations, solvent density helpers
│   ├── Utilities/          # Shared math utilities and splines
│   └── pybind/             # pybind11 bindings exposing the backend to Python
├── pycusaxs/               # Python package (CLI, GUI, MDAnalysis integration)
│   ├── main.py             # CLI/GUI front-end
│   ├── topology.py         # MDAnalysis-based topology/trajectory loader
│   └── saxs_widget.py      # PySide6 GUI components
├── main.py                 # Convenience launcher for the Python entry point
├── pyproject.toml          # scikit-build configuration for packaging
├── requirements.txt        # Python runtime dependencies
└── setup.sh                # Helper for setting up a local development env
```

> Note: Artefacts under `test/` are local fixtures and are not part of distributed packages.

## Core Concepts

CuSAXS consumes a GROMACS topology (`.tpr`) and trajectory (`.xtc` or compatible) and produces SAXS intensity curves. The process involves:

1. Streaming trajectory frames from Python via MDAnalysis.
2. Converting positions and box dimensions into CUDA-friendly structures.
3. Running GPU kernels that build density grids, apply padding/supersampling, and compute FFTs.
4. Accumulating scattering intensities and binning results into SAXS histograms.
5. Writing `saxs.dat` (or a user-provided output file) with q-value vs intensity pairs.

## C++ / CUDA Backend

The backend is organised into modular libraries loaded by pybind11.

- **Exec/RunSaxs**: Orchestrates frame iteration, manages double-buffered loading from Python, and coordinates kernel execution. It writes the final histogram and reports timing statistics.
- **Exec/Options**: Stores global configuration shared across kernels (grid shape, padding mode, file paths, histogram parameters).
- **System/Cell**: Converts triclinic simulation boxes into orthonormalised coordinate transformations (`CO` and `OC` matrices) used by CUDA kernels.
- **System/AtomCounter**: Estimates water and ion densities when explicit solvent weighting is enabled.
- **Saxs/saxsKernel**: Implements the GPU pipeline (coordinate density assignment, padding, FFT, scattering factor application, histogram accumulation). It also manages device memory buffers and CUFFT plans.
- **Utilities/Splines & opsfact**: Provide B-spline modulation factors and scattering form-factor data for different atom types.
- **pybind/cuda_bindings.cpp**: Exposes `run_cuda_saxs` through the `pycusaxs_cuda` module, returning configuration summaries to Python callers.

All CUDA code assumes single-precision coordinates and relies on CUFFT for Fourier transforms. Thread/block dimensions are controlled via `Options::nx/ny/nz` (primary grid) and `Options::nnx/nny/nnz` (scaled grid). Padding modes balance solvent density averaging versus explicit water model input.

## Python Tooling

The Python package glues MDAnalysis trajectory streaming to the CUDA backend and offers user interfaces.

- **pycusaxs.topology.Topology**
  - Wraps MDAnalysis to load `.tpr` and `.xtc` files.
  - Provides `iter_frames_stream()` for memory-efficient iteration.
  - Supplies atom indices grouped by element, box dimensions, and coordinates in Angstroms.

- **pycusaxs.main**
  - CLI: parses arguments, validates inputs, constructs configuration dictionaries, and invokes the CUDA backend via `pycusaxs_cuda.run()`.
  - GUI: launches a PySide6 window (`SaxsParametersWindow`) for configuring runs; displays backend summaries on completion.

- **pycusaxs.saxs_widget**
  - Defines reusable widgets for required and advanced parameters, plus basic validation.

`main.py` at the repository root aliases `pycusaxs.main:main` for direct execution without installing the package.

## Building and Installing

### Prerequisites

- NVIDIA GPU with a CUDA toolkit supported by CUFFT (tested with CUDA 11+)
  - The build scripts call `nvidia-smi` during configuration and abort if no device is detected.
- CMake 3.20 or newer
- A C++17-capable compiler and matching CUDA host compiler
- Python 3.9 or newer with `pip`
- System packages: development headers for Python, CUDA, and a recent GCC/Clang

### Quick Start (recommended)

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install build requirements and CuSAXS
pip install -r requirements.txt
pip install .
```

### Development Build

```bash
# Configure and build the C++/CUDA extension in-place
pip install -e .

# Alternatively, drive the CMake build manually
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`setup.sh` provides a convenience wrapper that installs Python dependencies and builds the extension using the active interpreter (`PYTHON_BIN` can be exported to override the default).

## Running the Pipeline

### Command-Line Interface

After installation (or from the repository root with the editable build):

```bash
python -m pycusaxs.main \
  --topology path/to/system.tpr \
  --trajectory path/to/trajectory.xtc \
  --grid 128,128,128 \
  --begin 0 \
  --end 999 \
  --dt 10 \
  --out results/saxs.dat
```

Use `python -m pycusaxs.main --help` for the full list of options. The CLI validates input files, checks frame bounds against the trajectory length, and prints the formatted backend summary returned by the CUDA layer.

### GUI

```bash
python -m pycusaxs.main gui
# or, after installation
saxs-widget
```

The GUI mirrors the CLI options, opening an “Advanced Parameters” dialog for optional fields. When the run completes, the CuSAXS backend summary appears in the output panel and standard output.

## Configuration Reference

| Parameter            | CLI / GUI field          | Description                                                                                       |
|----------------------|--------------------------|---------------------------------------------------------------------------------------------------|
| `topology`           | `--topology`, `-s`       | Path to the GROMACS `.tpr` file                                                                   |
| `trajectory`         | `--trajectory`, `-x`     | Path to the trajectory (`.xtc`, `.trr`, etc.)                                                     |
| `grid_size`          | `--grid`, `-g`           | Primary density grid size (1 value for cubic or 3 for anisotropic grids)                          |
| `grid_scaled`        | `--gridS`                | Optional supersampled grid size (defaults to automatic sigma-based scaling)                      |
| `order`              | `--order`                | B-spline interpolation order (default 4)                                                          |
| `initial_frame`      | `--begin`, `-b`          | Start frame index (inclusive)                                                                     |
| `last_frame`         | `--end`, `-e`            | End frame index (inclusive; defaults to `initial_frame`)                                          |
| `dt` / `stride`      | `--dt`                   | Frame stride when iterating the trajectory                                                        |
| `scale_factor`       | `--Scale`                | Sigma value controlling supersampled grid sizing                                                  |
| `bin_size`           | `--bin`, `--Dq`          | Histogram bin width in reciprocal space units                                                     |
| `qcut`               | `--qcut`, `-q`           | Fourier space cutoff (`0` lets CuSAXS pick a conservative limit)                                  |
| `water_model`        | `--water`                | Solvent model identifier (enables explicit padding weights via `AtomCounter`)                    |
| `sodium`, `chlorine` | `--na`, `--cl`           | Explicit counts used when `water_model` is provided                                               |
| `output`             | `--out`, `-o`            | Destination for the SAXS output (defaults to `saxs.dat` in the working directory)                 |
| `simulation`         | GUI advanced field       | Optional ensemble flag (`nvt` triggers cumulative histogram averaging)                            |

The Python binding applies validation to critical fields (non-empty paths, non-negative grid sizes, stride positivity) before forwarding configurations to the C++ layer.

## Output Files

- **`saxs.dat`** (or the configured output path): Two-column text file with q-values and averaged intensity.
- **Console summary**: CuSAXS prints a formatted configuration banner and timing statistics (CUDA time, frame read time, total time per step).

Intermediate GPU data structures reside on the device; no temporary host files are created.

## Developer Guide

- **Code style**: C++ uses C++17 with CUDA kernels; Python targets 3.9+ with type annotations where practical.
- **Directory boundaries**: Backend code lives under `cpp-src/`; Python code interacts with compiled modules via `pycusaxs_cuda`.
- **Extending kernels**: Modify `saxsKernel.cu` for new scattering behaviour. Update `saxsKernel.h` and ensure host/device buffers are sized in `createMemory()`.
- **Adding configuration fields**: Extend `CudaSaxsConfig` in `cpp-src/include/CudaSaxsInterface.h`, propagate to `run_cuda_saxs`, and expose through the pybind wrapper and Python CLI/GUI.
- **Building docs**: This README serves as the primary documentation entry point. Supplementary docs can be added under `docs/` if needed.
- **Testing**: End-to-end validation typically involves running the CLI/GUI against representative trajectories and confirming histogram outputs. (Test fixtures under `test/` are local-only.)

## Troubleshooting

- **CUDA linking errors**: Ensure `cudadevrt` is discoverable; the top-level `CMakeLists.txt` attempts to locate it but may need explicit `CUDAToolkit_ROOT` hints.
- **ImportError: pycusaxs_cuda**: Verify the extension is built (`pip install .` or `pip install -e .`) and that Python is loading the correct environment.
- **MDAnalysis warnings**: The topology loader suppresses “No coordinate reader found” warnings by design. Other MDAnalysis errors usually indicate mismatched topology/trajectory files.
- **Empty SAXS output**: Check that `qcut` and `Dq` are sensible. Allowing CuSAXS to choose defaults (by leaving the fields unset) is a safe starting point.

For additional questions or contributions, please open an issue or pull request in the repository.

## API Reference

Detailed class and function documentation lives under `docs/`:

- `docs/backend.md` covers the C++ / CUDA types (`RunSaxs`, `saxsKernel`, `Options`, device kernels, and the pybind interface).
- `docs/python.md` documents the Python package (`Topology`, CLI/GUI entry points, widgets, and helper utilities).
- `docs/index.md` links the sections above and describes documentation conventions.

Consult these files for deeper API-level insights when extending CuSAXS.
