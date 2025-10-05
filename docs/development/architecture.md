# Architecture

This page describes the overall architecture and design of pyCuSAXS, including the project structure, technology stack, and data flow.

## Project Overview

pyCuSAXS is designed as a hybrid Python/C++ application that combines:

- **Python Layer:** User interfaces (CLI/GUI) and trajectory handling
- **C++/CUDA Backend:** High-performance SAXS calculations on GPU
- **pybind11 Bindings:** Seamless Python-C++ interoperability

## Project Structure

```
pyCuSAXS/
├── pycusaxs/              # Python package
│   ├── __init__.py        # Package initialization
│   ├── main.py            # CLI/GUI entry point
│   ├── topology.py        # MDAnalysis integration
│   └── saxs_widget.py     # PySide6 GUI components
│
├── cpp-src/               # C++/CUDA backend
│   ├── pybind/            # Python bindings
│   │   └── cuda_bindings.cpp  # pybind11 wrapper
│   ├── Exec/              # Execution orchestration
│   │   ├── RunSaxs.cu     # Main pipeline controller
│   │   └── Options.cpp    # Global configuration
│   ├── Saxs/              # SAXS computation kernels
│   │   ├── saxsKernel.cu  # Host-side kernel manager
│   │   └── saxsDeviceKernels.cu  # GPU kernels
│   ├── System/            # Coordinate transformations
│   │   ├── Cell.cpp       # Box geometry handling
│   │   └── AtomCounter.cpp # Solvent density calculation
│   ├── Utilities/         # Mathematical utilities
│   │   ├── BSpline.cpp    # B-spline modulation
│   │   └── Scattering.cpp # Form factor database
│   └── include/           # Header files
│       ├── CudaSaxsInterface.h  # Main API
│       └── *.h            # Component headers
│
├── docs/                  # Documentation
│   ├── getting-started/   # Installation and quickstart
│   ├── user-guide/        # User documentation
│   ├── algorithm/         # Algorithm details
│   ├── api/               # API reference
│   └── development/       # Developer guides
│
├── CMakeLists.txt         # Build configuration
├── pyproject.toml         # Python packaging
├── requirements.txt       # Python dependencies
├── setup.py               # Build script
└── README.md              # Project overview
```

## Technology Stack

### Python Layer

#### PySide6

**Purpose:** Graphical user interface

**Components:**

- `QWidget` - Base GUI components
- `QMainWindow` - Application window
- `QFileDialog` - File selection
- `QMessageBox` - Error dialogs

**Files:**

- `pycusaxs/saxs_widget.py` - GUI widgets
- `pycusaxs/main.py` - Main window implementation

#### MDAnalysis

**Purpose:** Trajectory and topology parsing

**Features:**

- Multi-format support (GROMACS, AMBER, CHARMM, etc.)
- Efficient frame iteration
- Atom selection language
- Coordinate transformations

**Files:**

- `pycusaxs/topology.py` - Wrapper around MDAnalysis

**Key Classes:**

- `Universe` - System container
- `AtomGroup` - Atom selections
- `Timestep` - Frame data

#### NetworkX

**Purpose:** Molecular graph construction

**Usage:**

- Identify connected components (molecules)
- Classify molecular types (protein, water, ions)
- Count molecules

**Files:**

- `pycusaxs/topology.py` - Graph-based molecule analysis

#### NumPy

**Purpose:** Numerical operations

**Usage:**

- Coordinate arrays
- Mathematical operations
- Data type conversions

### C++/CUDA Backend

#### CUDA Runtime & cuFFT

**Purpose:** GPU acceleration and FFT

**Components:**

- **CUDA Runtime:** Kernel execution, memory management
- **cuFFT:** Optimized 3D Fourier transforms

**Files:**

- `cpp-src/Saxs/saxsKernel.cu` - cuFFT integration
- `cpp-src/Saxs/saxsDeviceKernels.cu` - CUDA kernels

**Key Operations:**

- Grid-based density assignment
- 3D real-to-complex FFT
- Histogram accumulation

#### Thrust

**Purpose:** GPU data structures and algorithms

**Features:**

- `device_vector` - GPU memory management (RAII)
- Parallel algorithms
- Exception-safe cleanup

**Files:**

- Most `.cu` files use Thrust containers

**Example:**

```cpp
thrust::device_vector<float> d_density(grid_size);
// Automatic cleanup when out of scope
```

#### pybind11

**Purpose:** Python-C++ bindings

**Features:**

- Automatic type conversion
- Exception translation
- GIL management
- NumPy integration

**Files:**

- `cpp-src/pybind/cuda_bindings.cpp` - Main binding module

**Interface:**

```cpp
PYBIND11_MODULE(pycusaxs_cuda, m) {
    m.def("run", &run_cuda_saxs, "Run SAXS calculation");
}
```

#### fmt Library

**Purpose:** Modern C++ string formatting

**Usage:**

- Configuration banners
- Error messages
- Output formatting

**Files:**

- `cpp-src/Exec/Options.cpp` - Banner formatting
- `cpp-src/Exec/RunSaxs.cu` - Timing output

## Component Architecture

### Python Components

#### Topology (`topology.py`)

**Responsibilities:**

- Load GROMACS topology and trajectory
- Parse atomic structure
- Build connectivity graph
- Classify molecules
- Stream frames efficiently

**Key Methods:**

```python
class Topology:
    def __init__(tpr_file, xtc_file)
    def count_molecules() -> (total, protein, water, ions, other)
    def get_atom_index() -> Dict[str, List[int]]
    def iter_frames_stream(start, stop, step) -> Iterator[Dict]
```

**Dependencies:**

- MDAnalysis
- NetworkX
- NumPy

#### Main Module (`main.py`)

**Responsibilities:**

- Argument parsing (CLI)
- Parameter validation
- Backend orchestration
- GUI instantiation

**Key Functions:**

```python
def cuda_connect(required_params, advanced_params) -> Iterable[str]
def _invoke_cuda_backend(...) -> List[str]
def main(argv) -> int
```

**Entry Points:**

- `python -m pycusaxs.main` - CLI or GUI
- `saxs-widget` - Console script

#### GUI Widgets (`saxs_widget.py`)

**Components:**

```
SaxsParametersWindow
├── RequiredParametersWidget
│   ├── Topology file selector
│   ├── Trajectory file selector
│   ├── Grid size input
│   └── Frame range inputs
├── AdvancedParametersDialog
│   ├── Output path
│   ├── Grid parameters
│   ├── Histogram parameters
│   └── Solvent parameters
└── Output display panel
```

**Inheritance:**

```
QWidget
└── SaxsParametersWindow
    └── SaxsMainWindow (overrides execute())
```

### C++/CUDA Components

#### Options (`Options.cpp/h`)

**Purpose:** Global configuration singleton

**Data:**

- Grid dimensions (`nx`, `ny`, `nz`)
- Scaled grid (`nnx`, `nny`, `nnz`)
- Parameters (`order`, `sigma`, `Dq`, `Qcut`)
- File paths
- Solvent settings

**Usage:** Accessed by all backend components

#### RunSaxs (`RunSaxs.cu/h`)

**Purpose:** Pipeline orchestration

**Responsibilities:**

- Frame iteration
- GIL management
- Double-buffered loading
- Kernel invocation
- Output writing

**Key Method:**

```cpp
void RunSaxs::Run(py::object Topol, int beg, int end, int dt)
```

**Threading:**

- Acquires GIL for Python access
- Releases GIL during GPU computation
- Synchronizes when needed

#### saxsKernel (`saxsKernel.cu/h`)

**Purpose:** GPU computation manager

**Responsibilities:**

- CUDA memory allocation
- cuFFT plan management
- Kernel launch orchestration
- Histogram accumulation

**Key Method:**

```cpp
void saxsKernel::runPKernel(FrameData &frame)
```

**Pipeline Stages:**

1. Coordinate transformation
2. Density assignment
3. Padding & supersampling
4. FFT
5. Form factor application
6. Intensity calculation
7. Histogram binning

#### Device Kernels (`saxsDeviceKernels.cu`)

**CUDA Kernels:**

- `rhoCartKernel` - Density grid assignment
- `paddingKernel` - Solvent padding calculation
- `superDensityKernel` - Grid supersampling
- `scatterKernel` - Form factor application
- `modulusKernel` - Intensity calculation
- `calculate_histogram` - Binning

**Thread Organization:**

```cpp
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
kernel<<<gridSize, blockSize>>>(...);
```

#### Cell (`Cell.cpp/h`)

**Purpose:** Coordinate system transformations

**Methods:**

- `calculateMatrices()` - Compute CO/OC matrices
- `getCO()` - Cell-to-orthonormal matrix
- `getOC()` - Orthonormal-to-cell matrix

**Supports:** Orthorhombic and triclinic boxes

#### BSpline (`BSpline.cpp/h`)

**Purpose:** B-spline interpolation utilities

**Components:**

- `BSpmod` class - B-spline modulation factors
- `generateModulation()` - Compute Fourier space corrections
- Order 1-8 support

#### Scattering (`Scattering.cpp/h`)

**Purpose:** Atomic form factor database

**Data:**

- Tabulated coefficients for all elements
- 9-parameter form factor fit: $f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c$

**Methods:**

- Look up by element symbol
- Interpolate for arbitrary q values

## Data Flow

### High-Level Flow

```
┌─────────────────┐
│   User Input    │ (CLI args or GUI)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Topology Load  │ (MDAnalysis)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Frame Streaming │ (Python iterator)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  CUDA Backend   │ (C++/CUDA)
│  ┌───────────┐  │
│  │ Transform │  │
│  ├───────────┤  │
│  │  Density  │  │
│  ├───────────┤  │
│  │  Padding  │  │
│  ├───────────┤  │
│  │    FFT    │  │
│  ├───────────┤  │
│  │ Scattering│  │
│  ├───────────┤  │
│  │ Histogram │  │
│  └───────────┘  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ SAXS Profile    │ (Output file)
└─────────────────┘
```

### Detailed Frame Processing

```
Python Thread                GPU Stream
─────────────────────────────────────────

Load Frame n
  │ GIL held
  ├─> Extract coords
  ├─> Extract box
  └─> Create FrameData
      │
      v
Transfer to GPU ──────────> Coords in device memory
  │ GIL released            │
  │                         v
  │                    Transform Kernel
  │                         │
  │                         v
Load Frame n+1         Density Kernel
  │ GIL held                │
  ├─> Extract coords        v
  │                    Padding Kernel
  └─> Create FrameData      │
      │                     v
      │                FFT Execution
      │                     │
      v                     v
Wait for GPU ←───────  Histogram Update
  │ GIL held
  v
Process Results
```

### Memory Layout

**CPU Side:**

```
Python Objects
  │
  v
NumPy Arrays (coordinates, box)
  │
  v
std::vector<float> (C++ staging)
```

**GPU Side:**

```
Device Memory Layout:

┌──────────────────┐
│  d_coords_x      │ Float array [n_atoms]
├──────────────────┤
│  d_coords_y      │ Float array [n_atoms]
├──────────────────┤
│  d_coords_z      │ Float array [n_atoms]
├──────────────────┤
│  d_density       │ Float array [nx×ny×nz]
├──────────────────┤
│  d_density_scaled│ Float array [nnx×nny×nnz]
├──────────────────┤
│  d_fft_buffer    │ Complex array [nnx×nny×(nnz/2+1)]
├──────────────────┤
│  d_histogram_I   │ Float array [n_bins]
├──────────────────┤
│  d_histogram_N   │ Int array [n_bins]
└──────────────────┘
```

## Communication Patterns

### Python → C++

**Via pybind11:**

```python
# Python side
results = pycusaxs_cuda.run(
    obj_topology=topology_object,
    topology="system.tpr",
    trajectory="trajectory.xtc",
    grid=[128, 128, 128],
    # ...
)
```

**Automatic conversions:**

- Python `list` → C++ `std::vector`
- Python `str` → C++ `std::string`
- Python `dict` → C++ function arguments
- Python `int/float` → C++ `int/float`

### C++ → Python

**Return values:**

```cpp
// C++ side
return py::dict(
    "summary"_a = summary_string,
    "nx"_a = Options::nx,
    "ny"_a = Options::ny,
    // ...
);
```

**Exceptions:**

```cpp
try {
    // C++ code
} catch (std::invalid_argument& e) {
    throw py::value_error(e.what());  // → Python ValueError
} catch (std::runtime_error& e) {
    throw py::runtime_error(e.what());  // → Python RuntimeError
}
```

### Python ↔ GPU

**Frame data transfer:**

```python
# Python (NumPy array)
positions = frame['positions']  # Shape: (n_atoms, 3)

# → C++ conversion
std::vector<float> x_coords, y_coords, z_coords;

# → GPU transfer
cudaMemcpy(d_coords_x, x_coords.data(), size, cudaMemcpyHostToDevice);
```

**Result retrieval:**

```cpp
// GPU → CPU
cudaMemcpy(h_histogram, d_histogram, size, cudaMemcpyDeviceToHost);

// CPU → File
write_output_file(h_histogram, n_bins);
```

## Build System

### CMake Configuration

**Top-level CMakeLists.txt:**

```cmake
cmake_minimum_required(VERSION 3.20)
project(pyCuSAXS LANGUAGES CXX CUDA)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Build shared library
add_library(pycusaxs_cuda MODULE
    cpp-src/pybind/cuda_bindings.cpp
    cpp-src/Exec/RunSaxs.cu
    cpp-src/Saxs/saxsKernel.cu
    # ... more sources
)

# Link libraries
target_link_libraries(pycusaxs_cuda PRIVATE
    CUDA::cudart
    CUDA::cufft
    pybind11::module
)
```

### Python Packaging

**setup.py:**

```python
from setuptools import setup
from setuptools.command.build_ext import build_ext

setup(
    name='pycusaxs',
    packages=['pycusaxs'],
    ext_modules=[CMakeExtension('pycusaxs_cuda')],
    cmdclass={'build_ext': CMakeBuild},
    # ...
)
```

**Installation Flow:**

1. `pip install .` invokes `setup.py`
2. `setup.py` calls CMake
3. CMake builds C++/CUDA code
4. Shared library installed as Python module
5. Python package installed

## Design Patterns

### RAII (Resource Acquisition Is Initialization)

**Used for:**

- GPU memory (Thrust vectors)
- cuFFT plans (custom destructor)
- File handles (std::ofstream)

**Example:**

```cpp
saxsKernel::~saxsKernel() {
    if (fft_plan != 0) {
        cufftDestroy(fft_plan);  // Automatic cleanup
    }
}
```

### Singleton Pattern

**Options class:**

```cpp
class Options {
public:
    static int nx, ny, nz;  // Global configuration
    // No instances needed
};
```

### Iterator Pattern

**Frame streaming:**

```python
def iter_frames_stream(self, start, stop, step=1):
    for ts in self.universe.trajectory[start:stop:step]:
        yield frame_data_dict
```

### Factory Pattern

**Cell matrix creation:**

```cpp
static Cell createCell(box_dimensions);  // Factory method
```

## Error Handling Strategy

### Layered Error Handling

1. **Python Layer:** Input validation, file checks
2. **Binding Layer:** Type conversion, exception translation
3. **C++ Layer:** Logic errors, CUDA errors
4. **CUDA Layer:** Device errors, memory failures

### Exception Hierarchy

```
Exception (Python base)
├── ValueError (invalid parameters)
├── FileNotFoundError (missing files)
├── RuntimeError (CUDA errors)
└── MemoryError (GPU OOM)
```

### Error Propagation

```
CUDA Error
  │
  v
C++ Exception (std::runtime_error)
  │
  v
pybind11 Translation
  │
  v
Python Exception (RuntimeError)
  │
  v
User-facing Error Message
```

## Threading Model

### Python GIL Management

**Pattern:**

```cpp
{
    py::gil_scoped_acquire acquire;
    // Python operations
    auto frame = load_python_frame();
}  // GIL released

// GPU operations (no GIL needed)
process_on_gpu();

{
    py::gil_scoped_acquire acquire;
    // Next Python operation
}
```

### CUDA Stream Model

**Single stream execution:**

- All kernels in same CUDA stream
- Automatic ordering within stream
- Minimal synchronization needed

**Future: Multi-stream:**

- Parallel frame processing
- Overlapped compute/transfer
- Requires refactoring

## See Also

- [Backend API](../api/backend.md) - C++/CUDA API reference
- [Python API](../api/python.md) - Python API reference
- [Pipeline Details](../algorithm/pipeline.md) - Implementation details
- [Contributing Guide](contributing.md) - Development workflow
