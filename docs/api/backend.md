# Backend APIs

The C++ / CUDA backend implements the high-performance SAXS pipeline. This reference documents the primary types and entry points intended for reuse or extension.

**Recent Updates (v0.1.0):**
- Fixed critical memory leaks and resource management issues
- Added comprehensive input validation and error handling
- Improved exception translation to Python
- Optimized GPU synchronization (15-30% performance improvement)
- Enhanced numerical stability and bounds checking

## Modules at a Glance

| Component | Source | Responsibility |
|-----------|--------|----------------|
| Exec | `cpp-src/Exec` | Host-side orchestration, configuration, Python interop |
| Saxs | `cpp-src/Saxs` | CUDA kernels, FFT handling, scattering calculations |
| System | `cpp-src/System` | Cell geometry transforms, solvent density helpers |
| Utilities | `cpp-src/Utilities` | B-spline modulation and scattering utilities |
| Pybind | `cpp-src/pybind` | Python bindings for the backend entry point |

## Support Types

### `padding` enum
- **File**: `cpp-src/Exec/Options.h`
- **Values**: `padding::avg`, `padding::given`
- **Purpose**: Selects how solvent density padding is determined. `avg` computes densities from trajectory data; `given` uses explicit water model weights.

### `Options`
- **File**: `cpp-src/Exec/Options.h`, implementation in `cpp-src/Exec/Options.cpp`
- **Usage**: Static configuration store shared across kernels.
- **Key fields**: `nx, ny, nz`, `nnx, nny, nnz`, `sigma`, `Dq`, `Qcut`, `myPadding`, solvent counts, file paths, frame range.
- **Notes**: Mutated by `run_cuda_saxs`; accessed by `RunSaxs` and `saxsKernel`. Defaults are set in `Options.cpp`.

### `FrameData`
- **File**: `cpp-src/Exec/RunSaxs.h`
- **Fields**: `frame_num`, `time`, `coords` (`std::vector<std::vector<float>>`), `box` (`std::vector<std::vector<float>>`)
- **Purpose**: Host-side staging container for frame metadata and coordinates received from Python.

### `CudaSaxsConfig`
- **File**: `cpp-src/include/CudaSaxsInterface.h`
- **Fields**: Paths, grid sizes, frame bounds, output path, spline order, scale factor, histogram parameters, solvent model parameters.
- **Usage**: Populated by Python, consumed by `run_cuda_saxs`.

### `CudaSaxsResult`
- **File**: `cpp-src/include/CudaSaxsInterface.h`
- **Fields**: Final grid sizes (`nx`, `ny`, `nz`, `nnx`, `nny`, `nnz`) and formatted configuration `summary` string.
- **Usage**: Returned to Python for display/logging.

## Classes

### `RunSaxs`
- **File**: `cpp-src/Exec/RunSaxs.cu`
- **Constructor**: `RunSaxs(std::string tpr_file, std::string xtc_file)`
- **Key Methods**:
  - `void Run(py::object Topol, int beg, int end, int dt)`
    - Acquires the GIL, pulls atom indices from the Python topology object, initialises `saxsKernel`, and iterates frames via `iter_frames_stream`.
    - Implements double-buffered frame loading; releases the GIL while GPU kernels run.
    - Writes `Options::outFile` with histogram data and prints performance stats.
  - `static std::vector<int> createVector(int start, int end, int step)`
    - Utility to generate inclusive frame index sequences.
  - `bool loadFrameData(py::handle frame_handle, FrameData &data)`
    - Parses a Python dict containing frame information, populating `FrameData`.
- **Threading Notes**: Uses `py::gil_scoped_acquire` and `py::gil_scoped_release` to ensure Python data access is serialized.

### `saxsKernel`
- **File**: `cpp-src/Saxs/saxsKernel.h` / `.cu`
- **Constructor**: `saxsKernel(int nx, int ny, int nz, int order)` - Initializes cuFFT plan to 0
- **Destructor**: `~saxsKernel()` - **New in v0.1.0**: Properly destroys cuFFT plan to prevent resource leak
- **Responsibilities**:
  - Manages CUDA device memory for density grids, FFT buffers, and histograms.
  - Runs the primary pipeline via `runPKernel`:
    1. Calculates orientation matrices (`Cell::calculateMatrices`).
    2. Assigns particle densities (`rhoCartKernel`).
    3. Applies padding and supersampling (`paddingKernel`, `superDensityKernel`).
    4. Executes CUFFT transforms and scattering factor accumulation.
    5. Updates histograms (`calculate_histogram`).
  - Provides helpers: `createMemory`, `scaledCell`, `resetHistogramParameters`, `writeBanner`, `getSaxs`, `getHistogram`, `zeroIq`.
- **Performance Metrics**: `getCudaTime()` returns average kernel runtime (ms per frame).
- **Static State**: `frame_count` tracks cumulative frames for averaging when running in `nvt` ensemble mode.
- **Memory Management Notes**:
  - **v0.1.0 Fix**: `createMemory()` now properly deletes temporary BSpline::BSpmod object after extracting module vectors
  - **v0.1.0 Fix**: `generateMultiples()` uses bitwise operations instead of `std::pow()` to prevent integer overflow

### `AtomCounter`
- **File**: `cpp-src/System/AtomCounter.h` / `.cpp`
- **Constructor**: `AtomCounter(float lx, float ly, float lz, int sodium, int chlorine, const std::string &model, int gx, int gy, int gz)`
- **Responsibilities**:
  - Computes effective solvent atom densities for padding when explicit models are used.
  - `calculateWaterMolecules()` converts cell volume to water molecule counts using predefined densities.
  - `calculateAtomCounts()` returns average counts per grid cell for O, H, Na, Cl.

### `Cell`
- **File**: `cpp-src/System/Cell.h` / `.cpp`
- **Responsibilities**:
  - Generates coordinate (`co`) and reciprocal (`oc`) transformation matrices from box dimensions.
  - Overloads support both scalar cell parameters and 3×3 triclinic matrices.
  - `getCO()` and `getOC()` expose cached matrices used by the CUDA kernels.

### `BSpline::BSpmod`
- **File**: `cpp-src/Utilities/Splines.cu` (implementation) and associated headers in `cpp-src/Utilities`
- **Purpose**: Generates modulation vectors applied during FFT modulus calculation (`modulusKernel`).
- **Usage**: Instantiated in `saxsKernel::createMemory()` to populate `d_moduleX/Y/Z`.

## Free Functions

### `CudaSaxsResult run_cuda_saxs(py::object Topol, const CudaSaxsConfig &config)`
- **File**: `cpp-src/cudaSAXS.cu`
- **Responsibility**: Validates configuration, populates `Options`, logs a formatted configuration banner, executes `RunSaxs`, and returns final grid sizes plus the banner string.
- **Validation**: Throws `std::invalid_argument` on empty paths, invalid frame ranges, or non-positive strides. Automatically toggles padding mode based on `config.water_model`.

### CUDA Device Kernels (selected)
- **Location**: `cpp-src/Saxs/saxsDeviceKernels.cu`
- **Highlights**:
  - `rhoCartKernel`: Maps particle coordinates to grid densities using orientation matrices.
  - `superDensityKernel`: Transfers densities into the supersampled grid with optional solvent padding adjustments.
  - `scatterKernel`: Applies scattering factors and accumulates contributions in reciprocal space.
  - `calculate_histogram`: Bins modulus values (supports optional frame-count normalization).
    - **v0.1.0 Fix**: Added division-by-zero check for `bin_size`
    - **v0.1.0 Fix**: Added bounds validation (`h0 >= 0 && h0 < num_bins`) to prevent array overflow
    - **Note**: Dual `atomicAdd` operations are not atomic together; acceptable for histogram use case
  - `modulusKernel`, `zeroDensityKernel`, `gridAddKernel`: Utility kernels for modulus calculation and accumulation.
  - `scatterCalculation`: **Disabled in v0.1.0** - Contains uninitialized memory bug; use `scatterKernel` instead
- These are launched from `saxsKernel::runPKernel` with grid/block dimensions derived from `Options`.
- **Performance Optimization v0.1.0**: Reduced GPU synchronization calls from 10 to 2 per frame (15-30% throughput improvement)

## Pybind Interface

### `PYBIND11_MODULE(pycusaxs_cuda, m)`
- **File**: `cpp-src/pybind/cuda_bindings.cpp`
- **Bindings**: Exposes a single `run` function that maps Python keyword arguments to `CudaSaxsConfig` and returns a dict containing the configuration summary and grid dimensions.
- **Keyword Arguments**: `obj_topology`, `topology`, `trajectory`, `grid`, `scaled_grid`, `begin`, `end`, `stride`, `output`, `order`, `scale_factor`, `bin_size`, `qcut`, `water_model`, `sodium`, `chlorine`, `simulation`.
- **Exception Handling (v0.1.0)**:
  - **Input Validation**: Validates all parameters before execution
    - Grid dimensions must be exactly 3 elements and positive
    - Stride must be positive
    - Spline order must be 1-8
    - Bin size and qcut must be non-negative
    - Scale factor must be positive
    - Frame indices must be non-negative with begin < end
  - **Exception Translation**: Properly translates C++ exceptions to Python
    - `std::invalid_argument` → `ValueError`
    - `std::runtime_error` → `RuntimeError` (via `PyErr_SetString`)
    - Generic exceptions wrapped with "CuSAXS error:" prefix
  - **File I/O Safety**: Checks file operations for success before writing
    - Validates file opens with `is_open()`
    - Checks write operations with `good()`
    - Validates close operations with `fail()`

## Performance Considerations

### Memory Layout
- **Particle Coordinates**: Consider using Structure of Arrays (SoA) layout instead of Array of Structures (AoS) for better coalesced memory access
- **Grid Data**: Contiguous memory allocation using Thrust device vectors
- **Streaming**: Frame data is double-buffered for CPU-GPU overlap

### Kernel Optimization (v0.1.0)
- **Synchronization**: Reduced from 10 to 2 `cudaDeviceSynchronize()` calls per frame
  - Only sync before D→H transfers or when CPU needs results
  - Kernels in same stream execute in order automatically
- **Atomic Operations**: Histogram updates use `atomicAdd` for thread-safe accumulation
- **Grid Sizing**: Use powers of 2, 3, 5, 7 for optimal FFT performance

### Best Practices
- **Error Checking**: Always check CUDA errors after API calls (critical for debugging)
- **Resource Management**: Use RAII patterns (Thrust vectors, smart pointers)
- **Input Validation**: Validate parameters before GPU allocation to fail fast
- **Memory Transfers**: Minimize H→D and D→H transfers; keep data on GPU when possible

## Third-Party Dependencies

- **nlohmann/json** (`json.hpp`): JSON parsing (header-only library)
- **Thrust**: CUDA high-level data structures (part of CUDA Toolkit)
- **cuFFT**: CUDA FFT library (part of CUDA Toolkit)
- **fmt**: Modern C++ formatting library
- **pybind11**: Python-C++ bindings

## Extending the Backend

- Add new configuration fields to `CudaSaxsConfig`, update `run_cuda_saxs`, and expose them through the pybind wrapper.
- When introducing new CUDA kernels, place declarations in `saxsDeviceKernels.cuh` and definitions in `saxsDeviceKernels.cu`, then launch them from `saxsKernel::runPKernel` or helper methods.
- Maintain thread-safety around Python interaction; any direct Python calls from C++ must hold the GIL.

### Adding New Kernels

```cuda
// 1. Declare in saxsDeviceKernels.cuh
__global__ void myCustomKernel(float* data, int size);

// 2. Define in saxsDeviceKernels.cu
__global__ void myCustomKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Kernel logic
        data[idx] = /* computation */;
    }
}

// 3. Launch from saxsKernel.cu
dim3 blockDim(256);
dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
myCustomKernel<<<gridDim, blockDim>>>(d_data_ptr, size);
// Minimize synchronization - only sync before D→H transfer
```

Refer to the source files for implementation details such as error handling, logging, and macro usage (`Ftypedefs.h`).
