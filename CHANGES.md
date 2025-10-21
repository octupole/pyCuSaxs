# Installation Improvements - Change Summary

## Issues Addressed

1. **CuPy not automatically installed** - CuPy is required for GPU operations but was not listed as a dependency
2. **No CUDA availability checks** - Installation would fail with cryptic errors if CUDA was not properly configured

## Changes Made

### 1. Added CuPy to Dependencies

**Files Modified:**
- [pyproject.toml](pyproject.toml:22)
- [requirements.txt](requirements.txt:6)

**Changes:**
- Added `cupy-cuda12x>=12.0` as a core dependency for Linux systems
- Created optional dependency groups `[cuda11]` and `[cuda12]` for flexibility

**Installation:**
```bash
# For CUDA 12.x (default)
pip install -e ".[cuda12]"

# For CUDA 11.x
pip install -e ".[cuda11]"

# Basic install (uses platform detection)
pip install .
```

### 2. Enhanced CUDA Detection in CMake

**File Modified:** [CMakeLists.txt](CMakeLists.txt:1-99)

**Added Pre-Build Checks:**

1. **CUDA Compiler Check** (lines 3-27)
   - Checks if CUDA compiler (nvcc) is available before project initialization
   - Provides clear error message with installation instructions if missing

2. **GPU Driver Check** (lines 48-71)
   - Validates that nvidia-smi is available
   - Provides helpful error message for driver installation

3. **GPU Device Check** (lines 73-99)
   - Verifies at least one NVIDIA GPU is detected
   - Displays detected GPU information
   - Provides troubleshooting guidance if no GPU found

**Error Messages:**
All error messages now include:
- Clear description of what's missing
- Step-by-step resolution instructions
- Links to official documentation
- Formatted boxes for easy readability

### 3. Updated Build Configuration

**File Modified:** [pyproject.toml](pyproject.toml:35-43)

**Added:**
- `cmake.verbose = true` for better build diagnostics
- `CUSAXS_CHECK_CUDA = "ON"` CMake define for future extensibility
- Optional dependency groups for different CUDA versions

### 4. Created Comprehensive Installation Guide

**File Created:** [INSTALL.md](INSTALL.md)

**Contents:**
- Prerequisites checklist with version requirements
- Three installation methods (Conda, Pip, Development)
- CuPy installation instructions for different CUDA versions
- Comprehensive troubleshooting section covering:
  - CUDA compiler not found
  - GPU driver issues
  - No GPU detected
  - CuPy installation problems
  - CMake build failures
  - CUDA version mismatches
- Verification steps to test installation
- Environment variable configuration
- System requirements table

### 5. Updated README

**File Modified:** [README.md](README.md:36-92)

**Changes:**
- Added prominent link to INSTALL.md
- Updated installation examples to include CuPy
- Added notes about automatic dependency installation
- Emphasized CUDA version selection with `[cuda11]` or `[cuda12]`

## Migration Guide

### For New Users

Simply use the new installation command:

```bash
# Conda method (recommended)
conda create -n pycusaxs python=3.11
conda activate pycusaxs
conda install -c conda-forge -c nvidia cuda-toolkit cmake pybind11 fmt scikit-build-core
pip install -e ".[cuda12]"

# Or pip method
pip install -e ".[cuda12]"  # Automatically installs CuPy
```

### For Existing Users

If you already have pyCuSAXS installed but missing CuPy:

```bash
# Reinstall with optional dependencies
pip install -e ".[cuda12]"

# Or manually install CuPy
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x
```

### For Users Cloning to a New Machine

The new installation process will:

1. **Check for CUDA compiler** before attempting to build
2. **Check for GPU drivers** and available GPUs
3. **Automatically install CuPy** matching your CUDA version
4. **Provide clear error messages** if requirements are not met

Example output when CUDA is missing:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA Compiler Not Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pyCuSAXS requires NVIDIA CUDA to build.

  Please ensure:
    1. NVIDIA CUDA Toolkit is installed (version 11.0 or later)
    2. nvcc compiler is in your PATH
    3. CUDA_HOME or CUDA_PATH environment variable is set

  Installation instructions:
    - https://developer.nvidia.com/cuda-downloads
    - Or use conda: conda install -c nvidia cuda-toolkit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Testing

To verify the changes work correctly:

```bash
# 1. Test dependency installation
pip install -e ".[cuda12]"
python -c "import cupy; print('CuPy version:', cupy.__version__)"

# 2. Test CUDA backend
python -c "import pycusaxs_cuda; print('CUDA backend OK')"

# 3. Test all commands
pycusaxs --help
saxs-widget --help
saxs-db --help
saxs-subtract --help
```

## Benefits

1. **Automatic Dependency Resolution** - CuPy is now installed automatically
2. **Early Failure Detection** - CUDA issues caught during configuration, not compilation
3. **Better User Experience** - Clear, actionable error messages
4. **Flexible CUDA Support** - Easy selection between CUDA 11.x and 12.x
5. **Comprehensive Documentation** - INSTALL.md provides complete guidance
6. **Platform Detection** - CuPy only installed on Linux (where it's needed)

## Backward Compatibility

These changes are fully backward compatible:

- Existing installations continue to work
- Build process remains the same for systems with CUDA
- No changes to runtime behavior
- No changes to Python API
- No changes to command-line interface

## Future Improvements

Potential enhancements for the future:

1. Add Windows and macOS support with appropriate CuPy packages
2. Create conda package for one-command installation
3. Add GPU capability detection (compute capability check)
4. Provide pre-built wheels for common CUDA versions
5. Add runtime CUDA version detection and warning if mismatch
