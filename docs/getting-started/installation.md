# Installation

This guide covers the installation of pyCuSAXS on your system.

## Prerequisites

### Hardware Requirements

!!! info "GPU Requirements"
    - NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
    - At least 4GB GPU memory recommended

### Software Requirements

Before installing pyCuSAXS, ensure you have the following software installed:

| Software | Minimum Version | Purpose |
|----------|----------------|---------|
| **CUDA Toolkit** | 11.0+ | GPU acceleration and cuFFT |
| **CMake** | 3.20+ | Build configuration |
| **C++ Compiler** | GCC 9+, Clang 10+, or MSVC 2019+ | Compiling C++17 code |
| **Python** | 3.9+ | Python interface and scripting |
| **NVIDIA Driver** | Compatible with CUDA version | GPU device control |

!!! tip "Checking CUDA Version"
    Verify your CUDA installation:
    ```bash
    nvcc --version
    ```

## Installation Methods

### Method 1: Standard Installation (Recommended)

This is the recommended method for most users:

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build and install pyCuSAXS
pip install .
```

!!! success "Virtual Environment"
    Using a virtual environment isolates pyCuSAXS dependencies from your system Python, preventing conflicts.

### Method 2: Development Installation

For developers who want to modify the code:

```bash
# Install in editable mode for development
pip install -e .

# Or build manually with CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

!!! warning "Development Mode"
    Editable installation (`-e` flag) allows you to modify Python and C++ code without reinstalling. However, C++ changes require rebuilding the extension module.

## Verify Installation

After installation, verify that everything is working correctly:

### Check CUDA Backend

```bash
# Check if CUDA backend is available
python -c "import pycusaxs_cuda; print('CUDA backend loaded successfully')"
```

### Launch GUI

```bash
# Launch graphical interface
saxs-widget
```

If the GUI launches without errors, your installation is successful!

### Check Python Package

```bash
# Verify Python package
python -c "import pycusaxs; print('pyCuSAXS package loaded successfully')"

# Check GPU availability
nvidia-smi
```

## Platform-Specific Notes

=== "Linux"

    Most Linux distributions work out of the box. Ensure your NVIDIA driver is up to date:

    ```bash
    # Check driver version
    nvidia-smi

    # Update driver (Ubuntu/Debian)
    sudo apt update
    sudo apt install nvidia-driver-525  # or latest version
    ```

=== "Windows"

    On Windows, you may need to:

    1. Install Visual Studio 2019 or newer (for MSVC compiler)
    2. Ensure CUDA Toolkit is in your PATH
    3. Use `venv\Scripts\activate` instead of `source venv/bin/activate`

=== "macOS"

    !!! warning "macOS Support"
        pyCuSAXS requires NVIDIA CUDA, which is not supported on macOS with Apple Silicon. macOS support is limited to older Intel Macs with NVIDIA GPUs.

## Troubleshooting Installation

### CUDA Toolkit Not Found

If CMake cannot find your CUDA installation:

```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Compiler Errors

If you encounter C++ compiler errors:

1. Verify your compiler version is compatible with your CUDA version
2. For CUDA 11.x, use GCC 9-11 (GCC 12+ may have issues)
3. Check [NVIDIA's CUDA compatibility matrix](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

### Build Failures

```bash
# Clean build and try again
rm -rf build/
pip install --force-reinstall .
```

### Missing Dependencies

```bash
# Ensure all Python dependencies are installed
pip install -r requirements.txt

# Install MDAnalysis extras for all trajectory formats
pip install MDAnalysis[all]
```

## Next Steps

Once installation is complete:

- [Quick Start Guide](quickstart.md) - Run your first SAXS calculation
- [Configuration](configuration.md) - Learn about configuration options
- [Command Line Interface](../user-guide/cli.md) - Explore CLI options
- [Graphical User Interface](../user-guide/gui.md) - Use the GUI application
