# Installation Guide for pyCuSAXS

## Prerequisites

pyCuSAXS requires:
1. **NVIDIA GPU** with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
2. **NVIDIA GPU Drivers** (version 450.80.02 or newer)
3. **CUDA Toolkit** (version 11.0 or newer)
4. **Python** 3.9 or newer

## Quick Check

Before installing, verify your system meets the requirements:

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check Python version
python --version
```

## Installation Methods

### Method 1: Conda/Mamba (Recommended)

This method automatically handles all dependencies including CUDA toolkit.

```bash
# Create a new conda environment
conda create -n pycusaxs python=3.11
conda activate pycusaxs

# Install CUDA toolkit and build dependencies
conda install -c conda-forge -c nvidia \
    cuda-toolkit \
    cmake \
    pybind11 \
    fmt \
    scikit-build-core

# Clone and install pyCuSAXS
git clone <repository-url>
cd pyCuSaxs

# Install with CuPy for CUDA 12.x
pip install -e ".[cuda12]"

# OR for CUDA 11.x
pip install -e ".[cuda11]"
```

### Method 2: Pip Installation (Manual CUDA Setup)

If you already have CUDA toolkit installed system-wide:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Ensure CUDA is in your PATH
export CUDA_HOME=/usr/local/cuda  # Adjust path as needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies (example for Ubuntu/Debian)
sudo apt-get install cmake libfmt-dev

# Clone and install
git clone <repository-url>
cd pyCuSaxs

# Install with appropriate CUDA version
pip install -e ".[cuda12]"  # For CUDA 12.x
# OR
pip install -e ".[cuda11]"  # For CUDA 11.x
```

### Method 3: Development Installation

For developers who want to modify the code:

```bash
# Clone repository
git clone <repository-url>
cd pyCuSaxs

# Create conda environment with dev dependencies
conda env create -f environment.yml  # If available
conda activate pycusaxs

# Install in editable mode
pip install -e ".[cuda12]"
```

## CuPy Installation

CuPy is now automatically installed as a dependency. The installation includes:

- `cupy-cuda11x` for CUDA 11.x systems
- `cupy-cuda12x` for CUDA 12.x systems (default)

To manually install CuPy for a specific CUDA version:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

## Troubleshooting

### CUDA Compiler Not Found

**Error:**
```
CUDA Compiler Not Found
```

**Solution:**
- Install CUDA toolkit: `conda install -c nvidia cuda-toolkit`
- Or download from: https://developer.nvidia.com/cuda-downloads
- Ensure `nvcc` is in your PATH: `which nvcc`

### NVIDIA GPU Driver Not Found

**Error:**
```
NVIDIA GPU Driver Not Found
nvidia-smi command not found
```

**Solution:**
- Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
- Verify installation: `nvidia-smi`

### No NVIDIA GPU Detected

**Error:**
```
No NVIDIA GPU Detected
```

**Solution:**
- Ensure your system has a compatible NVIDIA GPU
- Check if GPU is recognized: `lspci | grep -i nvidia`
- Verify drivers are loaded: `nvidia-smi`

### CuPy Not Installed

**Error:**
```
ModuleNotFoundError: No module named 'cupy'
```

**Solution:**
- Install with optional dependencies: `pip install -e ".[cuda12]"`
- Or manually: `pip install cupy-cuda12x`

### Build Fails with CMake Errors

**Solution:**
- Ensure all build dependencies are installed:
  ```bash
  conda install -c conda-forge cmake pybind11 fmt scikit-build-core
  ```
- Clear build cache and retry:
  ```bash
  rm -rf _skbuild build *.egg-info
  pip install -e ".[cuda12]" --no-build-isolation
  ```

### Wrong CUDA Version

**Error:**
```
CUDA version mismatch
```

**Solution:**
- Check your CUDA version: `nvcc --version`
- Install matching CuPy version:
  - CUDA 11.x: `pip install -e ".[cuda11]"`
  - CUDA 12.x: `pip install -e ".[cuda12]"`

## Verification

After installation, verify everything works:

```bash
# Test Python imports
python -c "import pycusaxs; print('pyCuSAXS OK')"
python -c "import cupy; print('CuPy version:', cupy.__version__)"
python -c "import pycusaxs_cuda; print('CUDA backend OK')"

# Check installed commands
pycusaxs --help
saxs-widget --help
saxs-db --help
saxs-subtract --help
```

## Environment Variables

For consistent CUDA setup, add to your `.bashrc` or `.zshrc`:

```bash
export CUDA_HOME=/usr/local/cuda  # Adjust to your CUDA installation
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## System Requirements Summary

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| GPU | CUDA Compute 6.0+ | RTX 3080+ |
| CUDA Toolkit | 11.0 | 12.x |
| GPU Drivers | 450.80.02 | Latest |
| Python | 3.9 | 3.11+ |
| CMake | 3.20 | 3.25+ |
| GPU Memory | 2 GB | 8 GB+ |

## Next Steps

After successful installation:
1. Read the [README.md](README.md) for usage examples
2. Check the [docs/](docs/) directory for detailed documentation
3. Run example calculations to verify GPU performance
