# pyCuSAXS - CUDA-Accelerated SAXS Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/octupole/pyCuSaxs/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/octupole/pyCuSaxs?style=social)](https://github.com/octupole/pyCuSaxs)

**pyCuSAXS** is a high-performance CUDA-accelerated pipeline for computing Small-Angle X-ray Scattering (SAXS) profiles from molecular dynamics trajectories. It combines a GPU-optimized C++/CUDA backend with Python-based trajectory processing, offering both command-line and graphical user interfaces.

## 🔬 What is Small-Angle X-ray Scattering (SAXS)?

Small-Angle X-ray Scattering is an experimental technique used to study the structure of materials at the nanometer scale. In molecular biology and biophysics, SAXS provides crucial information about:

- **Protein structure and dynamics** in solution
- **Conformational changes** during molecular dynamics simulations
- **Validation of MD simulations** against experimental data
- **Ensemble properties** of flexible biomolecules

pyCuSAXS bridges the gap between MD simulations and experimental SAXS by computing theoretical scattering profiles directly from trajectories, enabling:

- Direct comparison with experimental SAXS curves
- Validation of simulation force fields and parameters
- Analysis of conformational ensembles
- Time-resolved structural changes across MD trajectories

## 📦 What's in This Package?

pyCuSAXS provides a complete toolkit for SAXS analysis:

- **GPU-Accelerated Backend**: Optimized C++/CUDA code for high-performance calculations
- **Python Interface**: Easy-to-use API integrated with MDAnalysis for trajectory handling
- **Command-Line Tool**: Batch processing and scripting support
- **Graphical Interface**: Interactive PySide6-based GUI for exploratory analysis
- **Flexible Configuration**: Extensive parameters for customizing calculations
- **Production Ready**: Comprehensive validation, error handling, and memory safety

## 🚀 Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __GPU-Accelerated Computing__

    ---

    Leverages NVIDIA CUDA for high-performance SAXS calculations with 15-30% performance improvements in recent optimizations

-   :material-chart-line:{ .lg .middle } __MDAnalysis Integration__

    ---

    Native support for GROMACS and other MD trajectory formats with streaming processing

-   :material-console:{ .lg .middle } __Dual Interface__

    ---

    Command-line tool and beautiful PySide6-based GUI for flexible workflows

-   :material-shield-check:{ .lg .middle } __Production Ready__

    ---

    Comprehensive input validation, error handling, and exception translation

-   :material-memory:{ .lg .middle } __Memory Efficient__

    ---

    Streaming trajectory processing with double-buffered frame loading

-   :material-tune:{ .lg .middle } __Flexible Configuration__

    ---

    Extensive parameters for grid sizing, histogram binning, and solvent modeling

</div>

## 💻 Installation

### Prerequisites

Before installing pyCuSAXS, ensure you have:

- **NVIDIA GPU** with CUDA Compute Capability 5.0 or higher
- **CUDA Toolkit** 11.0 or later ([download here](https://developer.nvidia.com/cuda-downloads))
- **Python** 3.9 or later
- **C++ compiler** with C++17 support (GCC 7+, Clang 5+, or MSVC 2019+)
- **CMake** 3.18 or later

### Quick Install

```bash
# Clone the repository
git clone https://github.com/octupole/pyCuSaxs.git
cd pyCuSaxs

# Install Python dependencies
pip install -r requirements.txt

# Build and install pyCuSAXS
pip install .
```

For detailed installation instructions including troubleshooting, see the [Installation Guide](getting-started/installation.md).

## ⚡ Quick Start

### Command-Line Interface

Run a SAXS calculation on your MD trajectory:

```bash
python -m pycusaxs.main \
    --topology system.tpr \
    --trajectory trajectory.xtc \
    --grid 128,128,128 \
    --begin 0 --end 100 --dt 10 \
    --out saxs_profile.dat
```

### Graphical Interface

Launch the interactive GUI:

```bash
saxs-widget
```

### Python API

Use pyCuSAXS in your Python scripts:

```python
from pycusaxs.main import cuda_connect

result = cuda_connect(
    topology="system.tpr",
    trajectory="trajectory.xtc",
    grid=[128, 128, 128],
    begin=0,
    end=100,
    dt=10
)

# Access results
print(f"SAXS profile: {result['output_file']}")
```

For more examples, see the [Quick Start Guide](getting-started/quickstart.md).

## 📚 Documentation

<div class="grid cards" markdown>

-   [:material-download: **Installation Guide**](getting-started/installation.md)

    ---

    Step-by-step installation instructions for all platforms

-   [:material-rocket-launch: **Quick Start**](getting-started/quickstart.md)

    ---

    Get up and running in minutes with examples

-   [:material-console-line: **User Guide**](user-guide/cli.md)

    ---

    Complete CLI, GUI, and Python API documentation

-   [:material-cog: **API Reference**](api/backend.md)

    ---

    Detailed C++/CUDA and Python API documentation

</div>

## 📈 Performance

Recent optimizations (v0.1.0):

- **15-30% throughput improvement** for multi-frame trajectories
- Reduced GPU synchronization from 10 to 2 calls per frame
- Fixed critical memory leaks and resource management issues
- Enhanced numerical stability and bounds checking

Typical performance on NVIDIA RTX 3080:

| System Size | Grid | Frames | Time/Frame | Total Time |
|------------|------|--------|------------|------------|
| 50K atoms  | 64³  | 1000   | ~15 ms     | ~15 sec    |
| 50K atoms  | 128³ | 1000   | ~35 ms     | ~35 sec    |
| 200K atoms | 128³ | 1000   | ~85 ms     | ~85 sec    |

## 🧪 Algorithm

pyCuSAXS implements the SAXS intensity calculation:

$$
I(q) = |F(q)|^2 = \left|\sum_j f_j(q) \exp(iq \cdot r_j)\right|^2
$$

Where:

- $f_j(q)$: Atomic form factor
- $r_j$: Atomic position
- $q$: Scattering vector

The pipeline:

1. **Coordinate Transformation** - Convert triclinic box to orthonormal coordinates
2. **Density Grid Assignment** - Map atoms to 3D grid using B-spline interpolation
3. **Padding & Supersampling** - Add solvent and supersample grid
4. **Fourier Transform** - 3D FFT using cuFFT
5. **Scattering Intensity** - Apply form factors and compute $|F(q)|^2$
6. **Histogram Binning** - Bin by $|q|$ magnitude and average over frames

## 🤝 Contributing

We welcome contributions! See our [development guide](development/contributing.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/octupole/pyCuSaxs/blob/main/LICENSE) file for details.

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/octupole/pyCuSaxs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/octupole/pyCuSaxs/discussions)

## 📖 Citation

If you use pyCuSAXS in your research, please cite:

```bibtex
@software{pycusaxs2024,
  title = {pyCuSAXS: GPU-Accelerated SAXS Analysis for Molecular Dynamics},
  author = {pyCuSAXS Authors},
  year = {2024},
  url = {https://github.com/octupole/pyCuSaxs}
}
```

---

<div align="center">
  <sub>Built with ❤️ for the molecular dynamics community</sub>
</div>
