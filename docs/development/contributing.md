# Contributing Guide

Thank you for your interest in contributing to pyCuSAXS! This guide will help you get started with development.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Git
- CUDA Toolkit 11.0+
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- Python 3.9+
- NVIDIA GPU for testing

### Development Environment Setup

1. **Fork and Clone**

   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/pyCuSaxs.git
   cd pyCuSaxs
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install in Development Mode**

   ```bash
   pip install -e .
   ```

   This allows you to modify Python code without reinstalling.

5. **Verify Installation**

   ```bash
   python -c "import pycusaxs_cuda; print('OK')"
   saxs-widget  # Test GUI
   ```

## Development Workflow

### Branch Strategy

We use a simplified Git Flow:

- `main` - Stable release branch
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write Code**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation

2. **Test Locally**

   ```bash
   # Run tests
   pytest tests/

   # Test specific module
   pytest tests/test_topology.py

   # Check code style
   flake8 pycusaxs/
   black --check pycusaxs/
   ```

3. **Build and Test C++ Code**

   ```bash
   # Clean rebuild
   rm -rf build/
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build

   # Run with debug symbols
   gdb python
   > run -m pycusaxs.main ...
   ```

4. **Commit Changes**

   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

   **Commit Message Format:**

   ```
   <type>: <subject>

   <body>

   <footer>
   ```

   **Types:**
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `style:` Formatting
   - `refactor:` Code restructuring
   - `perf:` Performance improvement
   - `test:` Tests
   - `build:` Build system

   **Example:**

   ```
   feat: Add multi-GPU support for frame processing

   Implements parallel processing across multiple GPUs using CUDA streams.
   Each GPU processes a subset of frames independently.

   Closes #123
   ```

5. **Push and Create Pull Request**

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a Pull Request on GitHub targeting the `develop` branch.

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Quotes:** Prefer double quotes
- **Imports:** Grouped and sorted (use `isort`)

**Formatting:**

```bash
# Auto-format code
black pycusaxs/

# Sort imports
isort pycusaxs/

# Check style
flake8 pycusaxs/
```

**Example:**

```python
"""Module docstring.

Detailed description of module purpose.
"""

import os
import sys
from typing import List, Dict, Optional

import numpy as np
from PySide6.QtWidgets import QWidget


class MyClass:
    """Class docstring.

    Detailed description of class.
    """

    def __init__(self, param: int) -> None:
        """Initialize class.

        Args:
            param: Parameter description
        """
        self.param = param

    def my_method(self, arg: str) -> List[int]:
        """Method description.

        Args:
            arg: Argument description

        Returns:
            Return value description

        Raises:
            ValueError: When argument is invalid
        """
        if not arg:
            raise ValueError("Argument cannot be empty")
        return [1, 2, 3]
```

### C++ Style

We follow a modified [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):

- **Indentation:** 4 spaces (not tabs)
- **Braces:** Opening brace on same line
- **Naming:**
  - Classes: `PascalCase`
  - Functions: `camelCase`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Member variables: prefix with `m_` or suffix with `_`

**Example:**

```cpp
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace cusaxs {

/**
 * @brief Brief class description
 *
 * Detailed class description.
 */
class MyClass {
public:
    /**
     * @brief Constructor
     * @param size Grid size
     */
    explicit MyClass(int size);

    /**
     * @brief Compute SAXS profile
     * @param coords Atomic coordinates
     * @return SAXS intensity array
     */
    std::vector<float> computeSaxs(const std::vector<float>& coords);

private:
    int grid_size_;
    std::vector<float> data_;
};

}  // namespace cusaxs
```

### CUDA Style

- **Kernel names:** `camelCaseKernel` suffix
- **Device functions:** `__device__` prefix clearly marked
- **Grid/block sizing:** Use `dim3` for clarity

**Example:**

```cuda
/**
 * @brief Density assignment kernel
 * @param coords Atomic coordinates [n_atoms × 3]
 * @param density Output density grid [nx × ny × nz]
 * @param n_atoms Number of atoms
 */
__global__ void densityAssignmentKernel(
    const float* coords,
    float* density,
    int n_atoms,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_atoms) {
        // Kernel logic
        float3 pos = make_float3(
            coords[idx * 3 + 0],
            coords[idx * 3 + 1],
            coords[idx * 3 + 2]
        );

        // ... computation ...
    }
}
```

## Testing

### Python Tests

We use `pytest` for Python testing.

**Running Tests:**

```bash
# All tests
pytest

# Specific test file
pytest tests/test_topology.py

# Specific test function
pytest tests/test_topology.py::test_load_topology

# With coverage
pytest --cov=pycusaxs --cov-report=html
```

**Writing Tests:**

```python
# tests/test_topology.py

import pytest
from pycusaxs.topology import Topology


def test_load_topology():
    """Test topology loading."""
    topo = Topology("test_data/system.tpr", "test_data/trajectory.xtc")
    assert topo.n_atoms > 0
    assert topo.n_frames > 0


def test_invalid_file():
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        Topology("nonexistent.tpr", "nonexistent.xtc")


@pytest.fixture
def sample_topology():
    """Fixture for reusable topology."""
    return Topology("test_data/system.tpr", "test_data/trajectory.xtc")


def test_molecule_count(sample_topology):
    """Test molecule counting."""
    total, protein, water, ions, other = sample_topology.count_molecules()
    assert total > 0
    assert protein >= 0
```

### C++ Tests

For C++ code, consider using Google Test or Catch2.

**Example with Google Test:**

```cpp
#include <gtest/gtest.h>
#include "Cell.h"

TEST(CellTest, OrthogonalBox) {
    std::vector<float> box = {10.0, 20.0, 30.0};
    Cell cell = Cell::createCell(box);

    auto CO = cell.getCO();
    EXPECT_FLOAT_EQ(CO[0][0], 10.0);
    EXPECT_FLOAT_EQ(CO[1][1], 20.0);
    EXPECT_FLOAT_EQ(CO[2][2], 30.0);
}

TEST(CellTest, TriclinicBox) {
    std::vector<std::vector<float>> box = {
        {10.0, 0.0, 0.0},
        {5.0, 15.0, 0.0},
        {2.0, 3.0, 20.0}
    };
    Cell cell = Cell::createCell(box);

    auto CO = cell.getCO();
    EXPECT_FLOAT_EQ(CO[0][0], 10.0);
    EXPECT_FLOAT_EQ(CO[1][0], 5.0);
}
```

### Integration Tests

Test the full pipeline:

```python
def test_full_saxs_calculation():
    """Test complete SAXS workflow."""
    from pycusaxs.main import cuda_connect

    required = {
        "topology": "test_data/system.tpr",
        "trajectory": "test_data/trajectory.xtc",
        "grid_size": [64, 64, 64],
        "initial_frame": 0,
        "last_frame": 10
    }

    advanced = {
        "dt": 1,
        "out": "test_output/saxs.dat"
    }

    results = list(cuda_connect(required, advanced))
    assert len(results) > 0

    # Check output file
    import os
    assert os.path.exists("test_output/saxs.dat")

    # Validate output format
    import numpy as np
    data = np.loadtxt("test_output/saxs.dat")
    assert data.shape[1] == 2  # q and I(q)
    assert np.all(data[:, 0] > 0)  # q > 0
```

## Documentation

### Docstring Format

**Python (Google Style):**

```python
def my_function(param1: int, param2: str) -> List[int]:
    """Brief description.

    Detailed description with more information about what
    the function does and how it works.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = my_function(10, "test")
        >>> print(result)
        [1, 2, 3, 4, 5]
    """
    pass
```

**C++ (Doxygen):**

```cpp
/**
 * @brief Brief description
 *
 * Detailed description.
 *
 * @param param1 Description of param1
 * @param param2 Description of param2
 * @return Description of return value
 * @throws std::invalid_argument When param1 is negative
 *
 * @code
 * MyClass obj(10);
 * auto result = obj.myMethod(param1, param2);
 * @endcode
 */
ReturnType myMethod(int param1, const std::string& param2);
```

### Updating Documentation

When adding features:

1. **Update relevant .md files** in `docs/`
2. **Add examples** to user guide
3. **Update API reference** if public API changed
4. **Add to CHANGELOG.md**

**Building Docs Locally:**

```bash
# Install MkDocs
pip install mkdocs-material

# Serve documentation locally
mkdocs serve

# Open http://127.0.0.1:8000 in browser
```

## Adding New Features

### 1. New Python Feature

**Steps:**

1. Create feature branch
2. Add code to appropriate module
3. Add tests
4. Update docstrings
5. Update user guide
6. Create PR

**Example: Add new analysis method**

```python
# pycusaxs/topology.py

def calculate_radius_of_gyration(self) -> float:
    """Calculate radius of gyration.

    Returns:
        Radius of gyration in Angstroms

    Raises:
        RuntimeError: If no frame is loaded
    """
    if self.ts is None:
        raise RuntimeError("No frame loaded. Call read_frame() first.")

    coords = self.get_coordinates()
    center = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
    return rg
```

### 2. New C++ Feature

**Steps:**

1. Declare in header file
2. Implement in .cpp/.cu file
3. Add to pybind11 bindings if needed
4. Add tests
5. Update documentation

**Example: Add new kernel**

```cpp
// cpp-src/Saxs/saxsDeviceKernels.cuh
__global__ void myNewKernel(float* data, int size);

// cpp-src/Saxs/saxsDeviceKernels.cu
__global__ void myNewKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Kernel implementation
        data[idx] = /* computation */;
    }
}

// cpp-src/Saxs/saxsKernel.cu
void saxsKernel::runMyNewKernel() {
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    myNewKernel<<<gridSize, blockSize>>>(d_data, size);
}
```

### 3. New Configuration Parameter

**Steps:**

1. Add to `CudaSaxsConfig` struct
2. Update pybind11 wrapper
3. Add to CLI parser
4. Add to GUI widget
5. Update documentation

**Example:**

```cpp
// cpp-src/include/CudaSaxsInterface.h
struct CudaSaxsConfig {
    // ... existing fields ...
    float new_parameter = 1.0;
};

// cpp-src/pybind/cuda_bindings.cpp
m.def("run", [](
    // ... existing params ...
    float new_parameter = 1.0
) {
    config.new_parameter = new_parameter;
    // ...
}, py::arg("new_parameter") = 1.0);

// pycusaxs/main.py
parser.add_argument("--new-param", type=float, default=1.0,
                   help="Description of new parameter")
```

## Performance Optimization

When optimizing:

1. **Profile first** - Use `nvprof` or Nsight to identify bottlenecks
2. **Benchmark** - Measure before and after
3. **Document** - Explain optimization in comments
4. **Test** - Ensure results are still correct

**Example profiling:**

```bash
nsys profile -o profile python -m pycusaxs.main -s system.tpr -x traj.xtc -g 128 -b 0 -e 10
nsys-ui profile.qdrep
```

## Debugging

### Python Debugging

```bash
# IPython debugger
import ipdb; ipdb.set_trace()

# PDB
import pdb; pdb.set_trace()

# VS Code debugger
# Set breakpoint in editor, F5 to debug
```

### C++ Debugging

```bash
# GDB
gdb --args python -m pycusaxs.main -s system.tpr -x traj.xtc -g 64 -b 0 -e 1
> break saxsKernel.cu:123
> run
> backtrace
> print variable_name
```

### CUDA Debugging

```bash
# cuda-gdb
cuda-gdb --args python -m pycusaxs.main -s system.tpr -x traj.xtc -g 64 -b 0 -e 1

# cuda-memcheck
cuda-memcheck python -m pycusaxs.main -s system.tpr -x traj.xtc -g 64 -b 0 -e 1

# Compute sanitizer (newer)
compute-sanitizer python -m pycusaxs.main ...
```

## Pull Request Process

1. **Ensure CI passes** - All tests must pass
2. **Update CHANGELOG.md** - Document your changes
3. **Request review** - At least one maintainer review required
4. **Address feedback** - Respond to all comments
5. **Squash commits** - Clean up commit history if requested
6. **Merge** - Maintainer will merge when approved

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] Commit messages are descriptive

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn

## Getting Help

- **Issues:** GitHub Issues for bugs and feature requests
- **Discussions:** GitHub Discussions for questions
- **Email:** [Maintainer email]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## See Also

- [Architecture](architecture.md) - System design
- [Backend API](../api/backend.md) - C++/CUDA API
- [Python API](../api/python.md) - Python API
- [Changelog](changelog.md) - Version history
