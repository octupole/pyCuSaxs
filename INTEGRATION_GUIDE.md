# Integration Guide: How to Use the New Modules

This guide explains how to integrate the newly created modules into your existing pyCuSaxs workflow.

## Quick Start

### Option 1: Keep Both Versions (Recommended for Testing)

The safest approach is to keep the old `main.py` and run the new modules alongside:

```bash
# Old way still works
python -m pycusaxs.main -s system.tpr -x traj.xtc -g 64 -b 0 -e 100

# New way (using new modules)
python -m pycusaxs.main_new -s system.tpr -x traj.xtc -g 64 -b 0 -e 100

# Or directly invoke new CLI
python -m pycusaxs.cli -s system.tpr -x traj.xtc -g 64 -b 0 -e 100
```

### Option 2: Switch to New Main (Recommended after Testing)

Once you've tested the new modules, update `main.py` to use them:

```python
# In pycusaxs/main.py - replace entire content with:
from .main_new import main

__all__ = ['main']

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

Then reinstall:
```bash
pip install -e .
```

## Using Individual New Features

### 1. Using the Logger

Add to any existing Python file:

```python
# At the top of the file
from .logger import get_logger

logger = get_logger(__name__)

# Replace print statements
# Old: print("Starting calculation...")
logger.info("Starting calculation...")

# Old: print(f"Error: {error}")
logger.error(f"Error: {error}")

# Old: if verbose: print(f"Debug info: {data}")
logger.debug(f"Debug info: {data}")
```

### 2. Using Progress Bars

Add to any loop in your code:

```python
from .progress import progress_bar

# Old way
for i in range(1000):
    # Do work
    pass

# New way
with progress_bar(1000, desc="Processing", unit="item") as pbar:
    for i in range(1000):
        # Do work
        pbar.update(1)
```

Or for iterables:

```python
from .progress import iter_with_progress

# Old way
for frame in topology.iter_frames_stream(0, 1000, 10):
    # Process frame
    pass

# New way
for frame in iter_with_progress(
    topology.iter_frames_stream(0, 1000, 10),
    total=100,
    desc="Processing frames",
    unit="frame"
):
    # Process frame
    pass
```

### 3. Using the Configuration System

In your Python scripts:

```python
from pycusaxs.config import SaxsConfig

# Create configuration with validation
config = SaxsConfig(
    topology="system.tpr",
    trajectory="trajectory.xtc",
    grid_size=(128, 128, 128),
    initial_frame=0,
    last_frame=100,
    dt=10,
    order=4,
    bin_size=0.01,
    qcut=0.5
)

# Automatic validation happens in __post_init__
# Raises errors if files don't exist or parameters are invalid

# Use with core module
from pycusaxs.core import run_saxs_calculation

results = run_saxs_calculation(
    config.to_required_params(),
    config.to_advanced_params()
)
```

### 4. Using the Core Module Directly

If you want to programmatically run SAXS calculations:

```python
from pycusaxs.core import run_saxs_calculation
from pycusaxs.logger import setup_logging

# Setup logging first
setup_logging(verbose=True)

# Define parameters
required = {
    "topology": "system.tpr",
    "trajectory": "trajectory.xtc",
    "grid_size": (128, 128, 128),
    "initial_frame": 0,
    "last_frame": 100
}

advanced = {
    "dt": 10,
    "order": 4,
    "bin_size": 0.01,
    "qcut": 0.5,
    "out": "saxs.dat"
}

# Run calculation
try:
    results = run_saxs_calculation(required, advanced)
    for line in results:
        print(line)
except Exception as e:
    print(f"Error: {e}")
```

## Building Documentation

### Install Documentation Dependencies

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build HTML Documentation

```bash
cd docs
make html
```

View the documentation by opening `docs/_build/html/index.html` in your browser.

### Build PDF Documentation

```bash
cd docs
make latexpdf
```

The PDF will be in `docs/_build/latex/pycusaxs.pdf`.

## Testing the Changes

### 1. Test Imports

```bash
python << EOF
# Test all new modules import correctly
from pycusaxs.logger import setup_logging, get_logger
from pycusaxs.config import SaxsConfig, parse_grid_values
from pycusaxs.core import run_saxs_calculation
from pycusaxs.cli import run_cli, build_cli_parser
from pycusaxs.gui import run_gui
from pycusaxs.progress import progress_bar, iter_with_progress
print("✓ All imports successful")
EOF
```

### 2. Test Configuration System

```bash
python << EOF
from pycusaxs.config import SaxsConfig, parse_grid_values

# Test grid parsing
assert parse_grid_values([128]) == (128, 128, 128)
assert parse_grid_values([64, 64, 128]) == (64, 64, 128)
print("✓ Grid parsing works")

# Test configuration (will fail due to missing files, which is expected)
try:
    config = SaxsConfig(
        topology="test.tpr",
        trajectory="test.xtc",
        grid_size=(128, 128, 128),
        initial_frame=0,
        last_frame=100
    )
except FileNotFoundError:
    print("✓ Configuration validation works (correctly rejects missing files)")
EOF
```

### 3. Test Logger

```bash
python << EOF
from pycusaxs.logger import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger('test')

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
print("✓ Logger works")
EOF
```

### 4. Test Progress Bars

```bash
python << EOF
from pycusaxs.progress import progress_bar
import time

with progress_bar(50, desc="Testing", unit="item") as pbar:
    for i in range(50):
        time.sleep(0.02)
        pbar.update(1)
print("✓ Progress bars work")
EOF
```

## Updating Entry Points

If you want to permanently switch to the new structure, update `pyproject.toml`:

```toml
[project.scripts]
pycusaxs = "pycusaxs.main_new:main"
saxs-widget = "pycusaxs.gui:run_gui"
saxs-db = "pycusaxs.saxs_db_tool:main"
saxs-subtract = "pycusaxs.saxs_subtract:main"
```

Then reinstall:
```bash
pip install -e .
```

## Adding New Features Using the Modular Structure

### Example: Adding a New CLI Command

1. **Add to `cli.py`:**

```python
def build_cli_parser():
    parser = argparse.ArgumentParser(...)

    # Add new argument
    parser.add_argument(
        "--my-new-feature",
        action="store_true",
        help="Enable my new feature"
    )

    return parser

def run_cli(args):
    parser = build_cli_parser()
    namespace = parser.parse_args(args)

    # Handle new feature
    if namespace.my_new_feature:
        logger.info("Running new feature")
        # Call new function
        my_new_feature_function()
```

2. **Add implementation to `core.py`:**

```python
def my_new_feature_function():
    """Implementation of new feature."""
    logger.info("New feature executing...")
    # Your code here
```

3. **Test it:**

```bash
python -m pycusaxs.cli --my-new-feature
```

## Troubleshooting

### Import Errors

If you get import errors like `ModuleNotFoundError: No module named 'pycusaxs.logger'`:

```bash
# Reinstall in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/home/marchi/git/pyCuSaxs:$PYTHONPATH
```

### Type Hint Errors in Python 3.9

If you still see type hint errors:

1. Check that `from __future__ import annotations` is at the top of the file
2. Use `Tuple`, `List`, `Optional` from `typing` module, not built-in types
3. Replace `type | None` with `Optional[type]`

### Documentation Build Errors

If Sphinx build fails:

```bash
# Install all documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Clean build
cd docs
make clean
make html
```

### Progress Bar Not Showing

If progress bars don't appear:

```bash
# Install tqdm
pip install tqdm>=4.65

# Verify it's installed
python -c "import tqdm; print(tqdm.__version__)"
```

## Migrating Existing Scripts

If you have existing Python scripts using pyCuSaxs, here's how to update them:

### Before:
```python
from pycusaxs.main import cuda_connect
from pycusaxs.topology import Topology

topo = Topology("system.tpr", "traj.xtc")

required = {...}
advanced = {...}

results = cuda_connect(required, advanced)
```

### After:
```python
from pycusaxs.core import run_saxs_calculation
from pycusaxs.topology import Topology
from pycusaxs.logger import setup_logging

# Setup logging
setup_logging(verbose=False)

topo = Topology("system.tpr", "traj.xtc")

required = {...}
advanced = {...}

# Same interface
results = run_saxs_calculation(required, advanced)
```

## Need Help?

- Check `IMPLEMENTATION_SUMMARY.md` for detailed information about each change
- Look at the docstrings in each new module for usage examples
- Open an issue on GitHub if you encounter problems

---

**Last Updated**: 2025-01-13
**Compatible with pyCuSaxs**: 0.1.0+
