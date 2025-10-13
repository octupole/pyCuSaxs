# Quick Reference: New pyCuSaxs Features

## New Module Structure

```
pycusaxs/
├── logger.py          # Logging framework
├── config.py          # Configuration management
├── core.py            # Business logic
├── cli.py             # Command-line interface
├── gui.py             # GUI implementation
├── progress.py        # Progress bars
├── main_new.py        # New main entry point
└── (existing files...)
```

## Logging

```python
from pycusaxs.logger import setup_logging, get_logger

# Setup (do once at startup)
setup_logging(verbose=True, log_file="app.log")

# Use in modules
logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

## Configuration

```python
from pycusaxs.config import SaxsConfig, parse_grid_values

# Parse grid
grid = parse_grid_values([128])  # → (128, 128, 128)
grid = parse_grid_values([64, 64, 128])  # → (64, 64, 128)

# Create validated config
config = SaxsConfig(
    topology="system.tpr",
    trajectory="traj.xtc",
    grid_size=(128, 128, 128),
    initial_frame=0,
    last_frame=100,
    dt=10,
    order=4
)

# Convert to old format
required = config.to_required_params()
advanced = config.to_advanced_params()
```

## Progress Bars

```python
from pycusaxs.progress import progress_bar, iter_with_progress

# Context manager
with progress_bar(100, desc="Processing", unit="item") as pbar:
    for i in range(100):
        # Do work
        pbar.update(1)

# Iterator wrapper
for item in iter_with_progress(items, desc="Processing"):
    # Process item
    pass

# With topology
for frame in topology.iter_frames_stream(0, 1000, 10, show_progress=True):
    # Process frame
    pass
```

## Core Functions

```python
from pycusaxs.core import run_saxs_calculation

results = run_saxs_calculation(
    required_params={
        "topology": "system.tpr",
        "trajectory": "traj.xtc",
        "grid_size": (128, 128, 128),
        "initial_frame": 0,
        "last_frame": 100
    },
    advanced_params={
        "dt": 10,
        "order": 4,
        "bin_size": 0.01,
        "qcut": 0.5,
        "out": "saxs.dat"
    }
)
```

## CLI Usage

```bash
# New CLI with progress
python -m pycusaxs.cli -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 --progress

# Verbose logging
python -m pycusaxs.cli -s system.tpr -x traj.xtc -g 128 --verbose

# System info
python -m pycusaxs.cli -s system.tpr -x traj.xtc --info
```

## GUI Usage

```python
from pycusaxs.gui import run_gui
import sys

sys.exit(run_gui())
```

Or from command line:
```bash
python -m pycusaxs.gui
```

## Documentation

```bash
# Install dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build HTML docs
cd docs && make html

# Build PDF docs
cd docs && make latexpdf

# View docs
open docs/_build/html/index.html
```

## Testing Imports

```python
# Test all new modules
from pycusaxs.logger import get_logger
from pycusaxs.config import SaxsConfig
from pycusaxs.core import run_saxs_calculation
from pycusaxs.cli import run_cli
from pycusaxs.gui import run_gui
from pycusaxs.progress import progress_bar

print("✓ All imports successful")
```

## Migration Checklist

- [ ] Install new dependencies: `pip install tqdm>=4.65`
- [ ] Test imports: `python -c "from pycusaxs import logger, config, core"`
- [ ] Replace print statements with logger calls
- [ ] Add progress bars to long-running loops
- [ ] Use SaxsConfig for parameter validation
- [ ] Build and review documentation
- [ ] Update pyproject.toml entry points (optional)
- [ ] Run full test suite

## Key Changes

| Feature | Before | After |
|---------|--------|-------|
| Logging | `print()` | `logger.info()` |
| Progress | None | `progress_bar()` |
| Config | Dict | `SaxsConfig` dataclass |
| Main file | 606 lines | 5 modules, ~200 lines each |
| Type hints | Python 3.10+ | Python 3.9+ compatible |
| Docs | README only | Full Sphinx docs |

## Backward Compatibility

✅ Old `main.py` still works
✅ Old parameter dictionaries supported
✅ CLI arguments unchanged
✅ GUI interface unchanged
✅ Python API compatible

## Getting Help

- Read `IMPLEMENTATION_SUMMARY.md` for details
- Check `INTEGRATION_GUIDE.md` for step-by-step instructions
- Look at module docstrings for examples
- Build documentation: `cd docs && make html`
