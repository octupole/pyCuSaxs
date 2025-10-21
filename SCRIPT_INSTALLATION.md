# Script Installation Summary

## Question
*"Could you create a script that calls pycusaxs.main and is installed into the conda environment at installation?"*

## Answer

**Good news!** This functionality **already exists** and is properly configured. When you install pyCuSAXS, executable scripts are automatically created and installed into your conda/virtual environment.

## How It Works

### 1. Configuration in `pyproject.toml`

The `[project.scripts]` section defines four command-line tools:

```toml
[project.scripts]
pycusaxs = "pycusaxs.main:main"
saxs-widget = "pycusaxs.main:main"
saxs-db = "pycusaxs.saxs_db_tool:main"
saxs-subtract = "pycusaxs.saxs_subtract:main"
```

### 2. Automatic Installation

When you run:
```bash
pip install -e ".[cuda12]"
```

pip automatically:
- Reads the `[project.scripts]` section
- Creates executable wrapper scripts
- Installs them to `$CONDA_PREFIX/bin/` (in conda environments)
- Makes them executable with proper shebang
- Ensures they use the correct Python interpreter

### 3. Installed Scripts

After installation, these commands are available:

```bash
pycusaxs        # Main CLI/GUI - calls pycusaxs.main:main
saxs-widget     # GUI launcher - calls pycusaxs.main:main
saxs-db         # Database tool - calls pycusaxs.saxs_db_tool:main
saxs-subtract   # Subtraction tool - calls pycusaxs.saxs_subtract:main
```

## Verification

Check that scripts are installed:

```bash
# Activate environment
conda activate pycusaxs

# Check script locations
which pycusaxs
which saxs-widget
which saxs-db
which saxs-subtract

# Test help output
pycusaxs --help
```

Expected output:
```
/opt/miniforge3/envs/pycusaxs/bin/pycusaxs
/opt/miniforge3/envs/pycusaxs/bin/saxs-widget
/opt/miniforge3/envs/pycusaxs/bin/saxs-db
/opt/miniforge3/envs/pycusaxs/bin/saxs-subtract
```

## Script Format

Each installed script is a Python wrapper that looks like:

```python
#!/opt/miniforge3/envs/pycusaxs/bin/python3.13
import sys
from pycusaxs.main import main
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
```

## Additional Standalone Scripts

For convenience, standalone scripts are also provided in the `bin/` directory of the repository:

```
pyCuSaxs/
└── bin/
    ├── pycusaxs       - Main CLI/GUI launcher
    ├── saxs-widget    - GUI launcher
    ├── saxs-db        - Database tool
    ├── saxs-subtract  - Subtraction tool
    └── README.md      - Documentation
```

These can be run directly from the repository:
```bash
./bin/pycusaxs --help
```

However, **using the installed scripts** (via pip install) is recommended as they:
- Use the correct Python interpreter
- Don't require PYTHONPATH configuration
- Are available system-wide in the activated environment
- Work from any directory

## Usage Examples

### Main CLI Tool
```bash
# Launch GUI (no arguments)
pycusaxs

# Run SAXS calculation
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 100

# Explicitly launch GUI
pycusaxs gui
```

### GUI Widget
```bash
saxs-widget
```

### Database Tool
```bash
# List profiles
saxs-db list

# Query specific profile
saxs-db query --id 1
```

### Subtraction Tool
```bash
# Interactive mode
saxs-subtract

# With parameters
saxs-subtract --id 5 --dq 0.01
```

## Documentation

For more detailed information, see:
- [docs/SCRIPTS.md](docs/SCRIPTS.md) - Complete script documentation
- [bin/README.md](bin/README.md) - Standalone scripts info
- [INSTALL.md](INSTALL.md) - Installation guide
- [README.md](README.md) - Main documentation

## Summary

✅ **Scripts are automatically installed** when you run `pip install`
✅ **Four command-line tools** are available: `pycusaxs`, `saxs-widget`, `saxs-db`, `saxs-subtract`
✅ **Standalone scripts** also provided in `bin/` directory for advanced use
✅ **Properly configured** in `pyproject.toml` with `[project.scripts]`
✅ **Environment-aware** - uses correct Python interpreter
✅ **Cross-platform** - works on Linux, macOS, Windows

**No additional changes needed** - the functionality you requested is already implemented and working correctly!
