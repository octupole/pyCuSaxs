# Executable Scripts in pyCuSAXS

## Overview

When you install pyCuSAXS using `pip install .` or `pip install -e .`, several executable scripts are automatically installed into your conda/virtual environment's `bin/` directory. These scripts provide command-line access to all pyCuSAXS tools.

## Installed Scripts

The following scripts are automatically created during installation:

### 1. `pycusaxs`
Main command-line interface for SAXS calculations.

**Entry Point:** `pycusaxs.main:main`

**Usage:**
```bash
# Run GUI (if no arguments provided)
pycusaxs

# Run CLI with trajectory
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 100

# Explicitly launch GUI
pycusaxs gui
```

### 2. `saxs-widget`
Graphical user interface launcher.

**Entry Point:** `pycusaxs.main:main`

**Usage:**
```bash
saxs-widget
```

### 3. `saxs-db`
Database query and management tool.

**Entry Point:** `pycusaxs.saxs_db_tool:main`

**Usage:**
```bash
# List all profiles
saxs-db list

# Query specific profile
saxs-db query --id 1

# Export database
saxs-db export output.csv
```

### 4. `saxs-subtract`
Solvent subtraction tool.

**Entry Point:** `pycusaxs.saxs_subtract:main`

**Usage:**
```bash
# Interactive mode
saxs-subtract

# With specific profile ID
saxs-subtract --id 5 --scale 0.95

# With resampling
saxs-subtract --dq 0.01
```

## How Scripts Are Installed

### Configuration in `pyproject.toml`

Scripts are defined in the `[project.scripts]` section:

```toml
[project.scripts]
pycusaxs = "pycusaxs.main:main"
saxs-widget = "pycusaxs.main:main"
saxs-db = "pycusaxs.saxs_db_tool:main"
saxs-subtract = "pycusaxs.saxs_subtract:main"
```

### Installation Process

When you run:
```bash
pip install -e ".[cuda12]"
```

pip/setuptools automatically:
1. Reads the `[project.scripts]` section from `pyproject.toml`
2. Creates executable wrapper scripts for each entry
3. Installs them to `$CONDA_PREFIX/bin/` (or `$VIRTUAL_ENV/bin/`)
4. Makes them executable with the shebang pointing to your Python interpreter

### Generated Script Format

Each installed script looks like this:

```python
#!/path/to/your/conda/env/bin/python3
import sys
from pycusaxs.main import main
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
```

## Verifying Installation

After installing pyCuSAXS, verify the scripts are available:

```bash
# Check if scripts exist
which pycusaxs
which saxs-widget
which saxs-db
which saxs-subtract

# Test help output
pycusaxs --help
saxs-db --help
saxs-subtract --help
```

Expected output locations:
- Conda: `/path/to/conda/envs/pycusaxs/bin/pycusaxs`
- Virtualenv: `/path/to/venv/bin/pycusaxs`

## Script Locations

### In Development Mode (`pip install -e .`)

Scripts are installed to:
```
$CONDA_PREFIX/bin/pycusaxs          # Conda environment
$VIRTUAL_ENV/bin/pycusaxs           # Virtual environment
~/.local/bin/pycusaxs               # User installation
```

### Source Scripts (Optional)

For advanced users, standalone scripts are also available in the `bin/` directory of the source repository:

```
pyCuSaxs/
└── bin/
    ├── pycusaxs
    ├── saxs-widget
    ├── saxs-db
    └── saxs-subtract
```

These can be run directly from the repository:
```bash
cd pyCuSaxs
./bin/pycusaxs --help
```

## Troubleshooting

### Scripts Not Found After Installation

**Problem:** `pycusaxs: command not found`

**Solutions:**

1. **Ensure environment is activated:**
   ```bash
   conda activate pycusaxs
   # or
   source venv/bin/activate
   ```

2. **Check if scripts were installed:**
   ```bash
   ls -la $CONDA_PREFIX/bin/ | grep saxs
   ```

3. **Verify PATH includes bin directory:**
   ```bash
   echo $PATH | grep -o "$CONDA_PREFIX/bin"
   ```

4. **Reinstall in editable mode:**
   ```bash
   pip install -e ".[cuda12]" --force-reinstall
   ```

### ImportError When Running Scripts

**Problem:** `ModuleNotFoundError: No module named 'pycusaxs'`

**Solutions:**

1. **Ensure pyCuSAXS is installed:**
   ```bash
   pip list | grep pycusaxs
   ```

2. **Reinstall:**
   ```bash
   pip install -e ".[cuda12]"
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

### Wrong Python Interpreter

**Problem:** Scripts use system Python instead of conda Python

**Solutions:**

1. **Check shebang in installed script:**
   ```bash
   head -1 $(which pycusaxs)
   ```
   Should show: `#!/path/to/conda/envs/pycusaxs/bin/python3`

2. **Reinstall with correct environment active:**
   ```bash
   conda activate pycusaxs
   pip uninstall pycusaxs
   pip install -e ".[cuda12]"
   ```

## Advanced Usage

### Running Scripts from Python

You can also import and run these entry points from Python:

```python
# Run main CLI
from pycusaxs.main import main
main(['-s', 'topology.tpr', '-x', 'trajectory.xtc', '-g', '128', '-b', '0', '-e', '100'])

# Run database tool
from pycusaxs.saxs_db_tool import main as db_main
db_main(['list'])

# Run subtract tool
from pycusaxs.saxs_subtract import main as subtract_main
subtract_main(['--id', '5'])
```

### Custom Script Wrappers

You can create your own wrapper scripts for specialized workflows:

```python
#!/usr/bin/env python3
"""Custom wrapper for batch SAXS processing."""

import sys
from pycusaxs.main import main

# Set default parameters
default_args = [
    '-g', '128',
    '--order', '4',
    '--bin', '0.02',
]

if __name__ == '__main__':
    # Combine default args with user args
    args = sys.argv[1:] + default_args
    sys.exit(main(args))
```

## Adding New Scripts

To add a new executable script:

1. **Define the entry point function** in a Python module:
   ```python
   # pycusaxs/my_tool.py
   def main():
       print("My new tool!")
       return 0
   ```

2. **Add to `pyproject.toml`:**
   ```toml
   [project.scripts]
   my-tool = "pycusaxs.my_tool:main"
   ```

3. **Reinstall:**
   ```bash
   pip install -e ".[cuda12]"
   ```

4. **Test:**
   ```bash
   my-tool
   ```

## Environment Variables

Scripts respect the following environment variables:

- `CUDA_HOME` - CUDA installation directory
- `CUDA_PATH` - Alternative CUDA path
- `LD_LIBRARY_PATH` - Library search path for CUDA libraries

Set them in your shell config (`.bashrc`, `.zshrc`):
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Summary

- **Scripts are automatically installed** via `[project.scripts]` in `pyproject.toml`
- **No manual script installation needed** - pip handles everything
- **Scripts call Python entry points** - defined in pyCuSAXS modules
- **Platform independent** - works on Linux, macOS, Windows
- **Environment aware** - uses correct Python interpreter from activated environment

For more information, see:
- [README.md](../README.md) - General usage
- [INSTALL.md](../INSTALL.md) - Installation guide
- [docs/user-guide/](user-guide/) - Detailed usage documentation
