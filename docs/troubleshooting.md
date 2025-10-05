# Troubleshooting

This guide helps you diagnose and resolve common issues with pyCuSAXS.

## Common Issues

### Installation Problems

#### 1. CUDA Backend Import Error

**Symptom:**

```python
ImportError: No module named 'pycusaxs_cuda'
```

**Causes and Solutions:**

=== "CUDA Not Installed"

    **Check:**
    ```bash
    nvcc --version
    ```

    **Solution:**
    - Install CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)
    - Ensure CUDA is in PATH:
      ```bash
      export PATH=/usr/local/cuda/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      ```

=== "Build Failed"

    **Check:**
    ```bash
    pip install --force-reinstall -v .
    ```

    **Look for error messages in build output**

    **Common fixes:**
    - Install CMake: `pip install cmake`
    - Install C++ compiler: `sudo apt install build-essential`
    - Check CUDA version compatibility

=== "Wrong Python Environment"

    **Check:**
    ```bash
    which python
    pip list | grep pycusaxs
    ```

    **Solution:**
    - Activate correct virtual environment
    - Reinstall in correct environment

#### 2. Build Failures

**Symptom:**

```
ninja: build stopped: subcommand failed
```

**Diagnostic Steps:**

1. **Check CMake Output:**

   ```bash
   pip install -v . 2>&1 | tee build.log
   grep -i error build.log
   ```

2. **Verify CUDA Version:**

   ```bash
   nvcc --version
   ```

   Ensure version is 11.0+

3. **Check Compiler Compatibility:**

   ```bash
   gcc --version
   ```

   For CUDA 11.x, use GCC 9-11 (not GCC 12+)

**Solutions:**

=== "Compiler Too New"

    ```bash
    # Ubuntu: Install older GCC
    sudo apt install gcc-11 g++-11

    # Set as default for this session
    export CC=gcc-11
    export CXX=g++-11

    # Rebuild
    pip install --force-reinstall .
    ```

=== "Missing Dependencies"

    ```bash
    # Ubuntu/Debian
    sudo apt install cmake build-essential python3-dev

    # Fedora/RHEL
    sudo dnf install cmake gcc-c++ python3-devel

    # Rebuild
    pip install .
    ```

=== "CUDA Not Found"

    ```bash
    # Set CUDA path
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # Rebuild
    pip install --force-reinstall .
    ```

#### 3. MDAnalysis Warnings

**Symptom:**

```
No coordinate reader found for...
```

**Solution:**

```bash
# Install MDAnalysis with all extras
pip install MDAnalysis[all]
```

This installs additional dependencies for various trajectory formats.

### Runtime Errors

#### 1. GPU Memory Errors

**Symptom:**

```
CUDA error: out of memory
```

**Diagnostic:**

```bash
# Check GPU memory
nvidia-smi

# Check what's using GPU
nvidia-smi pmon
```

**Solutions:**

=== "Reduce Grid Size"

    ```bash
    # Instead of 128³
    python -m pycusaxs.main -g 64 ...

    # Or 96³
    python -m pycusaxs.main -g 96 ...
    ```

    **Memory savings:**
    - 128³ → 64³: **8× less memory**
    - 256³ → 128³: **8× less memory**

=== "Reduce Scale Factor"

    ```bash
    # Instead of scale=2.0
    python -m pycusaxs.main --Scale 1.0 ...
    ```

    **Memory savings:**
    - Scale 2.0 → 1.0: **8× less memory** (for scaled grid)

=== "Close Other GPU Applications"

    ```bash
    # Find GPU processes
    nvidia-smi

    # Kill specific process
    kill -9 <PID>
    ```

=== "Process in Batches"

    ```bash
    # Instead of processing all frames
    # Process in chunks
    for start in 0 250 500 750; do
        end=$((start + 249))
        python -m pycusaxs.main \
            -s system.tpr -x traj.xtc \
            -g 128 -b $start -e $end \
            -o saxs_${start}.dat
    done
    ```

#### 2. File Not Found Errors

**Symptom:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'system.tpr'
```

**Diagnostic:**

```bash
# Check file exists
ls -lh system.tpr trajectory.xtc

# Check permissions
ls -l system.tpr

# Try absolute path
python -m pycusaxs.main -s /full/path/to/system.tpr ...
```

**Solutions:**

- Verify file paths are correct
- Use absolute paths instead of relative
- Check file permissions (must be readable)
- Ensure files are not corrupted

#### 3. Invalid Frame Range

**Symptom:**

```
ValueError: Begin frame (1000) must be less than end frame (999)
```

or

```
IndexError: Frame index out of range
```

**Diagnostic:**

```python
from pycusaxs.topology import Topology

topo = Topology("system.tpr", "trajectory.xtc")
print(f"Total frames: {topo.n_frames}")
```

**Solutions:**

- Ensure `--begin < --end`
- Check total frames in trajectory
- Use valid 0-based indices

**Example:**

```bash
# For 1000-frame trajectory (frames 0-999)
python -m pycusaxs.main -b 0 -e 999 ...  # Correct
python -m pycusaxs.main -b 0 -e 1000 ... # Error!
```

#### 4. Grid Size Errors

**Symptom:**

```
ValueError: Grid size must be 1 or 3 positive integers
```

**Solutions:**

```bash
# Correct formats
--grid 128           # Cubic: 128×128×128
--grid 128,128,128   # Explicit
--grid 64,64,128     # Non-cubic

# Incorrect formats
--grid 128 128 128   # Wrong: use commas!
--grid 128,128       # Wrong: need 1 or 3 values
--grid -128          # Wrong: must be positive
```

### Performance Issues

#### 1. Slow Performance

**Diagnostic:**

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Profile execution
nsys profile -o profile python -m pycusaxs.main -s system.tpr -x traj.xtc -g 128 -b 0 -e 10
nsys-ui profile.qdrep
```

**Check:**

- GPU utilization should be > 80%
- Memory bandwidth utilization
- CPU-GPU transfer time

**Solutions:**

=== "GPU Underutilized"

    **Possible causes:**
    - Grid too small (not enough work for GPU)
    - Slow trajectory I/O

    **Solutions:**
    ```bash
    # Use larger grid if memory allows
    --grid 128  # instead of 64

    # Copy trajectory to local SSD
    cp trajectory.xtc /tmp/
    python -m pycusaxs.main -x /tmp/trajectory.xtc ...
    ```

=== "Slow I/O"

    **Check Read Time in output:**
    ```
    Read Time:     25.12 ms/per step  # Too high!
    CUDA Time:     10.43 ms/per step  # OK
    ```

    **Solutions:**
    - Copy trajectory to local storage
    - Use compressed `.xtc` format
    - Increase `--dt` stride

=== "High CPU Load"

    **Check:**
    ```bash
    htop
    ```

    **Solutions:**
    - Close other CPU-intensive applications
    - Use `--dt` to reduce frame processing
    - Reduce trajectory I/O

#### 2. Memory Leaks

**Symptom:**

Memory usage grows with trajectory length (fixed in v0.1.0)

**Diagnostic:**

```bash
# Monitor memory over time
watch -n 1 nvidia-smi
```

**Solution:**

- Ensure you're using v0.1.0+
- Upgrade: `pip install --force-reinstall .`

#### 3. Incorrect Results

**Symptoms:**

- NaN values in output
- Negative intensities
- Unrealistic profiles

**Diagnostic Steps:**

1. **Validate Input Files:**

   ```bash
   # Check topology with GROMACS tools
   gmx check -f trajectory.xtc

   # Visualize trajectory
   vmd -gro topology.gro -xtc trajectory.xtc
   ```

2. **Check Parameters:**

   ```bash
   # Verify histogram parameters
   --bin 0.01   # Not too small (< 0.001)
   --qcut 0.5   # Reasonable range

   # Check grid size
   --grid 128   # Not too small (< 32)
   ```

3. **Test with Small System:**

   ```bash
   # Test on first 10 frames only
   python -m pycusaxs.main \
       -s system.tpr -x trajectory.xtc \
       -g 64 -b 0 -e 10 \
       -o test.dat
   ```

**Common Fixes:**

- Increase grid size
- Adjust histogram binning
- Check for corrupted trajectory frames
- Verify trajectory units (nm vs Å)

## Validation

### Check Installation

```bash
# Verify Python package
python -c "import pycusaxs; print('Python package: OK')"

# Verify CUDA backend
python -c "import pycusaxs_cuda; print('CUDA backend: OK')"

# Check GPU availability
nvidia-smi
```

Expected output:

```
Python package: OK
CUDA backend: OK

[nvidia-smi output showing GPU]
```

### Test Calculation

```bash
# Run minimal test
python -m pycusaxs.main \
    -s test_data/system.tpr \
    -x test_data/trajectory.xtc \
    -g 64 -b 0 -e 0 \
    -o test.dat
```

Should complete without errors and create `test.dat`.

### Debug Mode

```bash
# Enable verbose output
export CUDA_LAUNCH_BLOCKING=1

# Run with error checking
python -m pycusaxs.main --verbose -s system.tpr -x traj.xtc ...

# Check for CUDA errors
cuda-memcheck python -m pycusaxs.main -s system.tpr -x traj.xtc -g 64 -b 0 -e 1
```

## Error Messages Reference

### Python Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Grid size must be...` | Invalid grid specification | Use format: `128` or `128,128,128` |
| `ValueError: Begin frame must be...` | Invalid frame range | Ensure `begin < end` |
| `FileNotFoundError` | Missing input file | Check file path and permissions |
| `RuntimeError: No frame loaded` | Called getter before `read_frame()` | Call `read_frame()` first |

### C++ Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA error: out of memory` | Insufficient GPU memory | Reduce grid size or scale factor |
| `CUDA error: invalid device function` | CUDA version mismatch | Rebuild with correct CUDA version |
| `CUDA error: unspecified launch failure` | Kernel error | Enable cuda-memcheck for details |
| `File write error` | Cannot write output | Check directory permissions |

### Build Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `nvcc: command not found` | CUDA not in PATH | Add CUDA to PATH |
| `No CMAKE_CUDA_COMPILER` | CMake can't find CUDA | Set `CUDA_HOME` environment variable |
| `unsupported GNU version` | GCC too new for CUDA | Use compatible GCC version |

## Getting Help

If you're still having issues:

1. **Check existing issues:** [GitHub Issues](https://github.com/yourusername/pyCuSaxs/issues)

2. **Search discussions:** [GitHub Discussions](https://github.com/yourusername/pyCuSaxs/discussions)

3. **Create new issue** with:
   - pyCuSAXS version
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi`)
   - Operating system
   - Full error message
   - Minimal reproducing example

4. **Email support:** [your.email@domain.com]

## Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
lsb_release -a

# CUDA information
nvcc --version
nvidia-smi

# Python information
python --version
pip list | grep -E "(pycusaxs|MDAnalysis|numpy|PySide6)"

# GPU information
nvidia-smi -L
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
```

## Known Issues

### Current Limitations

1. **Single GPU Only:** Multi-GPU not yet supported
2. **No Windows Support:** Tested on Linux only (may work on Windows with modifications)
3. **Large Grids:** 512³+ grids require significant GPU memory (16GB+)

### Workarounds

**Multi-GPU workaround:**

Run multiple instances on different GPUs:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m pycusaxs.main -b 0 -e 499 -o saxs_0.dat ...

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m pycusaxs.main -b 500 -e 999 -o saxs_1.dat ...
```

**Large grid workaround:**

Process with reduced grid, then interpolate:

```bash
# Use 256³ instead of 512³
python -m pycusaxs.main -g 256 ...
```

## See Also

- [Installation Guide](getting-started/installation.md)
- [Configuration](getting-started/configuration.md)
- [GitHub Discussions](https://github.com/octupole/pyCuSaxs/discussions) for questions
- [GitHub Issues](https://github.com/yourusername/pyCuSaxs/issues)
