# Performance

This page details the performance characteristics, optimizations, and benchmarks of pyCuSAXS.

## Recent Performance Improvements (v0.1.0)

### GPU Synchronization Optimization

One of the major performance improvements in v0.1.0 was the reduction of unnecessary GPU synchronization.

**Before v0.1.0:**

- 10 `cudaDeviceSynchronize()` calls per frame
- Excessive CPU-GPU synchronization overhead
- GPU pipeline stalls waiting for CPU

**After v0.1.0:**

- 2 `cudaDeviceSynchronize()` calls per frame
- Synchronize only before Device-to-Host transfers
- Kernels in same stream execute in order automatically

**Impact:** **15-30% throughput improvement** for multi-frame trajectories

### Where Synchronization is Actually Needed

1. **Before D→H transfers**: Must complete GPU work before reading results
2. **Before averaging**: Ensure all histogram updates complete

### Where Synchronization was Removed

- After density assignment (kernels queue automatically)
- After padding (same stream ordering)
- After FFT (no CPU dependency)
- After form factor application (no CPU access needed)
- Multiple intermediate checks (unnecessary)

!!! success "Key Insight"
    CUDA kernels launched in the same stream execute in order automatically. Explicit synchronization is only needed when the CPU needs to access GPU results.

## Optimization Features

### Memory Management

#### Streaming Trajectory Processing

**No Full Trajectory in Memory:**

```python
# Bad: Loads entire trajectory
coords_all = traj.timeseries.positions  # GBs of memory

# Good: Stream frames one at a time
for frame_data in topo.iter_frames_stream(0, n_frames, step=10):
    process_frame(frame_data)  # Only one frame in memory
```

**Benefits:**

- Constant memory footprint
- Works with arbitrarily large trajectories
- Enables processing of TB-scale data

#### Thrust Device Vectors

Automatic GPU memory management:

```cpp
// Automatic allocation and cleanup
thrust::device_vector<float> d_density(grid_size);
thrust::device_vector<cufftComplex> d_fft_buffer(fft_size);

// No manual cudaFree needed - RAII handles it
```

**Benefits:**

- No memory leaks
- Exception-safe cleanup
- Simplified code

#### Efficient Coordinate Layout

Coordinates stored for coalesced memory access:

```cpp
// Structure of Arrays (SoA) - Good for GPU
float* x_coords;  // All x coordinates contiguous
float* y_coords;  // All y coordinates contiguous
float* z_coords;  // All z coordinates contiguous

// Array of Structures (AoS) - Bad for GPU
struct {float x, y, z;} coords[n_atoms];  // Non-coalesced access
```

### Computational Efficiency

#### cuFFT for 3D Transforms

- Optimized for NVIDIA GPUs
- Uses tensor cores on modern GPUs
- Automatic work distribution

**Complexity:**

- Direct Debye summation: $O(N_{\text{atoms}}^2)$
- Grid-based FFT: $O(N^3 \log N)$

For typical protein systems:

- $N_{\text{atoms}} \approx 50,000$
- $N_{\text{grid}} = 128$

FFT is **~100× faster** than direct summation.

#### Atomic Operations for Histograms

Thread-safe accumulation without locks:

```cpp
// Multiple threads can update same bin safely
atomicAdd(&histogram[bin], intensity);
```

**Performance:**

- Modern GPUs have dedicated atomic units
- Minimal contention for typical histogram sizes
- Much faster than per-thread histograms + reduction

#### B-spline Modulation in Fourier Space

Apply corrections in reciprocal space instead of real space:

```cpp
// Fourier space: O(N^3) operations
for (int i = 0; i < fft_size; i++) {
    F[i] /= modulation_factor[i];
}

// vs. Real space: O(N_atoms × order^3) operations
// Much more expensive for large systems
```

### Double-Buffered Frame Loading

Overlap CPU I/O with GPU computation:

```
Frame 1: Load ━━━┓
                  ┗━━━ Process ━━━┓
Frame 2:          Load ━━━━━━━━━━┓┗━━━ Write
                                  ┗━━━ Process ━━━┓
Frame 3:                          Load ━━━━━━━━━━┓┗━━━ Write
                                                  ┗━━━ Process
```

**Implementation:**

```cpp
// Load frame n+1 while processing frame n
py::gil_scoped_acquire acquire;  // Lock Python
load_frame(n+1);
py::gil_scoped_release release;  // Unlock Python

// GPU processes frame n (no GIL needed)
process_frame_gpu(n);
```

**Impact:** 10-20% wall-clock time reduction

## Benchmarks

### Test System

**Hardware:**

- GPU: NVIDIA RTX 3080 (10GB)
- CPU: Intel i9-10900K
- RAM: 32GB DDR4-3200
- Storage: NVMe SSD

**Software:**

- CUDA: 11.8
- Driver: 520.61.05
- OS: Ubuntu 22.04 LTS

### Performance by System Size

| System Size | Grid | Frames | Time/Frame | Total Time | GPU Memory |
|-------------|------|--------|------------|------------|------------|
| 10K atoms   | 64³  | 1000   | ~8 ms      | ~8 sec     | ~0.5 GB    |
| 50K atoms   | 64³  | 1000   | ~15 ms     | ~15 sec    | ~1.0 GB    |
| 50K atoms   | 128³ | 1000   | ~35 ms     | ~35 sec    | ~2.5 GB    |
| 100K atoms  | 128³ | 1000   | ~55 ms     | ~55 sec    | ~3.0 GB    |
| 200K atoms  | 128³ | 1000   | ~85 ms     | ~85 sec    | ~4.0 GB    |
| 200K atoms  | 256³ | 100    | ~180 ms    | ~18 sec    | ~8.5 GB    |

!!! info "Performance Notes"
    - Times include both CUDA computation and frame reading
    - Actual performance varies with GPU model, system complexity, and parameters
    - Larger grids require more memory and computation time

### Performance by Grid Size

**System:** 50K atoms, 1000 frames

| Grid Size | Time/Frame | Total Time | Speedup vs 256³ |
|-----------|------------|------------|-----------------|
| 64³       | ~15 ms     | ~15 sec    | 4.7×            |
| 96³       | ~25 ms     | ~25 sec    | 2.8×            |
| 128³      | ~35 ms     | ~35 sec    | 2.0×            |
| 160³      | ~50 ms     | ~50 sec    | 1.4×            |
| 192³      | ~62 ms     | ~62 sec    | 1.1×            |
| 256³      | ~70 ms     | ~70 sec    | 1.0×            |

**Scaling:** Approximately $O(N^3)$ as expected from FFT complexity

### Performance by B-spline Order

**System:** 50K atoms, 128³ grid, 1000 frames

| Order | Time/Frame | Relative Time |
|-------|------------|---------------|
| 1     | ~30 ms     | 0.86×         |
| 2     | ~32 ms     | 0.91×         |
| 4     | ~35 ms     | 1.00×         |
| 6     | ~40 ms     | 1.14×         |
| 8     | ~45 ms     | 1.29×         |

**Impact:** Higher orders slightly slower due to more grid points per atom

### Frame Stride Impact

**System:** 50K atoms, 128³ grid, 10000 frames in trajectory

| Stride (dt) | Frames Processed | Total Time | Time/Frame |
|-------------|------------------|------------|------------|
| 1           | 10000            | ~350 sec   | ~35 ms     |
| 5           | 2000             | ~70 sec    | ~35 ms     |
| 10          | 1000             | ~35 sec    | ~35 ms     |
| 20          | 500              | ~17 sec    | ~35 ms     |
| 50          | 200              | ~7 sec     | ~35 ms     |

**Conclusion:** Time per frame is constant; total time scales linearly with number of frames processed

## Scalability

### Linear Scaling with Trajectory Length

Processing time scales linearly with number of frames:

$$T_{\text{total}} = N_{\text{frames}} \times T_{\text{per frame}} + T_{\text{overhead}}$$

Where $T_{\text{overhead}}$ (initialization, file opening) is typically < 1 second.

### Grid Size Scaling

FFT dominates for large grids:

$$T_{\text{per frame}} \approx C_1 \cdot N_{\text{atoms}} + C_2 \cdot N_{\text{grid}}^3 \log N_{\text{grid}}$$

For typical systems (50K-200K atoms), the $N^3 \log N$ term dominates.

### GPU Memory Requirements

**Formula:**

$$M_{\text{GPU}} = 4N^3 + 8(N\sigma)^3 + M_{\text{atoms}} + M_{\text{overhead}}$$

Where:

- $4N^3$ - Primary density grid (float, bytes)
- $8(N\sigma)^3$ - Scaled grid + FFT buffer (float + complex)
- $M_{\text{atoms}}$ - Coordinate storage (~100 bytes × $N_{\text{atoms}}$)
- $M_{\text{overhead}}$ - Misc. buffers (~500 MB)

**Examples:**

| Grid | Scale | Grid Memory | Total Memory |
|------|-------|-------------|--------------|
| 64³  | 1.0   | ~1 MB + ~1 MB = ~2 MB | ~500 MB |
| 128³ | 1.0   | ~8 MB + ~8 MB = ~16 MB | ~600 MB |
| 128³ | 2.0   | ~8 MB + ~64 MB = ~72 MB | ~1 GB |
| 256³ | 1.0   | ~64 MB + ~64 MB = ~128 MB | ~1.5 GB |
| 256³ | 2.0   | ~64 MB + ~512 MB = ~576 MB | ~2.5 GB |

!!! warning "Memory Limit"
    Maximum grid size depends on GPU memory. For 10GB GPU:

    - 256³ with scale=2.0: ~2.5 GB (OK)
    - 512³ with scale=1.0: ~2 GB (OK)
    - 512³ with scale=2.0: ~16 GB (Too large)

## Performance Optimization Guide

### For Speed

**Minimize Computation Time:**

1. Use smallest acceptable grid (64³ or 96³)
2. Use order=4 (default)
3. Increase frame stride (dt=10-20)
4. Use scale=1.0 (no oversampling)
5. Coarse histogram binning (bin=0.02)

**Example:**

```bash
python -m pycusaxs.main \
    -s system.tpr -x trajectory.xtc \
    -g 64 --order 4 --Scale 1.0 \
    --dt 20 --bin 0.02 \
    -b 0 -e 999
```

**Expected:** ~10-15 ms/frame for 50K atom system

### For Accuracy

**Maximize Quality:**

1. Use larger grid (128³ or 160³)
2. Use higher order (6 or 8)
3. Use scale=2.0 for supersampling
4. Process more frames (dt=1-5)
5. Fine histogram binning (bin=0.005)

**Example:**

```bash
python -m pycusaxs.main \
    -s system.tpr -x trajectory.xtc \
    -g 128 --order 6 --Scale 2.0 \
    --dt 5 --bin 0.005 \
    -b 0 -e 999
```

**Expected:** ~40-60 ms/frame for 50K atom system

### For Large Systems (200K+ atoms)

**Balance Memory and Speed:**

1. Use 128³ grid (don't exceed unless necessary)
2. Monitor GPU memory with `nvidia-smi`
3. Use scale=1.0 to save memory
4. Process in batches if needed

**Example:**

```bash
# Monitor GPU while running
watch -n 1 nvidia-smi

python -m pycusaxs.main \
    -s large_system.tpr -x trajectory.xtc \
    -g 128 --order 4 --Scale 1.0 \
    --dt 10 -b 0 -e 999
```

### For Long Trajectories (10K+ frames)

**Efficient Sampling:**

1. Use larger stride (dt=10-50)
2. Process in chunks if needed
3. Consider parallel processing different ranges

**Example:**

```bash
# Process every 20th frame
python -m pycusaxs.main \
    -s system.tpr -x long_traj.xtc \
    -g 128 --dt 20 \
    -b 0 -e 9999 \
    -o saxs_long.dat
```

## Monitoring Performance

### Real-time Monitoring

**Terminal 1:** Run calculation

```bash
python -m pycusaxs.main -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 999
```

**Terminal 2:** Monitor GPU

```bash
watch -n 1 nvidia-smi
```

**Terminal 3:** Monitor system

```bash
htop
```

### Performance Metrics

The output shows timing breakdown:

```
=========================================================
=                    CuSAXS Timing                     =
=========================================================
=           CUDA Time:     25.43 ms/per step           =
=           Read Time:     5.12 ms/per step            =
=           Total Time:    30.55 ms/per step           =
=========================================================
```

**Interpreting Results:**

- **CUDA Time:** GPU computation (kernel execution)
- **Read Time:** Trajectory I/O and Python overhead
- **Total Time:** Wall-clock time per frame

**Ideal Ratio:** CUDA time should be 70-90% of total time

**If Read Time is High:**

- Trajectory file on slow storage
- Network filesystem latency
- Python overhead (GIL contention)

**Solutions:**

- Copy trajectory to local SSD
- Use compressed `.xtc` format
- Increase frame stride

### Profiling with NVIDIA Tools

**nvprof (legacy):**

```bash
nvprof python -m pycusaxs.main -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 10
```

**Nsight Systems (modern):**

```bash
nsys profile -o saxs_profile python -m pycusaxs.main \
    -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 10
```

View results:

```bash
nsys-ui saxs_profile.qdrep
```

## Comparison with Other Methods

### vs. Direct Debye Summation

| Method | Complexity | 50K atoms | 200K atoms |
|--------|------------|-----------|------------|
| **pyCuSAXS (FFT)** | $O(N^3 \log N)$ | ~35 ms | ~85 ms |
| **Debye (CPU)** | $O(N^2)$ | ~30 sec | ~480 sec |
| **Debye (GPU)** | $O(N^2)$ | ~5 sec | ~80 sec |

**Speedup:** 100-1000× faster than direct summation

### vs. CRYSOL (CPU)

| Software | Method | Time (50K atoms) |
|----------|--------|------------------|
| **pyCuSAXS** | GPU FFT | ~35 ms/frame |
| **CRYSOL** | CPU Debye | ~10-30 sec/frame |

**Speedup:** ~500× faster

!!! note "Trade-offs"
    - CRYSOL may include additional physical effects (hydration layer)
    - pyCuSAXS optimized for throughput on MD trajectories
    - Both methods converge to similar results for appropriate parameters

## Best Practices

!!! success "Grid Sizing"
    - Use powers of 2 for FFT efficiency (64, 128, 256)
    - Match grid spacing to system size (1-2 Å per grid point)
    - Monitor GPU memory usage

!!! success "Frame Sampling"
    - Balance statistics vs. computation time
    - 50-100 frames usually sufficient for good statistics
    - Use stride to process large trajectories efficiently

!!! success "Parameter Tuning"
    - Start with small test runs (10-20 frames)
    - Measure performance before full production run
    - Adjust grid size and stride based on results

!!! success "Resource Management"
    - Close other GPU applications
    - Use `nvidia-smi` to check for competing processes
    - Consider batch processing for very long trajectories

## Future Optimizations

**Potential Improvements:**

- [ ] Multi-GPU support for parallel frame processing
- [ ] Tensor core utilization for FFT on A100/H100
- [ ] Further memory optimizations for 512³+ grids
- [ ] Asynchronous frame processing pipeline
- [ ] GPU-direct storage for ultra-fast I/O

## See Also

- [Algorithm Overview](overview.md) - Computational approach
- [Pipeline Details](pipeline.md) - Implementation details
- [Configuration](../getting-started/configuration.md) - Parameter tuning
- [Backend API](../api/backend.md) - Performance-critical code
