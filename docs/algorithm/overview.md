# Algorithm Overview

This page describes the theoretical foundation and computational approach used by pyCuSAXS to calculate Small-Angle X-ray Scattering (SAXS) profiles from molecular dynamics trajectories.

## SAXS Theory

### Scattering Intensity

The scattering intensity $I(q)$ is calculated as the Fourier transform of the electron density:

$$I(\mathbf{q}) = |F(\mathbf{q})|^2 = \left|\sum_j f_j(q) \exp(i\mathbf{q} \cdot \mathbf{r}_j)\right|^2$$

Where:

- $\mathbf{q}$ is the scattering vector
- $f_j(q)$ is the atomic form factor for atom $j$
- $\mathbf{r}_j$ is the position of atom $j$
- $|\mathbf{q}| = q = \frac{4\pi \sin\theta}{\lambda}$

For solution scattering, we average over all orientations, giving intensity as a function of $q = |\mathbf{q}|$ only.

### Form Factors

Atomic form factors $f(q)$ describe how strongly each atom type scatters X-rays:

$$f(q) = \sum_{i=1}^{4} a_i \exp\left(-b_i \frac{q^2}{16\pi^2}\right) + c$$

Where $a_i$, $b_i$, and $c$ are tabulated constants for each element.

## SAXS Calculation Pipeline

pyCuSAXS uses a grid-based FFT approach for efficient SAXS calculation:

### 1. Coordinate Transformation

Transform atomic coordinates from triclinic simulation box to orthonormal coordinates.

**Input:** Atomic positions $\mathbf{r}_j$ and box matrix $\mathbf{H}$

**Process:**

1. Calculate orientation matrix $\mathbf{CO}$ (cell-to-orthonormal)
2. Apply transformation: $\mathbf{r}'_j = \mathbf{CO} \cdot \mathbf{r}_j$
3. Center coordinates in grid

**Mathematical Foundation:**

For triclinic boxes with vectors $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$:

$$\mathbf{CO} = \begin{pmatrix}
a_x & b_x & c_x \\
0 & b_y & c_y \\
0 & 0 & c_z
\end{pmatrix}$$

And reciprocal transformation $\mathbf{OC} = \mathbf{CO}^{-1}$.

### 2. Density Grid Assignment

Map atomic electron density onto a 3D grid using B-spline interpolation.

**Input:** Atomic positions and form factors

**Process:**

1. For each atom at position $\mathbf{r}_j$:
   - Calculate grid cell indices
   - Compute B-spline weights $M_n(u)$
   - Distribute $f_j(0)$ to nearby grid points

**B-spline Interpolation:**

For order $n$, the B-spline weight at grid point $i$ is:

$$\rho(\mathbf{r}_i) = \sum_j f_j(0) \prod_{d \in \{x,y,z\}} M_n(u_d - i_d)$$

Where $M_n(u)$ is the $n$-th order B-spline basis function.

**Benefits:**

- Smooth density assignment
- Continuous derivatives
- Reduces grid artifacts
- Orders 4-6 provide good accuracy

### 3. Padding & Supersampling

Add solvent padding and supersample to scaled grid.

**Padding Modes:**

=== "Average (Default)"

    Compute average density from grid borders:

    $$\rho_{\text{solvent}} = \frac{1}{N_{\text{border}}} \sum_{i \in \text{border}} \rho_i$$

=== "Explicit (Water Model)"

    Use precomputed densities for water model:

    $$\rho_{\text{solvent}} = n_{\text{O}} f_{\text{O}}(0) + n_{\text{H}} f_{\text{H}}(0) + n_{\text{Na}} f_{\text{Na}}(0) + n_{\text{Cl}} f_{\text{Cl}}(0)$$

    Where $n_i$ are atom counts per grid cell from water model.

**Supersampling:**

Grid is expanded by scale factor $\sigma$:

$$\text{Scaled grid: } (n_x \sigma) \times (n_y \sigma) \times (n_z \sigma)$$

This improves reciprocal space resolution:

$$\Delta q = \frac{2\pi}{N \cdot \sigma \cdot L}$$

### 4. Fourier Transform

Compute 3D Fast Fourier Transform using cuFFT.

**Real-to-Complex FFT:**

$$F(\mathbf{q}) = \mathcal{F}[\rho(\mathbf{r})] = \int \rho(\mathbf{r}) \exp(i\mathbf{q} \cdot \mathbf{r}) d\mathbf{r}$$

**Grid FFT:**

$$F_{klm} = \sum_{i=0}^{N_x-1} \sum_{j=0}^{N_y-1} \sum_{k=0}^{N_z-1} \rho_{ijk} \exp\left(2\pi i \left(\frac{ki}{N_x} + \frac{lj}{N_y} + \frac{mk}{N_z}\right)\right)$$

**B-spline Modulation:**

Apply correction factors to account for B-spline interpolation:

$$F'(\mathbf{q}) = \frac{F(\mathbf{q})}{\prod_{d \in \{x,y,z\}} \tilde{M}_n(q_d)}$$

Where $\tilde{M}_n(q)$ is the Fourier transform of the B-spline basis.

### 5. Scattering Factor Application

Apply $q$-dependent atomic form factors in reciprocal space.

**Process:**

1. For each grid point $\mathbf{q}_{klm}$:
   - Calculate $q = |\mathbf{q}_{klm}|$
   - Compute form factors $f_j(q)$ for each element
   - Apply correction: $F'(\mathbf{q}) \rightarrow F'(\mathbf{q}) \cdot \frac{f(q)}{f(0)}$

**Form Factor Lookup:**

Form factors are tabulated and interpolated:

$$f(q) = \sum_{i=1}^{4} a_i \exp\left(-b_i \frac{q^2}{16\pi^2}\right) + c$$

### 6. Intensity Calculation

Compute scattering intensity from structure factor.

$$I(\mathbf{q}) = |F(\mathbf{q})|^2 = F(\mathbf{q}) \cdot F^*(\mathbf{q})$$

For complex $F = F_{\text{real}} + i F_{\text{imag}}$:

$$I(\mathbf{q}) = F_{\text{real}}^2 + F_{\text{imag}}^2$$

### 7. Histogram Binning

Bin intensities by $|\mathbf{q}|$ magnitude and average over frames.

**Process:**

1. For each reciprocal space point:
   - Calculate $q = |\mathbf{q}|$
   - Determine bin index: $b = \lfloor q / \Delta q \rfloor$
   - Accumulate: $I_b \mathrel{+}= I(\mathbf{q})$, $N_b \mathrel{+}= 1$

2. Average over frames (NVT ensemble):
   - For frame $f$: $I_b^{(f)} = \sum_{\mathbf{q} \in \text{bin}} I^{(f)}(\mathbf{q})$
   - Final average: $I_b = \frac{1}{N_{\text{frames}}} \sum_f I_b^{(f)}$

**Output:**

Binned profile $(q, I(q))$ written to output file.

## Grid Resolution

The relationship between grid parameters and reciprocal space resolution:

### Real Space Resolution

Grid spacing in real space:

$$\Delta r = \frac{L}{N}$$

Where $L$ is box length and $N$ is grid dimension.

### Reciprocal Space Resolution

Resolution in $q$-space:

$$\Delta q = \frac{2\pi}{N \cdot \sigma \cdot L}$$

Where $\sigma$ is the scale factor.

### Maximum q Value

The maximum accessible $q$ value is:

$$q_{\max} = \frac{\pi}{2 \Delta r} = \frac{\pi N}{2L}$$

**Guidelines:**

- For $q_{\max} = 0.5$ Å⁻¹ and $L = 100$ Å: Need $N \geq 32$
- For $q_{\max} = 1.0$ Å⁻¹ and $L = 100$ Å: Need $N \geq 64$
- Typical: $N = 128$ provides $q_{\max} \approx 2.0$ Å⁻¹

## Ensemble Averaging

### NVT Ensemble (Constant Volume)

Simple averaging over frames:

$$I(q) = \frac{1}{N_{\text{frames}}} \sum_{f=1}^{N_{\text{frames}}} I^{(f)}(q)$$

### NPT Ensemble (Constant Pressure)

Volume fluctuations require weighted averaging:

$$I(q) = \frac{\sum_f I^{(f)}(q) V_f}{\sum_f V_f}$$

Where $V_f$ is the box volume at frame $f$.

## Computational Complexity

### Time Complexity

- **Density Assignment**: $O(N_{\text{atoms}} \times n^3)$ where $n$ is B-spline order
- **FFT**: $O(N^3 \log N)$ where $N$ is grid dimension
- **Scattering Factors**: $O(N^3)$
- **Histogram**: $O(N^3)$

**Total per frame:** $O(N_{\text{atoms}} \times n^3 + N^3 \log N)$

Dominated by FFT for large grids.

### Space Complexity

GPU memory requirements:

- **Density grid**: $4 \times N^3$ bytes (float)
- **FFT buffer**: $8 \times N^3$ bytes (complex)
- **Scaled grid**: $4 \times (N \sigma)^3$ bytes

**Example:** For $N = 128$, $\sigma = 2$:

- Primary grid: 8 MB
- Scaled grid: 64 MB
- FFT buffer: 64 MB
- **Total: ~150 MB** (plus overhead)

## Advantages of Grid-based Method

1. **FFT Efficiency**: $O(N \log N)$ vs $O(N^2)$ for direct summation
2. **GPU Parallelization**: Grid operations map well to GPU architecture
3. **Scalability**: Linear scaling with number of atoms
4. **Accuracy**: B-spline interpolation reduces grid artifacts

## Limitations

1. **Grid Artifacts**: Finite grid introduces small artifacts (mitigated by B-splines)
2. **Memory**: Large grids require significant GPU memory
3. **Resolution**: Maximum $q$ limited by grid spacing
4. **Periodic Boundary**: Assumes periodic density (not physical for biomolecules)

## Validation

The algorithm has been validated against:

- **CRYSOL**: Reference SAXS calculation software
- **Direct Debye Summation**: Exact calculation for small systems
- **Experimental Data**: Protein SAXS profiles from synchrotron sources

Typical accuracy: < 1% error for $q < 0.5$ Å⁻¹ with appropriate grid parameters.

## See Also

- [Pipeline Details](pipeline.md) - Step-by-step implementation
- [Performance](performance.md) - Optimization and benchmarks
- [Backend API](../api/backend.md) - Implementation details
- [Configuration](../getting-started/configuration.md) - Parameter tuning
