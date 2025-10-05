#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Splines.h"
#include "Ftypedefs.h"
#include "opsfact.h"

__global__ void calculate_histogram(cuFloatComplex *d_array, double *d_histogram, double *d_nhist, float *oc, int nx, int ny, int nz,
                                    float bin_size, float qcut, int num_bins, float fact)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int npz = nz / 2 + 1;
    if (i < nx && j < ny && k < npz)
    { // Only consider the upper half in z-direction
        int nfx = (nx % 2 == 0) ? nx / 2 : nx / 2 + 1;
        int nfy = (ny % 2 == 0) ? ny / 2 : ny / 2 + 1;
        int nfz = (nz % 2 == 0) ? nz / 2 : nz / 2 + 1;

        int ia = (i < nfx) ? i : i - nx;
        int ja = (j < nfy) ? j : j - ny;
        int ka = (k < nfz) ? k : k - nz;
        int ib = i == 0 ? 0 : nx - i;
        int jb = j == 0 ? 0 : ny - j;
        float mw1, mw2, mw3, mw;
        mw1 = oc[XX * DIM + XX] * ia + oc[XX * DIM + YY] * ja + oc[XX * DIM + ZZ] * ka;
        mw1 = 2.0 * M_PI * mw1;
        mw2 = oc[YY * DIM + XX] * ia + oc[YY * DIM + YY] * ja + oc[YY * DIM + ZZ] * ka;
        mw2 = 2.0 * M_PI * mw2;
        mw3 = oc[ZZ * DIM + XX] * ia + oc[ZZ * DIM + YY] * ja + oc[ZZ * DIM + ZZ] * ka;
        mw3 = 2.0 * M_PI * mw3;
        mw = sqrtf(mw1 * mw1 + mw2 * mw2 + mw3 * mw3);
        if (mw > qcut)
            return;
        // Check for division by zero
        if (bin_size <= 0.0f)
            return;
        int h0 = static_cast<int>(mw / bin_size);
        // Check bounds: h0 must be non-negative and within num_bins
        if (h0 >= 0 && h0 < num_bins)
        {
            double v0;
            int idx = k + j * npz + i * npz * ny;
            int idbx = k + jb * npz + ib * npz * ny;
            if (i == 0 && j == 0 && k == 0)
            {
                v0 = d_array[idx].x * fact;
                double nv0{1.0};
                // NOTE: These two atomicAdd operations are not atomic together
                // In practice, the race condition impact is minimal for histograms
                // For stricter consistency, consider using warp-level aggregation
                atomicAdd(&d_histogram[h0], v0);
                atomicAdd(&d_nhist[h0], nv0);
            }
            else
            {
                double nv0{2.0};
                v0 = (d_array[idx].x + d_array[idbx].x) * fact;
                atomicAdd(&d_histogram[h0], v0);
                atomicAdd(&d_nhist[h0], nv0);
            }
        }
    }
}
__global__ void calculate_histogram(cuFloatComplex *d_array, double *d_histogram, double *d_nhist, float *oc, int nx, int ny, int nz,
                                    float bin_size, float qcut, int num_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int npz = nz / 2 + 1;
    if (i < nx && j < ny && k < npz)
    { // Only consider the upper half in z-direction
        int nfx = (nx % 2 == 0) ? nx / 2 : nx / 2 + 1;
        int nfy = (ny % 2 == 0) ? ny / 2 : ny / 2 + 1;
        int nfz = (nz % 2 == 0) ? nz / 2 : nz / 2 + 1;

        int ia = (i < nfx) ? i : i - nx;
        int ja = (j < nfy) ? j : j - ny;
        int ka = (k < nfz) ? k : k - nz;
        int ib = i == 0 ? 0 : nx - i;
        int jb = j == 0 ? 0 : ny - j;
        float mw1, mw2, mw3, mw;
        mw1 = oc[XX * DIM + XX] * ia + oc[XX * DIM + YY] * ja + oc[XX * DIM + ZZ] * ka;
        mw1 = 2.0 * M_PI * mw1;
        mw2 = oc[YY * DIM + XX] * ia + oc[YY * DIM + YY] * ja + oc[YY * DIM + ZZ] * ka;
        mw2 = 2.0 * M_PI * mw2;
        mw3 = oc[ZZ * DIM + XX] * ia + oc[ZZ * DIM + YY] * ja + oc[ZZ * DIM + ZZ] * ka;
        mw3 = 2.0 * M_PI * mw3;
        mw = sqrtf(mw1 * mw1 + mw2 * mw2 + mw3 * mw3);

        if (mw > qcut)
            return;
        // Check for division by zero
        if (bin_size <= 0.0f)
            return;
        int h0 = static_cast<int>(mw / bin_size);
        // Check bounds: h0 must be non-negative and within num_bins
        if (h0 >= 0 && h0 < num_bins)
        {
            double v0;
            int idx = k + j * npz + i * npz * ny;
            int idbx = k + jb * npz + ib * npz * ny;
            if (i == 0 && j == 0 && k == 0)
            {
                v0 = d_array[idx].x;
                double nv0{1.0};
                // NOTE: These two atomicAdd operations are not atomic together
                // In practice, the race condition impact is minimal for histograms
                // For stricter consistency, consider using warp-level aggregation
                atomicAdd(&d_histogram[h0], v0);
                atomicAdd(&d_nhist[h0], nv0);
            }
            else
            {
                double nv0{2.0};
                v0 = d_array[idx].x + d_array[idbx].x;
                atomicAdd(&d_histogram[h0], v0);
                atomicAdd(&d_nhist[h0], nv0);
            }
        }
    }
}

/**
 * @brief Applies a modulus calculation to a grid of complex values.
 *
 * This kernel function calculates the modulus of each complex value in the input grid
 * and stores the result in the output grid.
 *
 * @param grid_q The input grid of complex values.
 * @param modX The modulus values for the x-dimension.
 * @param modY The modulus values for the y-dimension.
 * @param modZ The modulus values for the z-dimension.
 * @param numParticles The number of particles.
 * @param nnx The number of grid points in the x-dimension.
 * @param nny The number of grid points in the y-dimension.
 * @param nnz The number of grid points in the z-dimension.
 */
__global__ void modulusKernel(cuFloatComplex *grid_q, float *modX, float *modY, float *modZ,
                              int nnx, int nny, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int nnpz = nnz / 2 + 1;
    if (i < nnx && j < nny && k < nnpz)
    {
        int idx = k + j * nnpz + i * nnpz * nny;
        float bsp_i = modX[i];
        float bsp_j = modY[j];
        float bsp_k = modZ[k];
        float bsp_ijk = bsp_i * bsp_j * bsp_k;
        cuFloatComplex bsp = make_cuComplex(bsp_ijk, 0.0f);
        // grid_q[idx] = cuCmulf(cuConjf(grid_q[idx]), grid_q[idx]);
        // grid_q[idx] = cuCmulf(grid_q[idx], bsp);
        auto mod0 = cuCmulf(cuConjf(grid_q[idx]), grid_q[idx]);
        grid_q[idx] = cuCmulf(mod0, bsp);
    }
}
/**
 * @brief Performs scattering calculations on a grid of complex values.
 *
 * This kernel function calculates the scattering contribution for each grid point
 * based on the provided scattering factors and the grid of complex values.
 *
 * @param grid_q The input grid of complex values.
 * @param grid_oq The output grid of complex values.
 * @param oc The orientation coefficients.
 * @param Scatter The scattering factors.
 * @param nnx The number of grid points in the x-dimension.
 * @param nny The number of grid points in the y-dimension.
 * @param nnz The number of grid points in the z-dimension.
 */
__global__ void scatterKernel(cuFloatComplex *grid_q, cuFloatComplex *grid_oq, float *oc,
                              float *Scatter, int nnx, int nny, int nnz, float qcut, float *numParticles)
{

    // if (idx >= nx0 * ny0 * (nz0 / 2 + 1))
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int nfx = (nnx % 2 == 0) ? nnx / 2 : nnx / 2 + 1;
    int nfy = (nny % 2 == 0) ? nny / 2 : nny / 2 + 1;
    int nfz = (nnz % 2 == 0) ? nnz / 2 : nnz / 2 + 1;
    int nnpz = nnz / 2 + 1;
    if (i < nnx && j < nny && k < nnpz)
    {
        int idx = k + j * nnpz + i * nnpz * nny;

        opsfact ff;
        ff.allocate_device(Scatter);
        int ia = (i < nfx) ? i : i - nnx;
        int ja = (j < nfy) ? j : j - nny;
        int ka = (k < nfz) ? k : k - nnz;
        float mw1, mw2, mw3, mw;
        mw1 = oc[XX * DIM + XX] * ia + oc[XX * DIM + YY] * ja + oc[XX * DIM + ZZ] * ka;
        mw2 = oc[YY * DIM + XX] * ia + oc[YY * DIM + YY] * ja + oc[YY * DIM + ZZ] * ka;
        mw3 = oc[ZZ * DIM + XX] * ia + oc[ZZ * DIM + YY] * ja + oc[ZZ * DIM + ZZ] * ka;
        mw1 = 2.0 * M_PI * mw1;
        mw2 = 2.0 * M_PI * mw2;
        mw3 = 2.0 * M_PI * mw3;
        mw = sqrt(mw1 * mw1 + mw2 * mw2 + mw3 * mw3);
        if (mw > qcut)
            return;
        cuFloatComplex fq = make_cuComplex(ff(mw), 0.0f);
        cuFloatComplex mult = cuCmulf(fq, grid_q[idx]);
        grid_oq[idx] = cuCaddf(grid_oq[idx], mult);
        if (idx == 0)
        {
            numParticles[0] = grid_q[idx].x;
        }
    }
}
__global__ void zeroDensityKernel(float *d_grid, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int)size)
    {
        d_grid[idx] = 0.0f;
    }
}
__global__ void zeroDensityKernel(cuFloatComplex *d_grid, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int)size)
    {
        d_grid[idx] = make_cuComplex(0.0f, 0.0f);
    }
}
__global__ void gridSumKernel(cuFloatComplex *d_grid, size_t size, float *gridsum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        gridsum[0] = d_grid[0].x;
    }
}
__global__ void gridAddKernel(cuFloatComplex *grid_i, cuFloatComplex *grid_o, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int)size)
    {
        grid_o[idx].x += grid_i[idx].x;
        grid_o[idx].y += grid_i[idx].y;
    }
}

/**
 * @brief Computes the density contribution of each particle to the grid.
 *
 * This kernel function calculates the density contribution of each particle to the grid
 * using B-spline interpolation. It iterates over the grid points within the support
 * of the particle and adds the contribution to the corresponding grid points.
 *
 * @param xa The array of particle coordinates.
 * @param grid The grid to store the density contributions.
 * @param order The order of the B-spline interpolation.
 * @param numParticles The number of particles.
 * @param nx The number of grid points in the x-dimension.
 * @param ny The number of grid points in the y-dimension.
 * @param nz The number of grid points in the z-dimension.
 */
__global__ void rhoKernel(float *xa, float *grid, int order, int numParticles, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        Splines bsplineX;
        Splines bsplineY;
        Splines bsplineZ;

        int nx0 = static_cast<int>(nx);
        int ny0 = static_cast<int>(ny);
        int nz0 = static_cast<int>(nz);
        float x1, y1, z1, r1, s1, t1, gx, gy, gz;
        int mx, my, mz;

        x1 = xa[idx * DIM + XX];
        y1 = xa[idx * DIM + YY];
        z1 = xa[idx * DIM + ZZ];
        r1 = static_cast<float>(nx0 * (x1 - rint(x1 - 0.5)));
        s1 = static_cast<float>(ny0 * (y1 - rint(y1 - 0.5)));
        t1 = static_cast<float>(nz0 * (z1 - rint(z1 - 0.5)));
        mx = static_cast<int>(r1);
        my = static_cast<int>(s1);
        mz = static_cast<int>(t1);

        gx = r1 - static_cast<float>(mx);
        gy = s1 - static_cast<float>(my);
        gz = t1 - static_cast<float>(mz);
        spline splX = bsplineX(gx);
        spline splY = bsplineX(gy);
        spline splZ = bsplineX(gz);
        int i0 = mx - order;
        for (auto o = 0; o < order; o++)
        {
            int i = i0 + (nx0 - ((i0 >= 0) ? nx0 : -nx0)) / 2;

            int j0 = my - order;
            for (auto p = 0; p < order; p++)
            {
                int j = j0 + (ny0 - ((j0 >= 0) ? ny0 : -ny0)) / 2;

                int k0 = mz - order;
                for (auto q = 0; q < order; q++)
                {
                    int k = k0 + (nz0 - ((k0 >= 0) ? nz0 : -nz0)) / 2;
                    float fact_o = splX.x[o];
                    float fact_p = fact_o * splY.x[p];
                    float fact_q = fact_p * splZ.x[q];
                    int ig = k + j * nz0 + i * nz0 * ny0;
                    atomicAdd(&grid[ig], fact_q);
                    k0++;
                }
                j0++;
            }
            i0++;
        }
    }
}
/**
 * @brief CUDA kernel function to compute the density of particles on a 3D grid using B-spline interpolation.
 *
 * This kernel function takes the particle positions and orientations, and computes the density of the particles on a 3D grid using B-spline interpolation. The grid is represented as a 1D array, and the kernel function calculates the 1D index from the 3D coordinates of each grid point.
 *
 * @param xa Array of particle positions.
 * @param oc Array of particle orientations.
 * @param grid Pointer to the 1D array representing the 3D grid of floating-point values.
 * @param order The order of the B-spline interpolation.
 * @param numParticles The number of particles.
 * @param nx The size of the grid in the x-dimension.
 * @param ny The size of the grid in the y-dimension.
 * @param nz The size of the grid in the z-dimension.
 */
__global__ void rhoCartKernel(float *xa, float *oc, float *grid, int order, int numParticles, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        Splines bsplineX;
        Splines bsplineY;
        Splines bsplineZ;

        int nx0 = static_cast<int>(nx);
        int ny0 = static_cast<int>(ny);
        int nz0 = static_cast<int>(nz);
        float x0, y0, z0, x1, y1, z1, r1, s1, t1, gx, gy, gz;
        int mx, my, mz;
        x0 = xa[idx * DIM + XX];
        y0 = xa[idx * DIM + YY];
        z0 = xa[idx * DIM + ZZ];
        x1 = oc[XX * DIM + XX] * x0 + oc[XX * DIM + YY] * y0 + oc[XX * DIM + ZZ] * z0;
        y1 = oc[YY * DIM + XX] * x0 + oc[YY * DIM + YY] * y0 + oc[YY * DIM + ZZ] * z0;
        z1 = oc[ZZ * DIM + XX] * x0 + oc[ZZ * DIM + YY] * y0 + oc[ZZ * DIM + ZZ] * z0;

        r1 = static_cast<float>(nx0 * (x1 - rint(x1 - 0.5)));
        s1 = static_cast<float>(ny0 * (y1 - rint(y1 - 0.5)));
        t1 = static_cast<float>(nz0 * (z1 - rint(z1 - 0.5)));
        mx = static_cast<int>(r1);
        my = static_cast<int>(s1);
        mz = static_cast<int>(t1);

        gx = r1 - static_cast<float>(mx);
        gy = s1 - static_cast<float>(my);
        gz = t1 - static_cast<float>(mz);
        spline splX = bsplineX(gx);
        spline splY = bsplineX(gy);
        spline splZ = bsplineX(gz);
        int i0 = mx - order;
        for (auto o = 0; o < order; o++)
        {
            int i = i0 + (nx0 - ((i0 >= 0) ? nx0 : -nx0)) / 2;

            int j0 = my - order;
            for (auto p = 0; p < order; p++)
            {
                int j = j0 + (ny0 - ((j0 >= 0) ? ny0 : -ny0)) / 2;

                int k0 = mz - order;
                for (auto q = 0; q < order; q++)
                {
                    int k = k0 + (nz0 - ((k0 >= 0) ? nz0 : -nz0)) / 2;
                    float fact_o = splX.x[o];
                    float fact_p = fact_o * splY.x[p];
                    float fact_q = fact_p * splZ.x[q];
                    int ig = k + j * nz0 + i * nz0 * ny0;
                    atomicAdd(&grid[ig], fact_q);
                    k0++;
                }
                j0++;
            }
            i0++;
        }
    }
}
/**
 * @brief Kernel function to initialize a 3D grid with a given density value.
 *
 * This kernel function is used to initialize a 3D grid with a given density value. The grid is represented as a 1D array, and the kernel function calculates the 1D index from the 3D coordinates of each grid point.
 *
 * @param d_grid Pointer to the 1D array representing the 3D grid.
 * @param myDens The density value to be assigned to the grid.
 * @param nx The size of the grid in the x-dimension.
 * @param ny The size of the grid in the y-dimension.
 * @param nz The size of the grid in the z-dimension.
 * @param nnx The size of the super-sampled grid in the x-dimension.
 * @param nny The size of the super-sampled grid in the y-dimension.
 * @param nnz The size of the super-sampled grid in the z-dimension.
 */
__global__ void superDensityKernel(float *d_grid, float *d_gridSup, float myDens, int nx, int ny, int nz, int nnx, int nny, int nnz)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nnx && y < nny && z < nnz)
    {
        int idx_s = z + y * nnz + x * nnz * nny;
        d_gridSup[idx_s] = myDens;
        if (x < nx && y < ny && z < nz)
        {
            int idx = z + y * nz + x * nz * ny;
            d_gridSup[idx_s] = d_grid[idx];
        }
    }
}

/**
 * @brief Performs padding on a 3D grid, computing the average density and count of points on the border.
 *
 * This CUDA kernel function performs padding on a 3D grid, computing the average density and count of points on the border of the grid. The grid is represented as a 1D array, and the kernel function calculates the 1D index from the 3D coordinates of each grid point.
 *
 * @param grid Pointer to the 1D array representing the 3D grid of floating-point values.
 * @param nx The size of the grid in the x-dimension.
 * @param ny The size of the grid in the y-dimension.
 * @param nz The size of the grid in the z-dimension.
 * @param dx The padding size in the x-dimension.
 * @param dy The padding size in the y-dimension.
 * @param dz The padding size in the z-dimension.
 * @param Dens Pointer to a device-side float variable to store the total density of the border points.
 * @param count Pointer to a device-side integer variable to store the count of border points.
 */
__global__ void paddingKernel(float *grid, int nx, int ny, int nz, int dx, int dy, int dz, float *Dens, int *count)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // int mx = nx - dx;
    // int my = ny - dy;
    int mz = nz - dz;
    if (x < nx && y < ny && z < nz)
    {
        int idx = z + y * nz + x * nz * ny;
        //        bool cond1 = (x > dx && x < mx) && (y > dy && y < my) && (z > dz && z < mz);
        bool cond1 = (z > dz && z < mz);
        if (!cond1)
        {
            atomicAdd(&count[0], 1);
            atomicAdd(&Dens[0], grid[idx]);
        }
    }
}
// Custom kernel to complete the calculation
__global__ void completeScatterKernel(cuFloatComplex *grid_q, cuFloatComplex *grid_oq,
                                      float *mw_values, float *Scatter,
                                      int nnx, int nny, int nnz, float qcut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nnx * nny * (nnz / 2 + 1);
    if (idx < total)
    {
        float mw1 = mw_values[idx];
        float mw2 = mw_values[idx + total];
        float mw3 = mw_values[idx + 2 * total];
        float mw = sqrt(mw1 * mw1 + mw2 * mw2 + mw3 * mw3);

        if (mw <= qcut)
        {
            opsfact ff;
            ff.allocate_device(Scatter);
            cuFloatComplex fq = make_cuComplex(ff(mw), 0.0f);
            cuFloatComplex mult = cuCmulf(fq, grid_q[idx]);
            grid_oq[idx] = cuCaddf(grid_oq[idx], mult);
        }
    }
}
// Host function - DISABLED: Contains uninitialized memory bug
// This function allocates d_indices but never initializes it before use
// TODO: Implement proper initialization of d_indices or remove if unused
/*
void scatterCalculation(cuFloatComplex *grid_q, cuFloatComplex *grid_oq, float *oc,
                        float *Scatter, int nnx, int nny, int nnz, float qcut)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set up matrices for cublasSgemm
    int m = 3;               // Number of rows in oc
    int n = nnx * nny * nnz; // Total number of grid points
    int k = 3;               // Number of columns in oc

    float *d_indices; // Device array to store [ia, ja, ka] for all points
    cudaMalloc(&d_indices, 3 * n * sizeof(float));
    // BUG: d_indices is never initialized! This causes undefined behavior
    // Need to fill d_indices with appropriate values using a custom kernel

    float *d_result;
    cudaMalloc(&d_result, m * n * sizeof(float));

    float alpha = 2.0f * M_PI;
    float beta = 0.0f;

    // Perform matrix multiplication: d_result = alpha * oc * d_indices + beta * d_result
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, oc, m, d_indices, k, &beta, d_result, m);

    // Custom kernel to complete the calculation
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    completeScatterKernel<<<grid, block>>>(grid_q, grid_oq, d_result, Scatter, nnx, nny, nnz, qcut);

    // Clean up
    cudaFree(d_indices);
    cudaFree(d_result);
    cublasDestroy(handle);
}
*/
