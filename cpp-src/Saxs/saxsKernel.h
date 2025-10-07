/**
 * @class saxsKernel
 * @brief Provides functionality for performing small-angle X-ray scattering (SAXS) calculations.
 *
 * The `saxsKernel` class is responsible for managing the memory and computation required for SAXS
 * calculations. It provides methods for setting the number of particles and grid dimensions, as well
 * as running the main SAXS kernel. The class also manages the allocation and deallocation of
 * various device memory buffers used in the SAXS computations.
 */
#ifndef SAXSKERNEL_H
#define SAXSKERNEL_H
#include "Splines.h"
#include "Options.h"
#include <vector>
#include <cufft.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <map>
#include "Ftypedefs.h"
#include <fmt/core.h>

#pragma once

class saxsKernel
{
public:
    saxsKernel(int _nx, int _ny, int _nz, int _order) : nx{_nx}, ny{_ny}, nz{_nz}, order(_order), cufftPlan(0) {};
    void setnpx(int _npx, int _npy, int _npz)
    {
        npx = _npx;
        npy = _npy;
        npz = _npz;
    }
    void setnpx(int _npx)
    {
        npx = _npx;
        npy = _npx;
        npz = _npx;
    }
    void runPKernel(int, float, std::vector<std::vector<float>> &, std::map<std::string, std::vector<int>> &, std::vector<std::vector<float>> &);
    double getCudaTime() { return cudaTime / cudaCalls; }
    void scaledCell();
    void zeroIq();
    void getHistogram(std::vector<std::vector<float>> &);
    std::vector<std::vector<double>> getSaxs(double &);
    void createMemory();
    void resetHistogramParameters(std::vector<std::vector<float>> &);
    void writeBanner();
    void setcufftPlan(int nnx, int nny, int nnz)
    {
        cufftPlan3d(&cufftPlan, nnx, nny, nnz, CUFFT_R2C);
    }
    cufftHandle &getPlan() { return cufftPlan; }

    ~saxsKernel();

private:
    int size;
    int order;
    int npx, npy, npz;
    int nx, ny, nz, nnx, nny, nnz;
    int numParticles;
    float sigma;
    float bin_size;
    float kcut;
    float dk;
    int num_bins;
    double cudaTime{0};
    double cudaCalls{0};
    static int frame_count;
    cufftHandle cufftPlan;

    thrust::device_vector<float> d_moduleX;
    thrust::device_vector<float> d_moduleY;
    thrust::device_vector<float> d_moduleZ;
    thrust::device_vector<float> d_grid;
    thrust::device_vector<float> d_gridSup;
    thrust::device_vector<cuFloatComplex> d_gridSupAcc;
    thrust::device_vector<cuFloatComplex> d_Iq;
    thrust::device_vector<cuFloatComplex> d_gridSupC;
    thrust::device_vector<double> d_histogram;
    thrust::device_vector<double> d_nhist;

    thrust::host_vector<float> h_moduleX;
    thrust::host_vector<float> h_moduleY;
    thrust::host_vector<float> h_moduleZ;
    thrust::host_vector<double> h_histogram;
    thrust::host_vector<double> h_nhist;

    float *d_grid_ptr{nullptr};
    float *d_gridSup_ptr{nullptr};
    cuFloatComplex *d_gridSupC_ptr{nullptr};
    cuFloatComplex *d_gridSupAcc_ptr{nullptr};
    cuFloatComplex *d_Iq_ptr{nullptr};
    // Do bspmod
    float *d_moduleX_ptr{nullptr};
    float *d_moduleY_ptr{nullptr};
    float *d_moduleZ_ptr{nullptr};
    double *d_histogram_ptr{nullptr};
    double *d_nhist_ptr{nullptr};
    std::function<int(int, float)> borderBins = [](int nx, float shell) -> int
    {
        return static_cast<int>(shell * nx / 2);
    };

    std::vector<long long> generateMultiples(long long limit);
    long long findClosestProduct(int n, float sigma);
};

#endif