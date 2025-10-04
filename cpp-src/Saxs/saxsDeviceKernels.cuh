#ifndef SAXSDEVICEKERNEL_H
#define SAXSDEVICEKERNEL_H
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
void scatterCalculation(cuFloatComplex *grid_q, cuFloatComplex *grid_oq, float *oc,
                        float *Scatter, int nnx, int nny, int nnz, float qcut);
__global__ void calculate_histogram(cuFloatComplex *d_array, double *d_histogram, double *nhist, float *oc, int nx, int ny, int nz,
                                    float bin_size, float Qcut, int num_bins, float fact);
__global__ void calculate_histogram(cuFloatComplex *d_array, double *d_histogram, double *nhist, float *oc, int nx, int ny, int nz,
                                    float bin_size, float Qcut, int num_bins);

__global__ void modulusKernel(cuFloatComplex *grid_q, float *modX, float *modY, float *modZ,
                              int nnx, int nny, int nnz);

__global__ void scatterKernel(cuFloatComplex *grid_q, cuFloatComplex *grid_oq, float *oc,
                              float *Scatter, int nnx, int nny, int nnz, float Qcut, float *numParticles);
__global__ void rhoKernel(float *xa, float *grid, int order,
                          int numParticles, int nx, int ny, int nz);
__global__ void rhoCartKernel(float *xa, float *oc, float *grid, int order,
                              int numParticles, int nx, int ny, int nz);
__global__ void superDensityKernel(float *d_grid, float *d_gridSup, float myDens,
                                   int nx, int ny, int nz, int nnx, int nny, int nnz);
__global__ void zeroDensityKernel(float *d_grid, size_t size);
__global__ void zeroDensityKernel(cuFloatComplex *d_grid, size_t size);
__global__ void gridSumKernel(cuFloatComplex *, size_t size, float *);
__global__ void paddingKernel(float *grid, int nx, int ny, int nz, int dx, int dy, int dz, float *Dens, int *count);
__global__ void gridAddKernel(cuFloatComplex *grid_i, cuFloatComplex *grid_o, size_t size);
__global__ void completeScatterKernel(cuFloatComplex *grid_q, cuFloatComplex *grid_oq,
                                      float *mw_values, float *Scatter,
                                      int nnx, int nny, int nnz, float qcut);
#endif