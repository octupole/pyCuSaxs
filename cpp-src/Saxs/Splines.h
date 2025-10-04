#ifndef SPLINES_H
#define SPLINES_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Parameters.h"
struct spline
{
    float x[MAX_ORDER];
    float dx[MAX_ORDER];
};

class Splines
{
public:
    __host__ __device__ Splines(int ord = 4) : order(ord)
    {
    }

    __host__ __device__ spline operator()(const float w);

    __host__ void allocate_host(float *d_x, float *d_dx);

    __device__ void allocate_device(float *d_x, float *d_dx);
    __host__ __device__ ~Splines(){};

private:
    __host__ __device__ void Init(const float w);
    __host__ __device__ void OnePass(const float w, const int k);
    __host__ __device__ void Diff();

    int order{4};
    spline theta;
};

#endif
