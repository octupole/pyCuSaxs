#ifndef OPSFACT_H
#define OPSFACT_H
#include <stdio.h>
#include <cuda_runtime.h> // Include CUDA runtime header

#pragma once
struct opsfact
{
    __device__ void allocate_device(float *);
    __host__ __device__ void allocate(float *);
    __host__ __device__ float operator()(float);

    // q is in A^{-1} and q1=4*Pi*q0  q1 and q0 are in nm!!
private:
    float v[9];
    float d{1};
};
#endif