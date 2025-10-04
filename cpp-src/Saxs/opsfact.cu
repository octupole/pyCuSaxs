#include "opsfact.h"

/**
 * Allocates device memory and copies the contents of the provided `d_v0` array to the `v` array.
 *
 * @param d_v0 A pointer to an array of 9 float values to be copied to the `v` array.
 */
__device__ void opsfact::allocate_device(float *d_v0)
{
    for (int i = 0; i < 9; i++)
        v[i] = d_v0[i];
}
/**
 * Allocates device memory and copies the contents of the provided `d_v0` array to the `v` array.
 *
 * @param d_v0 A pointer to an array of 9 float values to be copied to the `v` array.
 */
__host__ __device__ void opsfact::allocate(float *d_v0)
{
    for (int i = 0; i < 9; i++)
        v[i] = d_v0[i];
}
/**
 * Calculates the small-angle X-ray scattering (SAXS) intensity for a given momentum transfer value `q1`.
 *
 * @param q1 The momentum transfer value in inverse Angstroms.
 * @return The calculated SAXS intensity.
 */
__host__ __device__ float opsfact::operator()(float q1)
{
    float q = q1 / (4.0 * M_PI);
    float q2 = q * q;
    float result = d * (v[0] * expf(-v[4] * q2) + v[1] * expf(-v[5] * q2) + v[2] * expf(-v[6] * q2) + v[3] * expf(-v[7] * q2) + v[8]);
    return result;
}
