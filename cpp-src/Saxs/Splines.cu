#include "Splines.h"

/// Initializes the spline with the given weight parameter `w`.
///
/// This function sets the initial values of the spline coefficients `theta.x`.
/// The first coefficient `theta.x[0]` is set to `1.0 - w`, the second coefficient
/// `theta.x[1]` is set to `w`, and the remaining coefficients from `theta.x[2]` to
/// `theta.x[order-1]` are set to 0.0.
///
/// @param w The weight parameter for the spline initialization.
__host__ __device__ void Splines::Init(const float w)
{
    // Initialize the spline
    for (int i = 2; i < order; ++i)
    {
        theta.x[i] = 0.0;
    }
    theta.x[1] = w;
    theta.x[0] = 1.0 - w;
}

/// Performs one pass of the spline calculation.
///
/// This function updates the spline coefficients `theta.x` based on the current weight parameter `w` and the current pass index `k1`.
/// The first coefficient `theta.x[k-1]` is calculated using the previous coefficient `theta.x[k-2]`. The remaining coefficients
/// from `theta.x[k-2]` to `theta.x[0]` are calculated using a formula that involves the current and previous coefficients, as well as the weight parameter `w` and the current pass index `k`.
///
/// @param w The weight parameter for the spline calculation.
/// @param k1 The current pass index.
__host__ __device__ void Splines::OnePass(const float w, const int k1)
{
    // One pass of the spline
    {
        int k = k1 + 1;
        float div = 1.0 / static_cast<float>(k - 1);

        theta.x[k - 1] = div * w * theta.x[k - 2];
        for (int j = 1; j <= k - 2; ++j)
        {
            theta.x[k - j - 1] = div * ((w + j) * theta.x[k - j - 2] + (k - j - w) * theta.x[k - j - 1]);
        }
        theta.x[0] = div * (1.0 - w) * theta.x[0];
    }
}
/// Calculates the derivatives of the spline coefficients.
///
/// This function computes the derivatives of the spline coefficients `theta.x` and stores them in `theta.dx`. The first derivative `theta.dx[0]` is set to the negative of the first coefficient `theta.x[0]`. The remaining derivatives `theta.dx[1]` to `theta.dx[order-1]` are calculated as the difference between the current and previous coefficients in `theta.x`.
__host__ __device__ void Splines::Diff()
{

    theta.dx[0] = -theta.x[0];
    for (int o = 1; o < order; ++o)
    {
        theta.dx[o] = theta.x[o - 1] - theta.x[o];
    }
}
/// Calculates a spline using the provided weight parameter `w`.
///
/// This function initializes the spline coefficients, performs multiple passes of the spline calculation, computes the derivatives, and performs a final pass before returning the resulting spline.
///
/// @param w The weight parameter for the spline calculation.
/// @return The resulting spline coefficients.
__host__ __device__ spline Splines::operator()(const float w)
{

    Init(w);
    for (int o = 2; o < order - 1; ++o)
    {
        OnePass(w, o);
    }
    Diff();
    OnePass(w, order - 1);
    return theta;
}

__host__ void Splines::allocate_host(float *d_x, float *d_dx)
{
    for (int i = 0; i < order; ++i)
    {
        theta.x[i] = d_x[i];
        theta.dx[i] = d_dx[i];
    }
}

__device__ void Splines::allocate_device(float *d_x, float *d_dx)
{
    for (int i = 0; i < order; ++i)
    {
        theta.x[i] = d_x[i];
        theta.dx[i] = d_dx[i];
    }
}