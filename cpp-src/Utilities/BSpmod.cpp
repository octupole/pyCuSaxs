/*
 * BSpmod.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: marchi
 */

#include "BSpmod.h"
namespace BSpline
{
	/// Declares static member variables for the BSpmod class, including the dimensions of the spline
	/// coefficients (nx, ny, nz) and the total number of dimensions (ndim). Also declares the spline
	/// order (order) and some mathematical constants (twopi, tiny).
	size_t BSpmod::nx = 0;
	size_t BSpmod::ny = 0;
	size_t BSpmod::nz = 0;
	size_t BSpmod::ndim = 0;
	size_t BSpmod::order = 0;

	const float twopi = M_PI * 2.0;
	const float tiny = 1.0e-7;
	/// Computes the inverse of the values in the x, y, and z components of the BSp struct.
	/// This is used to invert the results of the DFTmod function, which computes the modulus
	/// of the Discrete Fourier Transform of the input vector.
	void BSpmod::Inverse()
	{
		for (auto &v : BSp.x)
		{
			v = 1.0 / v;
		}
		for (auto &v : BSp.y)
		{
			v = 1.0 / v;
		}
		for (auto &v : BSp.z)
		{
			v = 1.0 / v;
		}
	}
	/// Loads the moduli values for the x, y, and z components of the BSp struct.
	/// This function computes the modulus of the Discrete Fourier Transform of the
	/// input vector A0, which contains the spline coefficients. The Gamma function
	/// is then applied to the moduli values, and the inverse of the moduli values
	/// is computed and stored in the BSp struct.
	void BSpmod::load_moduli()
	{
		BSpline tmp;
		float w = 0.0;
		spline d = BSpline{}(w);

		vector<float> A0(ndim, 0.0);
		for (auto o = 1; o < order + 1; o++)
			A0[o] = d.x[o - 1];
		BSp.x = DFTmod(A0, nx);
		Gamma(BSp.x);
		BSp.y = DFTmod(A0, ny);
		Gamma(BSp.y);
		BSp.z = DFTmod(A0, nz);
		Gamma(BSp.z);
		Inverse();
	}
	/// Computes the modulus of the Discrete Fourier Transform (DFT) of the input vector A.
	/// The modulus is computed by taking the square root of the sum of the squares of the
	/// real and imaginary parts of the DFT. The resulting modulus values are stored in
	/// the output vector bsp.
	///
	/// @param A The input vector for which the DFT modulus is to be computed.
	/// @param Ndim The size of the input vector A.
	/// @return The vector of DFT modulus values.
	vector<float> BSpmod::DFTmod(const vector<float> &A, size_t Ndim)
	{
		vector<float> bsp(Ndim, 0.0);
		for (auto o = 0; o < Ndim; o++)
		{
			float sum1 = 0.0, sum2 = 0.0;
			for (auto p = 0; p < order + 1; p++)
			{
				float arg = twopi * static_cast<float>(o * p) / static_cast<float>(Ndim);
				sum1 += A[p] * cos(arg);
				sum2 += A[p] * sin(arg);
			}
			bsp[o] = sum1 * sum1 + sum2 * sum2;
		}
		for (auto o = 1; o < Ndim - 1; o++)
			bsp[o] = bsp[o] > tiny ? bsp[o] : 0.5 * (bsp[o - 1] + bsp[o + 1]);
		return bsp;
	}

	/// Applies the Gamma function to the moduli values in the input vector bsp.
	/// The Gamma function is computed as a sum of terms involving the ratio of the
	/// input value to the sum or difference of the input value and multiples of pi.
	/// The resulting Gamma-transformed moduli values are stored back in the input
	/// vector bsp.
	///
	/// @param bsp The vector of moduli values to which the Gamma function is applied.
	void BSpmod::Gamma(vector<float> &bsp)
	{

		size_t Ndim = bsp.size();
		size_t nf = Ndim / 2;

		auto GammaSum = [&Ndim](const int m, const int order) -> float
		{
			if (!m)
				return 1.0;
			float gsum = 1.0;
			float x = M_PI * static_cast<float>(m) / static_cast<float>(Ndim);
			for (auto k = 1; k <= KCUT; k++)
			{
				gsum += pow(x / (x + M_PI * k), order) + pow(x / (x - M_PI * k), order);
			}
			return gsum;
		};

		int order2 = 2 * order;
		for (auto k = 0; k < Ndim; k++)
		{
			float lambda = 1.0;
			int m = k < nf ? k : k - Ndim;
			if (m != 0)
			{
				float gsum = GammaSum(m, order);
				float gsum2 = GammaSum(m, order2);
				lambda = gsum / gsum2;
			}

			bsp[k] /= (lambda * lambda);
		}
	}

	BSpmod::~BSpmod()
	{
		// TODO Auto-generated destructor stub
	}
}