/*
 * BSpline.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: marchi
 */

#include "BSpline.h"
namespace BSpline
{

	int BSpline::order = 4;

	void BSpline::allocate()
	{
		theta.x = vector<float>(order);
		theta.dx = vector<float>(order);
	}
	/// Evaluates a B-spline curve at the given parameter value `w`.
	///
	/// This function initializes the B-spline coefficients, performs multiple passes to compute the B-spline values, computes the derivatives, and performs a final pass to compute the B-spline values at the given parameter value `w`.
	///
	/// @param w The parameter value at which to evaluate the B-spline curve.
	/// @return The B-spline curve evaluated at the given parameter value `w`.

	spline &BSpline::operator()(const float w)
	{
		Init(w);
		for (auto o = 2; o < order - 1; o++)
			OnePass(w, o);
		Diff();
		OnePass(w, order - 1);
		return theta;
	}

	/// Initializes the B-spline coefficients for a given parameter value `w`.
	///
	/// This function sets the initial values of the B-spline coefficients `theta.x` based on the given parameter value `w`. The first coefficient `theta.x[0]` is set to `1.0 - w`, the second coefficient `theta.x[1]` is set to `w`, and the remaining coefficients `theta.x[2]` through `theta.x[order-1]` are set to 0.0.
	///
	/// @param w The parameter value at which to initialize the B-spline coefficients.
	void BSpline::Init(const float w)
	{
		for (auto i = 2; i < order; i++)
			theta.x[i] = 0.0;
		theta.x[1] = w;
		theta.x[0] = 1.0 - w;
	};
	/// Performs a single pass of the B-spline computation for a given parameter value `w` and pass index `k1`.
	///
	/// This function computes the B-spline coefficients `theta.x` for the current pass of the B-spline evaluation. It updates the coefficients based on the previous pass's coefficients and the current parameter value `w`. The function handles the first and last coefficients separately, and updates the intermediate coefficients using a linear combination of the previous coefficients.
	///
	/// @param w The parameter value at which to perform the B-spline computation.
	/// @param k1 The index of the current pass of the B-spline computation.
	void BSpline::OnePass(const float w, const int k1)
	{
		int k = k1 + 1;
		float div = 1.0 / static_cast<float>(k - 1);
		theta.x[k - 1] = div * w * theta.x[k - 2];
		for (auto j = 1; j <= k - 2; j++)
			theta.x[k - j - 1] = div * ((w + j) * theta.x[k - j - 2] + (k - j - w) * theta.x[k - j - 1]);
		theta.x[0] = div * (1.0 - w) * theta.x[0];

		/*
			float div;
			int K1=k+1;
			div=1.0/ static_cast<float>(K1-1);
			theta.x[k]=div*w*theta.x[k-1];
			for(auto o=0;o<k-2;o++){
				theta.x[k-o-1]=div*((w+o+1)*theta.x[k-o-2]+(k-o-w)*theta.x[k-o-1]);
			}
			theta.x[0]*=div*(1.0-w)*theta.x[0];
			*/
	}
	/// Computes the derivatives of the B-spline coefficients.
	///
	/// This function computes the derivatives of the B-spline coefficients `theta.x` and stores them in `theta.dx`. The first derivative `theta.dx[0]` is set to the negative of the first coefficient `theta.x[0]`. The remaining derivatives `theta.dx[1]` through `theta.dx[order-1]` are computed as the difference between the current and previous coefficients in `theta.x`.
	void BSpline::Diff()
	{
		theta.dx[0] = -theta.x[0];
		for (auto o = 1; o < order; o++)
			theta.dx[o] = theta.x[o - 1] - theta.x[o];
	}

	BSpline::BSpline()
	{
		// TODO Auto-generated constructor stub
		allocate();
	}

	BSpline::BSpline(int myorder)
	{
		// TODO Auto-generated constructor stub
		order = myorder;
		allocate();
	}

	BSpline::~BSpline()
	{
		// TODO Auto-generated destructor stub
	}
}