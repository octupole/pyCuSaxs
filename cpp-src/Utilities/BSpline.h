/*
 * BSpline.h
 *
 *  Created on: Jun 26, 2015
 *      Author: marchi
 */

#ifndef SRC_BSPLINE_H_
#define SRC_BSPLINE_H_
#include <vector>

using std::vector;
namespace BSpline
{
	struct spline
	{
		vector<float> x;
		vector<float> dx;
	};
	class BSpline
	{
		static int order;
		spline theta;
		void allocate();
		void Init(const float);
		void OnePass(const float, const int);
		void Diff();

	public:
		BSpline();
		BSpline(int);
		virtual ~BSpline();
		spline &operator()(const float);
		static int Order() { return order; }
		static void SetOrder(int MyOrder) { order = MyOrder; }
	};
}
#endif /* SRC_BSPLINE_H_ */
