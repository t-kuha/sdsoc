/*
 * hls_remapping_function.h
 *
 *  Created on: 2017/10/11
 *      Author: kuriharat
 */

#ifndef HLS_REMAPPING_FUNCTION_H_
#define HLS_REMAPPING_FUNCTION_H_

#include "opencv2/core/core.hpp"

class hlsRemappingFunction {
public:
	hlsRemappingFunction(double alpha, double beta) {
		alpha_ = alpha;
		beta_ = beta;
	}

	template<typename T>
	void Evaluate(const cv::Mat& input, cv::Mat& output,
		const T& reference, double sigma_r) 
	{
		output.create(input.rows, input.cols, input.type());
		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++) {
				Evaluate(input.at<T>(i, j), reference, sigma_r, output.at<T>(i, j));
			}
		}
	}

private:
	void Evaluate(double value,
		double reference,
		double sigma_r,
		double& output) 
	{
		double delta = std::abs(value - reference);
		int sign = value < reference ? -1 : 1;

		if (delta < sigma_r) {
			output = reference + sign * sigma_r * DetailRemap(delta, sigma_r);
		}
		else {
			output = reference + sign * (EdgeRemap(delta - sigma_r) + sigma_r);
		}
	}

	void Evaluate(const cv::Vec3d& value,
		const cv::Vec3d& reference,
		double sigma_r,
		cv::Vec3d& output) 
	{
		cv::Vec3d delta = value - reference;
		double mag = cv::norm(delta);
		if (mag > 1e-10) delta /= mag;

		if (mag < sigma_r) {
			output = reference + delta * sigma_r * DetailRemap(mag, sigma_r);
		}
		else {
			output = reference + delta * (EdgeRemap(mag - sigma_r) + sigma_r);
		}
	}


	double SmoothStep(double x_min, double x_max, double x) 
	{
		double y = (x - x_min) / (x_max - x_min);
		y = std::max(0.0, std::min(1.0, y));
		return pow(y, 2) * pow(y - 2, 2);
	}

	/*inline*/ double DetailRemap(double delta, double sigma_r) 
	{
		double fraction = delta / sigma_r;
		double polynomial = pow(fraction, alpha_);
		if (alpha_ < 1) {
			const double kNoiseLevel = 0.01;
			double blend = SmoothStep(kNoiseLevel,
				2 * kNoiseLevel, fraction * sigma_r);
			polynomial = blend * polynomial + (1 - blend) * fraction;
		}
		return polynomial;
	}

	/*inline*/ double EdgeRemap(double delta) 
	{
		return beta_ * delta;
	}


	double alpha_, beta_;
};



#endif /* HLS_REMAPPING_FUNCTION_H_ */
