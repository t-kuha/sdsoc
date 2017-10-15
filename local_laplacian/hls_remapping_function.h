/*
 * hls_remapping_function.h
 *
 *  Created on: 2017/10/11
 *      Author: kuriharat
 */

#ifndef HLS_REMAPPING_FUNCTION_H_
#define HLS_REMAPPING_FUNCTION_H_

#include "opencv2/core/core.hpp"

#include "hls_video.h"


class hlsRemappingFunction {
public:
	hlsRemappingFunction(double alpha, double beta) {
		alpha_ = alpha;
		beta_ = beta;
	}

	template<typename T, int CH>
	void Evaluate(const cv::Mat& input, cv::Mat& output,
		hls::Scalar<CH, T>& reference, double sigma_r)
	{
		output.create(input.rows, input.cols, input.type());
		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++) {
				Evaluate(input.at< cv::Vec<T, CH> >(i, j), reference, sigma_r, output.at< cv::Vec<T, CH> >(i, j));
			}
		}
	}


private:
	template<typename T, int CH>
	void Evaluate(const cv::Vec<T, CH>& value,
		hls::Scalar<CH, T>& reference,
		double sigma_r,
		cv::Vec<T, CH>& output)
	{
		cv::Vec<T, CH> delta;
		for(int i = 0; i < CH; i++){
			delta[i] = value[i] - reference.val[i];
		}
		double mag = cv::norm(delta);
		if (mag > 1e-10) delta /= mag;

		if (mag < sigma_r) {
			for(int i = 0; i < CH; i++){
			output[i] = reference.val[i] + delta[i] * sigma_r * DetailRemap(mag, sigma_r);
			}
		}
		else {
			for(int i = 0; i < CH; i++){
			output[i] = reference.val[i] + delta[i] * (EdgeRemap(mag - sigma_r) + sigma_r);
			}
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
