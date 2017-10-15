/*
 * hls_remapping_function.h
 *
 *  Created on: 2017/10/11
 *      Author: kuriharat
 */

#ifndef HLS_REMAPPING_FUNCTION_H_
#define HLS_REMAPPING_FUNCTION_H_

#include <assert.h>

#include "hls_video.h"


class hlsRemappingFunction {
public:
	hlsRemappingFunction(double alpha, double beta) {
		alpha_ = alpha;
		beta_ = beta;
	}

	template<int ROWS, int COLS, int MAT_T>
	void Evaluate(
			hls::Mat<ROWS, COLS, MAT_T>& input, hls::Mat<ROWS, COLS, MAT_T>& output,
			hls::Scalar<HLS_MAT_CN(MAT_T), HLS_TNAME(MAT_T)>& reference, double sigma_r,
			int rows, int cols)
	{
		assert(rows <= ROWS);
		assert(cols <= COLS);

		cv::Vec<HLS_TNAME(MAT_T), HLS_MAT_CN(MAT_T)> tmp;
		hls::Scalar<HLS_MAT_CN(MAT_T), HLS_TNAME(MAT_T)> px1;
		hls::Scalar<HLS_MAT_CN(MAT_T), HLS_TNAME(MAT_T)> px2;

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				input >> px1;

				Evaluate<HLS_TNAME(MAT_T), HLS_MAT_CN(MAT_T)>(px1, reference, sigma_r, px2);

				output << px2;
			}
		}
	}


private:
	template<typename T, int CH>
	void Evaluate(const hls::Scalar<CH, T>& value,
		hls::Scalar<CH, T>& reference,
		double sigma_r,
		hls::Scalar<CH, T>& output)
	{
		hls::Scalar<CH, T> delta;
		for(int i = 0; i < CH; i++){
			delta.val[i] = value.val[i] - reference.val[i];
		}
		double mag = 0;
		for(int i = 0; i < CH; i++){
			mag += delta.val[i]*delta.val[i];
		}
		mag = std::sqrt(mag);

		if (mag > 1e-10){
			for(int i = 0; i < CH; i++){
				delta.val[i] /= mag;
			}
		}

		if (mag < sigma_r) {
			for(int i = 0; i < CH; i++){
				output.val[i] = reference.val[i] + delta.val[i] * sigma_r * DetailRemap(mag, sigma_r);
			}
		}
		else {
			for(int i = 0; i < CH; i++){
				output.val[i] = reference.val[i] + delta.val[i] * (EdgeRemap(mag - sigma_r) + sigma_r);
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
