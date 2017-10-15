/*
 * hls_filter.h
 *
 *  Created on: 2017/10/09
 *      Author: kuriharat
 */

#ifndef HLS_FILTER_H_
#define HLS_FILTER_H_

#include <iostream>
#include <assert.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "hls_laplacian_pyramid.h"
#include "opencv_utils.h"
#include "hls_remapping_function.h"

#include "hls_video.h"
#include "hls_util.h"
#include "hls_def.h"

template<typename T, int CH>
void accel_wrap(
	const cv::Mat& _output,
	const cv::Mat& gauss,
	const cv::Mat& input,
	const int l,
	const int subregion_r,
	const double sigma_r,
	hlsRemappingFunction &r);


template<typename T, int CH>
cv::Mat hlsLocalLaplacianFilter(const cv::Mat& input,
	double alpha,
	double beta,
	double sigma_r) {
	hlsRemappingFunction r(alpha, beta);

	int num_levels = LaplacianPyramid::GetLevelCount(input.rows, input.cols, 30);
	std::cout << "Number of levels: " << num_levels << std::endl;

	const int kRows = input.rows;
	const int kCols = input.cols;

	assert(kRows <= _MAX_IMG_ROWS_);
	assert(kCols <= _MAX_IMG_COLS_);

	GaussianPyramid gauss_input(input, num_levels);

	// Construct the unfilled Laplacian pyramid of the output. Copy the residual
	// over from the top of the Gaussian pyramid.
	LaplacianPyramid output(kRows, kCols, input.channels(), num_levels);
	gauss_input[num_levels].copyTo(output[num_levels]);

	// Calculate each level of the ouput Laplacian pyramid.
	for (int l = 0; l < num_levels; l++) {
		int subregion_size = 3 * ((1 << (l + 2)) - 1);
		int subregion_r = subregion_size / 2;

		// Show some info
		std::cout << "Layer" << l << ": " << std::endl;
		std::cout << "\t Input:  " << input.rows << " x " << input.cols << " x " << input.channels() << std::endl;
		std::cout << "\t Gauss:  " << gauss_input[l].rows << " x " << gauss_input[l].cols << " x " << gauss_input[l].channels() << std::endl;
		std::cout << "\t Output: " << output[l].rows << " x " << output[l].cols << " x " << output[l].channels() << std::endl;

		// HW-accelerated function
		accel_wrap<T, CH>(output[l], gauss_input[l], input,
			l, subregion_r, sigma_r, r);

		std::stringstream ss;
		ss << "hls_level" << l << ".png";
		cv::imwrite(ss.str(), ByteScale(cv::abs(output[l])));
		std::cout << std::endl;
	}

	return output.Reconstruct();
}


// HW function
template<typename T, int CH>
void accel(
	T* _output,
	T* _gauss,
	T* _input,
	const int l,
	const int subregion_r,
	const int rows_in,
	const int cols_in,
	const int rows_out,
	const int cols_out,
	const double sigma_r,
	hlsRemappingFunction &r)
{
	cv::Mat input(rows_in, cols_in, CV_MAKETYPE(CV_64F,CH) );
	memcpy(input.data, _input, rows_in*cols_in*CH*sizeof(T));

	// TODO: Apply DATAFLOW
	// TODO: Use hls::Mat
	hls::Mat<_MAX_IMG_ROWS_, _MAX_IMG_COLS_, HLS_MAKETYPE(HLS_64F, CH)> gauss(rows_out, cols_out);
	fb2hlsmat(_gauss, CH, gauss);

	hls::Scalar<CH, T> px_gauss;
	hls::Scalar<CH, T> px_out;

	for (int y = 0; y < rows_out; y++) {
		// Calculate the y-bounds of the region in the full-res image.
		int full_res_y = (1 << l) * y;
		int roi_y0 = full_res_y - subregion_r;
		int roi_y1 = full_res_y + subregion_r + 1;
		int row_start = std::max(0, roi_y0);
		int row_end = std::min(roi_y1, rows_in);

		int full_res_roi_y = full_res_y - row_start;

		for (int x = 0; x < cols_out; x++) {
			// Calculate the x-bounds of the region in the full-res image.
			int full_res_x = (1 << l) * x;
			int roi_x0 = full_res_x - subregion_r;
			int roi_x1 = full_res_x + subregion_r + 1;
			int col_start = std::max(0, roi_x0);
			int col_end = std::min(roi_x1, cols_in);

			int full_res_roi_x = full_res_x - col_start;

			// Remap the region around the current pixel.
			cv::Mat r0 = input({row_start, row_end}, {col_start, col_end});
			cv::Mat remapped;
			gauss >> px_gauss;
			r.Evaluate<T, CH>(r0, remapped, /*gauss.at< cv::Vec<T, CH> >(y, x)*/px_gauss, sigma_r);

			// Construct the Laplacian pyramid for the remapped region and copy the
			// coefficient over to the output Laplacian pyramid.
//			cv::Mat lap;
//			hlsLaplacianPyramid2(
//				remapped, lap, l + 1,
//				{ row_start, row_end - 1, col_start, col_end - 1 });
//
//			// Only the last one of laplacian pyramid is required
//			output.at< cv::Vec<T, CH> >(y, x) = lap.at< cv::Vec<T, CH> >(full_res_roi_y >> l, full_res_roi_x >> l);

			hlsLaplacianPyramid2<T, CH>(
				remapped, px_out, l + 1,
				{ row_start, row_end - 1, col_start, col_end - 1 },
				full_res_roi_y >> l, full_res_roi_x >> l);

			for(int i = 0; i < CH; i++){
				_output[(y*cols_out + x)*CH + i] = px_out.val[i];
			}
		}
		std::cout << "Level " << (l + 1) << " (" << rows_out << " x "
			<< cols_out << "), subregion: " << subregion_r << "x"
			<< subregion_r << " ... " << round(100.0 * y / rows_out)
			<< "%\r";
		std::cout.flush();
	}

}


// Wrapper for accel()
template<typename T, int CH>
void accel_wrap(
	const cv::Mat& output,
	const cv::Mat& gauss,
	const cv::Mat& input,
	const int l,
	const int subregion_r,
	const double sigma_r,
	hlsRemappingFunction &r)
{
	// Copy data to normal array
	T* _output = NULL;
	T* _gauss = NULL;
	T* _input = NULL;

	_input = new T [input.rows*input.cols*CH];
	_gauss = new T [gauss.rows*gauss.cols*CH];
	_output = new T [output.rows*output.cols*CH];

	memcpy(_input, input.data, input.rows*input.cols*CH*sizeof(T));
	memcpy(_gauss, gauss.data, gauss.rows*gauss.cols*CH*sizeof(T));

	accel<T, CH>(_output, _gauss, _input, l, subregion_r,
			input.rows, input.cols, output.rows, output.cols, sigma_r, r);

	// Copy back data
	memcpy(output.data, _output, output.rows*output.cols*CH*sizeof(T));

	if(_output){
		delete [] _output;
	}
	if(_gauss){
		delete [] _gauss;
	}
	if(_input){
		delete [] _input;
	}
}
#endif /* HLS_FILTER_H_ */
