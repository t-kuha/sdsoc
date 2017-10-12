/*
 * hls_laplacian_pyramid.h
 *
 *  Created on: 2017/10/11
 *      Author: kuriharat
 */

#ifndef HLS_LAPLACIAN_PYRAMID_H_
#define HLS_LAPLACIAN_PYRAMID_H_

#include "opencv2/core/core.hpp"

#include "gaussian_pyramid.h"


void hlsLaplacianPyramid2(const cv::Mat& input, cv::Mat& output, int num_levels,
	const std::vector<int>& subwindow)
{
	GaussianPyramid gauss_pyramid(input, num_levels, subwindow);

	cv::Mat _out;
	for (int i = 0; i < num_levels; i++) {
		_out = gauss_pyramid[i] - gauss_pyramid.Expand(i + 1, 1);
	}
	output = _out;
}



#endif /* HLS_LAPLACIAN_PYRAMID_H_ */
