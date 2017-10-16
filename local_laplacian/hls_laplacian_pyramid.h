/*
 * hls_laplacian_pyramid.h
 *
 *  Created on: 2017/10/11
 *      Author: kuriharat
 */

#ifndef HLS_LAPLACIAN_PYRAMID_H_
#define HLS_LAPLACIAN_PYRAMID_H_

#include "opencv2/core/core.hpp"

#include "hls_gaussian_pyramid.h"
#include "hls_video.h"

template<typename T, int CH>
void hlsLaplacianPyramid2(const cv::Mat& input, hls::Scalar<CH, T>& px_output, int num_levels,
	const std::vector<int>& subwindow, int r, int c)
{
	hlsGaussianPyramid<T, CH> gauss_pyramid(subwindow);
	gauss_pyramid.construct(input, num_levels/*, subwindow*/);

	cv::Mat output = gauss_pyramid[num_levels - 1] - gauss_pyramid.ExpandOnce(num_levels);
	cv::Vec<T, CH> px = output.at< cv::Vec<T, CH> >(r, c);

	for(int i = 0; i < CH; i++){
		px_output.val[i] = px[i];
	}
}

#endif /* HLS_LAPLACIAN_PYRAMID_H_ */
