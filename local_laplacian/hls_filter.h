/*
 * hls_filter.h
 *
 *  Created on: 2017/10/09
 *      Author: kuriharat
 */

#ifndef HLS_FILTER_H_
#define HLS_FILTER_H_

//class hls_filter {
//public:
//	hls_filter();
//	virtual ~hls_filter();
//};

#include "opencv2/core/core.hpp"

template<typename T>
cv::Mat hls_laplacian(const cv::Mat& input,
                             double alpha,
                             double beta,
                             double sigma_r);

#endif /* HLS_FILTER_H_ */
