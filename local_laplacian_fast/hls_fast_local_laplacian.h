#ifndef _HLS_FAST_LOCAL_LAPLACIAN_
#define _HLS_FAST_LOCAL_LAPLACIAN_


#include "opencv2/core/core.hpp"		// cv::Mat

#include "hls_def.h"


void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact);

void hls_local_laplacian(
		float* gau0, float* gau1, float* gau2, float* gau3,
		float* lap0, float* lap1, float* lap2, float* lap3,
		float* dst0, float* dst1, float* dst2, float* dst3,
		int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_],
		int num_levels, float ref);

void gaussian_pyramid(float* src, float* dst1, float* dst2, float* dst4, int num_levels,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);

void laplacian_pyramid(
	float* src,
	float* dst0, float* dst1, float* dst2, float* dst3, int num_levels,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);

void reconstruct(float* src0, float* src1, float* src2, float* src3, 
	data_out_t* dst, int num_levels, int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);

void remap(float* src, float* dst, float ref, float fact, float sigma, int rows, int cols);


#endif
