#ifndef _HLS_FAST_LOCAL_LAPLACIAN_
#define _HLS_FAST_LOCAL_LAPLACIAN_


#include "opencv2/core/core.hpp"		// cv::Mat

#include "hls_video.h"
#include "hls_def.h"


void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact);

#pragma SDS data access_pattern(gau0:SEQUENTIAL)
#pragma SDS data access_pattern(gau1:SEQUENTIAL)
#pragma SDS data access_pattern(gau2:SEQUENTIAL)
#pragma SDS data access_pattern(gau3:SEQUENTIAL)
#pragma SDS data access_pattern(lap0:SEQUENTIAL)
#pragma SDS data access_pattern(lap1:SEQUENTIAL)
#pragma SDS data access_pattern(lap2:SEQUENTIAL)
#pragma SDS data access_pattern(lap3:SEQUENTIAL)
//#pragma SDS data access_pattern(dst0:SEQUENTIAL)
//#pragma SDS data access_pattern(dst1:SEQUENTIAL)
//#pragma SDS data access_pattern(dst2:SEQUENTIAL)
//#pragma SDS data access_pattern(dst3:SEQUENTIAL)
#pragma SDS data copy(gau0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(gau1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(gau2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(gau3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data copy(lap0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(lap1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(lap2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(lap3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data zero_copy(dst0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data zero_copy(dst1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data zero_copy(dst2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data zero_copy(dst3[0:"pyr_rows_[3]*pyr_cols_[3]"])
void hls_local_laplacian(
		float* gau0, float* gau1, float* gau2, float* gau3,
		float* lap0, float* lap1, float* lap2, float* lap3,
		float* dst0, float* dst1, float* dst2, float* dst3,
		int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_],
		float ref);

#pragma SDS data access_pattern(src:SEQUENTIAL)
#pragma SDS data access_pattern(dst1:SEQUENTIAL)
#pragma SDS data access_pattern(dst2:SEQUENTIAL)
#pragma SDS data access_pattern(dst3:SEQUENTIAL)
#pragma SDS data copy(src[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(dst2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(dst3[0:"pyr_rows_[3]*pyr_cols_[3]"])
void hls_gaussian_pyramid(
		float* src,
		float* dst1, float* dst2, float* dst3,
		int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);

#pragma SDS data access_pattern(src:SEQUENTIAL)
#pragma SDS data access_pattern(dst0:SEQUENTIAL)
#pragma SDS data access_pattern(dst1:SEQUENTIAL)
#pragma SDS data access_pattern(dst2:SEQUENTIAL)
#pragma SDS data access_pattern(dst3:SEQUENTIAL)
#pragma SDS data copy(src[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(dst2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(dst3[0:"pyr_rows_[3]*pyr_cols_[3]"])
void hls_laplacian_pyramid(
	float* src,
	float* dst0, float* dst1, float* dst2, float* dst3,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);


#pragma SDS data access_pattern(src0:SEQUENTIAL)
#pragma SDS data access_pattern(src1:SEQUENTIAL)
#pragma SDS data access_pattern(src2:SEQUENTIAL)
#pragma SDS data access_pattern(src3:SEQUENTIAL)
#pragma SDS data access_pattern(dst:SEQUENTIAL)
#pragma SDS data copy(src0[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data copy(src1[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(src2[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(src3[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst[0:"pyr_rows_[0]*pyr_cols_[0]"])
void hls_reconstruct(float* src0, float* src1, float* src2, float* src3, 
	data_out_t* dst, int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_]);

void remap(float* src, float* dst, float ref, float fact, float sigma, int rows, int cols);


void my_split(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst1,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst2);

void downsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst);

void upsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst);

void add(
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src1,
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src2,
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst);

void load(float* src, hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst);

void save(hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src, float* dst);

#endif
