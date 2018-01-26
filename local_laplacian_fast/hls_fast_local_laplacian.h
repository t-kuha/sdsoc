#ifndef _HLS_FAST_LOCAL_LAPLACIAN_
#define _HLS_FAST_LOCAL_LAPLACIAN_


#include "opencv2/core/core.hpp"		// cv::Mat

#include "hls_video.h"

#include "hls_def.h"


void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact);

#pragma SDS data access_pattern(gau0:SEQUENTIAL, gau1:SEQUENTIAL, gau2:SEQUENTIAL)
#pragma SDS data access_pattern(lap0:SEQUENTIAL, lap1:SEQUENTIAL, lap2:SEQUENTIAL)
#pragma SDS data copy(gau0[0:"pyr_rows_[0]*pyr_cols_[0]"], gau1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(gau2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(lap0[0:"pyr_rows_[0]*pyr_cols_[0]"], lap1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(lap2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data zero_copy(dst0[0:"pyr_rows_[0]*pyr_cols_[0]"], dst1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data zero_copy(dst2[0:"pyr_rows_[2]*pyr_cols_[2]"])
void hls_local_laplacian(
		data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2,
		data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2,
		data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2,
		pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
		int step);

void hls_gaussian_pyramid(
	data_in_t* src,
	data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);

void hls_laplacian_pyramid(
	data_in_t* src,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);

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
void hls_laplacian_pyramid_remap(
    data_in_t* src,
    data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2,
    pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
	int step, float fact, float sigma2);

#pragma SDS data access_pattern(src0:SEQUENTIAL, src1:SEQUENTIAL)
#pragma SDS data access_pattern(src2:SEQUENTIAL, src3:SEQUENTIAL)
#pragma SDS data access_pattern(dst:SEQUENTIAL)
#pragma SDS data mem_attribute(src0:PHYSICAL_CONTIGUOUS, src1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(src2:PHYSICAL_CONTIGUOUS, src3:PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(src0[0:"pyr_rows_[3]*pyr_cols_[3]"], src1[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(src2[0:"pyr_rows_[1]*pyr_cols_[1]"], src3[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst[0:"pyr_rows_[0]*pyr_cols_[0]"])
void hls_reconstruct(data_pyr_t* src0, data_pyr_t* src1, data_pyr_t* src2, data_pyr_t* src3,
	data_out_t* dst, pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);

#pragma SDS data access_pattern(src:SEQUENTIAL)
#pragma SDS data access_pattern(gau0:SEQUENTIAL, gau1:SEQUENTIAL)
#pragma SDS data access_pattern(gau2:SEQUENTIAL, gau3:SEQUENTIAL)
#pragma SDS data access_pattern(lap0:SEQUENTIAL, lap1:SEQUENTIAL)
#pragma SDS data access_pattern(lap2:SEQUENTIAL, lap3:SEQUENTIAL)
#pragma SDS data copy(src[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(gau0[0:"pyr_rows_[0]*pyr_cols_[0]"], gau1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(gau2[0:"pyr_rows_[2]*pyr_cols_[2]"], gau3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data copy(lap0[0:"pyr_rows_[0]*pyr_cols_[0]"], lap1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(lap2[0:"pyr_rows_[2]*pyr_cols_[2]"], lap3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data mem_attribute(gau0:PHYSICAL_CONTIGUOUS, gau1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(gau2:PHYSICAL_CONTIGUOUS, gau3:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(lap0:PHYSICAL_CONTIGUOUS, lap1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(lap2:PHYSICAL_CONTIGUOUS, lap3:PHYSICAL_CONTIGUOUS)
void hls_construct_pyramid(
	data_in_t* src,
    data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2,
    data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2, data_pyr_t* lap3,
    pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);


// For debugging
bool hls_save_img(std::string name, data_pyr_t* img, int rows, int cols);
void hls_show_img(data_pyr_t* img, int rows, int cols, int delay = 0, std::string winname = "img");
void hls_print_value(data_pyr_t* img, int rows, int cols, std::string name = "");

#endif
