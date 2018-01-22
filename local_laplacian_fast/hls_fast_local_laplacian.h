#ifndef _HLS_FAST_LOCAL_LAPLACIAN_
#define _HLS_FAST_LOCAL_LAPLACIAN_


#include "opencv2/core/core.hpp"		// cv::Mat

#ifdef _WIN32

#include "ap_int.h"
//#include "ap_fixed.h"

#include "hls_stream.h"

#include "hls/utils/x_hls_utils.h"
#include "hls/utils/x_hls_traits.h"
#include "hls/utils/x_hls_defines.h"

#include "hls/hls_video_types.h"
#include "hls/hls_video_mem.h"
#include "hls/hls_video_core.h"
#include "hls/hls_video_imgbase.h"
#include "hls/hls_video_io.h"

//#include "hls_math.h"

#define ___HLS__VIDEO__
#include "hls/hls_video_imgproc.h"

#else

#include "hls_video.h"

#endif

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
		data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2,
		data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2,
		data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2,
		pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
		float ref);

#pragma SDS data access_pattern(src:SEQUENTIAL)
#pragma SDS data access_pattern(dst1:SEQUENTIAL)
#pragma SDS data access_pattern(dst2:SEQUENTIAL)
#pragma SDS data access_pattern(dst3:SEQUENTIAL)
#pragma SDS data copy(src[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(dst1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(dst2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(dst3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data mem_attribute(src:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(dst1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(dst2:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(dst3:PHYSICAL_CONTIGUOUS)
void hls_gaussian_pyramid(
	data_in_t* src,
	data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
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
void hls_laplacian_pyramid(
	data_in_t* src,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);


void hls_laplacian_pyramid_remap(
    data_in_t* src,
    data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
    pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
    float ref, float fact, float sigma2);

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
void hls_reconstruct(data_pyr_t* src0, data_pyr_t* src1, data_pyr_t* src2, data_pyr_t* src3,
	data_out_t* dst, pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);

#pragma SDS data access_pattern(src:SEQUENTIAL)
#pragma SDS data access_pattern(gau0:SEQUENTIAL, gau1:SEQUENTIAL)
#pragma SDS data access_pattern(gau2:SEQUENTIAL, gau3:SEQUENTIAL)
#pragma SDS data access_pattern(lap0:SEQUENTIAL, lap1:SEQUENTIAL)
#pragma SDS data access_pattern(lap2:SEQUENTIAL, lap3:SEQUENTIAL)
#pragma SDS data copy(src[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(gau0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(gau1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(gau2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(gau3[0:"pyr_rows_[3]*pyr_cols_[3]"])
#pragma SDS data copy(lap0[0:"pyr_rows_[0]*pyr_cols_[0]"])
#pragma SDS data copy(lap1[0:"pyr_rows_[1]*pyr_cols_[1]"])
#pragma SDS data copy(lap2[0:"pyr_rows_[2]*pyr_cols_[2]"])
#pragma SDS data copy(lap3[0:"pyr_rows_[3]*pyr_cols_[3]"])
void hls_construct_pyramid(
    data_pyr_t* src,
    data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2, data_pyr_t* gau3,
    data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2, data_pyr_t* lap3,
    pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_]);

void remap(data_in_t* src, data_in_t* dst, float ref, float fact, float sigma, int rows, int cols);

// For debugging
bool hls_save_img(std::string name, data_pyr_t* img, int rows, int cols);
void hls_show_img(data_pyr_t* img, int rows, int cols, int delay = 0, std::string winname = "img");
void hls_print_value(data_pyr_t* img, int rows, int cols, std::string name = "");

#endif
