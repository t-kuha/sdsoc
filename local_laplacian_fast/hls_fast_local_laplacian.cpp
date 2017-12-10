#include "hls_fast_local_laplacian.h"

//#include "hls_opencv.h"		// cvMat2hlsMat() etc.

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

#include "hls_util.h"


void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact)
{
	// Check input

	// Settings
	// num_levels: Max. 9 (for 1024 x 1024 image)
	int num_levels = 4;// std::ceil(std::log(std::min(src.rows, src.cols)) - log(2)) + 2;
//	float discretisation_step = 1.0f / (N - 1);

	// Original image
	data_in_t*  buf_src;
	data_out_t* buf_dst;
	buf_src = new data_in_t[src.rows*src.cols];
	buf_dst = new data_out_t[src.rows*src.cols];

	memcpy(buf_src, src.data, src.rows*src.cols*sizeof(data_in_t));


	// List for pyramid's widths & heights
	int pyr_rows[_MAX_LEVELS_] = { 0 };
	int pyr_cols[_MAX_LEVELS_] = { 0 };

	// Total memory size for pyramids (measured in num. of elements)
	int sz_gaussian_pyr = 0;
	int sz_laplacian_pyr = 0;
	int sz_temp_pyr = 0;

	int width = src.cols, height = src.rows;
	for (int l = 0; l < num_levels; l++) {
		pyr_cols[l] = width;
		pyr_rows[l] = height;

		sz_gaussian_pyr += width*height;
		sz_laplacian_pyr += width*height;
		sz_temp_pyr += width*height;

		height = std::ceil(height / 2.0);
		width = std::ceil(width / 2.0);
	}


	// Pyramids
	float* input_gaussian_pyr = NULL;
	float* output_laplace_pyr = NULL;
	float* temp_laplace_pyr = NULL;
	float* I_remap = NULL;

	input_gaussian_pyr = new float [sz_gaussian_pyr];
	output_laplace_pyr = new float [sz_laplacian_pyr];
	temp_laplace_pyr = new float [sz_temp_pyr];
	I_remap = new float [pyr_rows[0]*pyr_cols[0]];

	float* ptr[4];

	// Construct Laplacian pyramid
	ptr[0] = &(output_laplace_pyr[0]) + pyr_rows[0] * pyr_cols[0];
	ptr[1] = ptr[0] + pyr_rows[1] * pyr_cols[1];
	ptr[2] = ptr[1] + pyr_rows[2] * pyr_cols[2];
	laplacian_pyramid(buf_src, &(output_laplace_pyr[0]), ptr[0], ptr[1], ptr[2], num_levels, pyr_rows, pyr_cols);
#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < num_levels; l++) {
			std::string name = "L - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(&output_laplace_pyr[offset]);
			cv::imshow(name, tmp + 0.5);
			cv::waitKey(1.0 * 1000);
			cv::destroyWindow(name);

			offset += h_*w_;

			h_ = std::ceil(h_ / 2.0);
			w_ = std::ceil(w_ / 2.0);
		}
	}
#endif

	// Gaussian Pyramid
	ptr[0] = &(input_gaussian_pyr[0]) + pyr_rows[0] * pyr_cols[0];
	ptr[1] = ptr[0] + pyr_rows[1] * pyr_cols[1];
	ptr[2] = ptr[1] + pyr_rows[2] * pyr_cols[2];

	// Copy finest level
	memcpy(input_gaussian_pyr, buf_src, src.rows*src.cols * sizeof(float));
	gaussian_pyramid(buf_src, 
		ptr[0], ptr[1], ptr[2], 
		num_levels, pyr_rows, pyr_cols);
#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < num_levels; l++) {
			std::string name = "G - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(&input_gaussian_pyr[offset]);
			cv::imshow(name, tmp);
			cv::waitKey(1.0 * 1000);
			cv::destroyWindow(name);

			offset += h_*w_;

			h_ = std::ceil(h_ / 2.0);
			w_ = std::ceil(w_ / 2.0);
		}
	}
#endif

	//
	float* tmp[4];
	tmp[0] = &(temp_laplace_pyr[0]);
	tmp[1] = tmp[0] + pyr_rows[0] * pyr_cols[0];
	tmp[2] = tmp[0] + pyr_rows[1] * pyr_cols[1];
	tmp[3] = tmp[1] + pyr_rows[2] * pyr_cols[2];

	float* lap[4];
	lap[0] = &(input_gaussian_pyr[0]);
	lap[1] = lap[0] + pyr_rows[0] * pyr_cols[0];
	lap[2] = lap[0] + pyr_rows[1] * pyr_cols[1];
	lap[3] = lap[1] + pyr_rows[2] * pyr_cols[2];

	float* out[4];
	out[0] = &(output_laplace_pyr[0]);
	out[1] = out[0] + pyr_rows[0] * pyr_cols[0];
	out[2] = out[1] + pyr_rows[1] * pyr_cols[1];
	out[3] = out[2] + pyr_rows[2] * pyr_cols[2];

	for (int n = 0; n < _NUM_STEP_; n++) {
		float ref = ((float)n) / ((float)(_NUM_STEP_ - 1));

		// Remap original image
		remap(buf_src, I_remap, ref, fact, sigma, pyr_rows[0], pyr_cols[0]);

		// Create laplacian pyramid from remapped image
		laplacian_pyramid(I_remap, temp_laplace_pyr, ptr[0], ptr[1], ptr[2], num_levels, pyr_rows, pyr_cols);

		hls_local_laplacian(
			lap[0], lap[1], lap[2], lap[3],
			tmp[0], tmp[1], tmp[2], tmp[3],
			out[0], out[1], out[2], out[3],
			pyr_rows, pyr_cols, num_levels, ref);
	}


#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < num_levels; l++) {
			std::string name = "L - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(&output_laplace_pyr[offset]);
			cv::imshow(name, tmp + 0.5);
			cv::waitKey(1.0 * 1000);
			cv::destroyWindow(name);

			offset += h_*w_;

			h_ = std::ceil(h_ / 2.0);
			w_ = std::ceil(w_ / 2.0);
		}
	}
#endif

	// Reconstruct
	hls_reconstruct(out[3], out[2], out[1], out[0], buf_dst, num_levels, pyr_rows, pyr_cols);

	// Copy back
	dst.create(src.rows, src.cols, src.type());
	memcpy(dst.data, buf_dst, dst.rows*dst.cols * sizeof(data_out_t));

	// Release memory
	if (input_gaussian_pyr) {
		delete [] input_gaussian_pyr;
	}
	if (output_laplace_pyr) {
		delete [] output_laplace_pyr;
	}
	if(temp_laplace_pyr){
		delete [] temp_laplace_pyr;
	}
	if(I_remap){
		delete [] I_remap;
	}

	if (buf_src) {
		delete [] buf_src;
	}
	if (buf_dst) {
		delete [] buf_dst;
	}
}

void kernel(float* gau, float* temp_laplace_pyr, float* dst, int rows, int cols, float ref, float discretisation_step)
{
#pragma HLS INLINE
	float x_;
	int offset = 0;

	for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			if (std::abs(gau[offset] - ref) < discretisation_step) {
				x_ = 1 - std::abs(gau[offset] - ref) / discretisation_step;
				x_ = x_ * temp_laplace_pyr[offset];
			}
			else {
				x_ = 0;
			}
			x_ = x_ + dst[offset];

			dst[offset] = x_;
			offset++;
		}
	}
}

// Accelerated function
// I:    Original image
// gau:  Pre-built Gaussian pyramid
// dst:  Remapped Laplacian pyramid
void hls_local_laplacian(
		float* gau0, float* gau1, float* gau2, float* gau3,
		float* lap0, float* lap1, float* lap2, float* lap3,
		float* dst0, float* dst1, float* dst2, float* dst3,
		int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_],
		int num_levels, float ref)
{
	float discretisation_step = 1.0f / (_NUM_STEP_ - 1);

	int pyr_rows[_MAX_LEVELS_];
	int pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete

	for (int l = 0; l < num_levels; l++) {
#pragma HLS PIPELINE
		pyr_rows[l] = pyr_rows_[l];
		pyr_cols[l] = pyr_cols_[l];
	}

	// Parallel execution
	// Layer 0
	kernel(gau0, lap0, dst0, pyr_rows[0], pyr_cols[0], ref, discretisation_step);
	// Layer 1
	kernel(gau1, lap1, dst1, pyr_rows[1], pyr_cols[1], ref, discretisation_step);
	// Layer 2
	kernel(gau2, lap2, dst2, pyr_rows[2], pyr_cols[2], ref, discretisation_step);
	// Layer 3
	kernel(gau3, lap3, dst3, pyr_rows[3], pyr_cols[3], ref, discretisation_step);
}


// Marked for HW acceleration
void gaussian_pyramid(float* src, float* dst1, float* dst2, float* dst3, 
	int num_levels,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_])
{
	// Check range of input for determining trip count
	assert(num_levels <= _MAX_LEVELS_);

	int pyr_rows[_MAX_LEVELS_];
	int pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete
	for (int l = 0; l < num_levels; l++) {
#pragma HLS PIPELINE
		pyr_rows[l] = pyr_rows_[l];
		pyr_cols[l] = pyr_cols_[l];
	}

	// -------------------------------
	hls::Scalar<1, float> px;

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in(pyr_rows[0], pyr_cols[0]);

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp12(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp2(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp22(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp3(pyr_rows[3], pyr_cols[3]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp32(pyr_rows[3], pyr_cols[3]);

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out2(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out3(pyr_rows[3], pyr_cols[3]);

#pragma HLS DATAFLOW
	//	assert(pyr_rows[0] <= _MAX_ROWS_);
	//	assert(pyr_cols[0] <= _MAX_COLS_);

	load(src, in);

//#pragma HLS allocation instances=downsample limit=2 function
	downsample(in, tmp1);
	my_split(tmp1, out1, tmp12);
	downsample(tmp12, tmp2);
	my_split(tmp2, out2, tmp22);
	downsample(tmp22, out3);

	save(out1, dst1);
	save(out2, dst2);
	save(out3, dst3);
}


void lap_kernel(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst_down,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& lap)
{
#pragma HLS DATAFLOW

	// Source is split into: 1. for down-sampling, and 2. Difference
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> src_down(src.rows, src.cols);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> src_diff(src.rows, src.cols);

	// Down-sampled result
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp_down(dst_down.rows, dst_down.cols);
	// Input to up-sample
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp_up_i(dst_down.rows, dst_down.cols);
	// Output of up-sample
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp_up_o(src.rows, src.cols);

	my_split(src, src_down, src_diff);

	//
	downsample(src_down, tmp_down);
	my_split(tmp_down, dst_down, tmp_up_i);

	// Up-sampling
	upsample(tmp_up_i, tmp_up_o);

	hls::Scalar<1, float> px1;
	hls::Scalar<1, float> px2;
	for (int r = 0; r < src.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < src.cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			src_diff >> px1;
			tmp_up_o >> px2;

			lap << (px1 - px2);
		}
	}
}

void laplacian_pyramid(
	float* src,
	float* dst0, float* dst1, float* dst2, float* dst3, int num_levels,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_])
{
	// Check range of input for determining trip count
	assert(num_levels <= _MAX_LEVELS_);

	int pyr_rows[_MAX_LEVELS_];
	int pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete
	for (int l = 0; l < num_levels; l++) {
#pragma HLS PIPELINE
		pyr_rows[l] = pyr_rows_[l];
		pyr_cols[l] = pyr_cols_[l];
	}

	hls::Scalar<1, float> px;

	// Input
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in(pyr_rows[0], pyr_cols[0]);

	// Output laplacian
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap0(pyr_rows[0], pyr_cols[0]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap2(pyr_rows[2], pyr_cols[2]);

	// Down-sampled image
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down0(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down2(pyr_rows[3], pyr_cols[3]);

#pragma HLS DATAFLOW

	load(src, in);

	lap_kernel(in, down0, lap0);
	lap_kernel(down0, down1, lap1);
	lap_kernel(down1, down2, lap2);
	//	lap_kernel(down2, down3, lap3);

	// Transfer data
	save(lap0, dst0);
	save(lap1, dst1);
	save(lap2, dst2);
	save(down2, dst3);
}

void hls_reconstruct(
	float* src0, float* src1, float* src2, float* src3, data_out_t* dst, 
	int num_levels, int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_])
{
	assert(num_levels <= _MAX_LEVELS_);

	int pyr_rows[_MAX_LEVELS_];
	int pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete
	for (int l = 0; l < num_levels; l++) {
#pragma HLS PIPELINE
		pyr_rows[l] = pyr_rows_[l];
		pyr_cols[l] = pyr_cols_[l];
	}

#pragma HLS DATAFLOW

	// Input stream
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in0(pyr_rows[3], pyr_cols[3]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in3(pyr_rows[0], pyr_cols[0]);

	// Up-sampled image
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> up1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> up2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> up3(pyr_rows[0], pyr_cols[0]);

	// SUm of updampled image + 1-layer up in the pyramid
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> sum1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> sum2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> sum3(pyr_rows[0], pyr_cols[0]);

	hls::Scalar<1, float> px1;
	hls::Scalar<1, float> px2;

	// Load image data
	load(src0, in0);
	load(src1, in1);
	load(src2, in2);
	load(src3, in3);

	// 
	upsample(in0, up1);
	add(up1, in1, sum1);	// Add

	upsample(sum1, up2);
	add(up2, in2, sum2);

	upsample(sum2, up3);
//	add(up3, in3, dst);
	for (int r = 0; r < up3.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < up3.cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			up3 >> px1;
			in3 >> px2;

			dst[r*pyr_cols[0] + c] = (px1.val[0] + px2.val[0]);
		}
	}
}

void remap(float* src, float* dst, float ref, float fact, float sigma, int rows, int cols)
{
#pragma HLS INLINE
	float I;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			I = src[r*cols + c];
#ifdef _WIN32
			dst[r*cols + c] =
				fact*(I - ref)*std::exp(-(I - ref)*(I - ref) / (2 * sigma*sigma)); 
#else
			dst[r*cols + c] =
				fact*(I - ref)*hls::expf(-(I - ref)*(I - ref) / (2 * sigma*sigma));
#endif
		}
	}
}
