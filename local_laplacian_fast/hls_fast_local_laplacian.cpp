#include "hls_fast_local_laplacian.h"

#include "hls_opencv.h"		// cvMat2hlsMat() etc.

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

void my_split(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst1,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst2);


void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact, int N)
{
	// Check input
	if (N <= 0) {
		return;
	}

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

	int width = src.cols, height = src.rows;
	for (int l = 0; l < num_levels; l++) {
		pyr_cols[l] = width;
		pyr_rows[l] = height;

		sz_gaussian_pyr += width*height;
		sz_laplacian_pyr += width*height;

		height = std::ceil(height / 2.0);
		width = std::ceil(width / 2.0);
	}


	// Pyramids
	float* input_gaussian_pyr = NULL;
	float* output_laplace_pyr = NULL;
	input_gaussian_pyr = new float [sz_gaussian_pyr];
	output_laplace_pyr = new float [sz_laplacian_pyr];

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

	hls_local_laplacian(
		buf_src, input_gaussian_pyr, output_laplace_pyr, pyr_rows, pyr_cols,
		num_levels, sigma, fact, N);
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
	ptr[0] = &(output_laplace_pyr[0]) + pyr_rows[0] * pyr_cols[0];
	ptr[1] = ptr[0] + pyr_rows[1] * pyr_cols[1];
	ptr[2] = ptr[1] + pyr_rows[2] * pyr_cols[2];
	reconstruct(ptr[2], ptr[1], ptr[0], output_laplace_pyr, buf_dst, num_levels, pyr_rows, pyr_cols);

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

	if (buf_src) {
		delete [] buf_src;
	}
	if (buf_dst) {
		delete [] buf_dst;
	}
}


// Accelerated function
// I:    Original image
// gau:  Pre-built Gaussian pyramid
// dst:  Remapped Laplacian pyramid
void hls_local_laplacian(float* I, float* gau, float* dst,
		int pyr_rows[_MAX_LEVELS_], int pyr_cols[_MAX_LEVELS_],
		int num_levels, float sigma, float fact, int N)
{
	float discretisation_step = 1.0f / (N - 1);

	int sz_temp_pyr = 0;
	for (int l = 0; l < num_levels; l++) {
		sz_temp_pyr += pyr_rows[l] * pyr_cols[l];
	}

	float* temp_laplace_pyr = NULL;
	temp_laplace_pyr = new float [sz_temp_pyr];

	float* ptr[3];
	ptr[0] = &(temp_laplace_pyr[0]) + pyr_rows[0] * pyr_cols[0];
	ptr[1] = ptr[0] + pyr_rows[1] * pyr_cols[1];
	ptr[2] = ptr[1] + pyr_rows[2] * pyr_cols[2];

	// Copy
	int offset2 = 0;
	for (int l = 0; l < num_levels - 1; l++) {
		offset2 += pyr_rows[l] * pyr_cols[l];
	}

	for (int r = 0; r < pyr_rows[num_levels - 1]; r++) {
		for (int c = 0; c < pyr_cols[num_levels - 1]; c++) {
			dst[offset2 + r*pyr_cols[num_levels - 1] + c] = gau[offset2 + r*pyr_cols[num_levels - 1] + c];
		}
	}

	// Parallelize-able
	int rows = pyr_rows[0];
	int cols = pyr_cols[0];

	float* I_remap = NULL;
	I_remap = new float [rows*cols];
	for (int n = 0; n < N; n++) {
		float ref = ((float)n) / ((float)(N - 1));

		// Remap original image
		remap(I, I_remap, ref, fact, sigma, rows, cols);

		// Create laplacian pyramid from remapped image
		laplacian_pyramid(I_remap, temp_laplace_pyr, ptr[0], ptr[1], ptr[2], num_levels, pyr_rows, pyr_cols);

		int offset = 0;
		for (int l = 0; l < num_levels - 1; l++) {
			float x_ = 0;
			for (int r = 0; r < pyr_rows[l]; r++) {
				for (int c = 0; c < pyr_cols[l]; c++) {
					if (std::abs(gau[offset + r*pyr_cols[l] + c] - ref) < discretisation_step) {
						x_ = 1 - std::abs(gau[offset + r*pyr_cols[l] + c] - ref) / discretisation_step;
						x_ = x_ * temp_laplace_pyr[offset + r*pyr_cols[l] + c];
					}
					else {
						x_ = 0;
					}
					x_ = x_ + dst[offset + r*pyr_cols[l] + c];

					dst[offset + r*pyr_cols[l] + c] = x_;
				}
			}

			offset += pyr_rows[l] * pyr_cols[l];
		}
	}

	// Release memory
	if(temp_laplace_pyr){
		delete [] temp_laplace_pyr;
	}

	delete [] I_remap;
}


void downsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst,
	int rows, int cols, int rows2, int cols2)
{
	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);
	assert(rows2 <= _MAX_ROWS_ / 2);
	assert(cols2 <= _MAX_COLS_ / 2);

	//#pragma HLS INLINE
#pragma HLS DATAFLOW

	// Convolution Kernel
	// This sums to unity
	static const float x[25] = {
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;

	// Convolve
	hls::Point p(-1, -1);
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(/*src.*/rows, /*src.*/cols);
	hls::Filter2D(src, tmp, kernel, p);

	// Decimate
#if 01
	int cnt = 0;
	for (int r = 0; r < /*src.*/rows; r++) {
#pragma HLS PIPELINE
		for (int c = 0; c < /*src.*/cols; c++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max=1024
			tmp >> px;
			if ((r % 2 == 0) && (c % 2 == 0)) {
				dst << px;
			}
		}
	}
#else
	for (int r = 0; r < rows2; r++) {
		for (int c = 0; c < cols2; c++) {
			// No if() statements
#pragma HLS PIPELINE
			tmp >> px;
			dst << px;

			// Consume
			tmp >> px;
		}
		for (int c = 0; c < cols2; c++) {
#pragma HLS PIPELINE
			tmp >> px;
		}
	}
#endif
}

void upsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst,
	int rows, int cols)
{
	// Convolution Kernel
	// This sums to unity
	static const float x[25] = {
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
//#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

	// Up-scaling
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	hls::Window<1, _MAX_ROWS_, HLS_TNAME(_MAT_TYPE_)> buf;	// Line buffer
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if ((r % 2 == 0) && (c % 2 == 0)) {
				src >> px;
			}

			if (r % 2 == 0) {
				tmp << px;
				buf.val[0][c] = px.val[0];
			}
			else {
				tmp << buf.val[0][c];
			}
		}
	}

	// Convolve
	hls::Point p(-1, -1);
	hls::Filter2D(tmp, dst, kernel, p);
}

void upsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
{
	// Convolution Kernel
	// This sums to unity
	static const float x[25] = {
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

#pragma HLS DATAFLOW

	// Up-scaling
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(dst.rows, dst.cols);
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	hls::Window<1, _MAX_ROWS_, HLS_TNAME(_MAT_TYPE_)> buf;	// Line buffer
	for (int r = 0; r < dst.rows; r++) {
		for (int c = 0; c < dst.cols; c++) {
#pragma HLS PIPELINE
			if ((r % 2 == 0) && (c % 2 == 0)) {
				src >> px;
			}

			if (r % 2 == 0) {
				tmp << px;
				buf.val[0][c] = px.val[0];
			}
			else {
				tmp << buf.val[0][c];
			}
		}
	}

	// Convolve
	hls::Point p(-1, -1);
	hls::Filter2D(tmp, dst, kernel, p);
}

#if 01
void upsample( hls::stream<float>& src, hls::stream<float>& dst, int rows, int cols)
{
	// Convolution Kernel
	// This sums to unity
	static const float x[25] = {
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
			//#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> hls_src(rows, cols);
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> hls_dst(rows, cols);

	// Array to hls::Mat
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	float val;
	//for (int r = 0; r < rows; r++) {
	//	for (int c = 0; c < cols; c++) {
	//		src >> px.val[0];
	//		//px.val[0] = src[r*cols + c];
	//		hls_src << px;
	//	}
	//}

	// Up-scaling
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
	//hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	hls::Window<1, _MAX_ROWS_, HLS_TNAME(_MAT_TYPE_)> buf;	// Line buffer
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if ((r % 2 == 0) && (c % 2 == 0)) {
				//hls_src >> px;
				src >> val;
			}

			if (r % 2 == 0) {
				px.val[0] = val;
				tmp << px;
				buf.val[0][c] = px.val[0];
			}
			else {
				tmp << buf.val[0][c];
			}
		}
	}

	// Convolve
	hls::Point p(-1, -1);
	hls::Filter2D(tmp, hls_dst, kernel, p);

	// hls::Mat to array
	//hls::Scalar<1, float> px;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			hls_dst >> px;
			dst << px.val[0];//[r*cols + c] = px.val[0];
		}
	}
}
#endif


// Marked for HW acceleration
void gaussian_pyramid(float* src, float* dst1, float* dst2, float* dst3, 
	int num_levels,
	int pyr_rows_[_MAX_LEVELS_], int pyr_cols_[_MAX_LEVELS_])
{
#pragma HLS INTERFACE ap_fifo port=dst1
#pragma HLS INTERFACE ap_fifo port=dst2
#pragma HLS INTERFACE ap_fifo port=dst3
#pragma HLS INTERFACE ap_fifo port=dst4
#pragma HLS INTERFACE ap_fifo port=src

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

	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp(pyr_rows[1], pyr_cols[1]);
	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> tmp[2];

#pragma HLS DATAFLOW
	//
	//	assert(pyr_rows[0] <= _MAX_ROWS_);
	//	assert(pyr_cols[0] <= _MAX_COLS_);
	for (int r = 0; r < pyr_rows[0]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[0]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			px.val[0] = src[r*pyr_cols[0] + c];
			in << px;
		}
	}

//#pragma HLS allocation instances=downsample limit=2 function
	downsample(in, tmp1, in.rows, in.cols, tmp1.rows, tmp1.cols);
	my_split(tmp1, out1, tmp12);
	downsample(tmp12, tmp2, tmp12.rows, tmp12.cols, tmp2.rows, tmp2.cols);
	my_split(tmp2, out2, tmp22);
	downsample(tmp22, out3, tmp22.rows, tmp22.cols, out3.rows, out3.cols);
	//my_split(tmp3, out3, tmp32);
	//downsample(tmp32, out, tmp32.rows, tmp32.cols, out.rows, out.cols);


	for (int r = 0; r < pyr_rows[1]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=512
		for (int c = 0; c < pyr_cols[1]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=512
#pragma HLS PIPELINE
			out1 >> px;
			dst1[r*pyr_cols[1] + c] = px.val[0];
		}
	}

	for (int r = 0; r < pyr_rows[2]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=256
		for (int c = 0; c < pyr_cols[2]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=256
#pragma HLS PIPELINE
			out2 >> px;
			dst2[r*pyr_cols[2] + c] = px.val[0];
		}
	}

	for (int r = 0; r < pyr_rows[3]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=128
		for (int c = 0; c < pyr_cols[3]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=128
#pragma HLS PIPELINE
			out3 >> px;
			dst3[r*pyr_cols[3] + c] = px.val[0];
		}
	}

//	for (int r = 0; r < pyr_rows[4]; r++) {
//#pragma HLS LOOP_TRIPCOUNT max=64
//		for (int c = 0; c < pyr_cols[4]; c++) {
//#pragma HLS LOOP_TRIPCOUNT max=64
//#pragma HLS PIPELINE
//			out >> px;
//			dst4[r*pyr_cols[4] + c] = px.val[0];
//		}
//	}
	//		}
}

#if 0
void gaussian_pyramid(float* src, float* dst, int num_levels,
		int pyr_rows[_MAX_LEVELS_], int pyr_cols[_MAX_LEVELS_])
{
	// Check range of input for determining trip count
	assert(num_levels <= _MAX_LEVELS_);

	// Inter-loop buffer
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> buf_;
	hls::Scalar<1, float> px;

	int offset = 0;

	for (int l = 1; l < num_levels; l++) {
		int rows_ = pyr_rows[l - 1];
		int cols_ = pyr_cols[l - 1];

		offset += rows_*cols_;

		// Before downsampling
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in(rows_, cols_);

		if (l == 1) {
			// Copy from source
			for (int r = 0; r < rows_; r++) {
				for (int c = 0; c < cols_; c++) {
					px.val[0] = src[r*cols_ + c];
					in << px;
				}
			}
		}
		else {
			// Copy from inter-loop buffer
			for (int r = 0; r < rows_; r++) {
				for (int c = 0; c < cols_; c++) {
					buf_ >> px;
					in << px;
				}
			}
		}

		// Image size after down sampling
		int rows2_ = pyr_rows[l];
		int cols2_ = pyr_cols[l];

		// Perform down-sampling
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out(rows2_, cols2_);
		downsample(in, out);

		// Transfer data - Add to pyramid
		buf_.init(rows2_, cols2_);
		for (int r = 0; r < rows2_; r++) {
			for (int c = 0; c < cols2_; c++) {
				out >> px;

				// Output
				dst[offset + r*cols2_ + c] = px.val[0];

				// For next loop
				if (l != num_levels - 1) {
					// Prevent remaining data
					buf_ << px;
				}
			}
		}

#if 0
		// Debugging
		cv::Mat tmp(rows2_, cols2_, CV_32FC1);
		tmp.data = (unsigned char*)(dst[i]);
		cv::imshow("Down", tmp);
		cv::waitKey();
		cv::destroyWindow("Down");
#endif
	}
}

void laplacian_pyramid(float* src, float* dst, int num_levels,
	int pyr_rows[_MAX_LEVELS_], int pyr_cols[_MAX_LEVELS_])
{
	// Check range of input for determining trip count
	assert(num_levels <= _MAX_LEVELS_);

	// Inter-loop buffer
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> buf_;
	hls::Scalar<1, float> px;

	int offset = 0;

	for (int l = 0; l < num_levels - 1; l++) {
		int rows_ = pyr_rows[l];
		int cols_ = pyr_cols[l];

		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in(rows_, cols_);	// for downsampling
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in2(rows_, cols_);	// 

		if (l == 0) {
			// Copy from source
			for (int r = 0; r < rows_; r++) {
				for (int c = 0; c < cols_; c++) {
					px.val[0] = src[r*cols_ + c];
					in << px;
					in2 << px;
				}
			}
		}
		else {
			// Copy from inter-loop buffer
			for (int r = 0; r < rows_; r++) {
				for (int c = 0; c < cols_; c++) {
					buf_ >> px;
					in << px;
					in2 << px;
				}
			}
		}

		// Down-sample
		int rows2_ = pyr_rows[l + 1];
		int cols2_ = pyr_cols[l + 1];

		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out_down(rows2_, cols2_);	// For down-sampling
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out_down2(rows2_, cols2_);	// For up-sampling
																				//		downsample(in, out_down);

		buf_.init(rows2_, cols2_);

		my_split(out_down, buf_, out_down2);

		// Up-sample
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out_up(rows_, cols_);
		upsample(out_down2, out_up, rows_, cols_);

		// Diff
		hls::Scalar<1, float> px0, px1;
		hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> diff(rows_, cols_);
		for (int r = 0; r < rows_; r++) {
			for (int c = 0; c < cols_; c++) {
				in2 >> px0;
				out_up >> px1;

				diff << (px0 - px1);
			}
		}

		// Transfer
		for (int r = 0; r < rows_; r++) {
			for (int c = 0; c < cols_; c++) {
				diff >> px;
				dst[offset + r*cols_ + c] = px.val[0];
			}
		}

		offset += rows_*cols_;
	}

	// Transfer last layer
	int rows_ = pyr_rows[num_levels - 1];
	int cols_ = pyr_cols[num_levels - 1];
	for (int r = 0; r < rows_; r++) {
		for (int c = 0; c < cols_; c++) {
			buf_ >> px;
			dst[offset + r*cols_ + c] = px.val[0];
		}
	}
}
#endif


void lap_kernel(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst_down,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& lap)
{
#pragma HLS DATAFLOW

	// Source is split 1. for down-sampling, and 2. Difference
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
	downsample(src_down, tmp_down, src_down.rows, src_down.cols, tmp_down.rows, tmp_down.cols);
	my_split(tmp_down, dst_down, tmp_up_i);

	// Up-sampling
	upsample(tmp_up_i, tmp_up_o);

	hls::Scalar<1, float> px1;
	hls::Scalar<1, float> px2;
	for (int r = 0; r < lap.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < lap.cols; c++) {
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
#pragma HLS INTERFACE ap_fifo port=dst0
#pragma HLS INTERFACE ap_fifo port=dst1
#pragma HLS INTERFACE ap_fifo port=dst2
#pragma HLS INTERFACE ap_fifo port=dst3
#pragma HLS INTERFACE ap_fifo port=dst4
#pragma HLS INTERFACE ap_fifo port=src


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

	// Down-sample
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> in(pyr_rows[0], pyr_cols[0]);
	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> out(pyr_rows[4], pyr_cols[4]);

	// Output laplacian
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap0(pyr_rows[0], pyr_cols[0]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap2(pyr_rows[2], pyr_cols[2]);
	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap3(pyr_rows[0], pyr_cols[0]);
	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> lap4(pyr_rows[0], pyr_cols[0]);

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down0(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down2(pyr_rows[3], pyr_cols[3]);
	//	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE_> down3(pyr_rows[0], pyr_cols[0]);


#pragma HLS DATAFLOW

	for (int r = 0; r < pyr_rows[0]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[0]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			px.val[0] = src[r*pyr_cols[0] + c];
			in << px;
			//down0 << px;
		}
	}

	lap_kernel(in, down0, lap0);
	lap_kernel(down0, down1, lap1);
	lap_kernel(down1, down2, lap2);
	//	lap_kernel(down2, down3, lap3);

	// Transfer data
	for (int r = 0; r < pyr_rows[0]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[0]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			lap0 >> px;
			dst0[r*pyr_cols[0] + c] = px.val[0];
		}
	}

	for (int r = 0; r < pyr_rows[1]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=512
		for (int c = 0; c < pyr_cols[1]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=512
#pragma HLS PIPELINE
			lap1 >> px;
			dst1[r*pyr_cols[1] + c] = px.val[0];
		}
	}

	for (int r = 0; r < pyr_rows[2]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=256
		for (int c = 0; c < pyr_cols[2]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=256
#pragma HLS PIPELINE
			lap2 >> px;
			dst2[r*pyr_cols[2] + c] = px.val[0];
		}
	}

	for (int r = 0; r < pyr_rows[3]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=128
		for (int c = 0; c < pyr_cols[3]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=128
#pragma HLS PIPELINE
			down2 >> px;
			dst3[r*pyr_cols[3] + c] = px.val[0];
		}
	}
}

void reconstruct(
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
	for (int r = 0; r < pyr_rows[3]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[3]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			px1.val[0] = src0[r*pyr_cols[3] + c];
			in0 << px1;
		}
	}

	for (int r = 0; r < pyr_rows[2]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=512
		for (int c = 0; c < pyr_cols[2]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=512
#pragma HLS PIPELINE
			px1.val[0] = src1[r*pyr_cols[2] + c];
			in1 << px1;
		}
	}

	for (int r = 0; r < pyr_rows[1]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=256
		for (int c = 0; c < pyr_cols[1]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=256
#pragma HLS PIPELINE
			px1.val[0] = src2[r*pyr_cols[1] + c];
			in2 << px1;
		}
	}

	for (int r = 0; r < pyr_rows[0]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=128
		for (int c = 0; c < pyr_cols[0]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=128
#pragma HLS PIPELINE
			px1.val[0] = src3[r*pyr_cols[0] + c];
			in3 << px1;
		}
	}

	// 
	upsample(in0, up1);

	// Add
	for (int r = 0; r < up1.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < up1.cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			up1 >> px1;
			in1 >> px2;

			sum1 << (px1 + px2);
		}
	}

	upsample(sum1, up2);
	for (int r = 0; r < up2.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < up2.cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			up2 >> px1;
			in2 >> px2;

			sum2 << (px1 + px2);
		}
	}

	upsample(sum2, up3);
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

#if 0
	hls::stream<float> buf_[_MAX_LEVELS_];
	hls::stream<float> tmp_[_MAX_LEVELS_];

	//	hls::stream<float> in;
	float px;

	// Last layer in the pyramid
	int offset = 0;
	for (int l = 0; l < num_levels - 1; l++) {
		offset += pyr_rows[l] * pyr_cols[l];
	}

	for (int r = 0; r < pyr_rows[num_levels - 1]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[num_levels - 1]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
			//px.val[0] = src[offset + r*pyr_cols[num_levels - 1] + c];
			//in << px;
			buf_[num_levels - 1] << src[offset + r*pyr_cols[num_levels - 1] + c];
		}
	}

	//hls::Scalar<1, float> px2;
	//	hls::stream<float> out;
	for (int l = num_levels - 2; l >= 0; l--) {
#pragma HLS UNROLL
		//hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> out(pyr_rows[l], pyr_cols[l]);
		//float out[pyr_rows[l]*pyr_cols[l]];

		offset -= pyr_rows[l] * pyr_cols[l];

		// Upsample
		//		if (l == num_levels - 2) {
		//			upsample(in, out, pyr_rows[l], pyr_cols[l]);
		//		}
		//		else {
		upsample(buf_[l + 1], tmp_[l], pyr_rows[l], pyr_cols[l]);
		//		}

		//buf_.init(pyr_rows[l], pyr_cols[l]);

		// Load data
		for (int r = 0; r < pyr_rows[l]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
			for (int c = 0; c < pyr_cols[l]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
				tmp_[l] >> px;
				//px.val[0] += src[offset + r*pyr_cols[l] + c];
				px += src[offset + r*pyr_cols[l] + c];
				buf_[l] << px;
			}
		}
	}

	// Output
	for (int r = 0; r < pyr_rows[0]; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < pyr_cols[0]; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
			buf_[0] >> px;
			dst[r*pyr_cols[0] + c] = px;
		}
	}
//#else
	// Inter-loop buffer
	//hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> buf_;	// Inter-loop buffer
	//hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> in(pyr_rows[num_levels - 1], pyr_cols[num_levels - 1]);
	//hls::Scalar<1, float> px;
	hls::stream<float> buf_;
	//hls::stream<float> in;
	float px;

	// Last layer in the pyramid
	int offset = 0;
	for (int l = 0; l < num_levels - 1; l++) {
		offset += pyr_rows[l] * pyr_cols[l];
	}

	for (int r = 0; r < pyr_rows[num_levels - 1]; r++) {
		for (int c = 0; c < pyr_cols[num_levels - 1]; c++) {
			//px.val[0] = src[offset + r*pyr_cols[num_levels - 1] + c];
			//in << px;
			/*in*/buf_ << src[offset + r*pyr_cols[num_levels - 1] + c];
		}
	}

	//hls::Scalar<1, float> px2;
	for (int l = num_levels - 2; l >= 0; l--) {
		hls::stream<float> out;
		//hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> out(pyr_rows[l], pyr_cols[l]);
		//float out[pyr_rows[l]*pyr_cols[l]];

		offset -= pyr_rows[l] * pyr_cols[l];

		// Upsample
		//if (l == num_levels - 2) {
		//	upsample(in, out, pyr_rows[l], pyr_cols[l]);
		//}
		//else {
			upsample(buf_, out, pyr_rows[l], pyr_cols[l]);
		//}

		//buf_.init(pyr_rows[l], pyr_cols[l]);
		
		// Load data
		for (int r = 0; r < pyr_rows[l]; r++) {
			for (int c = 0; c < pyr_cols[l]; c++) {
				out >> px;
				//px.val[0] += src[offset + r*pyr_cols[l] + c];
				px += src[offset + r*pyr_cols[l] + c];
				buf_ << px;
			}
		}
	}

	// Output
	for (int r = 0; r < pyr_rows[0]; r++) {
		for (int c = 0; c < pyr_cols[0]; c++) {
			buf_ >> px;
			dst[r*pyr_cols[0] + c] = px;
		}
	}
#endif
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

void my_split(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst1,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst2)
{
	int rows_ = src.rows;
	int cols_ = src.cols;

	assert(rows_ <= _MAX_ROWS_);
	assert(cols_ <= _MAX_COLS_);

	for (int r = 0; r < rows_; r++) {
		for (int c = 0; c < cols_; c++) {
//#pragma HLS PIPELINE
			hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
			src >> px;

			dst1 << px;
			dst2 << px;
		}
	}
}
