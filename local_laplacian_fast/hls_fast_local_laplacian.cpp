#include "hls_fast_local_laplacian.h"

//#include "hls_opencv.h"		// cvMat2hlsMat() etc.
#include "opencv2/highgui/highgui.hpp"

#ifdef __SDSCC__
#include "sds_lib.h"
#endif

void hls_local_laplacian_wrap(cv::Mat& src, cv::Mat& dst, float sigma, float fact)
{
	// Check input
	if (src.data == NULL) {
		std::cout << "Input image is empty..." << std::endl;
		return;
	}

	if (src.channels() != 1) {
		std::cout << "Input must be a single-channel image..." << std::endl;
		return;
	}

	// Settings
	// num_levels: Max. 9 (for 1024 x 1024 image)
//	int num_levels = 4;// std::ceil(std::log(std::min(src.rows, src.cols)) - log(2)) + 2;
//	float discretisation_step = 1.0f / (N - 1);

	// Original image
	data_in_t*  buf_src;
	data_out_t* buf_dst;

	// Pyramids
	float* input_gaussian_pyr[_MAX_LEVELS_] = {NULL};
	float* output_laplace_pyr[_MAX_LEVELS_] = {NULL};
	float* temp_laplace_pyr[_MAX_LEVELS_] = {NULL};

	// Remapped image
	float* I_remap = NULL;

	// Allocation
#ifdef __SDSCC__
	buf_src = (data_in_t*) sds_alloc(src.rows*src.cols*sizeof(data_in_t));
	buf_dst = (data_out_t*) sds_alloc(src.rows*src.cols*sizeof(data_out_t));

	I_remap = (float*) sds_alloc(src.rows*src.cols*sizeof(float));
#else
	buf_src = new data_in_t[src.rows*src.cols];
	buf_dst = new data_out_t[src.rows*src.cols];

	I_remap = new float [src.rows*src.cols];
#endif

	// List for pyramid's widths & heights
	pyr_sz_t pyr_rows[_MAX_LEVELS_] = { 0 };
	pyr_sz_t pyr_cols[_MAX_LEVELS_] = { 0 };

	int rows = src.rows;
	int cols = src.cols;
	for (int l = 0; l < _MAX_LEVELS_; l++) {
		pyr_cols[l] = (pyr_sz_t) cols;
		pyr_rows[l] = (pyr_sz_t) rows;

#ifdef __SDSCC__
		input_gaussian_pyr[l] = (float*) sds_alloc(rows*cols*sizeof(float));
		output_laplace_pyr[l] = (float*) sds_alloc(rows*cols*sizeof(float));
		temp_laplace_pyr[l] = (float*) sds_alloc(rows*cols*sizeof(float));
#else
		input_gaussian_pyr[l] = new float [rows*cols];
		output_laplace_pyr[l] = new float [rows*cols];
		temp_laplace_pyr[l] = new float [rows*cols];
#endif
		rows = (pyr_sz_t) std::ceil(rows / 2.0);
		cols = (pyr_sz_t) std::ceil(cols / 2.0);
	}

	// Copy input image data
	memcpy(buf_src, src.data, src.rows*src.cols*sizeof(data_in_t));

	// Construct Laplacian pyramid
#pragma SDS resource(1)
	hls_laplacian_pyramid(buf_src,
			output_laplace_pyr[0], output_laplace_pyr[1], output_laplace_pyr[2], output_laplace_pyr[3],
			pyr_rows, pyr_cols);
#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			std::string name = "L - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(output_laplace_pyr[l]);
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
	// Copy finest level
	memcpy(input_gaussian_pyr[0], buf_src, src.rows*src.cols * sizeof(float));
	hls_gaussian_pyramid(buf_src,
			input_gaussian_pyr[1], input_gaussian_pyr[2], input_gaussian_pyr[3],
			pyr_rows, pyr_cols);
#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			std::string name = "G - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(input_gaussian_pyr[l]);
			cv::imshow(name, tmp);
			cv::waitKey(1.0 * 1000);
			cv::destroyWindow(name);

			offset += h_*w_;

			h_ = std::ceil(h_ / 2.0);
			w_ = std::ceil(w_ / 2.0);
		}
	}
#endif

	for (int n = 0; n < _NUM_STEP_; n++) {
		float ref = ((float)n) / ((float)(_NUM_STEP_ - 1));

		// Remap original image
		remap(buf_src, I_remap, ref, fact, sigma, pyr_rows[0], pyr_cols[0]);

		// Create laplacian pyramid from remapped image
#pragma SDS resource(1)
		hls_laplacian_pyramid(I_remap,
				temp_laplace_pyr[0], temp_laplace_pyr[1], temp_laplace_pyr[2], temp_laplace_pyr[3],
				pyr_rows, pyr_cols);

		hls_local_laplacian(
				input_gaussian_pyr[0], input_gaussian_pyr[1], input_gaussian_pyr[2], input_gaussian_pyr[3],
				temp_laplace_pyr[0], temp_laplace_pyr[1], temp_laplace_pyr[2], temp_laplace_pyr[3],
				output_laplace_pyr[0], output_laplace_pyr[1], output_laplace_pyr[2], output_laplace_pyr[3],
				pyr_rows, pyr_cols, ref);
	}

#if 0
	{
		// Show pyramid image
		int h_ = src.rows, w_ = src.cols;
		int offset = 0;
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			std::string name = "L - ";
			name += std::to_string(l);

			cv::Mat tmp(h_, w_, CV_32FC1);
			tmp.data = (unsigned char*)(output_laplace_pyr[l]);
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
	hls_reconstruct(output_laplace_pyr[3], output_laplace_pyr[2], output_laplace_pyr[1], output_laplace_pyr[0],
			buf_dst, pyr_rows, pyr_cols);

	// Copy back
	dst.create(src.rows, src.cols, src.type());
	memcpy(dst.data, buf_dst, dst.rows*dst.cols * sizeof(data_out_t));

	// Release memory
#ifdef __SDSCC__
	for (int l = 0; l < _MAX_LEVELS_; l++){
		sds_free(input_gaussian_pyr[l]);
		sds_free(output_laplace_pyr[l]);
		sds_free(temp_laplace_pyr[l]);
	}

	if(I_remap){
		sds_free(I_remap);
	}

	if (buf_src) {
		sds_free(buf_src);
	}
	if (buf_dst) {
		sds_free(buf_dst);
	}
#else
	for (int l = 0; l < _MAX_LEVELS_; l++){
		delete [] input_gaussian_pyr[l];
		delete [] output_laplace_pyr[l];
		delete [] temp_laplace_pyr[l];
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
#endif
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
		pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
		float ref)
{
	float discretisation_step = 1.0f / (_NUM_STEP_ - 1);

	pyr_sz_t pyr_rows[_MAX_LEVELS_];
	pyr_sz_t pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete

	for (int l = 0; l < _MAX_LEVELS_; l++) {
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
void hls_gaussian_pyramid(
		float* src,
		float* dst1, float* dst2, float* dst3,
		pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_])
{
	pyr_sz_t pyr_rows[_MAX_LEVELS_];
	pyr_sz_t pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete

	for (int l = 0; l < _MAX_LEVELS_; l++) {
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

void hls_laplacian_pyramid(
	float* src,
	float* dst0, float* dst1, float* dst2, float* dst3,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_])
{
	pyr_sz_t pyr_rows[_MAX_LEVELS_];
	pyr_sz_t pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete
	for (int l = 0; l < _MAX_LEVELS_; l++) {
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
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_])
{
	pyr_sz_t pyr_rows[_MAX_LEVELS_];
	pyr_sz_t pyr_cols[_MAX_LEVELS_];
#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
#pragma HLS ARRAY_PARTITION variable=pyr_cols complete

	for (int l = 0; l < _MAX_LEVELS_; l++) {
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

void my_split(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst1,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst2)
{
	int rows = src.rows;
	int cols = src.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			src >> px;

			dst1 << px;
			dst2 << px;
		}
	}
}


void downsample(
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
{
	int rows = src.rows;
	int cols = src.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	//#pragma HLS INLINE
#pragma HLS DATAFLOW

	// Convolution Kernel - This sums to unity
	static const float x[25] = {
		0.0025f, 0.0125f, 0.0200f, 0.0125f, 0.0025f,
		0.0125f, 0.0625f, 0.1000f, 0.0625f, 0.0125f,
		0.0200f, 0.1000f, 0.1600f, 0.1000f, 0.0200f,
		0.0125f, 0.0625f, 0.1000f, 0.0625f, 0.0125f,
		0.0025f, 0.0125f, 0.0200f, 0.0125f, 0.0025f };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

	// Convolve
	hls::Point p(-1, -1);
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
	hls::Filter2D(src, tmp, kernel, p);

	// Decimate
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;

#if 01
	for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < cols; c++) {
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
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
{
	int rows = dst.rows;
	int cols = dst.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	// Convolution Kernel - This sums to unity
	static const float x[25] = {
		0.0025f, 0.0125f, 0.0200f, 0.0125f, 0.0025f,
		0.0125f, 0.0625f, 0.1000f, 0.0625f, 0.0125f,
		0.0200f, 0.1000f, 0.1600f, 0.1000f, 0.0200f,
		0.0125f, 0.0625f, 0.1000f, 0.0625f, 0.0125f,
		0.0025f, 0.0125f, 0.0200f, 0.0125f, 0.0025f };
	hls::Window<5, 5, float> kernel;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
#pragma HLS PIPELINE
			kernel.val[r][c] = x[r * 5 + c];
		}
	}

#pragma HLS DATAFLOW

	// Up-scaling
	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
	hls::Window<1, _MAX_ROWS_, HLS_TNAME(_MAT_TYPE_)> buf;	// Line buffer

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
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


void add(
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src1,
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src2,
		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
{
	int rows = src1.rows;
	int cols = src1.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	hls::Scalar<1, float> px1;
	hls::Scalar<1, float> px2;

	for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			src1 >> px1;
			src2 >> px2;

			dst << (px1 + px2);
		}
	}
}


void load(float* src, hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
{
	int rows = dst.rows;
	int cols = dst.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	hls::Scalar<1, float> px;

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			px.val[0] = src[r*cols + c];
			dst << px;
		}
	}
}


void save(hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src, float* dst)
{
	int rows = src.rows;
	int cols = src.cols;

	assert(rows <= _MAX_ROWS_);
	assert(cols <= _MAX_COLS_);

	hls::Scalar<1, float> px;

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			src >> px;
			dst[r*cols + c] = px.val[0];
		}
	}
}

