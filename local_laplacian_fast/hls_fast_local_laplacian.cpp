#include "hls_fast_local_laplacian.h"

#include "opencv2/highgui/highgui.hpp"

#ifdef __SDSCC__
#include "sds_lib.h"
#endif

#include "hls_util.h"

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

	// Convert input image to 16-bit (signed)
	// 2047: 11 bit range
	src.convertTo(src, CV_16UC1, ( ( 1 << (HLS_TBITDEPTH(_MAT_TYPE2_) - 1)) - 1 ) /* = 2047.0*/);


	// Original image
	data_in_t*  buf_src;
	data_out_t* buf_dst;

	// Pyramids
	data_pyr_t* input_gaussian_pyr[_MAX_LEVELS_] = {NULL};
	data_pyr_t* output_laplace_pyr[_MAX_LEVELS_] = {NULL};
	data_pyr_t* temp_laplace_pyr[_MAX_LEVELS_] = {NULL};

	// Remapped image
	data_in_t* I_remap = NULL;

	// Allocation
#ifdef __SDSCC__
	buf_src = (data_in_t*) sds_alloc(src.rows*src.cols*sizeof(data_in_t));
	buf_dst = (data_out_t*) sds_alloc(src.rows*src.cols*sizeof(data_out_t));

	I_remap = (data_in_t*) sds_alloc(src.rows*src.cols*sizeof(data_in_t));
#else
	buf_src = new data_in_t[src.rows*src.cols];
	buf_dst = new data_out_t[src.rows*src.cols];

	I_remap = new data_in_t [src.rows*src.cols];
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
		input_gaussian_pyr[l] = (data_pyr_t*) sds_alloc(rows*cols*sizeof(data_pyr_t));
		output_laplace_pyr[l] = (data_pyr_t*) sds_alloc(rows*cols*sizeof(data_pyr_t));
		temp_laplace_pyr[l] = (data_pyr_t*) sds_alloc(rows*cols*sizeof(data_pyr_t));
#else
		input_gaussian_pyr[l] = new data_pyr_t [rows*cols];
		output_laplace_pyr[l] = new data_pyr_t [rows*cols];
		temp_laplace_pyr[l] = new data_pyr_t [rows*cols];
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
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			cv::Mat tmp(pyr_rows[l], pyr_cols[l], CV_16SC1);
			tmp.data = (unsigned char*)(output_laplace_pyr[l]);
			
            tmp = cv::abs(tmp);
            tmp = (tmp/2047)*255;
            tmp.convertTo(tmp, CV_8UC1);

            cv::imwrite("hls_laplacian_" + std::to_string(l) + ".tif", tmp);
            std::string name = "L - ";
//            name += std::to_string(l);
//          cv::imshow(name, 16*tmp + (1 << 14));
//			cv::waitKey(1.0 * 1000);
//			cv::destroyWindow(name);
		}
	}
#endif

	// Gaussian Pyramid
	// Copy finest level
	memcpy(input_gaussian_pyr[0], buf_src, src.rows*src.cols * sizeof(data_pyr_t));

	hls_gaussian_pyramid(buf_src,
			input_gaussian_pyr[1], input_gaussian_pyr[2], input_gaussian_pyr[3],
			pyr_rows, pyr_cols);

#if 0
	{
		// Show pyramid image
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			cv::Mat tmp(pyr_rows[l], pyr_cols[l], CV_16SC1);
			tmp.data = (unsigned char*)(input_gaussian_pyr[l]);
            
            tmp = cv::abs(tmp);
            tmp = (tmp/2047)*255;
            tmp.convertTo(tmp, CV_8UC1);
            cv::imwrite("hls_gauss_" + std::to_string(l) + ".tif", tmp);
            
//            std::string name = "G - ";
//            name += std::to_string(l);
//			cv::imshow(name, tmp*16);
//			cv::waitKey(1.0 * 1000);
//			cv::destroyWindow(name);
		}
	}
#endif

#if 01
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
#endif

#if 0
	{
		// Show pyramid image
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			std::string name = "L - ";
			name += std::to_string(l);

			cv::Mat tmp(pyr_rows[l], pyr_cols[l], CV_16SC1);
			tmp.data = (unsigned char*)(output_laplace_pyr[l]);
			cv::imshow(name, 16*tmp + (1 << 14));
			cv::waitKey(1.0 * 1000);
			cv::destroyWindow(name);
		}
	}
#endif

	// Reconstruct
	hls_reconstruct(output_laplace_pyr[3], output_laplace_pyr[2], output_laplace_pyr[1], output_laplace_pyr[0],
			buf_dst, pyr_rows, pyr_cols);

	// Copy back
	dst.create(src.rows, src.cols, CV_16SC1);
	memcpy(dst.data, buf_dst, dst.rows*dst.cols * sizeof(signed short));
	dst.convertTo(dst, CV_32FC1, 1/2047.0);

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


void kernel(data_pyr_t* gau, data_pyr_t* temp_laplace_pyr, data_pyr_t* dst, int rows, int cols, float ref, float discretisation_step)
{
#pragma HLS INLINE
	float x_;
	int offset = 0;

	float g = 0;

	for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
		for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
			g = gau[offset] / 2047.0f;
			if (std::abs(g - ref) < discretisation_step) {
				x_ = 1 - std::abs(g - ref) / (discretisation_step );
				x_ = x_ * temp_laplace_pyr[offset];
			}
			else {
				x_ = 0;
			}
			dst[offset] = x_ + dst[offset];
			offset++;
		}
	}
}

// Accelerated function
// I:    Original image
// gau:  Pre-built Gaussian pyramid
// dst:  Remapped Laplacian pyramid
void hls_local_laplacian(
	data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2, data_pyr_t* gau3,
	data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2, data_pyr_t* lap3,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
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
		data_in_t* src,
		data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
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
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in(pyr_rows[0], pyr_cols[0]);

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp12(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp2(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp22(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp3(pyr_rows[3], pyr_cols[3]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp32(pyr_rows[3], pyr_cols[3]);

	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> out1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> out2(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> out3(pyr_rows[3], pyr_cols[3]);

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


void hls_laplacian_pyramid(
	data_in_t* src,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, data_pyr_t* dst3,
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

	// Input
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in(pyr_rows[0], pyr_cols[0]);

	// Output laplacian
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap0(pyr_rows[0], pyr_cols[0]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap1(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap2(pyr_rows[2], pyr_cols[2]);

	// Down-sampled image
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> down0(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> down1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> down2(pyr_rows[3], pyr_cols[3]);

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



//void hls_reconstruct(
//	float* src0, float* src1, float* src2, float* src3, data_out_t* dst, 
//	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_])
void hls_reconstruct(
	data_pyr_t* src0, data_pyr_t* src1, data_pyr_t* src2, data_pyr_t* src3, 
	signed short* dst,
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
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in0(pyr_rows[3], pyr_cols[3]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> in3(pyr_rows[0], pyr_cols[0]);

	// Up-sampled image
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> up1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> up2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> up3(pyr_rows[0], pyr_cols[0]);

	// SUm of updampled image + 1-layer up in the pyramid
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> sum1(pyr_rows[2], pyr_cols[2]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> sum2(pyr_rows[1], pyr_cols[1]);
	hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> sum3(pyr_rows[0], pyr_cols[0]);

	hls::Scalar<HLS_MAT_CN(_MAT_TYPE2_), HLS_TNAME(_MAT_TYPE2_)> px1;
	hls::Scalar<HLS_MAT_CN(_MAT_TYPE2_), HLS_TNAME(_MAT_TYPE2_)> px2;

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

	// Add & output
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


void remap(data_in_t* src, float* dst, float ref, float fact, float sigma, int rows, int cols)
{
#pragma HLS INLINE
	float I;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			I = src[r*cols + c]/2047.0; // [0, 1]
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

void remap(data_in_t* src, data_in_t* dst, float ref, float fact, float sigma, int rows, int cols)
{
#pragma HLS INLINE
    float tmp;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
            float I = src[r*cols + c]/2047.0; // [0, 1]
//            I = src[r*cols + c] / (float) ( ( 1 << (HLS_TBITDEPTH(_MAT_TYPE2_) - 1)) - 1 );
#ifdef _WIN32
            tmp = fact*(I - ref)*std::exp(-(I - ref)*(I - ref) / (2 * sigma*sigma));
#else
            tmp = fact*(I - ref)*hls::expf(-(I - ref)*(I - ref) / (2 * sigma*sigma));
#endif
            dst[r*cols + c] = (unsigned short)(tmp* 2047.0f);
        }
    }
}
