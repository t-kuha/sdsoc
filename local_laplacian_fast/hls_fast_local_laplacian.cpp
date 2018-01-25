#include "stdafx.h"

#include <iostream>
#include <iomanip>

#include "hls_fast_local_laplacian.h"

//#include "hls_math.h"

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

	// Convert input image to 16-bit (signed)
	cv::Mat _src;		// Use '_src' hereafter
#ifdef _DATA_IS_FLOAT_
	src.convertTo(_src, CV_32FC1, _MAT_RANGE_);
#else
	src.convertTo(_src, CV_16SC1, _MAT_RANGE_ );
#endif

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
	buf_src = (data_in_t*) sds_alloc(_src.rows*_src.cols*sizeof(data_in_t));
	buf_dst = (data_out_t*) sds_alloc(_src.rows*_src.cols*sizeof(data_out_t));

	I_remap = (data_in_t*) sds_alloc(_src.rows*_src.cols*sizeof(data_in_t));
#else
	buf_src = new data_in_t[_src.rows*_src.cols];
	buf_dst = new data_out_t[_src.rows*_src.cols];

	I_remap = new data_in_t [_src.rows*_src.cols];
#endif

	// List for pyramid's widths & heights
	pyr_sz_t pyr_rows[_MAX_LEVELS_] = { 0 };
	pyr_sz_t pyr_cols[_MAX_LEVELS_] = { 0 };

	int rows = _src.rows;
	int cols = _src.cols;
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
	memcpy(buf_src, _src.data, _src.rows*_src.cols*sizeof(data_in_t));

#if 1
    hls_construct_pyramid(
        buf_src,
        input_gaussian_pyr[0], input_gaussian_pyr[1], input_gaussian_pyr[2], //input_gaussian_pyr[3],
        output_laplace_pyr[0], output_laplace_pyr[1], output_laplace_pyr[2], output_laplace_pyr[3],
        pyr_rows, pyr_cols);
#else
	// Construct Laplacian pyramid
#pragma SDS resource(1)
	hls_laplacian_pyramid(buf_src,
		output_laplace_pyr[0], output_laplace_pyr[1], output_laplace_pyr[2], output_laplace_pyr[3],
		pyr_rows, pyr_cols);

	// Gaussian Pyramid
	// Copy finest level
	memcpy(input_gaussian_pyr[0], buf_src, _src.rows*_src.cols * sizeof(data_pyr_t));

	hls_gaussian_pyramid(buf_src,
			input_gaussian_pyr[1], input_gaussian_pyr[2], input_gaussian_pyr[3],
			pyr_rows, pyr_cols);
#endif
    

#if 0
	{
		// Show pyramid image
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			hls_show_img(input_gaussian_pyr[l], pyr_rows[l], pyr_cols[l], 500, "hls_gauss_" + std::to_string(l));
			hls_save_img("hls_gauss_" + std::to_string(l), input_gaussian_pyr[l], pyr_rows[l], pyr_cols[l]);
		}
		for (int l = 0; l < _MAX_LEVELS_; l++) {
			hls_show_img(output_laplace_pyr[l], pyr_rows[l], pyr_cols[l], 500, "hls_laplacian_" + std::to_string(l));
			hls_save_img("hls_laplacian_" + std::to_string(l), output_laplace_pyr[l], pyr_rows[l], pyr_cols[l]);
		}
	}
#endif
	
	for (int n = 0; n < _NUM_STEP_; n++) {
		//float ref = ((float)n) / ((float)(_NUM_STEP_ - 1));

#if 1
        hls_laplacian_pyramid_remap(buf_src,
            temp_laplace_pyr[0], temp_laplace_pyr[1], temp_laplace_pyr[2], //temp_laplace_pyr[3],
            pyr_rows, pyr_cols, n/*ref*/, fact, 2*sigma*sigma);
#else
		// Remap original image
		remap(buf_src, I_remap, ref, fact, 2*sigma*sigma, pyr_rows[0], pyr_cols[0]);

		// Create laplacian pyramid from remapped image
#pragma SDS resource(1)
		hls_laplacian_pyramid(I_remap,
			temp_laplace_pyr[0], temp_laplace_pyr[1], temp_laplace_pyr[2], temp_laplace_pyr[3],
			pyr_rows, pyr_cols);
#endif
		
		hls_local_laplacian(
			input_gaussian_pyr[0], input_gaussian_pyr[1], input_gaussian_pyr[2], 
			temp_laplace_pyr[0], temp_laplace_pyr[1], temp_laplace_pyr[2], 
			output_laplace_pyr[0], output_laplace_pyr[1], output_laplace_pyr[2], 
			pyr_rows, pyr_cols, n/*, ref*/);
	}

	// Reconstruct
	hls_reconstruct(output_laplace_pyr[3], output_laplace_pyr[2], output_laplace_pyr[1], output_laplace_pyr[0],
			buf_dst, pyr_rows, pyr_cols);


	// Copy back
#ifdef _DATA_IS_FLOAT_
	dst.create(_src.rows, _src.cols, CV_32FC1);
	memcpy(dst.data, buf_dst, dst.rows*dst.cols * sizeof(float));
#else
	dst.create(_src.rows, _src.cols, CV_16SC1);
	memcpy(dst.data, buf_dst, dst.rows*dst.cols * sizeof(signed short));
	dst.convertTo(dst, CV_32FC1, 1/ ((float) _MAT_RANGE_) );
#endif

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

#if 0
// Kernel for hls_local_laplacian()
void kernel(data_pyr_t* gau, data_pyr_t* temp_laplace_pyr, data_pyr_t* dst, int rows, int cols, int step /*float ref, float discretisation_step*/)
{
#pragma HLS INLINE
	//const float discretisation_step = 1.0f / (_NUM_STEP_ - 1);

	float x_;

	int offset = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
			float g = gau[offset] / ((float) _MAT_RANGE_);	// [0, 1]
			g = (_NUM_STEP_ - 1)*g - /*ref*//*(float) */step;
			if (g < 0) {
				g = -g;
			}

			if (/*g < discretisation_step*/g/**(_NUM_STEP_ - 1)*/ < 1) {
				x_ = /*(_NUM_STEP_ - 1) */ (1 - g) /* (discretisation_step ) */;
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
#endif

// Accelerated function
// I:    Original image
// gau:  Pre-built Gaussian pyramid
// dst:  Remapped Laplacian pyramid
void hls_local_laplacian(
	data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2,
	data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
	/*float ref*/int step)
{
	//float discretisation_step = 1.0f / (_NUM_STEP_ - 1);

	pyr_sz_t pyr_rows[_MAX_LEVELS_];
	pyr_sz_t pyr_cols[_MAX_LEVELS_];
//#pragma HLS ARRAY_PARTITION variable=pyr_rows complete
//#pragma HLS ARRAY_PARTITION variable=pyr_cols complete

	for (int l = 0; l < _MAX_LEVELS_; l++) {
#pragma HLS PIPELINE
		pyr_rows[l] = pyr_rows_[l];
		pyr_cols[l] = pyr_cols_[l];
	}

	// Parallel execution
#ifdef _DATA_IS_FLOAT_
	// Layer 0
	hls::kernel(gau0, lap0, dst0, pyr_rows[0], pyr_cols[0], step /*ref, discretisation_step*/);
	// Layer 1
	hls::kernel(gau1, lap1, dst1, pyr_rows[1], pyr_cols[1], step /*ref, discretisation_step*/);
	// Layer 2
	hls::kernel(gau2, lap2, dst2, pyr_rows[2], pyr_cols[2], step /*ref, discretisation_step*/);
	// Not necessary for Layer 3
#else
	// Layer 0
	hls::kernel<HLS_TBITDEPTH(_MAT_TYPE2_)>(gau0, lap0, dst0, pyr_rows[0], pyr_cols[0], step /*ref, discretisation_step*/);
	// Layer 1
	hls::kernel<HLS_TBITDEPTH(_MAT_TYPE2_)>(gau1, lap1, dst1, pyr_rows[1], pyr_cols[1], step /*ref, discretisation_step*/);
	// Layer 2
	hls::kernel<HLS_TBITDEPTH(_MAT_TYPE2_)>(gau2, lap2, dst2, pyr_rows[2], pyr_cols[2], step /*ref, discretisation_step*/);
	// Not necessary for Layer 3
#endif
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

	// Transfer data
	save(lap0, dst0);
	save(lap1, dst1);
	save(lap2, dst2);
	save(down2, dst3);
}


void hls_laplacian_pyramid_remap(
	data_in_t* src,
	data_pyr_t* dst0, data_pyr_t* dst1, data_pyr_t* dst2, //data_pyr_t* dst3,
	pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_],
	/*float ref*/ int step, float fact, float sigma2)
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
    
    // Remapped image
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp(pyr_rows[0], pyr_cols[0]);
    
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
    
#ifdef _DATA_IS_FLOAT_
    hls::remap(in, tmp, step/*ref*/, fact, sigma2);
#else
	hls::remap<_MAX_ROWS_, _MAX_ROWS_, HLS_MAT_DEPTH(_MAT_TYPE2_)>(in, tmp, step, fact, sigma2);
#endif

    lap_kernel(tmp, down0, lap0);
    lap_kernel(down0, down1, lap1);
    lap_kernel(down1, down2, lap2);
    
    // Transfer data
    save(lap0, dst0);
    save(lap1, dst1);
    save(lap2, dst2);
    //save(down2, dst3);
	consume(down2);
}


void hls_reconstruct(
	data_pyr_t* src0, data_pyr_t* src1, data_pyr_t* src2, data_pyr_t* src3, 
	data_out_t* dst,
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

			dst[r*pyr_cols[0] + c] = hls::sr_cast<data_pyr_t>(px1.val[0] + px2.val[0]);
		}
	}
}


void hls_construct_pyramid(
                           data_in_t* src,
                           data_pyr_t* gau0, data_pyr_t* gau1, data_pyr_t* gau2, //data_pyr_t* gau3,
                           data_pyr_t* lap0, data_pyr_t* lap1, data_pyr_t* lap2, data_pyr_t* lap3,
                           pyr_sz_t pyr_rows_[_MAX_LEVELS_], pyr_sz_t pyr_cols_[_MAX_LEVELS_])
{
#pragma HLS DATAFLOW
    // Input stream
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> src_(pyr_rows_[0], pyr_cols_[0]);
    
    // Output
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> gau0_(pyr_rows_[0], pyr_cols_[0]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> gau1_(pyr_rows_[1], pyr_cols_[1]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> gau2_(pyr_rows_[2], pyr_cols_[2]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> gau3_(pyr_rows_[3], pyr_cols_[3]);
    
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap0_(pyr_rows_[0], pyr_cols_[0]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap1_(pyr_rows_[1], pyr_cols_[1]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap2_(pyr_rows_[2], pyr_cols_[2]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> lap3_(pyr_rows_[3], pyr_cols_[3]);
    
    // Input to lap_kernel()
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp0(pyr_rows_[0], pyr_cols_[0]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp1(pyr_rows_[1], pyr_cols_[1]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp2(pyr_rows_[2], pyr_cols_[2]);
    
    // Output from my_split()
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp_gau1(pyr_rows_[1], pyr_cols_[1]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp_gau2(pyr_rows_[2], pyr_cols_[2]);
    hls::Mat<_MAX_ROWS_, _MAX_ROWS_, _MAT_TYPE2_> tmp_gau3(pyr_rows_[3], pyr_cols_[3]);

    // Load image data
    load(src, src_);
    
    my_split(src_, gau0_, tmp0);  // G0
    
    lap_kernel(tmp0, tmp_gau1, lap0_);   // G1 & L0
    my_split(tmp_gau1, gau1_, tmp1);
    
    lap_kernel(tmp1, tmp_gau2, lap1_);   // G2 & L1
    my_split(tmp_gau2, gau2_, tmp2);
    
    lap_kernel(tmp2, lap3_/*tmp_gau3*/, lap2_);   // G3 (= L3) & L2
//    my_split(tmp_gau3, gau3_, lap3_);   // L3 (= G3)
    
    // Transfer data
    save(gau0_, gau0);
    save(gau1_, gau1);
    save(gau2_, gau2);
//    save(gau3_, gau3);

    save(lap0_, lap0);
    save(lap1_, lap1);
    save(lap2_, lap2);
    save(lap3_, lap3);
}


#if 0
// src: [0, 1]
// ref: [0, 1]
void remap(data_in_t* src, data_in_t* dst, float ref, float fact, float sigma2, int rows, int cols)
{
#pragma HLS INLINE
    float tmp;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
            float I = src[r*cols + c] / ((float)_MAT_RANGE_); // [0, 1]
			I = I - ref;
#ifdef __SDSVHLS__
            tmp = fact*I*hls::exp(-I*I / sigma2);
#else
            tmp = fact*I*std::exp(-I*I / sigma2);
#endif
            dst[r*cols + c] = (data_in_t)(tmp* ((float)_MAT_RANGE_));
        }
    }
}
#endif


bool hls_save_img(std::string name, data_pyr_t* img, int rows, int cols)
{
	cv::Mat tmp(rows, cols, CV_16SC1);

	short* buf = new short[rows * cols];
	memcpy(buf, img, rows * cols * sizeof(short));

	tmp.data = (unsigned char*)buf;

	tmp = cv::abs(tmp);
	tmp = (tmp / _MAT_RANGE_) * 255;
	tmp.convertTo(tmp, CV_8UC1);
	return cv::imwrite(name + ".tif", tmp);
}

void hls_show_img(data_pyr_t* img, int rows, int cols, int delay, std::string winname)
{
	cv::Mat tmp(rows, cols, CV_16SC1);

	short* buf = new short[rows * cols];
	memcpy(buf, img, rows * cols * sizeof(short));

	tmp.data = (unsigned char*)buf;

	tmp = cv::abs(tmp);
	tmp = (tmp / _MAT_RANGE_) * 255;
	tmp.convertTo(tmp, CV_8UC1);

	cv::imshow(winname, tmp);
	cv::waitKey(delay);
	cv::destroyWindow(winname);
}

void hls_print_value(data_pyr_t* img, int rows, int cols, std::string name)
{
	std::cout << "------- " << name << " --------" << std::endl;
	for (int r = 0; r < 10; r++) {
		for (int c = 0; c < 10; c++) {
			float tmp = img[r*cols + c];
			tmp /= _MAT_RANGE_;
			std::cout << std::setw(6) << std::setprecision(4) << tmp << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
