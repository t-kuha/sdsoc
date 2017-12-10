/*
 * hls_util.cpp
 *
 *  Created on: 2017/12/09
 *
 */

#include "hls_util.h"


//void my_split(
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst1,
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst2)
//{
//	int rows = src.rows;
//	int cols = src.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
//			src >> px;
//
//			dst1 << px;
//			dst2 << px;
//		}
//	}
//}
//
//
//void downsample(
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
//{
//	int rows = src.rows;
//	int cols = src.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	//#pragma HLS INLINE
//#pragma HLS DATAFLOW
//
//	// Convolution Kernel - This sums to unity
//	static const float x[25] = {
//		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
//		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
//		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
//		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
//		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
//	hls::Window<5, 5, float> kernel;
//	for (int r = 0; r < 5; r++) {
//		for (int c = 0; c < 5; c++) {
//#pragma HLS PIPELINE
//			kernel.val[r][c] = x[r * 5 + c];
//		}
//	}
//
//	// Convolve
//	hls::Point p(-1, -1);
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
//	hls::Filter2D(src, tmp, kernel, p);
//
//	// Decimate
//	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
//
//#if 01
//	for (int r = 0; r < rows; r++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
//		for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
//#pragma HLS LOOP_TRIPCOUNT max=1024
//			tmp >> px;
//			if ((r % 2 == 0) && (c % 2 == 0)) {
//				dst << px;
//			}
//		}
//	}
//#else
//	for (int r = 0; r < rows2; r++) {
//		for (int c = 0; c < cols2; c++) {
//			// No if() statements
//#pragma HLS PIPELINE
//			tmp >> px;
//			dst << px;
//
//			// Consume
//			tmp >> px;
//		}
//		for (int c = 0; c < cols2; c++) {
//#pragma HLS PIPELINE
//			tmp >> px;
//		}
//	}
//#endif
//}
//
//
//void upsample(
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src,
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
//{
//	int rows = dst.rows;
//	int cols = dst.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	// Convolution Kernel - This sums to unity
//	static const float x[25] = {
//		0.0025, 0.0125, 0.0200, 0.0125, 0.0025,
//		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
//		0.0200, 0.1000, 0.1600, 0.1000, 0.0200,
//		0.0125, 0.0625, 0.1000, 0.0625, 0.0125,
//		0.0025, 0.0125, 0.0200, 0.0125, 0.0025 };
//	hls::Window<5, 5, float> kernel;
//	for (int r = 0; r < 5; r++) {
//		for (int c = 0; c < 5; c++) {
//#pragma HLS PIPELINE
//			kernel.val[r][c] = x[r * 5 + c];
//		}
//	}
//
//#pragma HLS DATAFLOW
//
//	// Up-scaling
//	hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_> tmp(rows, cols);
//	hls::Scalar<HLS_MAT_CN(_MAT_TYPE_), HLS_TNAME(_MAT_TYPE_)> px;
//	hls::Window<1, _MAX_ROWS_, HLS_TNAME(_MAT_TYPE_)> buf;	// Line buffer
//
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
//			if ((r % 2 == 0) && (c % 2 == 0)) {
//				src >> px;
//			}
//
//			if (r % 2 == 0) {
//				tmp << px;
//				buf.val[0][c] = px.val[0];
//			}
//			else {
//				tmp << buf.val[0][c];
//			}
//		}
//	}
//
//	// Convolve
//	hls::Point p(-1, -1);
//	hls::Filter2D(tmp, dst, kernel, p);
//}
//
//
//void add(
//		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src1,
//		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src2,
//		hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
//{
//	int rows = src1.rows;
//	int cols = src1.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	hls::Scalar<1, float> px1;
//	hls::Scalar<1, float> px2;
//
//	for (int r = 0; r < rows; r++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
//		for (int c = 0; c < cols; c++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
//#pragma HLS PIPELINE
//			src1 >> px1;
//			src2 >> px2;
//
//			dst << (px1 + px2);
//		}
//	}
//}
//
//
//void load(float* src, hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& dst)
//{
//	int rows = dst.rows;
//	int cols = dst.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	hls::Scalar<1, float> px;
//
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
//			px.val[0] = src[r*cols + c];
//			dst << px;
//		}
//	}
//}
//
//
//void save(hls::Mat<_MAX_ROWS_, _MAX_COLS_, _MAT_TYPE_>& src, float* dst)
//{
//	int rows = src.rows;
//	int cols = src.cols;
//
//	assert(rows <= _MAX_ROWS_);
//	assert(cols <= _MAX_COLS_);
//
//	hls::Scalar<1, float> px;
//
//	for (int r = 0; r < rows; r++) {
//		for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
//			src >> px;
//			dst[r*cols + c] = px.val[0];
//		}
//	}
//}
