/*
* hls_def.h
*
*  Created on: 2017/12/11
*
*/

#ifndef _HLS_UTIL_H_
#define _HLS_UTIL_H_

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

namespace hls
{
	// Split a stream into two
	template<int ROWS, int COLS, int TYPE>
	void my_split(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst1,
		hls::Mat<ROWS, COLS, TYPE>& dst2)
	{
		int rows = src.rows;
		int cols = src.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
				src >> px;

				dst1 << px;
				dst2 << px;
			}
		}
	}


	// Elementwise addition
	template<int ROWS, int COLS, int TYPE>
	void add(
		hls::Mat<ROWS, COLS, TYPE>& src1,
		hls::Mat<ROWS, COLS, TYPE>& src2,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
		int rows = src1.rows;
		int cols = src1.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px1;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px2;

		for (int r = 0; r < rows; r++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
			for (int c = 0; c < cols; c++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
				src1 >> px1;
				src2 >> px2;

				dst << (px1 + px2);
			}
		}
	}


	template<int ROWS, int COLS, int TYPE, typename SRC_T>
	void load(SRC_T* src, hls::Mat<ROWS, COLS, TYPE>& dst)
	{
		int rows = dst.rows;
		int cols = dst.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
				px.val[0] = src[r*cols + c];
				dst << px;
			}
		}
	}

	template<int ROWS, int COLS, int TYPE, typename DST_T>
	void save(hls::Mat<ROWS, COLS, TYPE>& src, DST_T* dst)
	{
		int rows = src.rows;
		int cols = src.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
				src >> px;
				dst[r*cols + c] = px.val[0];
			}
		}
	}


	template<int ROWS, int COLS, int TYPE>
	void downsample2(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
		// Separable convolution
		// This weights sum to 20
		static const ap_uint<4> x[5] = { 1, 5, 8, 5, 1 };

		int rows = src.rows;
		int cols = src.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		int cols2 = cols >> 1;

#pragma HLS DATAFLOW
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_in;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_out;

		hls::Mat<ROWS, COLS, TYPE>			tmp0(rows, cols2);
		hls::Window<1, 5, HLS_TNAME(TYPE)>	buf1;	// Line buffer

		// Horizontal
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols + 2; c++) {
#pragma HLS PIPELINE
				if (c < cols) {
					// Load pixel from source
					src >> px_in;

					// Add to window buffer
					buf1.val[0][4] = px_in.val[0];
				}
				if (c > 0 && c < 2) {
					buf1.val[0][4 - c*2] = buf1.val[0][4];
				}

				// Convolution
				if (c % 2 == 0) {
					if (c >= 2) {
						px_out.val[0] = 
							(x[0] * buf1.val[0][0] +
							x[1] * buf1.val[0][1] +
							x[2] * buf1.val[0][2] +
							x[3] * buf1.val[0][3] +
							x[4] * buf1.val[0][4])/20;

						tmp0 << px_out;
					}

					// Shift to left
				}
					buf1.shift_pixels_left();
			}
		}

		// Vertical & sub sampling
		hls::LineBuffer<5, COLS/2, HLS_TNAME(TYPE)>	buf2;	// Line buffer
		for (int r = 0; r < rows + 2; r++) {
			for (int c = 0; c < cols2; c++) {
#pragma HLS PIPELINE
				if (r < rows) {
					// Load pixel
					tmp0 >> px_in;

					buf2.insert_bottom_row(px_in.val[0], c);
				}

				if (r > 0 && r < 2) {
					buf2.val[4 - r*2][c] = px_in.val[0];
				}

				if (r % 2 == 0 ) {
					if (r >= 2) {
						px_out.val[0] =
							(x[0] * buf2.val[0][c] +
							x[1] * buf2.val[1][c] +
							x[2] * buf2.val[2][c] +
							x[3] * buf2.val[3][c] +
							x[4] * buf2.val[4][c]) / 20;

						dst << px_out;
					}
				}
				
				buf2.shift_pixels_up(c);
			}
		}
	}

	template<int ROWS, int COLS, int TYPE>
	void downsample(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
		int rows = src.rows;
		int cols = src.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

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
		hls::Mat<ROWS, COLS, TYPE> tmp(rows, cols);
		hls::Filter2D(src, tmp, kernel, p);

		// Decimate
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;

#if 01
		for (int r = 0; r < rows; r++) {
//#pragma HLS LOOP_TRIPCOUNT max=1024
			for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
//#pragma HLS LOOP_TRIPCOUNT max=1024
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


	template<int ROWS, int COLS, int TYPE>
	void upsample(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
		int rows = dst.rows;
		int cols = dst.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

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
		hls::Mat<ROWS, COLS, TYPE>						tmp(rows, cols);
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px;
		hls::Window<1, ROWS, HLS_TNAME(TYPE)>			buf;	// Line buffer

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


	template<int ROWS, int COLS, int TYPE>
	void lap_kernel(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst_down,
		hls::Mat<ROWS, COLS, TYPE>& lap)
	{
#pragma HLS DATAFLOW

		// Source is split into: 1. for down-sampling, and 2. Difference
		hls::Mat<ROWS, COLS, TYPE> src_down(src.rows, src.cols);
		hls::Mat<ROWS, COLS, TYPE> src_diff(src.rows, src.cols);

		// Down-sampled result
		hls::Mat<ROWS, COLS, TYPE> tmp_down(dst_down.rows, dst_down.cols);
		// Input to up-sample
		hls::Mat<ROWS, COLS, TYPE> tmp_up_i(dst_down.rows, dst_down.cols);
		// Output of up-sample
		hls::Mat<ROWS, COLS, TYPE> tmp_up_o(src.rows, src.cols);

		my_split(src, src_down, src_diff);

		//
		downsample(src_down, tmp_down);
		my_split(tmp_down, dst_down, tmp_up_i);

		// Up-sampling
		upsample(tmp_up_i, tmp_up_o);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px1;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px2;
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
}


#endif	// _HLS_UTIL_H_