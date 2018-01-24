/*
* hls_def.h
*
*  Created on: 2017/12/11
*
*/

#ifndef _HLS_UTIL_H_
#define _HLS_UTIL_H_

#include "hls_video.h"
#include "hls_math.h"


namespace hls
{
	// Show part of hls::Mat values
	template<int ROWS, int COLS, int TYPE>
	void print_value(hls::Mat<ROWS, COLS, TYPE>& mat, std::string name = "mat")
	{
		hls::Mat<ROWS, COLS, TYPE> tmp;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;

		std::cout << "----------- " << name << " ------------------" << std::endl;
		for (int r = 0; r < mat.rows; r++) {
			for (int c = 0; c < mat.cols; c++) {
				mat >> px;
				if ((r < 10) && (c < 10)) {
					std::cout << px.val[0] << " ";
				}

				tmp << px;
			}
			if (r < 10) {
				std::cout << std::endl;
			}
		}
		std::cout << std::endl;

		for (int r = 0; r < mat.rows; r++) {
			for (int c = 0; c < mat.cols; c++) {
				tmp >> px;
				mat << px;
			}
		}
	}


	// Split a stream into two
	template<int ROWS, int COLS, int TYPE>
	void my_split(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst1,
		hls::Mat<ROWS, COLS, TYPE>& dst2)
	{
#pragma HLS INLINE

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
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px3;
        
        for (int r = 0; r < rows; r++) {
            //#pragma HLS LOOP_TRIPCOUNT max=1024
            for (int c = 0; c < cols; c++) {
                //#pragma HLS LOOP_TRIPCOUNT max=1024
#pragma HLS PIPELINE
                src1 >> px1;
                src2 >> px2;
                
				px3.val[0] = hls::sr_cast<HLS_TNAME(TYPE)>(px1.val[0] + px2.val[0]);
				dst << px3;
                //dst << (px1 + px2);
            }
        }
    }
    
    
    // Elementwise subtraction
    // dst = src1 - src2
    template<int ROWS, int COLS, int TYPE>
    void sub(
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
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px3;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
                src1 >> px1;
                src2 >> px2;
                
				px3.val[0] = hls::sr_cast<HLS_TNAME(TYPE)>(px1.val[0] - px2.val[0]);
				dst << px3;
                //dst << (px1 - px2);
            }
        }
    }
    
    
    template<int ROWS, int COLS, int TYPE, typename SRC_T>
	void load(SRC_T* src, hls::Mat<ROWS, COLS, TYPE>& dst)
	{
#pragma HLS DATAFLOW
#pragma HLS INLINE

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
#pragma HLS DATAFLOW
#pragma HLS INLINE

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
	void downsample(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
#pragma HLS DATAFLOW

		// Separable convolution
		// This weights sum to 1
		//static const float x[5] = { .05f, .25f, .4f, .25f, .05f };
		static const int x[5] = { 1, 5, 8, 5, 1 };
		const int x_sum = 20;
		
		int rows = src.rows;
		int cols = src.cols;
		
		assert(rows <= ROWS);
		assert(cols <= COLS);

		// Width after horizontal convolution
		int cols2 = cols >> 1;

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_in;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_out;

		// Intermediate buffer
		hls::Mat<ROWS, COLS, TYPE>			tmp(rows, cols2);

		// Horizontal
		hls::Window<1, 5 - 1, HLS_TNAME(TYPE)>	buf_h;	// Line buffer (for horizoltal convolution)
		hls::Window<1, 5, HLS_TNAME(TYPE)>		cal_h;	// Calculation

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols + 2; c++) {
//#pragma HLS PIPELINE
				if (c < cols) {
					// Load pixel from source
					src >> px_in;
					// Add to window buffer
					cal_h.val[0][4] = px_in.val[0];
				}

				// Copy data to calculation buffer
				for (int j = 0; j < 4; j++) {
					cal_h.val[0][j] = buf_h.val[0][j];
				}

#if 0	// REEFLECT 101
				// Left edge
				if (c > 0 && c <= 2) {
					cal_h.val[0][4 - c * 2] = px_in.val[0];
				}
				// Right edge
				if (c >= cols) {
					cal_h.val[0][4] = cal_h.val[0][2 - (c - cols)*2];
				}
#else	// REFLECT
				// Left edge
				if (c >= 0 && c < 2) {
					cal_h.val[0][3 - c * 2] = px_in.val[0];
				}
				// Right edge
				if (c >= cols) {
					cal_h.val[0][4] = cal_h.val[0][3 - (c - cols) * 2];
				}
#endif

				// Convolution
				if (c % 2 == 0) {
					if (c >= 2) {
						px_out.val[0] =
							(x[0] * cal_h.val[0][0] +
								x[1] * cal_h.val[0][1] +
								x[2] * cal_h.val[0][2] +
								x[3] * cal_h.val[0][3] +
								x[4] * cal_h.val[0][4]) / x_sum;
						tmp << px_out;
					}
				}

				// Shift to left
				// Copy back from calculation buffer
				for (int j = 0; j < 4; j++) {
					buf_h.val[0][j] = cal_h.val[0][j + 1];
				}
			}
		}

		// Vertical
		hls::LineBuffer<5 - 1, COLS / 2, HLS_TNAME(TYPE)>	buf_v;	// Line buffer
		hls::Window<5, 1, HLS_TNAME(TYPE)>				cal_v;	// Calculation

		for (int r = 0; r < rows + 2; r++) {
			for (int c = 0; c < cols2; c++) {
#pragma HLS PIPELINE
				if (r < rows) {
					// Load pixel
					tmp >> px_in;
					cal_v.val[4][0] = px_in.val[0];
				}

				// Copy from buffer
				for (int i = 0; i < 5 - 1; i++) {
					 cal_v.val[i][0] = buf_v.val[i][c];
				}

#if 0	// REFLECT101
				// Top edge
				if (r > 0 && r <= 2) {
					cal_v.val[4 - r * 2][0] = px_in.val[0];
				}

				// Bottom edge
				if (r >= rows) {
					cal_v.val[4][0] = cal_v.val[2 - (r - rows) * 2][0];
				}
#else	// REFLECT
				// Top edge
				if (r >= 0 && r < 2) {
					cal_v.val[3 - r * 2][0] = px_in.val[0];
				}

				// Bottom edge
				if (r >= rows) {
					cal_v.val[4][0] = cal_v.val[3 - (r - rows) * 2][0];
				}
#endif

				if (r % 2 == 0) {
					if (r >= 2) {
						px_out.val[0] =
							(x[0] * cal_v.val[0][0] +
								x[1] * cal_v.val[1][0] +
								x[2] * cal_v.val[2][0] +
								x[3] * cal_v.val[3][0] +
								x[4] * cal_v.val[4][0]) / x_sum;
						dst << px_out;
					}
				}
		
				// Shift-up
				for (int i = 0; i < 5 - 1; i++) {
					buf_v.val[i][c] = cal_v.val[i + 1][0];
				}
			}
		}
	}

#if 0
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
		hls::Filter2D<hls::BORDER_REFLECT>(src, tmp, kernel, p);

		// Decimate
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px;

#if 1
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
#endif


	template<int ROWS, int COLS, int TYPE>
	void upsample(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst)
	{
#pragma HLS DATAFLOW

		// Size of up-sampled image
		int rows = dst.rows;
		int cols = dst.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		// Convolution Kernel - This sums to unity
		//static const float x[5] = { .05f, .25f, .4f, .25f, .05f };
		static const int x[5] = { 1, 5, 8, 5, 1 };
		const int x_sum = 20;

		// Width of input image
		int cols2 = src.cols;

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_in;
		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)>	px_out;

		// Intermediate buffer
		hls::Mat<ROWS, COLS, TYPE>			tmp(rows, cols2);


		// Veritcal up-sampling
		hls::Window<5, 1, HLS_TNAME(TYPE)>				cal_v;	// Calculation
		hls::LineBuffer<5 - 1, COLS / 2, HLS_TNAME(TYPE)>	buf_v;	// Line buffer

		for (int r = 0; r < rows + 2; r++) {
			for (int c = 0; c < cols2; c++) {
#pragma HLS PIPELINE
				if (r < rows) {
					if (r % 2 == 0) {
						// Load pixel
						src >> px_in;
						cal_v.val[4][0] = px_in.val[0];
					}
					else {
						// Fill 0 otherwise
						cal_v.val[4][0] = 0;
						px_in.val[0] = 0;
					}
				}

				// Copy from buffer
				for (int i = 0; i < 5 - 1; i++) {
					cal_v.val[i][0] = buf_v.val[i][c];
				}

				// Top edge
				if (r < 2) {
					cal_v.val[2][0] = px_in.val[0];
				}

				// Bottom edge
				if (r >= rows) {
					cal_v.val[4][0] = cal_v.val[2][0];
				}

				if (r >= 2) {
					px_out.val[0] =
						(x[0] * cal_v.val[0][0] +
							x[1] * cal_v.val[1][0] +
							x[2] * cal_v.val[2][0] +
							x[3] * cal_v.val[3][0] +
							x[4] * cal_v.val[4][0]) / x_sum;

					tmp << px_out;
				}

				// Shift-up
				for (int i = 0; i < 5 - 1; i++) {
					buf_v.val[i][c] = cal_v.val[i + 1][0];
				}
			}
		}

		// Horizontal
		hls::Window<1, 5 - 1, HLS_TNAME(TYPE)>	buf_h;	// Line buffer (for horizoltal convolution)
		hls::Window<1, 5, HLS_TNAME(TYPE)>		cal_h;	// Calculation

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols + 2; c++) {
#pragma HLS PIPELINE
				if (c < cols) {
					if (c % 2 == 0) {
						// Load pixel from source
						tmp >> px_in;
						// Add to window buffer
						cal_h.val[0][4] = px_in.val[0];
					}
					else {
						cal_h.val[0][4] = 0;
						px_in.val[0] = 0;
					}
				}

				// Copy data to calculation buffer
				for (int j = 0; j < 4; j++) {
					cal_h.val[0][j] = buf_h.val[0][j];
				}

				// Left edge
				if (c < 2) {
					cal_h.val[0][2] = px_in.val[0];
				}

				// Right edge
				if (c >= cols) {
					cal_h.val[0][4] = cal_h.val[0][2];
				}

				// Convolution
				if (c >= 2) {
					px_out.val[0] =
						(x[0] * cal_h.val[0][0] +
							x[1] * cal_h.val[0][1] +
							x[2] * cal_h.val[0][2] +
							x[3] * cal_h.val[0][3] +
							x[4] * cal_h.val[0][4]) * 4 / x_sum;
					dst << px_out;
				}

				// Shift to left
				// Copy back from calculation buffer
				for (int j = 0; j < 4; j++) {
					buf_h.val[0][j] = cal_h.val[0][j + 1];
				}
			}
		}

	}


	template<int ROWS, int COLS, int TYPE>
	void lap_kernel(
		hls::Mat<ROWS, COLS, TYPE>& src,
		hls::Mat<ROWS, COLS, TYPE>& dst_down,
		hls::Mat<ROWS, COLS, TYPE>& lap)
	{
#pragma HLS DATAFLOW
#pragma HLS INLINE

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
    
    
    template<int ROWS, int COLS, int TYPE, typename PARAM_TYPE>
    void remap(
        hls::Mat<ROWS, COLS, TYPE>& src,
        hls::Mat<ROWS, COLS, TYPE>& dst,
        /*PARAM_TYPE ref*/int step, PARAM_TYPE fact, PARAM_TYPE sigma2)
    {
#pragma HLS DATAFLOW

		PARAM_TYPE ref = (float)step / (_NUM_STEP_ - 1);

        int rows = dst.rows;
        int cols = dst.cols;
        
        assert(rows <= ROWS);
        assert(cols <= COLS);
        
        hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px_in;
        hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px_out;
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
                src >> px_in;
                
                // Remap
                float I = px_in.val[0] / ((float)_MAT_RANGE_); // [0, 1]
                I = I - ref;
#ifdef __SDSVHLS__
                float tmp = fact*I*hls::exp(-I*I / sigma2);
#else
                float tmp = fact*I*std::exp(-I*I / sigma2);
#endif
                px_out.val[0] = (data_in_t)(tmp* ((float)_MAT_RANGE_));
                
				dst << px_out;
            }
        }
    }


	template<int ROWS, int COLS, int TYPE>
	void consume(hls::Mat<ROWS, COLS, TYPE>& src)
	{
		int rows = src.rows;
		int cols = src.cols;

		assert(rows <= ROWS);
		assert(cols <= COLS);

		hls::Scalar<HLS_MAT_CN(TYPE), HLS_TNAME(TYPE)> px_in;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				src >> px_in;
			}
		}
	}
}


#endif	// _HLS_UTIL_H_
