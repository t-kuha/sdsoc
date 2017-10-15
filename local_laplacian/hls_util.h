/*
 * hls_util.h
 *
 *  Created on: 2017/08/01
 *
 */

#ifndef HLS_UTIL_H_
#define HLS_UTIL_H_

// HLS (HLS Video Library) functions for SDSoC

#include	"hls/hls_video_types.h"
#include	"hls/hls_video_core.h"
#include	"hls/hls_axi_io.h"
#include	"ap_int.h"

#include	<assert.h>

namespace hls
{

template<int ROWS, int COLS, int DATA_T, typename FB_T>
void fb2hlsmat(FB_T* fb, int ch, hls::Mat<ROWS, COLS, DATA_T>& mat)
{
#pragma HLS INLINE

	HLS_SIZE_T rows = mat.rows;
	HLS_SIZE_T cols = mat.cols;

    assert(rows <= ROWS);
    assert(cols <= COLS);

    int fb_BitWidth = Type_BitWidth<FB_T>::Value;	//
    int depth = HLS_TBITDEPTH(DATA_T);
//    int _ch = HLS_MAT_CN(DATA_T);
//    std::cout << fb_BitWidth << " " << depth << " " << _ch << std::endl;
//    assert(fb_BitWidth >= _ch*depth && "Bit-Width of frame buffer must be greater than the total number of bits in a pixel");
    assert(ch == HLS_MAT_CN(DATA_T));

    Scalar<HLS_MAT_CN(DATA_T), HLS_TNAME(DATA_T)> pix;

	for(HLS_SIZE_T r = 0; r < rows; r++){
		for(HLS_SIZE_T c = 0; c < cols; c++){
#pragma HLS PIPELINE
			// Unpack
            for (HLS_CHANNEL_T k = 0; k < HLS_MAT_CN(DATA_T); k++) {
#pragma HLS UNROLL
            		pix.val[k] = fb[(r*cols + c)*HLS_MAT_CN(DATA_T) + k];
            }
			mat << pix;
		}
	}
}

// Copy only as required - no padding (margin) unlike hls::array2Mat()
//#pragma SDS data access_pattern (fb:SEQUENTIAL)
//#pragma SDS data copy (fb[0:"mat.rows*mat.cols"])
template<int ROWS, int COLS, int DATA_T, typename FB_T>
void fb2hlsmat(FB_T* fb, hls::Mat<ROWS, COLS, DATA_T>& mat)
{
#pragma HLS INLINE

	HLS_SIZE_T rows = mat.rows;
	HLS_SIZE_T cols = mat.cols;

    assert(rows <= ROWS);
    assert(cols <= COLS);

    int fb_BitWidth = Type_BitWidth<FB_T>::Value;	//
    int depth = HLS_TBITDEPTH(DATA_T);
    int ch = HLS_MAT_CN(DATA_T);
    std::cout << fb_BitWidth << " " << depth << " " << ch << std::endl;
    assert(fb_BitWidth >= ch*depth && "Bit-Width of frame buffer must be greater than the total number of bits in a pixel");

    Scalar<HLS_MAT_CN(DATA_T), HLS_TNAME(DATA_T)> pix;

	for(HLS_SIZE_T r = 0; r < rows; r++){
		for(HLS_SIZE_T c = 0; c < cols; c++){
#pragma HLS PIPELINE
			FB_T data = fb[r*cols + c];

            ap_uint<HLS_MAT_CN(DATA_T)*HLS_TBITDEPTH(DATA_T)> pix_value;
            AXISetBitFields(pix_value, 0, HLS_MAT_CN(DATA_T)*HLS_TBITDEPTH(DATA_T), data);

			// Unpack
            for (HLS_CHANNEL_T k = 0; k < HLS_MAT_CN(DATA_T); k++) {
#pragma HLS UNROLL
                AXIGetBitFields(pix_value, k*depth, depth, pix.val[k]);
            }

			mat << pix;
		}
	}
}


//#pragma SDS data access_pattern (fb:SEQUENTIAL)
//#pragma SDS data copy (fb[0:"mat.rows*mat.cols"])
template<int ROWS, int COLS, int DATA_T, typename FB_T>
void hlsmat2fb(hls::Mat<ROWS, COLS, DATA_T>& mat, FB_T* fb)
{
#pragma HLS INLINE

    HLS_SIZE_T rows = mat.rows;
    HLS_SIZE_T cols = mat.cols;

    assert(rows <= ROWS);
    assert(cols <= COLS);

    int fb_BitWidth = Type_BitWidth<FB_T>::Value;
    int depth = HLS_TBITDEPTH(DATA_T);
    int ch = HLS_MAT_CN(DATA_T);

    assert(fb_BitWidth >= ch*depth && "Bit-Width of frame buffer must be greater than the total number of bits in a pixel");

    Scalar<HLS_MAT_CN(DATA_T), HLS_TNAME(DATA_T)> pix;

    for (HLS_SIZE_T r = 0; r < rows; r++) {
        for (HLS_SIZE_T c = 0; c < cols; c++) {
#pragma HLS PIPELINE
            ap_uint<HLS_MAT_CN(DATA_T)*HLS_TBITDEPTH(DATA_T)> pix_value;
            mat >> pix;

            for (HLS_CHANNEL_T k = 0; k < HLS_MAT_CN(DATA_T); k++) {
#pragma HLS UNROLL
                AXISetBitFields(pix_value, k*depth, depth, pix.val[k]);
            }

            FB_T fb_pix;
            AXIGetBitFields(pix_value, 0, HLS_MAT_CN(DATA_T)*HLS_TBITDEPTH(DATA_T), fb_pix);
            fb[r*cols + c] = fb_pix;
        }
    }
}

};

#endif /* HLS_UTIL_H_ */
