/*
 * hls_def.h
 *
 *  Created on: 2017/12/09
 *
 */

#ifndef SRC_HLS_DEF_H_
#define SRC_HLS_DEF_H_


// Max. image size
#define		_MAX_ROWS_		1024
#define		_MAX_COLS_		1024

// Max. number of pyramid levels
#define		_MAX_LEVELS_		4

// Num. of descretization step
#define		_NUM_STEP_		10

//#define		_DATA_IS_FLOAT_

#ifdef _DATA_IS_FLOAT_
// Floating point implementation

// hls::Mat type
#define		_MAT_TYPE2_		HLS_32FC1

// Range of hls::Mat ( = 2^(16 - 1) - 1 )
#define		_MAT_RANGE_		1.0f

// Data types
typedef		float			data_in_t;		// Input data type
typedef		float 			data_out_t;		// Output data type
typedef		float			data_pyr_t;		// Data type for intermediate pyramid

#else
// Fixed point implementation
#include "ap_int.h"
#include "hls_math.h"				// hls::__isnan()
#include "hls/hls_video_types.h"	// HLS_TBITDEPTH()

// hls::Mat type
#define		_MAT_TYPE2_		HLS_16SC1

// Range of hls::Mat ( = 2^(16 - 1) - 1 )
#define		_MAT_RANGE_		( ( 1 << (HLS_TBITDEPTH(_MAT_TYPE2_) - 1)) - 1 )

// Data types
typedef		signed short	data_in_t;		// Input data type
typedef		signed short	data_out_t;		// output data type
typedef		ap_int<HLS_TBITDEPTH(_MAT_TYPE2_)>	data_pyr_t;		// Data type for intermediate pyramid

#endif


typedef		unsigned short	pyr_sz_t;		// Data type for pyramid size 

#endif /* SRC_HLS_DEF_H_ */
