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

// Data types
#define		_MAT_TYPE_			HLS_32FC1

// Num. of descretization step
#define		_NUM_STEP_				10


// Data types
typedef		float			data_in_t;		// Input data type
typedef		float			data_out_t;		// output data type

typedef		float			pipe_t;

typedef		unsigned short	pyr_sz_t;		// Data type for pyramid size 


#endif /* SRC_HLS_DEF_H_ */
