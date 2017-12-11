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

// hls::Mat type
#define		_MAT_TYPE2_		HLS_12SC1

// Num. of descretization step
#define		_NUM_STEP_		10

// Data types
typedef		signed short		data_in_t;		// Input data type
typedef		signed short		data_out_t;		// output data type
typedef		signed short		data_pyr_t;		// Data type for intermediate pyramid


typedef		unsigned short	pyr_sz_t;		// Data type for pyramid size 


#endif /* SRC_HLS_DEF_H_ */
