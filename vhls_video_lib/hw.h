
#ifndef _HW_H_
#define _HW_H_

#define		_MAX_ROWS_		1080
#define		_MAX_COLS_		1920

typedef 	unsigned int 	data_t;

// Accelerated function
void hw_top(data_t* video_in, data_t* video_out, int rows, int cols);

#endif	// _HW_H_
