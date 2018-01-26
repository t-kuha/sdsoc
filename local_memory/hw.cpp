
#include "hw.h"

#if 0
//
// We cannot do this, which causes "Illegal memory access" error
// Refer to https://forums.xilinx.com/t5/SDSoC-Environment-and-reVISION/sdsoc-hardware-function-global-array-access-between-functions/td-p/706322
//
static int mem[NUM_ELEM];

void hw_write(int src[NUM_ELEM])
{
#pragma HLS RESOURCE variable=mem core=RAM_1P
	for(int i = 0; i < NUM_ELEM; i++){
		mem[i] = src[i];
	}
}


void hw_read(int dst[NUM_ELEM])
{
#pragma HLS RESOURCE variable=mem core=RAM_1P
	for(int i = 0; i < NUM_ELEM; i++){
		dst[i] = mem[i];
	}
}

#endif


#pragma SDS data zero_copy(data[0:NUM_ELEM])
void hw(int data[NUM_ELEM], int type){
	static int mem[NUM_ELEM] = {0};

	if (type == 0) {
		// Write
		for(int i = 0; i < NUM_ELEM; i++){
			mem[i] = data[i];
		}
	}else{
		// Read
		for(int i = 0; i < NUM_ELEM; i++){
			data[i] = mem[i];
		}
	}
}

