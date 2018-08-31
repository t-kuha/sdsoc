#include <iostream>

#include "sds_lib.h"

#define		NUM_DATA		4096

void hw_func_1(int* src, int* dst, int count);
void hw_func_2(int* src, int* dst, int count);

int main(int argc, char* argv[])
{
	std::cout << "--------------------" << std::endl;

	int* buf1 = NULL;
	int* buf2 = NULL;
	int* tmp = NULL;

	// Allocate memory
	buf1 = (int*) sds_alloc(NUM_DATA * sizeof(int));
	if(!buf1){
		std::cerr << "Could not allocate memory..." << std::endl;
		return -1;
	}

	buf2 = (int*) sds_alloc(NUM_DATA * sizeof(int));
	if(!buf2){
		std::cerr << "Could not allocate memory..." << std::endl;
		return -1;
	}

#if 1
	tmp = (int*) sds_alloc(NUM_DATA * sizeof(int));
#else
	tmp = (int*) sds_alloc_non_cacheable(NUM_DATA * sizeof(int));
#endif
	if(!tmp){
		std::cerr << "Could not allocate memory..." << std::endl;
		return -1;
	}

	// Initialize
	for(int i = 0; i < NUM_DATA; i++){
		buf1[i] = i * 2;
		buf2[i] = 0;
	}

	// Accelerator 1
	hw_func_1(buf1, tmp, NUM_DATA);

	std::cout << "... Partition ..." << std::endl;

	// Accelerator 2
	hw_func_2(tmp, buf2, NUM_DATA);

	// Release memory
	if(buf1){
		sds_free(buf1);
	}

	if(buf2){
		sds_free(buf2);
	}

	if(tmp){
		sds_free(tmp);
	}

	std::cout << "--------------------" << std::endl;

	return 0;
}

#pragma SDS data copy(src[0:"count"])
#pragma SDS data copy(dst[0:"count"])
#pragma SDS data access_pattern(src:SEQUENTIAL, dst:SEQUENTIAL)
void hw_func_1(int* src, int* dst, int count)
{
	int v;
	for(unsigned int i = 0; i < count; i++){
#pragma HLS LOOP_TRIPCOUNT max=4096
		v = src[i];
		v = v * v;
		dst[i] = v;
	}
}

#pragma SDS data copy(src[0:"count"])
#pragma SDS data copy(dst[0:"count"])
#pragma SDS data access_pattern(src:SEQUENTIAL, dst:SEQUENTIAL)
void hw_func_2(int* src, int* dst, int count)
{
	int v;
	for(unsigned int i = 0; i < count; i++){
#pragma HLS LOOP_TRIPCOUNT max=4096
		v = src[i];
		v = v + 8;
		dst[i] = v;
	}

}
