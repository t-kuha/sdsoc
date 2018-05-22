#include <iostream>
#include <assert.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "hw.h"

#ifdef __SDSCC__
#include "sds_lib.h"	// sds_***()
#endif


int main(int argc, char* argv[])
{
	std::cout << "-------- Vivado HLS Video Library --------" << std::endl;

	assert(argc == 2);

	// Image data
	cv::Mat src;
	cv::Mat dst;

	data_t* buf_src = NULL;
	data_t* buf_dst = NULL;

	// Load image
	src = cv::imread(argv[1]);

	if(src.data == NULL){
		// Is there image data?
		std::cerr << "" << std::endl;
		return -1;
	}

	// Accept only RGB image
	if(src.channels() != 3){
		std::cerr << "" << std::endl;
		return -1;
	}

	// Allocate contiguous memory & copy image data to it
#ifdef __SDSCC__
	buf_src = (data_t *) sds_alloc_non_cacheable(src.rows*src.cols*sizeof(data_t));
	buf_dst = (data_t *) sds_alloc_non_cacheable(src.rows*src.cols*sizeof(data_t));

	if( (buf_src == NULL) || (buf_dst == NULL) ){
		return -1;
	}
#else
	buf_src = new data_t [src.rows*src.cols];
	buf_dst = new data_t [src.rows*src.cols];
#endif

	// Copy data to buffer
	for(int r = 0; r < src.rows; r++){
		unsigned char* ptr = src.ptr<unsigned char>(r);
		for(int c = 0; c < src.cols; c++){
			buf_src[r*src.cols + c] =
					(ptr[3*c + 0] << 16) +
					(ptr[3*c + 1] << 8) +
					ptr[3*c + 2];
		}
	}

	// Processing
	std::cout << "\t HW Function Started..." << std::endl;

#ifdef __SDSCC__
	unsigned long long start = sds_clock_counter();
#endif

	hw_top(buf_src, buf_dst, src.rows, src.cols);

#ifdef __SDSCC__
	// Show processing time
	unsigned long long stop = sds_clock_counter();

	std::cout << "Average CPU Cycle: " << (stop - start) << std::endl;
#endif

	std::cout << "\t HW Function Finished..." << std::endl;

	// Retrieve processed data
	dst = cv::Mat(src.size(), src.type());

	// Copy data to buffer
	for(int r = 0; r < src.rows; r++){
		unsigned char* ptr = dst.ptr<unsigned char>(r);
		for(int c = 0; c < src.cols; c++){
			ptr[3*c + 0] = (buf_dst[r*src.cols + c] >> 16) & 0xFF;
			ptr[3*c + 1] = (buf_dst[r*src.cols + c] >>  8) & 0xFF;
			ptr[3*c + 2] = (buf_dst[r*src.cols + c]      ) & 0xFF;
		}
	}

	// Save processed image
	cv::imwrite("sdsoc.tif", dst);

	// Release memory
#ifdef __SDSCC__
	sds_free(buf_src);
	sds_free(buf_dst);
#else
	delete [] buf_src;
	delete [] buf_dst;
#endif

	std::cout << "----------------------------------------" << std::endl;

	return 0;
}
