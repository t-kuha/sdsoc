/*
 * main.cpp
 *
 *  Created on: Sep 21, 2017
 */

#include <iostream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "common/xf_common.h"
#include "imgproc/xf_histogram.hpp"

#define WIDTH 	1920
#define HEIGHT	1080
#define _NPPC 	XF_NPPC1


#pragma SDS data access_pattern("imgInput.data":SEQUENTIAL)
#pragma SDS data access_pattern(histogram:SEQUENTIAL)

#pragma SDS data copy("imgInput.data"[0:"imgInput.size"])
#pragma SDS data copy(histogram[0:256])
void histogram_accel(xf::Mat<XF_8UC1, HEIGHT, WIDTH, _NPPC> &imgInput, unsigned int *histogram)
{
//	xf::calcHist<XF_8UC1, HEIGHT, WIDTH, _NPPC> (imgInput, histogram);

	xf::xFHistogram<XF_8UC1, HEIGHT, WIDTH, XF_DEPTH(XF_8UC1,_NPPC), _NPPC, XF_WORDWIDTH(XF_8UC1,_NPPC)>(imgInput, histogram, imgInput.rows, imgInput.cols);

}


int main(int argc, char* argv[])
{
	std::cout << "------------------------------" << std::endl;

	if(argc != 2){
		std::cerr << "Invalid Number of Arguments!" << std::endl;
		std::cerr << "Usage:" << std::endl;
		std::cerr << "<Executable Name> <input image path>" << std::endl;;
		return -1;
	}

	cv::Mat src;
	cv::Mat gray;

#if __SDSCC__
	uint32_t *histogram = (uint32_t *)sds_alloc/*_non_cacheable*/(256*sizeof(uint32_t));
#else
	uint32_t histogram[256];
#endif

	//
	src = cv::imread(argv[1], 1);
	if (src.data == NULL){
		std::cerr << "Cannot open image: " << argv[1] << std::endl;
		return 0;
	}

	cvtColor(src, gray, CV_BGR2GRAY);

	xf::Mat<XF_8UC1, HEIGHT, WIDTH, _NPPC> xf_src(gray.rows, gray.cols);

	xf_src.copyTo(gray.data);

	histogram_accel (xf_src, histogram);

	// Show histogram shape
	int max_freq = -1;	// Max. frequency of histogram
	for(int i = 0; i < 256; i++){
		if(histogram[i] > max_freq){
			max_freq = histogram[i];
		}
	}

	if(max_freq <= 0){
		std::cerr << "Cannot calculate (normalized) histogram..." << std::endl;
		return 0;
	}

	for(int i = 0; i < 256; i++){
		int n = (histogram[i]*20)/max_freq;
		std::cout << std::setw(4) << i;
		for(int j = 0; j < n; j++){
			std::cout << "*";
		}
		std::cout << std::endl;
	}

#if __SDSCC__
	sds_free(histogram);
#endif

	return 0;
}
