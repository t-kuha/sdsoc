/*
 * main.cpp
 *
 *  Created on: 2017/09/13
 *      Author: kuriharat
 */


#include <iostream>
#include <assert.h>
#include <string.h>

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

// xFOpenCV
#include "common/xf_common.h"

#include "ap_int.h"

//
#define  _MAX_IMG_COLS_	1920
#define  _MAX_IMG_ROWS_	1080


// HW-accelerated function
#pragma SDS data mem_attribute("_src.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute("_dst.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern("_src.data":SEQUENTIAL, "_dst.data":SEQUENTIAL)
#pragma SDS data copy("_src.data"[0:"_src.size"], "_dst.data"[0:"_dst.size"])
void hw(
		xf::Mat<XF_8UC4, _MAX_IMG_ROWS_, _MAX_IMG_COLS_, XF_NPPC1>& _src,
		xf::Mat<XF_8UC4, _MAX_IMG_ROWS_, _MAX_IMG_COLS_, XF_NPPC1>& _dst)
{
	// Use ap_int<12> instead of int to improve performance & resource utilization
	ap_uint<11> rows = _src.rows;
	ap_uint<11> cols = _src.cols;

	assert(rows <= _MAX_IMG_ROWS_);
	assert(cols <= _MAX_IMG_COLS_);

	// Just copy data...
#if 1
	for(ap_uint<11> r = 0; r < rows; r++){
#pragma HLS PIPELINE
		for(ap_uint<11> c = 0; c < cols; c++){
			_dst.data[r*cols + c] = _src.data[r*cols + c];
		}
	}
#else
	for(ap_uint<22> i = 0; i < rows*cols; i++){
#pragma HLS PIPELINE
		_dst.data[i] = _src.data[i];
	}
#endif
}

int main(int argc, char* argv[])
{
	std::cout << "---------------" << std::endl;

	if(argc != 2){
		std::cout << "\t Usage : " << argv[0] << "<input image>" << std::endl;
		return -1;
	}

	cv::Mat img;
	cv::Mat dst;

	// Load image (as RGB image)
	img = cv::imread(argv[1]);
	assert(img.data);

	std::cout << "\t Input image size: " << img.cols << " x " << img.rows << std::endl;

	// Copy to xF::Mat
	xf::Mat<XF_8UC4, _MAX_IMG_ROWS_, _MAX_IMG_COLS_, XF_NPPC1> xf_src;
	xf::Mat<XF_8UC4, _MAX_IMG_ROWS_, _MAX_IMG_COLS_, XF_NPPC1> xf_dst;

	xf_src.init(img.rows, img.cols);
	xf_dst.init(img.rows, img.cols);

	std::cout << xf_src.size << std::endl;	// Number of pixels
	// Data size in Byte
	std::cout << img.rows*(img.cols>>XF_BITSHIFT(XF_NPPC1))*(sizeof(XF_TNAME(XF_8UC4,XF_NPPC1))) << std::endl;

	//
	unsigned char* tmp = new unsigned char [xf_src.size*4];
	memset(tmp, 0, xf_src.size*4);

//	for(int r = 0; r < img.rows; r++){
//		unsigned char* ptr = img.ptr(r);
//		for(int c = 0; c < img.cols; c++){
//			for(int ch = 0; ch < img.channels(); ch++){
//				tmp[img.channels()*(r*img.cols + c) + ch] = ptr[img.channels()*c + ch];
//			}
//		}
//	}
	memcpy(tmp, img.data, img.rows*img.cols*img.channels());

	xf_src.copyTo(/*img.data*/tmp);

	// Run HW function
	hw(xf_src, xf_dst);

	/*dst.data*/tmp = xf_src.copyFrom();

	// Copy back to cv::Mat
	dst.create(xf_dst.rows, xf_dst.cols, CV_8UC3);

	memcpy(dst.data, tmp, dst.rows*dst.cols*dst.channels());
//	for(int r = 0; r < dst.rows; r++){
//		unsigned char* ptr = dst.ptr(r);
//		for(int c = 0; c < dst.cols; c++){
//			for(int ch = 0; ch < dst.channels(); ch++){
//				ptr[dst.channels()*c + ch] = tmp[dst.channels()*(r*dst.cols + c) + ch];
//			}
//		}
//	}

	// Save image
	bool ret = cv::imwrite("out.tif", dst);
	if(!ret){
		std::cerr << "Failed to save image..." << std::endl;
	}

	std::cout << "---------------" << std::endl;

	return 0;
}
