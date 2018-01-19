#include "stdafx.h"

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "hls_def.h"
#include "cv_fast_local_laplacian.h"
#include "hls_fast_local_laplacian.h"


void rgb2gray(cv::Mat& rgb, cv::Mat& gray, std::vector<cv::Mat>& color);
void gray2rgb(cv::Mat& gray, std::vector<cv::Mat>& color, cv::Mat& rgb);


int main(int argc, char* argv[])
{
	std::cout << "----- Fast Local Laplacian Pyramid -----" << std::endl;

	if(argc != 2){
		std::cerr << "Usage: " << argv[0] << " <input image>" << std::endl;
		return -1;
	}

	// Parameters
	const float sigma = 0.1f;
	const float fact = 5.0f;

	// Load image
	cv::Mat img_input;
	img_input = cv::imread(std::string(argv[1]));
	if (img_input.data == NULL) {
		std::cerr << "Failed to load image: " << argv[1] << std::endl;
		return -1;
	}

	// Scaling: [0, 255] -> [0, 1]
	img_input.convertTo(img_input, CV_32FC3, 1.0f/255.0f);

	// Convert to grayscale
	cv::Mat gray;
	std::vector<cv::Mat> color;
	rgb2gray(img_input, gray, color);

	// Input for processing
	cv::Mat cv_in;
	cv::Mat hls_in;

	cv_in = gray.clone();
	hls_in = gray.clone();

#if 01
	// OpenCV implementation
	cv::Mat cv_out;
	cv::Mat cv_rgb;		// RGB image for output

	local_laplacian(cv_in, cv_out, sigma, fact, _NUM_STEP_);

	gray2rgb(cv_out, color, cv_rgb);

	cv_rgb.convertTo(cv_rgb, CV_8UC3, 255.0);

	// Show output image
	cv::imshow("Output - OpenCV", cv_rgb);
	cv::waitKey();
	cv::destroyWindow("Output - OpenCV");

	// Save output image
	cv::imwrite("cv.tif", cv_rgb);
#endif

	// HLS implementation
	cv::Mat hls_out;		// Enhanced grayscale image
	cv::Mat hls_rgb;		// RGB image for output

	hls_local_laplacian_wrap(hls_in, hls_out, sigma, fact);

	gray2rgb(hls_out, color, hls_rgb);

	hls_rgb.convertTo(hls_rgb, CV_8UC3, 255.0);

	cv::imshow("Output - HLS", hls_rgb);
	cv::waitKey();
	cv::destroyWindow("Output - HLS");

	cv::imwrite("hls.tif", hls_rgb);

#if 01
    cv::Mat diff = cv::abs(cv_rgb - hls_rgb)*255;
    
    cv::imshow("Difference", diff);
    cv::waitKey();
    cv::destroyWindow("Difference");
#endif

	std::cout << "----- Done -----" << std::endl;

    return 0;
}


#define _MATLAB
void rgb2gray(cv::Mat& rgb, cv::Mat& gray, std::vector<cv::Mat>& color)
{
#ifdef _MATLAB
	// Original MATLAB implementation
	cv::split(rgb, color);

	gray = 0.2989*color.at(2) + 0.5870*color.at(1) + 0.1140*color.at(0);

	for (int c = 0; c < color.size(); c++) {
		cv::divide(color.at(c), gray, color.at(c));
	}

#else
	// Standard RGB -> YUV conversion
	cv::Mat yuv;
	cv::cvtColor(rgb, yuv, CV_BGR2YUV);

	cv::split(yuv, color);

	gray = color.at(0).clone();
#endif
}

void gray2rgb(cv::Mat& gray, std::vector<cv::Mat>& color, cv::Mat& rgb)
{
#ifdef _MATLAB
	// Original MATLAB implementation
	std::vector<cv::Mat> color2;

	for (int c = 0; c < color.size(); c++) {
		cv::Mat tmp;
		cv::multiply(color.at(c), gray, tmp);
		color2.push_back(tmp);
	}

	cv::merge(color2, rgb);

#else
	// Standard YUV -> RGB conversion
	cv::Mat tmp;
	color.at(0) = gray;
	cv::merge(color, tmp);

	// YUV -> RGB
	cv::cvtColor(tmp, rgb, CV_YUV2BGR);
#endif
}