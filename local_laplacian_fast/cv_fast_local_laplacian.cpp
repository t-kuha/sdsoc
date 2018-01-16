#include "stdafx.h"

#include "cv_fast_local_laplacian.h"
#include "hls_def.h"    // for _MAX_LEVELS_

#include <cmath>		// ceil() / exp()

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


void local_laplacian(cv::Mat& src, cv::Mat& dst, float sigma, float fact, int N)
{
	// Check input
	if (N <= 0) {
		return;
	}

	// Settings
	int num_levels = _MAX_LEVELS_;
	float discretisation_step = 1.0f / (N - 1);

	// Pyramids
	std::vector< cv::Mat > input_gaussian_pyr;
	std::vector< cv::Mat > output_laplace_pyr;

#if 0
	gaussian_pyramid(src, input_gaussian_pyr, num_levels);
	laplacian_pyramid(src, output_laplace_pyr, num_levels);
#else
	construct_pyramid(src, input_gaussian_pyr, output_laplace_pyr, num_levels);
#endif
	output_laplace_pyr.at(num_levels - 1) = input_gaussian_pyr.at(num_levels - 1);


#if 0
	// Show pyramid images
	for(int l = 0; l < num_levels; l++){
		save_img("cv_gauss_" + std::to_string(l), input_gaussian_pyr.at(l));
		//show_img(input_gaussian_pyr.at(l), 0.3 * 1000, "cv_gauss_" + std::to_string(l));
	}
	for (int l = 0; l < num_levels; l++) {
		save_img("cv_laplacian_" + std::to_string(l), output_laplace_pyr.at(l));
		//show_img(output_laplace_pyr.at(l), 0.3 * 1000, "cv_laplacian_" + std::to_string(l));
	}
#endif

#if 01
	cv::Mat I_remap;
	cv::Mat one_1 = cv::Mat::ones(src.rows, src.cols, src.type());
	for (int i = 0; i < N; i++) {
		float ref = ((float)i) / ((float)(N - 1));

		cv::Mat e;
		cv::Mat d = src - ref*one_1;
		e = d.mul(d);
		cv::exp(-e / (2 * sigma*sigma), e);
		I_remap = fact*d.mul(e);

		std::vector< cv::Mat > temp_laplace_pyr;
		laplacian_pyramid(I_remap, temp_laplace_pyr, num_levels);

		//save_img("cv_remap_" + std::to_string(i), I_remap);
		//save_img("cv_temp_0_" + std::to_string(i), temp_laplace_pyr.at(0));
		//save_img("cv_temp_1_" + std::to_string(i), temp_laplace_pyr.at(1));
		//save_img("cv_temp_2_" + std::to_string(i), temp_laplace_pyr.at(2));
		//save_img("cv_temp_3_" + std::to_string(i), temp_laplace_pyr.at(3));
		
		for (int level = 0; level < num_levels - 1; level++) {
			cv::Mat one_2 = cv::Mat::ones(input_gaussian_pyr.at(level).rows, input_gaussian_pyr.at(level).cols, input_gaussian_pyr.at(level).type());

			cv::Mat tmp = one_2 - cv::abs(input_gaussian_pyr.at(level) - ref*one_2) / discretisation_step;
			tmp = tmp.mul(temp_laplace_pyr.at(level));

			cv::Mat tmp2;
			// cv::compare() returns CV_8UC1 [0, 255]:
			cv::compare(cv::abs(input_gaussian_pyr.at(level) - ref*one_2), discretisation_step/**ref*/, tmp2, cv::CMP_LT);
			tmp2.convertTo(tmp2, CV_32FC1, 1.0f / 255.0f);

			output_laplace_pyr.at(level) += tmp.mul(tmp2);
		}
        
		//save_img("cv_out_lap_" + std::to_string(i), output_laplace_pyr.at(3));
 	}
#endif

#if 0
	for(int l = 0; l < num_levels; l++){
		save_img("cv_out_lap_" + std::to_string(l), output_laplace_pyr.at(l));
	}
#endif

	// Reconstruct
	dst = output_laplace_pyr.at(num_levels - 1);
	for (int i = num_levels - 2; i >= 0; i--) {
		int rows = output_laplace_pyr.at(i).rows;
		int cols = output_laplace_pyr.at(i).cols;
		dst = output_laplace_pyr.at(i) + upsample(dst, rows, cols);
	}
}

cv::Mat downsample(cv::Mat& src)
{
	// Convolution Kernel
	const float x[5] = { .05, .25, .4, .25, .05 };
	cv::Mat kernel = cv::Mat(5, 5, CV_32FC1);
	for (int r = 0; r < kernel.rows; r++) {
		for (int c = 0; c < kernel.rows; c++) {
			kernel.at<float>(r, c) = x[r] * x[c];
		}
	}

	// Convolve
	cv::Mat dst;
	cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    
	// Decimate
	cv::Size sz;
	sz.height = std::ceil(dst.rows / 2.0);
	sz.width = std::ceil(dst.cols / 2.0);
    
#if 0
    cv::resize(dst, dst, sz, 0.0, 0.0, cv::INTER_NEAREST);
	return dst;
#else
    // Create output matrix
    cv::Mat dst2;
    dst2.create(sz, dst.type());

	for (int r = 0; r < src.rows; r += 2) {
		float* ptr = dst.ptr<float>(r);
		float* ptr2 = dst2.ptr<float>(r/2);

		for (int c = 0; c < src.cols; c+=2) {
			ptr2[c/2] = ptr[c];
		}
	}
    
    return dst2;
#endif
}

// rows, cols: size after upsampling
cv::Mat upsample(cv::Mat& src, int rows, int cols)
{
	// FIXME: Dupliacted code
	const float x[5] = { .05, .25, .4, .25, .05 };
	cv::Mat kernel = cv::Mat(5, 5, CV_32FC1);
	for (int r = 0; r < kernel.rows; r++) {
		for (int c = 0; c < kernel.rows; c++) {
			kernel.at<float>(r, c) = x[r] * x[c];
		}
	}

	// Resize - Increase resolution
	cv::Mat R;
	cv::Size sz;

#if 0
	sz.width = cols;
	sz.height = rows;
	cv::resize(src, R, sz, 0.0, 0.0, cv::INTER_NEAREST);
#else
	// Original MATLAB implementation
	cv::Mat tmp;
	cv::copyMakeBorder(src, tmp, 1, 1, 1, 1, cv::BORDER_REPLICATE);

	sz.width = cols + 4;
	sz.height = rows + 4;
	
	R = cv::Mat::zeros(sz, src.type());

    for(int r = 0; r < R.rows; r+=2){
        float* ptr = tmp.ptr<float>(r/2);
        float* ptr2 = R.ptr<float>(r);
        
        for(int c = 0; c < R.cols; c+=2){
            ptr2[c] = ptr[c/2]*4;
        }
    }
    
	cv::filter2D(R, R, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

	// Crop
	cv::Rect roi = cv::Rect(2, 2, R.cols - 4, R.rows - 4);

	R = R(roi);

#endif

	return R;
}

void gaussian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels)
{
	// Finest level
	dst.push_back(src);

	// Down sample
	cv::Mat R;
	R = src.clone();
	for (int i = 1; i < num_levels; i++) {
		// Add to pyramid
		R = downsample(R);
		dst.push_back(R);
	}
}

void laplacian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels)
{
	cv::Mat J = src.clone();
	cv::Mat I = src.clone();
	for (int l = 0; l < num_levels - 1; l++) {
		// apply low pass filter, and downsample
		I = downsample(J);

		// in each level, store difference between image and upsampled low pass version
		cv::Mat tmp = upsample(I, J.rows, J.cols);
		dst.push_back(J - tmp);

		J = I; // continue with low pass image
	}

	dst.push_back(J);	// the coarsest level contains the residual low pass image
}

// gau: output gaussian pyramid
// lap: output laplacian pyramid
void construct_pyramid(const cv::Mat& src, std::vector< cv::Mat >& gau, std::vector< cv::Mat >& lap, int num_levels)
{
	// Finest level
	gau.push_back(src);

	cv::Mat J = src;
	cv::Mat I = src;
	for (int l = 0; l < num_levels - 1; l++) {
		// apply low pass filter, and downsample
		I = downsample(J);

		gau.push_back(I);

		// in each level, store difference between image and upsampled low pass version
		lap.push_back(J - upsample(I, J.rows, J.cols));

		J = I; // continue with low pass image
	}

	lap.push_back(J);	// the coarsest level contains the residual low pass image
}

bool save_img(std::string name, cv::Mat& img)
{
	cv::Mat tmp;
	img.copyTo(tmp);
	
	tmp = cv::abs(tmp) * 255;
	tmp.convertTo(tmp, CV_8UC1);
	
	return imwrite(name + ".tif", tmp);
}

void show_img(cv::Mat& img, int delay, std::string winname)
{
	cv::imshow(winname, img);
	cv::waitKey(delay);
	cv::destroyWindow(winname);
}
