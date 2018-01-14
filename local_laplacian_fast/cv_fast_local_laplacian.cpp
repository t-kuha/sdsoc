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
    int num_levels = _MAX_LEVELS_;//std::ceil(std::log(std::min(src.rows, src.cols)) - log(2)) + 2;
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

#if 01
	// Show pyramid images
	for(int i = 0; i < num_levels; i++){
        cv::Mat s;
        input_gaussian_pyr.at(i).copyTo(s);
        s = s*255;
        s.convertTo(s, CV_8UC1);
        cv::imwrite("cv_gauss_" + std::to_string(i) + ".tif", s);
//		cv::imshow(std::to_string(i), input_gaussian_pyr.at(i));
//		cv::waitKey(0.3 * 1000);
//		cv::destroyWindow(std::to_string(i));
	}
	for (int i = 0; i < num_levels; i++) {
        cv::Mat s;
        output_laplace_pyr.at(i).copyTo(s);
        s = cv::abs(s)*255;
        s.convertTo(s, CV_8UC1);
        cv::imwrite("cv_laplacian_" + std::to_string(i) + ".tif", s);
//		cv::imshow(std::to_string(i), output_laplace_pyr.at(i) + 0.5);
//		cv::waitKey();
//		cv::destroyWindow(std::to_string(i));
	}
#endif

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

		for (int level = 0; level < num_levels - 1; level++) {
			cv::Mat one_2 = cv::Mat::ones(input_gaussian_pyr.at(level).rows, input_gaussian_pyr.at(level).cols, input_gaussian_pyr.at(level).type());

			cv::Mat tmp = one_2 - cv::abs(input_gaussian_pyr.at(level) - ref*one_2) / discretisation_step;
			tmp = tmp.mul(temp_laplace_pyr.at(level));

			cv::Mat tmp2;

			// cv::compare() returns CV_8UC1 [0, 255]:
			cv::compare(cv::abs(input_gaussian_pyr.at(level) - ref*one_2), discretisation_step/**ref*/, tmp2, cv::CMP_LT);
			tmp2.convertTo(tmp2, CV_32FC1, 1.0f / 255.0f);

			tmp = tmp.mul(tmp2);
			tmp = tmp + output_laplace_pyr.at(level);

			output_laplace_pyr.at(level) = tmp;
		}
	}

#if 0
	for(int i = 0; i < num_levels; i++){
		cv::imshow(std::to_string(i), output_laplace_pyr.at(i) + 0.5);
		cv::waitKey(0.3 * 1000);
		cv::destroyWindow(std::to_string(i));
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
	cv::filter2D(src, dst, -1, kernel);
    
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
    
    for(int r = 0; r < dst2.rows; r++){
        float* ptr = dst.ptr<float>(2*r);
        float* ptr2 = dst2.ptr<float>(r);
        
        for(int c = 0; c < dst2.cols; c++){
            ptr2[c] = ptr[2*c];
        }
    }
    
    return dst2;
#endif
}

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
	sz.width = cols;
	sz.height = rows;
	cv::resize(src, R, sz, 0.0, 0.0, cv::INTER_NEAREST);

	cv::filter2D(R, R, -1, kernel);

	return R;
}

void gaussian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels)
{
	// Finest level
	dst.push_back(src);

	// Down sample
	cv::Mat R = src;
	for (int i = 1; i < num_levels; i++) {
		// Add to pyramid
		R = downsample(R);
		dst.push_back(R);
	}
}

void laplacian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels)
{
	cv::Mat J = src;
	cv::Mat I = src;
	for (int l = 0; l < num_levels - 1; l++) {
		// apply low pass filter, and downsample
		I = downsample(J);

		// in each level, store difference between image and upsampled low pass version
		dst.push_back(J - upsample(I, J.rows, J.cols));

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
