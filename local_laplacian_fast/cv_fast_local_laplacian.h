#ifndef _CV_FAST_LOCAL_LAPLACIAN_
#define _CV_FAST_LOCAL_LAPLACIAN_


#include "opencv2/core/core.hpp"


void local_laplacian(cv::Mat& src, cv::Mat& dst, float sigma, float fact, int N);

cv::Mat downsample(cv::Mat& src);
cv::Mat upsample(cv::Mat& src, int rows, int cols);

void gaussian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels);
void laplacian_pyramid(const cv::Mat& src, std::vector< cv::Mat >& dst, int num_levels);
void construct_pyramid(const cv::Mat& src, std::vector< cv::Mat >& gau, std::vector< cv::Mat >& lap, int num_levels);

// For debugging
bool save_img(std::string name, cv::Mat& img);
void show_img(cv::Mat& img, int delay = 0, std::string winname = "img");

#endif
