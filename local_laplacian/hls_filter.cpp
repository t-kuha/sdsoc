/*
 * hls_filter.cpp
 *
 *  Created on: 2017/10/09
 *      Author: kuriharat
 */

#include "hls_filter.h"

//hls_filter::hls_filter() {
//	// TODO Auto-generated constructor stub
//
//}
//
//hls_filter::~hls_filter() {
//	// TODO Auto-generated destructor stub
//}


#include "hls_video.h"

// SW only
#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "gaussian_pyramid.h"
#include "laplacian_pyramid.h"
//#include "opencv_utils.h"
#include "remapping_function.h"

template<typename T>
cv::Mat hls_laplacian(const cv::Mat& input,
                             double alpha,
                             double beta,
                             double sigma_r) {
  std::cout << "..." << __FUNCTION__ << "..." << std::endl;

  RemappingFunction r(alpha, beta);

  int num_levels = LaplacianPyramid::GetLevelCount(input.rows, input.cols, 30);
  std::cout << "Number of levels: " << num_levels << std::endl;

  const int kRows = input.rows;
  const int kCols = input.cols;

  GaussianPyramid gauss_input(input, num_levels);

  // Construct the unfilled Laplacian pyramid of the output. Copy the residual
  // over from the top of the Gaussian pyramid.
  LaplacianPyramid output(kRows, kCols, input.channels(), num_levels);
  gauss_input[num_levels].copyTo(output[num_levels]);

  // Calculate each level of the ouput Laplacian pyramid.
  for (int l = 0; l < num_levels; l++) {

	  // Accelerate this region
    int subregion_size = 3 * ((1 << (l + 2)) - 1);
    int subregion_r = subregion_size / 2;

    for (int y = 0; y < output[l].rows; y++) {
      // Calculate the y-bounds of the region in the full-res image.
      int full_res_y = (1 << l) * y;
      int roi_y0 = full_res_y - subregion_r;
      int roi_y1 = full_res_y + subregion_r + 1;
      cv::Range row_range(std::max(0, roi_y0), std::min(roi_y1, kRows));
      int full_res_roi_y = full_res_y - row_range.start;

      for (int x = 0; x < output[l].cols; x++) {
        // Calculate the x-bounds of the region in the full-res image.
        int full_res_x = (1 << l) * x;
        int roi_x0 = full_res_x - subregion_r;
        int roi_x1 = full_res_x + subregion_r + 1;
        cv::Range col_range(std::max(0, roi_x0), std::min(roi_x1, kCols));
        int full_res_roi_x = full_res_x - col_range.start;

        // Remap the region around the current pixel.
        cv::Mat r0 = input(row_range, col_range);
        cv::Mat remapped;
        r.Evaluate<T>(r0, remapped, gauss_input[l].at<T>(y, x), sigma_r);

        // Construct the Laplacian pyramid for the remapped region and copy the
        // coefficient over to the ouptut Laplacian pyramid.
        LaplacianPyramid tmp_pyr(remapped, l + 1,
            {row_range.start, row_range.end - 1,
             col_range.start, col_range.end - 1});
        output.at<T>(l, y, x) = tmp_pyr.at<T>(l, full_res_roi_y >> l,
                                                 full_res_roi_x >> l);
      }
      std::cout << "Level " << (l+1) << " (" << output[l].rows << " x "
           << output[l].cols << "), footprint: " << subregion_size << "x"
           << subregion_size << " ... " << round(100.0 * y / output[l].rows)
           << "%\r";
      std::cout.flush();
    }
    std::stringstream ss;
    ss << "level" << l << "_hls.png";
    cv::imwrite(ss.str(), ByteScale(cv::abs(output[l])));
    std::cout << std::endl;
  }

  return output.Reconstruct();
}
