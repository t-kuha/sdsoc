/*
 * hls_gaussian_pyramid.h
 *
 *  Created on: 2017/10/12
 *      Author: kuriharat
 */

#ifndef HLS_GAUSSIAN_PYRAMID_H_
#define HLS_GAUSSIAN_PYRAMID_H_

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>


template<typename T, int CH>
class hlsGaussianPyramid {
public:
	// Indicates that this is a subimage. If the start index is odd, this is
	// necessary to make the higher levels the correct size.
	hlsGaussianPyramid(const std::vector<int>& subwindow)
	: pyramid_(), subwindow_(subwindow)
	{
		// DO NOTHING
		for (int i = -2; i <= 2; i++) {
			for (int j = -2; j <= 2; j++) {
				filter_[i + 2][j + 2] =
					WeightingFunction(i, kA) * WeightingFunction(j, kA);
			}
		}
	};


	void construct(const cv::Mat& image, int num_levels/*,
		const std::vector<int>& subwindow*/)

	{
		pyramid_.reserve(num_levels + 1);
		pyramid_.emplace_back();
		image.convertTo(pyramid_.back(), CV_64F);

		// This test verifies that the image is large enough to support the requested
		// number of levels.
		if (image.cols >> num_levels == 0 || image.rows >> num_levels == 0) {
			std::cerr << "Warning: Too many levels requested. Image size "
				<< image.cols << " x " << image.rows << " and  " << num_levels
				<< " levels wer requested." << std::endl;
		}

		for (int l = 0; l < num_levels; l++) {
			const cv::Mat& previous = pyramid_.back();

			// Get the subwindows of the previous level and the current one.
			std::vector<int> prev_subwindow, current_subwindow;
			GetLevelSize(pyramid_.size() - 1, &prev_subwindow);
			GetLevelSize(pyramid_.size(), &current_subwindow);

			const int kRows = current_subwindow[1] - current_subwindow[0] + 1;
			const int kCols = current_subwindow[3] - current_subwindow[2] + 1;

			// If the subwindow starts on even indices, then (0,0) of the new level is
			// centered on (0,0) of the previous level. Otherwise, it's centered on
			// (1,1).
			int row_offset = ((prev_subwindow[0] % 2) == 0) ? 0 : 1;
			int col_offset = ((prev_subwindow[2] % 2) == 0) ? 0 : 1;

			// Push a new level onto the top of the pyramid.
			pyramid_.emplace_back(kRows, kCols, previous.type());
			//cv::Mat& next = pyramid_.back();

			// Populate the next level.
			PopulateTopLevel/*< cv::Vec<T, CH> >*/(row_offset, col_offset);
//			if (next.channels() == 1) {
//				PopulateTopLevel<double>(row_offset, col_offset);
//			}
//			else if (next.channels() == 3) {
//				PopulateTopLevel<cv::Vec3d>(row_offset, col_offset);
//			}
		}
	}

	// 
	const cv::Mat& operator[](int level) const { return pyramid_[level]; }


	static void GetLevelSize(const std::vector<int> base_subwindow,
		int level,
		std::vector<int>* subwindow) {
		subwindow->clear();
		subwindow->insert(begin(*subwindow),
			begin(base_subwindow), end(base_subwindow));

		for (int i = 0; i < level; i++) {
			(*subwindow)[0] = ((*subwindow)[0] >> 1) + (*subwindow)[0] % 2;
			(*subwindow)[1] = (*subwindow)[1] >> 1;
			(*subwindow)[2] = ((*subwindow)[2] >> 1) + (*subwindow)[2] % 2;
			(*subwindow)[3] = (*subwindow)[3] >> 1;
		}
	}


	void GetLevelSize(int level, std::vector<int>* subwindow) const {
		GetLevelSize(subwindow_, level, subwindow);
	}

//	template<typename T>
	void PopulateTopLevel(int row_offset, int col_offset) {
		cv::Mat& previous = pyramid_[pyramid_.size() - 2];
		cv::Mat& top = pyramid_.back();

		// Calculate the end indices, based on where (0,0) is centered on the
		// previous level.
		const int kEndRow = row_offset + 2 * top.rows;
		const int kEndCol = col_offset + 2 * top.cols;
		for (int y = row_offset; y < kEndRow; y += 2) {
			for (int x = col_offset; x < kEndCol; x += 2) {
				cv::Vec<T, CH> value = 0;
				double total_weight = 0;

				int row_start = std::max(0, y - 2);
				int row_end = std::min(previous.rows - 1, y + 2);
				for (int n = row_start; n <= row_end; n++) {
					double row_weight = WeightingFunction(n - y, kA);

					int col_start = std::max(0, x - 2);
					int col_end = std::min(previous.cols - 1, x + 2);
					for (int m = col_start; m <= col_end; m++) {
						double weight = row_weight * WeightingFunction(m - x, kA);
						total_weight += weight;
						value += weight * previous.at< cv::Vec<T, CH> >(n, m);
					}
				}
				top.at< cv::Vec<T, CH> >(y >> 1, x >> 1) = value / total_weight;
			}
		}
	}

//	template<typename T>
	static void Expand(const cv::Mat& input,
		int row_offset,
		int col_offset,
		cv::Mat& output) {
		cv::Mat upsamp = cv::Mat::zeros(output.rows, output.cols, input.type());
		cv::Mat norm = cv::Mat::zeros(output.rows, output.cols, CV_64F);

		for (int i = row_offset; i < output.rows; i += 2) {
			for (int j = col_offset; j < output.cols; j += 2) {
				upsamp.at< cv::Vec<T, CH> >(i, j) = input.at< cv::Vec<T, CH> >(i >> 1, j >> 1);
				norm.at<double>(i, j) = 1;
			}
		}

		//cv::Mat filter(5, 5, CV_64F);
		//for (int i = -2; i <= 2; i++) {
		//	for (int j = -2; j <= 2; j++) {
		//		filter.at<double>(i + 2, j + 2) =
		//			WeightingFunction(i, kA) * WeightingFunction(j, kA);
		//	}
		//}

		for (int i = 0; i < output.rows; i++) {
			int row_start = std::max(0, i - 2);
			int row_end = std::min(output.rows - 1, i + 2);
			for (int j = 0; j < output.cols; j++) {
				int col_start = std::max(0, j - 2);
				int col_end = std::min(output.cols - 1, j + 2);

				cv::Vec<T, CH> value = 0;
				double total_weight = 0;
				for (int n = row_start; n <= row_end; n++) {
					for (int m = col_start; m <= col_end; m++) {
						double weight = filter_[n - i + 2][m - j + 2];
						value += weight * upsamp.at< cv::Vec<T, CH> >(n, m);
						total_weight += weight * norm.at<double>(n, m);
					}
				}
				output.at< cv::Vec<T, CH> >(i, j) = value / total_weight;
			}
		}
	}


	// Expand the given level a set number of times. The argument times must be
	// less than or equal to level, since the pyramid is used to determine the
	// size of the output. Having level equal to times will upsample the image to
	// the initial pixel dimensions.
//	template<typename T, int CH>
	cv::Mat ExpandOnce(int level) const {
		cv::Mat base = pyramid_[level], expanded;

		std::vector<int> subwindow;
		GetLevelSize(/*subwindow_,*/ level - 1, &subwindow);

		int out_rows = pyramid_[level - 1].rows;
		int out_cols = pyramid_[level - 1].cols;
		expanded.create(out_rows, out_cols, base.type());

		int row_offset = ((subwindow[0] % 2) == 0) ? 0 : 1;
		int col_offset = ((subwindow[2] % 2) == 0) ? 0 : 1;

		Expand/*< cv::Vec<T, CH> >*/(base, row_offset, col_offset, expanded);

		return expanded;
	}


	// 3x3 kernel
	// i = -2, -1, 0, 1, 2
	// a = 0.3 - Broad blurring Kernel
	// s = 0.4   Gaussian-like kernel
	// a = 0.5 - Triangle
	// a = 0.6 - Trimodal (Negative lobes)
	static double WeightingFunction(int i, double a)
	{
		switch (i) {
		case 0: return a;
		case -1: case 1: return 0.25;
		case -2: case 2: return 0.25 - 0.5 * a;
		}
		return 0;
	}


private:
	std::vector<cv::Mat> pyramid_;
	std::vector<int> subwindow_;

	constexpr static const double kA = 0.4;

	//cv::Mat filter_;
	double filter_[5][5];
};

#endif /* HLS_GAUSSIAN_PYRAMID_H_ */
