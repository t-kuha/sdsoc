/*
 * main.cpp
 *
 *  Created on: 2017/08/19
 *      Author: kuriharat
 */



#include <iostream>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <algorithm>

#include "top.h"

#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include "foldedmv-offload.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;


void makeNetwork(network<mse, adagrad> & nn) {
  nn
#ifdef OFFLOAD
      << chaninterleave_layer<identity>(3, 32*32, false)
      << offloaded_layer(3*32*32, 10, &FixedFoldedMVOffload<8, 1>, 0xdeadbeef, 0)
#endif
      ;
}


int main(int argc, char* argv[])
{
	std::cout << "... BNN in SDSoC ..." << std::endl;

	std::string path = "";
	unsigned int number_class = 10;

	FoldedMVInit("cnv-pynq");

	network<mse, adagrad> nn;

	makeNetwork(nn);
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;

	parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
	std::vector<unsigned int> class_result;
	float usecPerImage_int;
	unsigned int results[64] = {0};
	class_result=testPrebuiltCIFAR10_from_image<8, 16>(test_images, number_class, usecPerImage_int);
	if(results)
		std::copy(class_result.begin(),class_result.end(), results);
//	if (usecPerImage)
//	    *usecPerImage = usecPerImage_int;
	int a = std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end()));

	return 0;
}
