/*
 * main.cpp
 *
 *  Created on: 2017/08/19
 *
 *      Define OFFLOAD & RAWHLS
 *      Top-Level function: BlackBoxJam()
 *
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


	FoldedMVInit("cnv-pynq");

	network<mse, adagrad> nn;

	makeNetwork(nn);

	std::string path_weight = "/Users/kuriharat/Downloads/BNN-PYNQ-master/bnn/params/cifar10";

	std::cout << "Setting network weights and thresholds in accelerator..." << std::endl;
    FoldedMVLoadLayerMem(path_weight , 0, L0_PE, L0_WMEM, L0_TMEM);
    FoldedMVLoadLayerMem(path_weight , 1, L1_PE, L1_WMEM, L1_TMEM);
    FoldedMVLoadLayerMem(path_weight , 2, L2_PE, L2_WMEM, L2_TMEM);
    FoldedMVLoadLayerMem(path_weight , 3, L3_PE, L3_WMEM, L3_TMEM);
    FoldedMVLoadLayerMem(path_weight , 4, L4_PE, L4_WMEM, L4_TMEM);
    FoldedMVLoadLayerMem(path_weight , 5, L5_PE, L5_WMEM, L5_TMEM);
    FoldedMVLoadLayerMem(path_weight , 6, L6_PE, L6_WMEM, L6_TMEM);
    FoldedMVLoadLayerMem(path_weight , 7, L7_PE, L7_WMEM, L7_TMEM);
    FoldedMVLoadLayerMem(path_weight , 8, L8_PE, L8_WMEM, L8_TMEM);


	std::string path = "/Users/kuriharat/Desktop/neuralnet/data/cifar10/cifar-10-batches-bin/test_batch.bin";
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;
	parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);

	std::cout << test_labels.size() << std::endl;
	for (int i = 0; i < 20; i++){
		std::cout << test_labels.at(i) << " ";
	}
	std::cout << std::endl;

	std::cout << test_images.size() << std::endl;

//	return 0;

	const unsigned int number_class = 10;
	std::vector<unsigned int> class_result;
	float usecPerImage_int = 0;
//	unsigned int results[64] = {0};
	class_result=testPrebuiltCIFAR10_from_image<8, 16>(test_images, number_class, usecPerImage_int);

	for(int i = 0; i < class_result.size(); i++){
		std::cout << class_result.at(i) << std::endl;
	}

//	std::copy(class_result.begin(),class_result.end(), results);
//	if (usecPerImage)
//	    *usecPerImage = usecPerImage_int;
	std::cout << "Output Class = " <<
			std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())) << std::endl;

	FoldedMVDeinit();

	std::cout << "--------------------" << std::endl;

	return 0;
}
