/*
 * main.cpp
 *
 *  Created on: 2017/10/01
 *
 */

#include <iostream>

#include "net.h"
#include "ap_int.h"

#define INPUT_IS_8

// HW-accelerated function
void accel(ap_uint<32> pbuf[32], unsigned char *predict_num);

int main(int argc, char* argv[])
{
	std::cout << "Binary Net (MNIST) test..." << std::endl;

	// Load input data
	const int input[32][32] = {
#include "input.h"
	};

	ap_uint<32> pbuf[32];

	std::cout << "READ TEST DATA" << std::endl;
	for (int yy = 0; yy < 32; yy++) {
		pbuf[yy] = 0;
	}
	for (int yy = 0; yy < 32; yy++) {
		ap_uint<32> temp = 0;
		for (int xx = 0; xx < 32; xx++) {
			if (input[yy][xx] == 1) {
				std::cout << "#";

				temp = (temp << 1) | 1;
			} else {
				std::cout << " ";

				temp = (temp << 1);
			}
		}

		pbuf[yy] = temp;
		std::cout << std::endl;
	}

	std::cout <<"START PREDICTION" << std::endl;

	unsigned char est = 0;

	accel(pbuf, &est);

	std::cout << "ESTIMATION = " << (int) est << std::endl; ;

	return 0;
}



#include "weight.h"
void accel(ap_uint<32> pbuf[32], unsigned char *predict_num)
{
	// Resource directive for weights
#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM

	// Tensor
	ap_uint<1> src[32][32];
#pragma HLS ARRAY_PARTITION variable=src complete dim=2

	ap_uint<1> tensor01[6*28*28];
	ap_uint<1> tensor12[6*14*14];
	ap_uint<1> tensor23[16*10*10];
	ap_uint<1> tensor34[16*5*5];
	ap_uint<1> tensor45[120];

	ap_int<24> result[10];		// Output score
#pragma HLS ARRAY_PARTITION variable=result complete

	// 入力された画像データをバ?��?ファメモリ(ping-pongメモリ)に格?��?
LOOP_INPUT_DATA:
	for (int yy = 0; yy < 32; yy++) {
#pragma HLS PIPELINE
		ap_uint<32> pict = pbuf[yy];
		for (int xx = 0; xx < 32; xx++) {
			if ((pict & 0x1) == 1) {
				src[yy][31 - xx] = 1;
			} else {
				src[yy][31 - xx] = 0;
			}
			pict = pict >> 1;
		}
	}

//	int LAYER, 	// Layer No.
//	int WX, 		// Width of convolution kernel
//	int WY,		// Height
//	int DX, 		// Stride
//	int DY,		//
//	int N_SMAP,	// Num. of input feature map
//	int SMAP_X,	// Width of feature map
//	int SMAP_Y,	// Height
//	int N_DMAP,	// Num. of output feature map
//	int DMAP_X,	// Width of output feature map
//	int DMAP_Y,	// Height
//	int SF,		// Scaling factor
//	int NUM_W,	// Num. of weight
//	int NUM_B	// Num. of bias

	// Layer 0
	net::layer0<0, 5, 5, 1, 1,  1, 32, 32,   6, 28, 28, 32,   150, 6>(&(src[0][0]), tensor01, coef_w_0, bias_0);

	// Layer 1
 	net::pool<1, 2, 2, 2, 2,  1, 28, 28,   6, 14, 14,  8,     6, 6>(tensor01, tensor12, coef_w_1, bias_1);

	// Layer 2
	net::conv<2, 5, 5, 1, 1, 6, 14, 14, 16, 10, 10, 32, 2400, 16>(tensor12, tensor23, coef_w_2, bias_2);

	// Layer 3
	net::pool<3, 2, 2, 2, 2,  1, 10, 10,  16,  5,  5,  8, 16, 16>(tensor23, tensor34, coef_w_3, bias_3);

	// Layer 4
	net::fc<4, 5, 5, 1, 1, 16,  5,  5, 120,  1,  1, 32,    48000,  120>(tensor34, tensor45, coef_w_4, bias_4);

	// Layer 5
	net::layer5<5, 1, 1, 1, 1, 120, 1, 1, 10, 1, 1, 1, 1200, 10>(tensor45, result, coef_w_5, bias_5);


	// Prediction ----------------------------------------------------
	ap_int<24> max_val = result[0];
	unsigned char max_idx = 0;
LOOP_OUTPUT:
	for (ap_uint<4> i = 1; i < 10; i++) {
#pragma HLS PIPELINE
		if (max_val < result[i]) {
			max_val = result[i];
			max_idx = i;
		}
//		std::cout <<"idx=" << i << "  \t" << result[i] << std::endl;;
	}
//	std::cout << "max index = " << (int) max_idx << std::endl;

	*predict_num = max_idx;
}
