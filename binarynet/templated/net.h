/*
 * net.h
 *
 *  Created on: 2017/10/01
 *
 */

#ifndef NET_H_
#define NET_H_

#include "ap_int.h"

namespace net {

template<
int LAYER, 	// Layer No.
int WX, 		// Width of convolution kernel
int WY,		// Height
int DX, 		// Stride
int DY,		//
int N_SMAP,	// Num. of input feature map
int SMAP_X,	// Width of feature map
int SMAP_Y,	// Height
int N_DMAP,	// Num. of output feature map
int DMAP_X,	// Width of output feature map
int DMAP_Y,	// Height
int SF,		// Scaling factor
int NUM_W,	// Num. of weight
int NUM_B	// Num. of bias
>
void conv(ap_uint<1>* ping, ap_uint<1>* pong, const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
{
#pragma HLS INLINE

	ap_uint<16>/*int*/ idx = 0;

	int x = 0, y = 0;

LAYER_CONV:
	for(int dmap = 0; dmap < N_DMAP; dmap++){
#pragma HLS LOOP_FLATTEN off
		for(int i = 0; i < DMAP_X * DMAP_Y; i++){
#pragma HLS PIPELINE
			ap_int<24> temp = 0;
			ap_int<2> dat;
			ap_int<7> coef;

			int s = 0;
			for(int smap = 0; smap < N_SMAP; smap++){
				for(int oy = 0; oy < WY; oy++){
					for(int ox = 0; ox < WX; ox++){
						if( ping[smap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)] == 1){
							dat = 1;
						}else{
							dat = -1;
						}

						coef = weight[N_SMAP*WX*WY*dmap + s];
						s++;
						temp += (dat * coef);

					} // end for oy
				} // end for ox
			} // end for smap

			// Activation function for the BinaryNet
			temp = temp * SF;
			temp = temp + bias[dmap];

			if( temp >= 0){
				temp = 1;
			}else{
				temp = -1;
			}

			// Store ping-pong memory
			if( temp == 1){
				pong[idx] = 1;
			}else{
				pong[idx] = 0;
			}

			// Update indices
			idx++;
			x += DX;
			if( x > (SMAP_X - WX)){
				x = 0;
				y += DY;
				if( y > (SMAP_Y - WY)){
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}


template<
int LAYER, 	// Layer No.
int WX, 		// Width of convolution kernel
int WY,		// Height
int DX, 		// Stride
int DY,		//
int N_SMAP,	// Num. of input feature map
int SMAP_X,	// Width of feature map
int SMAP_Y,	// Height
int N_DMAP,	// Num. of output feature map
int DMAP_X,	// Width of output feature map
int DMAP_Y,	// Height
int SF,		// Scaling factor
int NUM_W,	// Num. of weight
int NUM_B	// Num. of bias
>
void pool(ap_uint<1>* ping, ap_uint<1>* pong, const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
{
#pragma HLS INLINE

//#pragma HLS RESOURCE variable=coef_w_1 core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=bias_1 core=ROM_2P_LUTRAM

	ap_uint<14> x = 0, y = 0;
	ap_uint<16> idx = 0;

LAYER_POOL:
	for (/*ap_uint<3>*/int dmap = 0; dmap < N_DMAP; dmap++) {
#pragma HLS LOOP_FLATTEN off
		for (/*ap_uint<8>*/int i = 0; i < DMAP_X * DMAP_Y; i++) {
#pragma HLS PIPELINE
			ap_int<24> temp = 0;
//#pragma HLS RESOURCE variable=temp core=AddSub
			for (/*ap_uint<2>*/int  oy = 0; oy < WY; oy++) {
				for (/*ap_uint<2>*/int  ox = 0; ox < WX; ox++) {
					ap_int<2> dat;
					ap_int<7> coef;

					if (ping[dmap * (SMAP_X * SMAP_Y) + (y + oy) * SMAP_X + (x + ox)]
							== 1){
						dat = 1;
					}else{
						dat = -1;
					}

					coef = weight[dmap];

					temp += (dat * coef);
				} // end for oy
			} // end for ox

			ap_int<7> bi = bias[dmap];

			ap_int<24> temp0 = temp * SF;
//#pragma HLS RESOURCE variable=temp0 core=Mul
			ap_int<24> temp2 = temp0 + bi;
//#pragma HLS RESOURCE variable=temp2 core=AddSub
			if (temp2 >= 0) {
				pong[idx] = 1;
			} else {
				pong[idx] = 0;
			}

			// Update indices
			idx++;
			x += WX;
			if (x > (SMAP_X - WX)) {
				x = 0;
				y += WY;
				if (y > (SMAP_Y - WY)) {
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}


template<
int LAYER, 	// Layer No.
int WX, 		// Width of convolution kernel
int WY,		// Height
int DX, 		// Stride
int DY,		//
int N_SMAP,	// Num. of input feature map
int SMAP_X,	// Width of feature map
int SMAP_Y,	// Height
int N_DMAP,	// Num. of output feature map
int DMAP_X,	// Width of output feature map
int DMAP_Y,	// Height
int SF,		// Scaling factor
int NUM_W,	// Num. of weight
int NUM_B	// Num. of bias
>
void fc(ap_uint<1>* ping, ap_uint<1>* pong, const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
{
#pragma HLS INLINE

//#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM
//
//#pragma HLS ARRAY_PARTITION variable=coef_w_4 cyclic factor=4
//#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM

	ap_int<22> tmp[NUM_B];// = {0};
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=4

	ap_uint<24> idx0 = 0;
	ap_uint<12> idx1 = 0;

LAYER_FC:
	for (int/*ap_uint<5>*/ smap = 0; smap < N_SMAP; smap++) {
#pragma HLS LOOP_FLATTEN off
		ap_int<22> temp = 0;
		for (ap_uint<3> oy = 0; oy < WX; oy++) {
#pragma HLS LOOP_FLATTEN off
			for (ap_uint<3> ox = 0; ox < WY; ox++) {
#pragma HLS LOOP_FLATTEN off
				ap_int<2> dat;
				ap_int<7> coef;

				if (ping[idx1] == 1){
					dat = 1;
				}else{
					dat = -1;
				}
				for (ap_uint<7> dmap = 0; dmap < N_DMAP; dmap++) {
#pragma HLS UNROLL factor=4
#pragma HLS PIPELINE
					coef = weight[idx0++];

					tmp[dmap] += (dat * coef);
				}// end for dmap
				idx1++;
			} // end for oy
		} // end for ox
	} // end for smap

	//
	for (/*ap_uint<7>*/int dmap = 0; dmap < N_DMAP; dmap++) {
#pragma HLS PIPELINE
		ap_int<7> bi = bias[dmap];

		ap_int<24> temp0 = tmp[dmap] * SF;
//#pragma HLS RESOURCE variable=temp0 core=Mul
		ap_int<24> temp2 = temp0 + bi;
		if (temp2 >= 0) {
			pong[dmap] = 1;
		} else {
			pong[dmap] = 0;
		}
	}
}


template<
int LAYER, 	// Layer No.
int WX, 		// Width of convolution kernel
int WY,		// Height
int DX, 		// Stride
int DY,		//
int N_SMAP,	// Num. of input feature map
int SMAP_X,	// Width of feature map
int SMAP_Y,	// Height
int N_DMAP,	// Num. of output feature map
int DMAP_X,	// Width of output feature map
int DMAP_Y,	// Height
int SF,		// Scaling factor
int NUM_W,	// Num. of weight
int NUM_B	// Num. of bias
>
void layer0(ap_uint<1>* ping, ap_uint<1>* pong, const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
{
#pragma HLS INLINE

	ap_uint<1> win[5][5];
#pragma HLS ARRAY_PARTITION variable=win complete

LAYER0:
	for(int r = 0; r < 32; r++){		// Input - Row
#pragma HLS LOOP_FLATTEN off
		for(int c = 0; c < 32; c++){	// Input - Column
#pragma HLS LOOP_FLATTEN off
			// Shift - left
			for(int c0 = 0; c0 < 5 - 1; c0++){
#pragma HLS PIPELINE
				for(int r0 = 0; r0 < 5; r0++){
					win[r0][c0] = win[r0][c0 + 1];
				}
			}

			// Fill Window with values
			if(r >= 4){
				for(int r0 = 0; r0 < 5; r0++){
#pragma HLS PIPELINE
					if( (r - r0) >= 0){
						win[4 - r0][4] = ping[(r - r0)*32 + c];
					}else{
						win[4 - r0][4] = 0;
					}
				}
			}

			// Convolution
			if( (r >= 4) && (c >= 4) ){
				for (ap_uint<3> dmap = 0; dmap < 6; dmap++) {
#pragma HLS PIPELINE
					int temp = 0;
					for(int r0 = 0; r0 < 5; r0++){
						for(int c0 = 0; c0 < 5; c0++){
							if(win[r0][c0] == 1){
								temp += weight[ 5*5*dmap + r0*5 + c0 ];
							}else{
								temp -= weight[ 5*5*dmap + r0*5 + c0 ];
							}
						}
					}

					temp = temp*32 + bias[dmap];

					if (temp >= 0) {
						pong[dmap*28*28 + (r - 4)*28 + (c - 4)] = 1;
					} else {
						pong[dmap*28*28 + (r - 4)*28 + (c - 4)] = 0;
					}

				}	// dmap
			}	// if()
		}	// c

		// Shift - up
		for(int c0 = 0; c0 < 5; c0++){
			for(int r0 = 0; r0 < 5 - 1; r0++){
#pragma HLS PIPELINE
				win[r0][c0] = win[r0 + 1][c0];
			}
		}
	}	// r
}


template<
int LAYER, 	// Layer No.
int WX, 		// Width of convolution kernel
int WY,		// Height
int DX, 		// Stride
int DY,		//
int N_SMAP,	// Num. of input feature map
int SMAP_X,	// Width of feature map
int SMAP_Y,	// Height
int N_DMAP,	// Num. of output feature map
int DMAP_X,	// Width of output feature map
int DMAP_Y,	// Height
int SF,		// Scaling factor
int NUM_W,	// Num. of weight
int NUM_B	// Num. of bias
>
void layer5(ap_uint<1> ping[120], ap_int<24> result[10], const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
{
#pragma HLS INLINE

LAYER5:
	ap_uint<11> idx = 0;
	for (ap_uint<7> smap = 0; smap < N_SMAP; smap++) {
#pragma HLS PIPELINE
		for (ap_uint<4> dmap = 0; dmap < N_DMAP; dmap++) {
			ap_int<2> dat;
			ap_int<7> coef;

			if (ping[smap] == 1){
				dat = 1;
			}else{
				dat = -1;
			}

			coef = weight[idx];
			idx++;

			result[dmap] += (dat * coef);
		} // end for dmap
	} // end for smap

LOOP_SUM:
	for (ap_uint<4> dmap = 0; dmap < N_DMAP; dmap++) {
#pragma HLS PIPELINE
		result[dmap] += bias[dmap];
	}
}

};

#endif
