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
//#pragma HLS INLINE

	ap_uint<16>/*int*/ idx = 0;

	int x = 0, y = 0;
//	int coef_offset = 0;

LAYER_CONV:
	for(int dmap = 0; dmap < N_DMAP; dmap++){
//#pragma HLS LOOP_FLATTEN off
		for(int i = 0; i < DMAP_X * DMAP_Y; i++){
//#pragma HLS PIPELINE
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

						coef = weight[/*coef_offset + oy*WY + ox*/N_SMAP*WX*WY*dmap + s];
						s++;

//						std::cout << (smap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)) << "|" <<
//								(coef_offset + oy*WY + ox) << std::endl;
						// Perform an ADD-MUL operation
						temp += (dat * coef);

//						if(dmap ==0 && i == 0 && smap == 0){
//							std::cout << "\t" << dat << " | " << coef << std::endl;
//						}
					} // end for oy
				} // end for ox

//				coef_offset += (WX * WY);
			} // end for smap



			// Activation function for the BinaryNet
			temp = temp * SF;
			temp = temp + bias[dmap];

			std::cout << temp << std::endl;

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
//#pragma HLS INLINE
//
//#pragma HLS RESOURCE variable=coef_w_1 core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=bias_1 core=ROM_2P_LUTRAM

	ap_uint<14> x = 0, y = 0;
	ap_uint<16> idx = 0;

LAYER_POOL:
	for (/*ap_uint<3>*/int dmap = 0; dmap < N_DMAP; dmap++) {
//#pragma HLS LOOP_FLATTEN off
		for (/*ap_uint<8>*/int i = 0; i < DMAP_X * DMAP_Y; i++) {
//#pragma HLS PIPELINE
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
//#pragma HLS INLINE
//
//#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM
//
//#pragma HLS ARRAY_PARTITION variable=coef_w_4 cyclic factor=4
//#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM

	ap_int<22> tmp[NUM_B] = {0};
//#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=4

	ap_uint<24> idx0 = 0;
	ap_uint<12> idx1 = 0;

LAYER_FC:
	for (int/*ap_uint<5>*/ smap = 0; smap < N_SMAP; smap++) {
//#pragma HLS LOOP_FLATTEN off

		ap_int<22> temp = 0;
		for (ap_uint<3> oy = 0; oy < WX; oy++) {
//#pragma HLS LOOP_FLATTEN off

			for (ap_uint<3> ox = 0; ox < WY; ox++) {
//#pragma HLS LOOP_FLATTEN off
				ap_int<2> dat;
				ap_int<7> coef;

				if (ping[idx1] == 1){
					dat = 1;
				}else{
					dat = -1;
				}
				for (ap_uint<7> dmap = 0; dmap < N_DMAP; dmap++) {
//#pragma HLS UNROLL factor=4
//#pragma HLS PIPELINE
					coef = weight[idx0++];

					tmp[dmap] += (dat * coef);
				}// end for dmap
				idx1++;
			} // end for oy
		} // end for ox
	} // end for smap

	//
	for (/*ap_uint<7>*/int dmap = 0; dmap < N_DMAP; dmap++) {
//#pragma HLS PIPELINE
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
//#pragma HLS INLINE

	ap_uint<1> win[5][5];
//#pragma HLS ARRAY_PARTITION variable=win complete

LAYER0:
	for(int r = 0; r < 32; r++){		// Input - Row
		for(int c = 0; c < 32; c++){	// Input - Column

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
#if 0
	ap_uint<16>/*int*/ idx = 0;

	int x = 0, y = 0;
	int coef_offset = 0;

	for(int dmap = 0; dmap < N_DMAP; dmap++){	// 10
#pragma HLS LOOP_FLATTEN off
//		for(int i = 0; i < DMAP_X * DMAP_Y; i++){	// 1
			ap_int<24> temp = 0;
			ap_int<2> dat;
			ap_int<7> coef;

			for(int smap = 0; smap < N_SMAP; smap++){	// 120
#pragma HLS PIPELINE
				for(int oy = 0; oy < WY; oy++){
						for(int ox = 0; ox < WX; ox++){
							if( LAYER == 1 || LAYER == 3){
								// average pooling layer
								if( buf[LAYER & 0x1][dmap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)] == 1){
									dat = 1;
								}else{
									dat = -1;
								}

								coef = weight[idx * (WX*WY*N_SMAP) + (smap * WX * WY) + oy * WY + ox];
							} else {
								// convolutional and fully-connected layer
								if( buf[LAYER & 0x1][smap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)] == 1){
									dat = 1;
								}else{
									dat = -1;
								}

								coef = weight[coef_offset + oy*WY + ox];
							}

							// Perform an ADD-MUL operation
							temp += (dat * coef);
						} // end for oy
					} // end for ox
					// Update offset, since the LeCun's table requires uniformaly connection
					coef_offset += (WX * WY);
			} // end for smap

			// Activation function for the BinaryNet
			// 最終層のみ, ２値化せずにそのまま計算結果を格納しています.
			// 幸い手書き数字認識なので10ニューロンで済みましたので、そのまま配列で書いてレジスタに合成
			temp = temp + bias[idx];
			result[idx] = temp;

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
//		} // end for i
	} // end for dmap
#else

LAYER5:
	ap_uint<11> idx = 0;
	for (ap_uint<7> smap = 0; smap < N_SMAP; smap++) {
//#pragma HLS PIPELINE
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
//#pragma HLS PIPELINE
		result[dmap] += bias[dmap];
	}

#endif
}

//template<
//int LAYER, 	// Layer No.
//int WX, 		// Width of convolution kernel
//int WY,		// Height
//int DX, 		// Stride
//int DY,		//
//int N_SMAP,	// Num. of input feature map
//int SMAP_X,	// Width of feature map
//int SMAP_Y,	// Height
//int N_DMAP,	// Num. of output feature map
//int DMAP_X,	// Width of output feature map
//int DMAP_Y,	// Height
//int SF,		// Scaling factor
//int NUM_W,	// Num. of weight
//int NUM_B	// Num. of bias
//>
//void layer2(ap_uint<1>* ping, ap_uint<1>* pong, /*const ap_int<1>* cnct_tbl[16][6],*/ const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
//{
//#pragma HLS INLINE
//
//#pragma HLS RESOURCE variable=bias_2 core=ROM_1P_LUTRAM
//
//	const static ap_uint<1> cnct_tbl[16][6] = {
//			{ 1, 1, 1, 0, 0, 0 },	//3
//			{ 0, 1, 1, 1, 0, 0 },	//3
//			{ 0, 0, 1, 1, 1, 0 },	//3
//			{ 0, 0, 0, 1, 1, 1 },	//3
//			{ 1, 0, 0, 0, 1, 1 },	//3
//			{ 1, 1, 0, 0, 0, 1 },	//3		18
//			{ 1, 1, 1, 1, 0, 0 },	//4
//			{ 0, 1, 1, 1, 1, 0 },	//4
//			{ 0, 0,	1, 1, 1, 1 },	//4
//			{ 1, 0, 0, 1, 1, 1 },	//4
//			{ 1, 1, 0, 0, 1, 1 },	//4
//			{ 1, 1, 1, 0, 0, 1 },	//4
//			{ 1, 1, 0, 1, 1, 0 },	//4
//			{ 0, 1, 1, 0, 1, 1 },	//4
//			{ 1, 0, 1, 1, 0, 1 },	//4		36
//			{ 1, 1, 1, 1, 1, 1 } };	//6
//
//	const static ap_uint<6> cumsum[16] = {
//			 0, 3, 6, 9, 12, 15, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54/*, 60*/
//	};
//
//	ap_uint<13> x = 0, y = 0;
//	ap_uint<22> coef_offset = 0;
//	ap_uint<16> idx = 0;
//
//LAYER2:
//	for (ap_uint<5> dmap = 0; dmap < 16; dmap++) {
//#pragma HLS LOOP_FLATTEN off
//		for (ap_uint<8> i = 0; i < 10 * 10; i++) {
//#pragma HLS PIPELINE
//			ap_int<24> temp = 0;
//			ap_uint<8> s = 0;
//			for (ap_uint<3> smap = 0; smap < 6; smap++) {
//				if (cnct_tbl[dmap][smap]) {
//					for (ap_uint<3>  oy = 0; oy < 5; oy++) {
//						for (ap_uint<3>  ox = 0; ox < 5; ox++) {
//							ap_int<2> dat;
//							ap_int<7> coef;
//
//							if (ping[smap * (14 * 14) + (y + oy) * 14
//									+ (x + ox)] == 1) {
//								dat = 1;
//							} else {
//								dat = -1;
//							}
//
//							coef = weight[s + cumsum[dmap]*25];
//							s++;
//							temp += (dat * coef);
//
////							if(dmap ==0 && i == 0 && smap == 0){
////								std::cout << "\t" << dat << " | " << coef << std::endl;
////							}
//						} // end for oy
//					} // end for ox
//				} // end for is_connect
//			} // end for smap
//
//
//			ap_int<7> bi = bias[dmap];
//
//			ap_int<24> temp0 = temp * SF;
////#pragma HLS RESOURCE variable=temp0 core=Mul
//			ap_int<24> temp2 = temp0 + bi;
//			std::cout << temp2 << std::endl;
//			if (temp2 >= 0) {
//				pong[idx] = 1;
//			} else {
//				pong[idx] = 0;
//			}
//
//			// Update indices
//			idx++;
//			x += 1;
//			if (x > (14 - 5)) {
//				x = 0;
//				y += 1;
//				if (y > (14 - 5)) {
//					y = 0;
//				}
//			}
//		} // end for i
//	} // end for dmap
//}

};

#endif /* NET_H_ */

//template<
//int LAYER, 	// Layer No.
//int WX, 		// Width of convolution kernel
//int WY,		// Height
//int DX, 		// Stride
//int DY,		//
//int N_SMAP,	// Num. of input feature map
//int SMAP_X,	// Width of feature map
//int SMAP_Y,	// Height
//int N_DMAP,	// Num. of output feature map
//int DMAP_X,	// Width of output feature map
//int DMAP_Y,	// Height
//int SF,		// Scaling factor
//int NUM_W,	// Num. of weight
//int NUM_B	// Num. of bias
//>
//void hw(ap_uint<1>* ping, ap_uint<1>* pong, const ap_int<7> weight[NUM_W], const ap_int<7> bias[NUM_B])
//{
//#pragma HLS INLINE
//
//	ap_uint<16>/*int*/ idx = 0;
////	ap_int<24>/*int*/ result[10];
//
//	int x = 0, y = 0;
//	int coef_offset = 0;
//
//	for(int dmap = 0; dmap < N_DMAP; dmap++){
//#pragma HLS LOOP_FLATTEN off
//		for(int i = 0; i < DMAP_X * DMAP_Y; i++){
//#pragma HLS PIPELINE
//			ap_int<24> temp = 0;
//			ap_int<2> dat;
//			ap_int<7> coef;
//
//			for(int smap = 0; smap < N_SMAP; smap++){
//				for(int oy = 0; oy < WY; oy++){
//					for(int ox = 0; ox < WX; ox++){
//						if( LAYER == 1 || LAYER == 3){
//							// average pooling layer
//							if( ping[dmap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)] == 1){
//								dat = 1;
//							}else{
//								dat = -1;
//							}
//
//							coef = weight[idx * (WX*WY*N_SMAP) + (smap * WX * WY) + oy * WY + ox];
//						} else {
//							// convolutional and fully-connected layer
//							if( ping[smap * (SMAP_X*SMAP_Y) + (y + oy) * SMAP_Y + (x + ox)] == 1){
//								dat = 1;
//							}else{
//								dat = -1;
//							}
//
//							coef = weight[coef_offset + oy*WY + ox];
//						}
//
//						// Perform an ADD-MUL operation
//						temp += (dat * coef);
//					} // end for oy
//				} // end for ox
//
//				// Update offset, since the LeCun's table requires uniformaly connection
//				coef_offset += (WX * WY);
////			} // end for is_connect
//			} // end for smap
//
//			// Activation function for the BinaryNet
////			if( LAYER != 5){
//				temp = temp * SF;
//				temp = temp + bias[idx];
//				if( temp >= 0){
//					temp = 1;
//				}else{
//					temp = -1;
//				}
////			} else {
////				// 最終層のみ, ２値化せずにそのまま計算結果を格納しています.
////				// 幸い手書き数字認識なので10ニューロンで済みましたので、そのまま配列で書いてレジスタに合成
////				temp = temp + bias[idx];
////				result[idx] = temp;
////			}
//
//			// Store ping-pong memory
//			if( temp == 1){
//				pong[idx] = 1;
//			}else{
//				pong[idx] = 0;
//			}
//
//			// Update indices
//			idx++;
//			x += DX;
//			if( x > (SMAP_X - WX)){
//				x = 0;
//				y += DY;
//				if( y > (SMAP_Y - WY)){
//					y = 0;
//				}
//			}
//		} // end for i
//	} // end for dmap
//}
