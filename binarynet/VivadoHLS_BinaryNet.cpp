// VivadoHLS_BinaryNet.c
// Deep Neural Networkのバイナリ版(今回は6ビット)
// 手書き数字MNISTを学習した重みを読み込んで, 認識を行います
// Vivado HLS用にビット幅を調整しました.
// またmain()関数をテストベンチとして記述しています.
// 合成対象はBinaryNet()関数です.
// Developed by H. Nakahara

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// FPGA実装のためにビット幅を調整します. そのためのヘッダ
#include "ap_int.h"

#define INPUT_IS_8

#if defined(__SYNTHESIS__)
#include "weight.h"
#else
ap_int<7> coef_w_0[117600];
ap_int<7> coef_w_1[4704];
ap_int<7> coef_w_2[240000];
ap_int<7> coef_w_3[6400];
ap_int<7> coef_w_4[48000];
ap_int<7> coef_w_5[1200];

ap_int<7> scale_f_0[4704];
ap_int<7> scale_f_1[1176];
ap_int<7> scale_f_2[1600];
ap_int<7> scale_f_3[400];
ap_int<7> scale_f_4[120];
ap_int<7> scale_f_5[10];

ap_int<7> bias_0[4704];
ap_int<7> bias_1[1176];
ap_int<7> bias_2[1600];
ap_int<7> bias_3[400];
ap_int<7> bias_4[120];
ap_int<7> bias_5[10];


void load_weight(const char* path, int numel, ap_int<7>* dst)
{
	signed char* tmp;
	tmp = new signed char [numel];

	FILE* fp = NULL;
	fp = fopen(path, "rb");
	assert(fp != NULL);

	size_t r = fread(tmp, sizeof(signed char), numel, fp);
	assert(r == numel);

	for(int i = 0; i < numel; i++){
		dst[i] = tmp[i];
	}

	fclose(fp);

	delete [] tmp;
}

#endif

// Cソース版はping-pongバッファを外部に持っていましたが, 内部に持つように
// BinaryNet()内部で宣言しました.
void BinaryNet(unsigned char *predict_num, // 認識した数字のインデックス
		ap_uint<32> pbuf[32] // 入力画像32x32ピクセル白黒画像
		);

// メイン関数. 認識用のBinaryNetのテストベンチになっています
int main(void) {
	// Test data by MNIST benchmark --------------------------------------------
	int input[32][32] = {
#include "input.h"
			};

	// Read test data -----------------------------------------------------
	ap_uint<32> pbuf[32];
	ap_uint<32> temp;

	printf("READ TEST DATA\n");
	for (int yy = 0; yy < 32; yy++) {
		pbuf[yy] = 0;
	}
	for (int yy = 0; yy < 32; yy++) {
		temp = 0;
		for (int xx = 0; xx < 32; xx++) {
			if (input[yy][xx] == 1) {
				printf("#");

				temp = (temp << 1) | 1;
			} else {
				printf(" ");

				temp = (temp << 1);
			}
		}

		pbuf[yy] = temp;
		printf("\n");
	}

#ifndef __SYNTHESIS__
    printf("READING WEIGHT DATA...\n");

    load_weight("weight/coef_w_0.bin",  117600, coef_w_0);
    load_weight("weight/coef_w_1.bin",    4704, coef_w_1);
    load_weight("weight/coef_w_2.bin",  240000, coef_w_2);
    load_weight("weight/coef_w_3.bin",    6400, coef_w_3);
    load_weight("weight/coef_w_4.bin",   48000, coef_w_4);
    load_weight("weight/coef_w_5.bin",    1200, coef_w_5);

    load_weight("weight/scale_f_0.bin",   4704, scale_f_0);
    load_weight("weight/scale_f_1.bin",   1176, scale_f_1);
    load_weight("weight/scale_f_2.bin",   1600, scale_f_2);
    load_weight("weight/scale_f_3.bin",    400, scale_f_3);
    load_weight("weight/scale_f_4.bin",    120, scale_f_4);
    load_weight("weight/scale_f_5.bin",     10, scale_f_5);

    load_weight("weight/bias_0.bin",      4704, bias_0);
    load_weight("weight/bias_1.bin",      1176, bias_1);
    load_weight("weight/bias_2.bin",      1600, bias_2);
    load_weight("weight/bias_3.bin",       400, bias_3);
    load_weight("weight/bias_4.bin",       120, bias_4);
    load_weight("weight/bias_5.bin",        10, bias_5);
#endif


	// Perform prediction -------------------------------------------------
	printf("START PREDICTION\n");
	unsigned char est;

	BinaryNet(&est, pbuf);

	printf("ESTIMATION = %d\n", est);

	// Printout result ----------------------------------------------------
	// 未使用ですが、テスト用にコメントを残しておきました.
	// デバッグ時にはHLSで合成した回路の値とCプログラムの出力を突き合わせていました。
	/*
	 int i
	 int max_val = 0, max_idx = 0;
	 for( i = 0; i < 10; i++){
	 if( max_val < buf[6 & 0x1][i]){
	 max_val = buf[6 & 0x1][i];
	 max_idx = i;
	 }
	 }
	 for( i = 0; i < 10; i++){
	 printf("NUMBER'%2d' VALUE=%8d", i, buf[6 & 0x1][i]);
	 if( i == max_idx)
	 printf(" (Predicted)");
	 printf("\n");
	 }
	 */

	return 0;
}

void layer0(ap_uint<1> buf[2][6 * 28 * 28]);
void layer1(ap_uint<1> buf[2][6 * 28 * 28]);
void layer2(ap_uint<1> buf[2][6 * 28 * 28]);
void layer3(ap_uint<1> buf[2][6 * 28 * 28]);
void layer4(ap_uint<1> buf[2][6 * 28 * 28]);
void layer5(ap_uint<1> buf[2][6 * 28 * 28], ap_int<24> result[10]);


// ディープニューラルネットワーク本体. ここから先を高位合成してRTLを出力します.
// あえて関数のインタフェースをコメントして残しています.
// CソースからHLSで設計するとき, このようにリソース消費と設計時間を考えながら
// 高位合成対象の部分を調整しました. 高位合成で設計しない場合はRTLを
// 自力で書かないといけない（多くはバッファやFIFOといったインタフェース部分）ので
// 設計時間との兼ね合いになると思います.
// 幸い, ディープニューラルネットワークが全てFPGAに納まって余裕もありましたので
// インタフェースも全てHLSに任せることができました.
// RTLはホストPCとの通信に使うUART部のみ記述しました.
void BinaryNet(unsigned char *predict_num, // 認識した数字のインデックス. 学習時にインデックス=数字としたのでそのまま利用できる.
		ap_uint<32> pbuf[32] //, 入力画像32x32ピクセル２値化済み.
		) {

	// This version uses a ping-pong memory
	// 途中結果は2値(Binarized)した値を保持します. なのでuint1で済みました.
	// 最初は18ビットでDSP48Eを使うネットワークだったのでVirtexクラスのFPGAが必要で
	// Artixには入りませんでした. 18ビットと比べて認識精度は若干落ちますがBinaryNetやっぱりすごい。
	// なお、今回の実装は重み係数は6ビットとしています. BinaryNetの論文は重み係数も1ビットです.
	// (ちょっとトリックが入ってるので精度はそれほど落ちない)
	ap_uint<1> buf[2][6 * 28 * 28];

	// 入力された画像データをバッファメモリ(ping-pongメモリ)に格納
//LOOP_INPUT_DATA:
	for (int yy = 0; yy < 32; yy++) {
//#pragma HLS PIPELINE
		ap_uint<32> pict = pbuf[yy];
		//printf("yy=%d pict=%X ", yy, pict);
		for (int xx = 0; xx < 32; xx++) {
//#pragma HLS UNROLL
			buf[0][yy * 32 + 31 - xx] = pict.get_bit(xx);
//			if ((pict & 0x1) == 1) {
//				buf[0][yy * 32 + 31 - xx] = 1;
//			} else {
//				buf[0][yy * 32 + 31 - xx] = 0;    //-1;
//			}
//			pict = pict >> 1;
		}
	}

	// Vivado HLSでは、C検証時に内部のデータをこうやって↓みれるので便利ですね。
	// RTLシミュレーションだと… (-_-)
	// for( yy = 0; yy < 32; yy++){
	// 	for( xx = 0; xx < 32; xx++){
	// 		if( buf[0][yy * 32 + xx] == 1){
	// 			printf("#");
	// 		} else {
	// 			printf(" ");
	// 		}
	// 	}
	// 	printf("\n");
	// }
	 
	ap_int<24> result[10];		// Output score



	// Layer 0
	layer0(buf);

	// Layer 1
	layer1(buf);

	// Layer 2
	layer2(buf);

	// Layer 3
	layer3(buf);

	// Layer 4
	layer4(buf);

	// Layer 5
	layer5(buf, result);


	// Prediction ----------------------------------------------------
	ap_int<24> max_val = result[0];
	unsigned char max_idx = 0;
//LOOP_OUTPUT:
	for (ap_uint<4> i = 1; i < 10; i++) {
//#pragma HLS PIPELINE
		if (max_val < result[i]) {
			max_val = result[i];
			max_idx = i;
		}
#if !defined(__SDSVHLS__) || !defined(__SYNTHESIS__)
		printf("idx=%d %d\n", i.to_uint(), result[i].to_int());
#endif
	}
#if !defined(__SDSVHLS__) && !defined(__SYNTHESIS__)
	printf("max index = %d\n", max_idx);
#endif

	// for( i = 0; i < 10; i++){
	// 	if( max_val < buf[6 & 0x1][i]){
	// 		max_val = buf[6 & 0x1][i];
	// 		max_idx = i;
	// 	}
	// 	//printf("idx=%d %d\n", i, buf[6 & 0x1][i]);
	// }
	// //printf("max index = %d\n", max_idx);

	*predict_num = max_idx;
}


void layer0(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE
	int x = 0, y = 0;
	int coef_offset = 0;

	ap_uint<16> idx = 0;

	for (int dmap = 0; dmap < 6; dmap++) {
		for (int i = 0; i < 28 * 28; i++) {
			ap_int<24> temp = 0;
//			for (int smap = 0; smap < 1; smap++) {
			for (int oy = 0; oy < 5; oy++) {
				for (int ox = 0; ox < 5; ox++) {
// #pragma HLS PIPELINE
					ap_int<18> dat;
					ap_int<8> coef;

					if (buf[0 & 0x1][/*smap*/0 * (32 * 32) + (y + oy) * 32
							+ (x + ox)] == 1) {
						dat = 1;
					} else {
						dat = -1;
					}

					coef = coef_w_0[coef_offset + oy * 5 + ox];

					temp += (dat * coef);
				} // end for oy
			} // end for ox

			coef_offset += (5 * 5);
//			} // end for smap

			ap_int<8> sf, bi;

			sf = scale_f_0[idx];
			bi = bias_0[idx];

			temp = temp * sf; // 8b x 8b = 16b
			temp = temp + bi;
			if (temp >= 0) {
				buf[(0 + 1) & 0x1][idx] = 1;
			} else {
				buf[(0 + 1) & 0x1][idx] = 0;
			}

			// Update indices
			idx++;
			x += 1;
			if (x > (32 - 5)) {
				x = 0;
				y += 1;
				if (y > (32 - 5)) {
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}

void layer1(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE
	int x = 0, y = 0;
	int coef_offset = 0;

	ap_uint<16> idx = 0;

	for (int dmap = 0; dmap < 6; dmap++) {
		for (int i = 0; i < 14 * 14; i++) {
			ap_int<24> temp = 0;
//			for (int smap = 0; smap < 1; smap++) {
			for (int oy = 0; oy < 2; oy++) {
				for (int ox = 0; ox < 2; ox++) {
// #pragma HLS PIPELINE
					ap_int<18> dat;
					ap_int<8> coef;

					if (buf[1 & 0x1][dmap * (28 * 28) + (y + oy) * 28 + (x + ox)]
							== 1)
						dat = 1;
					else
						dat = -1;

					coef = coef_w_1[idx * (2 * 2 * 1) + (0 * 2 * 2) + oy * 2
							+ ox];

					temp += (dat * coef);
				} // end for oy
			} // end for ox
			coef_offset += (2 * 2);
//			} // end for smap

			ap_int<8> sf, bi;

			sf = scale_f_1[idx];
			bi = bias_1[idx];

			temp = temp * sf; // 8b x 8b = 16b
			temp = temp + bi;
			if (temp >= 0) {
				buf[(1 + 1) & 0x1][idx] = 1;
			} else {
				buf[(1 + 1) & 0x1][idx] = 0;
			}

			// Update indices
			idx++;
			x += 2;
			if (x > (28 - 2)) {
				x = 0;
				y += 2;
				if (y > (28 - 2)) {
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}

void layer2(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE
	ap_uint<1> cnct_tbl[16][6] = {
			{ 1, 1, 1, 0, 0, 0 },
			{ 0, 1, 1, 1, 0, 0 },
			{ 0, 0, 1, 1, 1, 0 },
			{ 0, 0, 0, 1, 1, 1 },
			{ 1, 0, 0, 0, 1, 1 },
			{ 1, 1, 0, 0, 0, 1 },
			{ 1, 1, 1, 1, 0, 0 },
			{ 0, 1, 1, 1, 1, 0 },
			{ 0, 0,	1, 1, 1, 1 },
			{ 1, 0, 0, 1, 1, 1 },
			{ 1, 1, 0, 0, 1, 1 },
			{ 1, 1, 1, 0, 0, 1 },
			{ 1, 1, 0, 1, 1, 0 },
			{ 0, 1, 1, 0, 1, 1 },
			{ 1, 0, 1, 1, 0, 1 },
			{ 1, 1, 1, 1, 1, 1 } };

	int x = 0, y = 0;
	int coef_offset = 0;

	ap_uint<16> idx = 0;

	for (int dmap = 0; dmap < 16; dmap++) {
		for (int i = 0; i < 10 * 10; i++) {
			ap_int<24> temp = 0;
			for (int smap = 0; smap < 6; smap++) {
				// Read connection from LeCun's table
				ap_uint<1> is_connect = 0;
				if (cnct_tbl[dmap][smap]) {
					is_connect = 1;
				}

				if (is_connect) {
					for (int oy = 0; oy < 5; oy++) {
						for (int ox = 0; ox < 5; ox++) {
// #pragma HLS PIPELINE
							ap_int<18> dat;
							ap_int<8> coef;

							if (buf[2 & 0x1][smap * (14 * 14) + (y + oy) * 14
									+ (x + ox)] == 1) {
								dat = 1;
							} else {
								dat = -1;
							}

							coef = coef_w_2[coef_offset + oy * 5 + ox];

							temp += (dat * coef);
						} // end for oy
					} // end for ox

					coef_offset += (5 * 5);
				} // end for is_connect
			} // end for smap

			ap_int<8> sf, bi;

			sf = scale_f_2[idx];
			bi = bias_2[idx];
			temp = temp * sf; // 8b x 8b = 16b
			temp = temp + bi;
			if (temp >= 0) {
				buf[(2 + 1) & 0x1][idx] = 1;
			} else {
				buf[(2 + 1) & 0x1][idx] = 0;
			}

			// Update indices
			idx++;
			x += 1;
			if (x > (14 - 5)) {
				x = 0;
				y += 1;
				if (y > (14 - 5)) {
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}

void layer3(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE
	int x = 0, y = 0;
	int coef_offset = 0;

	ap_uint<16> idx = 0;

	for (int dmap = 0; dmap < 16; dmap++) {
		for (int i = 0; i < 5 * 5; i++) {
			ap_int<24> temp = 0;
//			for (int smap = 0; smap < 1; smap++) {
				for (int oy = 0; oy < 2; oy++) {
					for (int ox = 0; ox < 2; ox++) {
						ap_int<18> dat;
						ap_int<8> coef;

						if (buf[3 & 0x1][dmap * (10 * 10)
								+ (y + oy) * 10 + (x + ox)] == 1) {
							dat = 1;
						} else {
							dat = -1;
						}

						coef = coef_w_3[idx * (2 * 2 * 1)
								+ (0 * 2 * 2) + oy * 2 + ox];

						// Perform an ADD-MUL operation
						temp += (dat * coef);
					} // end for oy
				} // end for ox

				coef_offset += (2 * 2);
//			} // end for smap

			ap_int<8> sf, bi;

			sf = scale_f_3[idx];
			bi = bias_3[idx];

			temp = temp * sf; // 8b x 8b = 16b
			temp = temp + bi;
			if (temp >= 0) {
				buf[(3 + 1) & 0x1][idx] = 1;
			} else {
				buf[(3 + 1) & 0x1][idx] = 0;
			}

			// Update indices
			idx++;
			x += 2;
			if (x > (10 - 2)) {
				x = 0;
				y += 2;
				if (y > (10 - 2)) {
					y = 0;
				}
			}
		} // end for i
	} // end for dmap
}

void layer4(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE
	int x = 0, y = 0;
	int coef_offset = 0;

	ap_uint<16> idx = 0;

	LOOP_DMAP: for (ap_uint<7> dmap = 0; dmap < 120; dmap++) {
		ap_int<24> temp = 0;
		LOOP_SMAP: for (ap_uint<5> smap = 0; smap < 16; smap++) {
#pragma HLS PIPELINE
			LOOP_OY: for (ap_uint<3> oy = 0; oy < 5; oy++) {
				LOOP_OX: for (ap_uint<3> ox = 0; ox < 5; ox++) {
					ap_int<18> dat;
					ap_int<8> coef;

					if (buf[4 & 0x1][smap * (5 * 5) + (y + oy) * 5 + (x + ox)]
							== 1){
						dat = 1;
					}else{
						dat = -1;
					}

					coef = coef_w_4[coef_offset + oy * 5 + ox];

					temp += (dat * coef);
				} // end for oy
			} // end for ox

			coef_offset += (5 * 5);

		} // end for smap

		ap_int<8> sf, bi;

		sf = scale_f_4[idx];
		bi = bias_4[idx];

		temp = temp * sf; // 8b x 8b = 16b
		temp = temp + bi;
		if (temp >= 0) {
			buf[(4 + 1) & 0x1][idx] = 1;
		} else {
			buf[(4 + 1) & 0x1][idx] = 0;
		}

		// Update indices
		idx++;
		x += 1;
		if (x > (5 - 5)) {
			x = 0;
			y += 1;
			if (y > (5 - 5)) {
				y = 0;
			}
		}
	} // end for dmap
}

void layer5(ap_uint<1> buf[2][6 * 28 * 28], ap_int<24> result[10]){
#pragma HLS INLINE
	int x = 0, y = 0;
	int coef_offset = 0;
	ap_uint<16> idx = 0;

	for (ap_uint<4> dmap = 0; dmap < 10; dmap++) {
		ap_int<24> temp = 0;
		for (ap_uint<7> smap = 0; smap < 120; smap++) {
#pragma HLS PIPELINE
			const int ox = 0, oy = 0;
			ap_int<18> dat;
			ap_int<8> coef;

			if (buf[5 & 0x1][smap * (1 * 1) + (y + oy) * 1 + (x + ox)] == 1){
				dat = 1;
			}else{
				dat = -1;
			}

			coef = coef_w_5[coef_offset + oy * 1 + ox];

			temp += (dat * coef);

			coef_offset += (1 * 1);
		} // end for smap

		ap_int<8> bi;
		bi = bias_5[idx];
		temp = temp + bi;
		result[idx] = temp;

		// Update indices
		idx++;
		x += 1;
		if (x > (1 - 1)) {
			x = 0;
			y += 1;
			if (y > (1 - 1)) {
				y = 0;
			}
		}
	} // end for dmap
}

/* #############################################################*/
/*                         END OF PROGRAM                        */
/* #############################################################*/
