// VivadoHLS_BinaryNet.c
// Deep Neural Networkのバイナリ版(今回は6ビット)
// 手書き数字MNISTを学習した重みを読み込んで, 認識を行います
// Vivado HLS用にビット幅を調整しました.
// またmain()関数をテストベンチとして記述しています.
// 合成対象はBinaryNet()関数です.
// Developed by H. Nakahara

#include <stdio.h>

// FPGA実装のためにビット幅を調整します. そのためのヘッダ
#include "ap_int.h"

#define INPUT_IS_8

#if defined(__SYNTHESIS__)
#include "weight.h"
#else
ap_int<7> coef_w_0[6 * 5 * 5] = {
	0,  2,  2,   2,   0,   0,  -1,  -2,  -1,   0,   1,   0,   0,   2,  -1,   0,   0,   2,  -2,   2,   0,   1,   0,   1,   1,
	0,  0,  1,  -1,   0,  -5,   0,  -3,  -3,  -2,  -6,  -5,  -4,   0,  -3,  -1,   0,  -5,   0,  -1,  -1,   0,   1,   0,  -3,
	-29, -3, 12,   9,   6,  -8,  14,  18,  11,  15,  -3,   8,  13,  14,  17,  -7,  -2,  10,   7,   0, -11,  -9, -14, -10, -22,
	-7, -5, -5,  -7, -13,  17,   5,  10,   6, -10,  24,  10,  24,  11, -10,   8,   7,  18,   8, -11,  -3,  -3,   0,  -7, -14,
	0,  1,  0,   0,   1,   0,   2,   1,   3,   1,   0,   3,  -1,   2,   0,   1,  -1,  -1,   3,   0,   1,   0,  -2,  -1,   0,
	0,  0,  1,  -1,   0,   1,   2,  -3,  -3,  -1,   0,  -4,   1,  -2,  -2,  -4,  -4,   2,  -2,   0,  -2,  -3,  -1,   0,   2
};

ap_int<7> coef_w_1[4704];
ap_int<7> coef_w_2[240000];
ap_int<7> coef_w_3[1600];
ap_int<7> coef_w_4[48000];
ap_int<7> coef_w_5[1200] = {
	-6,4,-3,4,0,2,5,-7,1,13,5,0,-3,3,-1,2,1,-3,-2,4,3,3,3,-5,-6,9,-6,-2,-5,2,0,0,0,0,0,1,3,0,0,1,0,-1,-2,0,0,0,0,1,-1,-4,0,0,0,0,0,0,0,0,0,1,-1,0,0,-1,1,0,0,-1,0,0,-1,0,0,1,1,1,-1,0,-1,0,0,0,0,0,-1,0,-1,0,0,-1,5,-3,1,14,7,-3,0,10,-7,-9,0,0,0,0,0,2,1,1,1,0,7,3,0,10,0,-4,8,1,0,9,-1,-2,2,3,0,-2,-4,-3,0,0,-1,-3,2,0,-1,1,0,0,-1,1,0,-1,-2,0,-2,0,0,-1,-1,-1,7,3,0,0,4,2,-1,0,-2,-1,-1,-4,-1,-1,-3,-1,0,0,-1,-2,1,0,1,0,-3,0,0,-4,-1,3,0,-1,0,-1,-3,-4,1,0,3,2,-1,-3,0,0,-3,1,0,-4,-2,-2,-2,0,2,-1,0,0,-1,1,-1,0,-2,-4,-3,-1,-2,0,-2,-4,-2,-1,6,5,5,0,7,4,-13,-3,0,-2,2,0,2,-3,0,-1,-4,0,0,2,4,0,2,2,2,2,4,0,1,3,0,0,0,1,0,1,0,2,0,1,0,0,-3,0,-2,-1,-2,-1,-1,-1,3,4,1,1,0,0,0,-1,1,0,0,1,-1,1,0,2,0,0,0,0,0,-1,0,0,2,0,1,-3,0,0,4,3,5,5,3,1,0,0,3,1,0,-2,0,0,-2,1,1,0,0,0,0,-3,0,-1,0,-1,-1,0,0,0,0,2,0,0,-1,-1,-2,-2,2,-1,6,-6,-6,6,-6,4,-2,-2,-4,3,2,2,0,3,0,-1,-1,0,1,0,1,-2,0,-2,1,2,-3,2,4,-3,0,2,-2,3,2,1,1,0,0,0,-1,1,1,-3,1,0,-1,1,0,-2,0,-2,-1,-4,0,0,-1,-1,1,-2,1,3,3,0,0,2,0,4,3,1,-2,0,1,-1,-2,1,-2,0,-3,-2,1,0,0,1,2,0,3,5,1,3,3,1,0,0,3,1,4,1,3,3,0,0,0,0,-1,0,0,1,0,0,1,1,1,1,1,0,0,2,2,-1,3,5,2,2,3,3,1,1,4,2,0,0,0,0,-2,-1,-2,0,-1,-1,3,0,3,-3,-1,1,-4,-3,0,4,-1,-2,-4,-1,-1,0,-2,-2,-2,0,0,-7,3,-3,0,0,-4,-2,2,3,-1,-1,-1,0,-1,1,1,-2,-2,-2,0,0,0,-1,0,0,0,0,0,-1,2,0,0,0,0,-1,0,0,0,2,1,1,1,1,1,0,0,3,3,2,-1,-1,-1,0,0,0,0,0,0,0,4,0,-2,0,1,-1,1,2,1,1,0,0,1,1,2,-1,0,0,0,0,-2,-1,0,-1,0,0,-3,-1,-3,-2,-3,-5,3,1,-1,-2,-1,-1,-3,0,-1,-1,0,-1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,0,0,7,7,0,0,-9,-4,2,-2,1,-4,0,0,1,0,-1,1,1,1,0,0,1,-3,-3,2,5,-1,2,4,4,-3,-1,-1,2,0,-2,0,0,0,0,0,-2,9,5,4,-5,-6,-5,-7,5,9,0,2,0,0,0,1,1,0,2,1,-3,-2,-2,-3,-1,-3,-4,0,-4,-3,0,2,0,0,1,-1,0,1,1,0,-1,-1,-1,-5,-8,0,-3,1,0,6,0,1,0,2,3,0,1,-3,0,-2,0,0,0,0,2,-1,-1,0,0,-1,-1,-2,0,-2,-2,-1,0,0,-1,0,0,0,0,0,1,1,0,0,0,0,-1,7,-4,-6,-1,-5,3,6,-15,11,2,1,1,0,0,0,3,0,2,0,0,0,1,0,0,0,0,2,1,0,0,1,1,0,-1,0,0,-1,-1,1,2,2,1,2,0,1,0,1,0,0,2,0,5,3,1,2,1,1,2,3,0,0,2,0,1,-3,0,1,2,1,2,-3,-5,-9,-3,13,0,7,7,-6,-9,6,2,6,-8,2,8,-2,-4,-6,4,2,3,2,2,-5,0,-3,3,1,0,0,0,-1,1,-3,0,1,3,0,2,3,1,2,0,4,2,4,4,2,1,0,0,0,0,0,0,0,0,-1,-4,-3,-1,-1,0,0,-1,-1,-1,0,-1,2,-1,0,1,0,0,0,0,0,11,9,4,11,5,10,2,5,16,8,-5,-1,-2,5,0,3,-1,2,1,-10,0,0,0,0,-2,0,-1,0,0,0,4,1,2,2,2,-1,0,1,3,1,1,0,0,3,1,2,2,1,2,2,-1,0,0,-3,-2,3,1,5,0,-2,4,2,-12,9,1,-11,0,0,0,-3,0,-1,0,0,-3,0,-1,-3,-1,-1,1,4,0,-2,1,0,0,1,0,2,1,1,-4,2,4,0,0,0,2,4,-1,0,0,0,0,-2,-1,0,1,-1,0,0,-1,-1,-1,-2,-2,-1,-1,0,-3,0,10,-6,-2,5,1,-1,-7,3,5,0,2,-5,3,4,-5,-8,-5,11,5,-5,-8,4,0,-1,0,-4,-3,2,0,-1,0,0,1,0,0,-1,0,0,0,0,-1,-3,-2,-1,-2,3,0,0,0,0,1,2,1,2,0,2,1,2,-2,-1,1,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-9,-10,-6,-9,-8,-4,-7,-8,-14,-10,10,14,0,3,-1,10,0,7,3,5,3,-3,0,-10,-9,-6,4,8,3,-12,1,0,0,2,1,0,0,1,1,1,0,-1,1,0,0,-1,0,0,1,0,10,-1,-3,2,3,-2,2,0,2,3,-5,5,-12,5,-1,8,-8,8,13,-5,5,4,2,-2,4,0,-4,-2,-1,2,-1,0,0,-1,-1,-1,-1,1,0,0,1,-2,4,-4,8,3,0,2,0,-5
};

ap_int<7> bias_0[6] = {
		2,  0, 13, 18,  2,  2
};
ap_int<7> bias_1[1176];
ap_int<7> bias_2[1600];
ap_int<7> bias_3[400];
ap_int<7> bias_4[120];
ap_int<7> bias_5[10] = {
	0,0,0,-1,0,-1,-1,-1,-1,0
};


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
//
// predict_num: 認識した数字のインデックス. 学習時にインデックス=数字としたのでそのまま利用できる.
// pbuf:        入力画像32x32ピクセル２値化済み.
void BinaryNet(unsigned char *predict_num, ap_uint<32> pbuf[32]);

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

#if !defined(__SYNTHESIS__) && !defined(__SDSCC__)
    printf("READING WEIGHT DATA...\n");

//    load_weight("weight/coef_w_0.bin",  117600, coef_w_0);
    load_weight("weight/coef_w_1.bin",    4704, coef_w_1);
    load_weight("weight/coef_w_2.bin",  240000, coef_w_2);
    load_weight("weight/coef_w_3.bin",    6400, coef_w_3);
    load_weight("weight/coef_w_4.bin",   48000, coef_w_4);
//    load_weight("weight/coef_w_5.bin",    1200, coef_w_5);

//    load_weight("weight/bias_0.bin",      4704, bias_0);
    load_weight("weight/bias_1.bin",      1176, bias_1);
    load_weight("weight/bias_2.bin",      1600, bias_2);
    load_weight("weight/bias_3.bin",       400, bias_3);
    load_weight("weight/bias_4.bin",       120, bias_4);
//    load_weight("weight/bias_5.bin",        10, bias_5);
#endif

	// Perform prediction -------------------------------------------------
	printf("START PREDICTION\n");
	unsigned char est;

	BinaryNet(&est, pbuf);

	printf("ESTIMATION = %d\n", est);

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
void BinaryNet(unsigned char *predict_num, ap_uint<32> pbuf[32])
{
	// This version uses a ping-pong memory
	// 途中結果は2値(Binarized)した値を保持します. なのでuint1で済みました.
	// 最初は18ビットでDSP48Eを使うネットワークだったのでVirtexクラスのFPGAが必要で
	// Artixには入りませんでした. 18ビットと比べて認識精度は若干落ちますがBinaryNetやっぱりすごい。
	// なお、今回の実装は重み係数は6ビットとしています. BinaryNetの論文は重み係数も1ビットです.
	// (ちょっとトリックが入ってるので精度はそれほど落ちない)
	ap_uint<1> buf[2][6 * 28 * 28];

	// 入力された画像データをバッファメモリ(ping-pongメモリ)に格納
LOOP_INPUT_DATA:
	for (int yy = 0; yy < 32; yy++) {
#pragma HLS PIPELINE
		ap_uint<32> pict = pbuf[yy];
		for (int xx = 0; xx < 32; xx++) {
//#pragma HLS UNROLL
//			buf[0][yy * 32 + 31 - xx] = pict.get_bit(xx);
			if ((pict & 0x1) == 1) {
				buf[0][yy * 32 + 31 - xx] = 1;
			} else {
				buf[0][yy * 32 + 31 - xx] = 0;    //-1;
			}
			pict = pict >> 1;
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
LOOP_OUTPUT:
	for (ap_uint<4> i = 1; i < 10; i++) {
#pragma HLS PIPELINE
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

	*predict_num = max_idx;
}


void layer0(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE

	ap_uint<1> win[5][5];

LAYER0:
	for(int r = 0; r < 32; r++){		// Input - Row

		for(int c = 0; c < 32; c++){	// Input - Column

			// Shift - left
			for(int c0 = 0; c0 < 5 - 1; c0++){
				for(int r0 = 0; r0 < 5; r0++){
					win[r0][c0] = win[r0][c0 + 1];
				}
			}

			// 値を補充
			if(r >= 4){
				for(int r0 = 0; r0 < 5; r0++){
					if( (r - r0) >= 0){
						win[ 4 -r0][4] = src[r - r0][c];
					}else{
						win[ 4 -r0][4] = 0;
					}
				}
			}

			// Convolution
			if( (r >= 4) && (c >= 4) ){
				for (ap_uint<3> dmap = 0; dmap < 6; dmap++) {
					int temp = 0;
					for(int r0 = 0; r0 < 5; r0++){
						for(int c0 = 0; c0 < 5; c0++){
							if(win[r0][c0] == 1){
								temp += coef_w_0[ 5*5*dmap + r0*5 + c0 ];
							}else{
								temp -= coef_w_0[ 5*5*dmap + r0*5 + c0 ];
							}
						}
					}

					temp = temp*32 + bias_0[dmap];

					if (temp >= 0) {
						buf[(0 + 1) & 0x1][dmap*28*28 + (r - 4)*28 + (c - 4)] = 1;
					} else {
						buf[(0 + 1) & 0x1][dmap*28*28 + (r - 4)*28 + (c - 4)] = 0;
					}

				}	// dmap
			}	// if()
		}	// c

		// Shift - up
		for(int c0 = 0; c0 < 5; c0++){
			for(int r0 = 0; r0 < 5 - 1; r0++){
				win[r0][c0] = win[r0 + 1][c0];
			}
		}
	}	// r
}

void layer1(ap_uint<1> buf[2][6 * 28 * 28]){
#pragma HLS INLINE

#pragma HLS RESOURCE variable=coef_w_1 core=ROM_2P_LUTRAM
#pragma HLS RESOURCE variable=bias_1 core=ROM_2P_LUTRAM

	ap_uint<14> x = 0, y = 0;
	ap_uint<16> idx = 0;

LAYER1:
	for (ap_uint<3> dmap = 0; dmap < 6; dmap++) {
#pragma HLS LOOP_FLATTEN off
		for (ap_uint<8> i = 0; i < 14 * 14; i++) {
#pragma HLS PIPELINE
			ap_int<24> temp = 0;
#pragma HLS RESOURCE variable=temp core=AddSub
//			for (int smap = 0; smap < 1; smap++) {
			for (ap_uint<2>  oy = 0; oy < 2; oy++) {
				for (ap_uint<2>  ox = 0; ox < 2; ox++) {
					ap_int<2> dat;
					ap_int<7> coef;

					if (buf[1 & 0x1][dmap * (28 * 28) + (y + oy) * 28 + (x + ox)]
							== 1){
						dat = 1;
					}else{
						dat = -1;
					}

					coef = coef_w_1[idx * (2 * 2 * 1) + (0 * 2 * 2) + oy * 2
							+ ox];

					temp += (dat * coef);
				} // end for oy
			} // end for ox
//			} // end for smap

			ap_int<7> bi = bias_1[idx];

			ap_int<24> temp0 = temp * 8;
//#pragma HLS RESOURCE variable=temp0 core=Mul
			ap_int<24> temp2 = temp0 + bi;
#pragma HLS RESOURCE variable=temp2 core=AddSub
			if (temp2 >= 0) {
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

#pragma HLS RESOURCE variable=bias_2 core=ROM_1P_LUTRAM

	const static ap_uint<1> cnct_tbl[16][6] = {
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

	ap_uint<13> x = 0, y = 0;
	ap_uint<22> coef_offset = 0;
	ap_uint<16> idx = 0;

LAYER2:
	for (ap_uint<5> dmap = 0; dmap < 16; dmap++) {
#pragma HLS LOOP_FLATTEN off
		for (ap_uint<8> i = 0; i < 10 * 10; i++) {
#pragma HLS PIPELINE
			ap_int<24> temp = 0;
			for (ap_uint<3> smap = 0; smap < 6; smap++) {
				if (cnct_tbl[dmap][smap]) {
					for (ap_uint<3>  oy = 0; oy < 5; oy++) {
						for (ap_uint<3>  ox = 0; ox < 5; ox++) {
							ap_int<2> dat;
							ap_int<7> coef;

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

			ap_int<7> bi = bias_2[idx];

			ap_int<24> temp0 = temp * 32;
#pragma HLS RESOURCE variable=temp0 core=Mul
			ap_int<24> temp2 = temp0 + bi;
			if (temp2 >= 0) {
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

#pragma HLS RESOURCE variable=coef_w_3 core=ROM_2P_LUTRAM

	ap_uint<12> x = 0, y = 0;
	ap_uint<9> idx = 0;

LAYER3:
	for (ap_uint<5> dmap = 0; dmap < 16; dmap++) {
#pragma HLS LOOP_FLATTEN off
		for (ap_uint<5> i = 0; i < 5 * 5; i++) {
#pragma HLS PIPELINE
			ap_int<24> temp = 0;
#pragma HLS RESOURCE variable=temp core=AddSub
//			for (int smap = 0; smap < 1; smap++) {
				for (ap_uint<2> oy = 0; oy < 2; oy++) {
					for (ap_uint<2> ox = 0; ox < 2; ox++) {
						ap_int<2> dat;
						ap_int<7> coef;

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
//			} // end for smap

			ap_int<7> bi = bias_3[idx];

			ap_int<24> temp0 = temp * 8;
#pragma HLS RESOURCE variable=temp0 core=Mul
			ap_int<24> temp2 = temp0 + bi;
#pragma HLS RESOURCE variable=temp2 core=AddSub
			if (temp2 >= 0) {
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

#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM

#pragma HLS ARRAY_PARTITION variable=coef_w_4 cyclic factor=4
#pragma HLS RESOURCE variable=coef_w_4 core=ROM_2P_LUTRAM

	ap_int<22> tmp[120] = {0};
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=4

	ap_uint<24> idx0 = 0;
	ap_uint<12> idx1 = 0;

LAYER4:
	for (ap_uint<5> smap = 0; smap < 16; smap++) {
#pragma HLS LOOP_FLATTEN off

		ap_int<22> temp = 0;
		for (ap_uint<3> oy = 0; oy < 5; oy++) {
#pragma HLS LOOP_FLATTEN off

			for (ap_uint<3> ox = 0; ox < 5; ox++) {
#pragma HLS LOOP_FLATTEN off
				ap_int<2> dat;
				ap_int<7> coef;

				if (buf[4 & 0x1][idx1] == 1){
					dat = 1;
				}else{
					dat = -1;
				}
				for (ap_uint<7> dmap = 0; dmap < 120; dmap++) {
#pragma HLS UNROLL factor=4
#pragma HLS PIPELINE
					coef = coef_w_4[idx0++];

					tmp[dmap] += (dat * coef);
				}// end for dmap
				idx1++;
			} // end for oy
		} // end for ox
	} // end for smap

	//
	for (ap_uint<7> dmap = 0; dmap < 120; dmap++) {
#pragma HLS PIPELINE
		ap_int<7> bi = bias_4[dmap];

		ap_int<24> temp0 = tmp[dmap] * 32;
#pragma HLS RESOURCE variable=temp0 core=Mul
		ap_int<24> temp2 = temp0 + bi;
		if (temp2 >= 0) {
			buf[(4 + 1) & 0x1][dmap] = 1;
		} else {
			buf[(4 + 1) & 0x1][dmap] = 0;
		}
	}

}

void layer5(ap_uint<1> buf[2][6 * 28 * 28], ap_int<24> result[10]){
#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=coef_w_5 cyclic factor=10

LAYER5:
	ap_uint<11> idx = 0;
	for (ap_uint<7> smap = 0; smap < 120; smap++) {
#pragma HLS PIPELINE
		for (ap_uint<4> dmap = 0; dmap < 10; dmap++) {
			ap_int<2> dat;
			ap_int<7> coef;

			if (buf[5 & 0x1][smap] == 1){
				dat = 1;
			}else{
				dat = -1;
			}

			coef = coef_w_5[idx];
			idx++;

			result[dmap] += (dat * coef);
		} // end for dmap
	} // end for smap

LOOP_SUM:
	for (ap_uint<4> dmap = 0; dmap < 10; dmap++) {
#pragma HLS PIPELINE
		result[dmap] += bias_5[dmap];
	}
}

/* #############################################################*/
/*                         END OF PROGRAM                        */
/* #############################################################*/
