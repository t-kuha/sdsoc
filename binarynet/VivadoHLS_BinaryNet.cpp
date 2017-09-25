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

#ifdef __SYNTHESIS__
#include "weight.h"
#else
ap_int<7> coef_w_0[117600];	// 117600
ap_int<7> coef_w_1[4704];// 4704
ap_int<7> coef_w_2[240000];// 240000
ap_int<7> coef_w_3[6400];// 6400
ap_int<7> coef_w_4[4000];// 48000
ap_int<7> coef_w_5[1200];// 1200

ap_int<7> scale_f_0[4704];// 4704
ap_int<7> scale_f_1[1176];// 1176
ap_int<7> scale_f_2[1600];// 1600
ap_int<7> scale_f_3[400];// 400
ap_int<7> scale_f_4[120];// 120
ap_int<7> scale_f_5[10];// 10

ap_int<7> bias_0[4704];// 4704
ap_int<7> bias_1[1176];// 1176
ap_int<7> bias_2[1600];// 1600
ap_int<7> bias_3[400];// 400
ap_int<7> bias_4[120];// 120
ap_int<7> bias_5[10];// 10
#endif

// Cソース版はping-pongバッファを外部に持っていましたが, 内部に持つように
// BinaryNet()内部で宣言しました.
void BinaryNet(unsigned char *predict_num, // 認識した数字のインデックス
		ap_uint<32> pbuf[32] // 入力画像32x32ピクセル白黒画像
		);

// メイン関数. 認識用のBinaryNetのテストベンチになっています
#define INPUT_IS_8

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

	// Connection Table by LeCun ------------------------------
	// LeCunの論文[LeCun98]に書かれているとおりに接続テーブルを実装しました.
	// http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
	// 認識精度を維持しつつ, 過学習を避けて計算量の削減が達成できます.
	ap_uint<1> cnct_tbl[16][6] = { { 1, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, {
			0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 1, 1 }, { 1, 0, 0, 0, 1, 1 }, { 1,
			1, 0, 0, 0, 1 }, { 1, 1, 1, 1, 0, 0 }, { 0, 1, 1, 1, 1, 0 }, { 0, 0,
			1, 1, 1, 1 }, { 1, 0, 0, 1, 1, 1 }, { 1, 1, 0, 0, 1, 1 }, { 1, 1, 1,
			0, 0, 1 }, { 1, 1, 0, 1, 1, 0 }, { 0, 1, 1, 0, 1, 1 }, { 1, 0, 1, 1,
			0, 1 }, { 1, 1, 1, 1, 1, 1 } };

	// 入力された画像データをバッファメモリ(ping-pongメモリ)に格納
LOOP_INPUT_DATA:
	for (int yy = 0; yy < 32; yy++) {
		ap_uint<32> pict = pbuf[yy];
		//printf("yy=%d pict=%X ", yy, pict);
		for (int xx = 0; xx < 32; xx++) {
#pragma HLS PIPELINE
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
	 

    ap_uint<3> wx, wy;			// Kernel/Window size
    ap_uint<6> smap_x, smap_y;	// Size of input feature map
    ap_uint<6> dmap_x, dmap_y;	// Size of output feature map
    ap_uint<7> n_dmap, n_smap;	// n_dmap: # output channel | n_smap: # input channel
    ap_uint<2> dx, dy;			// Stride

	ap_int<24> result[10];		// Output score

	// Prediction --------------------------------------------
	// 認識本体のルーチン
	// 多重ループになっています. フル結合層も2次元畳込み層のループで記述できます.
	// (カーネルサイズ=フル結合層のサイズn, とみなして, n x 1 回の２重ループとして計算する)

	// ６層レイヤのループ
LOOP_LAYER:
	for (ap_uint<3> layer = 0; layer < 6; layer++) {
		// set layer parameters
		// レイヤ毎にパラメータを設定します.
		// 俗に, ハイパーパラメータというのですが、なぜこう決めたのかはノウハウと経験です.
		// 少し変えるだけでも認識精度はガクッと落ちることが多く, 大変苦労する部分です.
		// 予め上手くいったパラメータ（構造）を一部利用して, 目的のニューラルネットワークを
		// 設計する転移学習という方法もあります.
		switch (layer) {
		case 0: // Convolutional layer
			wx = 5;
			wy = 5;
			n_smap = 1;
			smap_x = 32;
			smap_y = 32;
			n_dmap = 6;
			dx = 1;
			dy = 1;
			break;
		case 1: // Average pooling layer
			wx = 2;
			wy = 2;
			n_smap = 1;
			smap_x = 28;
			smap_y = 28;
			n_dmap = 6;
			dx = 2;
			dy = 2;
			break;
		case 2: // Convolutional layer
			wx = 5;
			wy = 5;
			n_smap = 6;
			smap_x = 14;
			smap_y = 14;
			n_dmap = 16;
			dx = 1;
			dy = 1;
			break;
		case 3: // Average pooling layer
			wx = 2;
			wy = 2;
			n_smap = 1;
			smap_x = 10;
			smap_y = 10;
			n_dmap = 16;
			dx = 2;
			dy = 2;
			break;
		case 4: // Convolutional layer
			wx = 5;
			wy = 5;
			n_smap = 16;
			smap_x = 5;
			smap_y = 5;
			n_dmap = 120;
			dx = 1;
			dy = 1;
			break;
		case 5: // Fully connection layer
			wx = 1;
			wy = 1;
			n_smap = 120;
			smap_x = 1;
			smap_y = 1;
			n_dmap = 10;
			dx = 1;
			dy = 1;
			break;
		default:
			//fprintf( stderr, "ERROR: UNDEFIND LAYER CALLED\n");
			break;
		}

		// 出力となる特徴マップのサイズは, ずらし量(stride)と入力特徴マップの
		// 大きさから一意に決まります
		// オリジナルのCソースは単に除算を行っていましたが, FPGA(ハードウェア)では
		// ものすごく遅く大きな除算回路を合成してしまうため, 2のべき乗の除算であっても
		// 明示的にシフタを記述しています..
		// (Vivado HLSちゃん, もうちょっと賢くなって！)
		if (dx == 1) {
			dmap_x = (smap_x - wx + dx);
			dmap_y = (smap_y - wy + dy);
		} else {
			dmap_x = (smap_x - wx + dx) >> 1;
			dmap_y = (smap_y - wy + dy) >> 1;
		}

		int x = 0, y = 0;
		int coef_offset = 0;

		ap_uint<16> idx = 0;

		// 出力特徴マップの値を求めます
		// ここから先のループをどこでインライン展開(#pragma unroll)するかが
		// 高位合成設計者の腕の見せ所.
		// (今回は載せていません)
		// なお、ディープニューラルネットワークに限ってはインライン展開を解析した論文があって
		// とても勉強になります.
		// http://cadlab.cs.ucla.edu/~cong/slides/fpga2015_chen.pdf
		// 著者のJ.CongはVivado HLSで使われている高位合成アルゴリズムの
		// 開発者の一人.
LOOP_DMAP:
		for (int dmap = 0; dmap < n_dmap; dmap++) {
LOOP_I:
			for (int i = 0; i < dmap_x * dmap_y; i++) {
				ap_int<24> temp = 0;
LOOP_SMAP:
				for (int smap = 0; smap < n_smap; smap++) {
					// Read connection from LeCun's table
					ap_uint<1> is_connect = 0;
					if (layer != 2)
						is_connect = 1;
					else if (cnct_tbl[dmap][smap])
						is_connect = 1;

					// If a source map is connected,
					//  then apply a convolutional operation
					if (is_connect) {
						// window size is wy x wx
LOOP_OY:
						for (int oy = 0; oy < wy; oy++) {
LOOP_OX:
							for (int ox = 0; ox < wx; ox++) {
//#pragma HLS LOOP_TRIPCOUNT max=5
#pragma HLS UNROLL
								ap_int<18> dat;
								ap_int<8> coef;
								if (layer == 1 || layer == 3) {
									// average pooling layer

									if (buf[layer & 0x1][dmap
											* (smap_x * smap_y)
											+ (y + oy) * smap_y + (x + ox)]
											== 1)
										dat = 1;
									else
										dat = -1;

									if (layer == 0) {
										coef = coef_w_0[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									} else if (layer == 1) {
										coef = coef_w_1[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									} else if (layer == 2) {
										coef = coef_w_2[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									} else if (layer == 3) {
										coef = coef_w_3[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									} else if (layer == 4) {
										coef = coef_w_4[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									} else {
										coef = coef_w_5[idx * (wx * wy * n_smap)
												+ (smap * wx * wy) + oy * wy
												+ ox];
									}
								} else {
									// convolutional and fully-connected layer

									if (buf[layer & 0x1][smap
											* (smap_x * smap_y)
											+ (y + oy) * smap_y + (x + ox)]
											== 1)
										dat = 1;
									else
										dat = -1;

									if (layer == 0) {
										coef = coef_w_0[coef_offset + oy * wy
												+ ox];
									} else if (layer == 1) {
										coef = coef_w_1[coef_offset + oy * wy
												+ ox];
									} else if (layer == 2) {
										coef = coef_w_2[coef_offset + oy * wy
												+ ox];
									} else if (layer == 3) {
										coef = coef_w_3[coef_offset + oy * wy
												+ ox];
									} else if (layer == 4) {
										coef = coef_w_4[coef_offset + oy * wy
												+ ox];
									} else {
										coef = coef_w_5[coef_offset + oy * wy
												+ ox];
									}

								}

								// Perform an ADD-MUL operation
								temp += (dat * coef);
							} // end for oy
						} // end for ox

						// Update offset, since the LeCun's table requires
						// uniformaly connection
						coef_offset += (wx * wy);
					} // end for is_connect
				} // end for smap

				ap_int<8> sf, bi;

				if (layer == 0) {
					sf = scale_f_0[idx];
					bi = bias_0[idx];
				} else if (layer == 1) {
					sf = scale_f_1[idx];
					bi = bias_1[idx];
				} else if (layer == 2) {
					sf = scale_f_2[idx];
					bi = bias_2[idx];
				} else if (layer == 3) {
					sf = scale_f_3[idx];
					bi = bias_3[idx];
				} else if (layer == 4) {
					sf = scale_f_4[idx];
					bi = bias_4[idx];
				} else {
					sf = scale_f_5[idx];
					bi = bias_5[idx];
				}

				// Activation function for the BinaryNet
				// 活性化関数を省略して2値化しています.
				// ビット精度の調整もいらないので便利♪
				if (layer != 5) {
					temp = temp * sf; // 8b x 8b = 16b
					temp = temp + bi;
					if (temp >= 0)
						temp = 1; //1.0;
					else
						temp = -1; //-1.0;
				} else {
					// 最終層のみ, ２値化せずにそのまま計算結果を格納しています.
					// 幸い手書き数字認識なので10ニューロンで済みましたので、そのまま配列で書いて
					// レジスタに合成
					temp = temp + bi;
					result[idx] = temp;
				}

				// Store ping-pong memory
				// ２値化した結果をping-pongメモリに格納します
				// ただし, 最終層は２値化すると精度に影響がでるのでやっていません.
				if (temp == 1)
					buf[(layer + 1) & 0x1][idx] = 1;
				else
					buf[(layer + 1) & 0x1][idx] = 0;

				// Update indices
				idx++;
				x += dx;
				if (x > (smap_x - wx)) {
					x = 0;
					y += dy;
					if (y > (smap_y - wy)) {
						y = 0;
					}
				}
			} // end for i
		} // end for dmap

	} // end for layer

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
#if !(!defined(__SDSVHLS__) || !defined(__SYNTHESIS__))
		printf("idx=%d %d\n", i, result[i].to_int());
#endif
	}
#if defined(__SDSVHLS__) && defined(__SYNTHESIS__)
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

//	//　認識した回数をインクリメント
//	// ホストでこの値の変化を調べて, 認識が終わったと判断します.
//	steps++;
//	return steps;
}

/* #############################################################*/
/*                         END OF PROGRAM                        */
/* #############################################################*/
