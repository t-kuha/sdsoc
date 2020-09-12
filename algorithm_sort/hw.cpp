//
// Based on:
// I. Skliarova, V. Sklyarov, A. Sudnitson, M. Kruus,
// "Integration of High-Level Synthesis to the Courses on Reconfigurable Digital Systems",
// Proceedings of the 38th International Conference on Information and Communication Technology, Electronics and Microelectronics
// - MIPRO, Opatija, Croatia, May, 2015, pp. 172-177.
//

#include "hw.h"

void ex_sort(int in[N], int out[N])
{
	ap_uint<M> work_array[N];
#pragma HLS ARRAY_PARTITION variable=work_array cyclic factor=2 dim=0
	ap_uint<M> work_array2[N];
#pragma HLS ARRAY_PARTITION variable=work_array2 cyclic factor=2 dim=0

	//1. Fill in the work array
init_loop:
	for (unsigned i = 0; i < N; i++) {
#pragma HLS PIPELINE
		work_array[i] = in[i];
	}

	//2. Sort the data
proc_loop:
	for (int n = 0; n < N / 2; n++) {
		// processing even pairs of registers
		// [0, 1], [2, 3], [4 ...
	sort_even:
		for (unsigned j = 0; j < (N / 2); j++){
#pragma HLS UNROLL
			if (work_array[2 * j] < work_array[2 * j + 1]) {
				work_array2[2 * j + 1] = work_array[2 * j];
				work_array2[2 * j]     = work_array[2 * j + 1];
			}else{
				work_array2[2 * j]     = work_array[2 * j];
				work_array2[2 * j + 1] = work_array[2 * j + 1];
			}

			if(N % 2 == 1){
				work_array[N - 1] = work_array2[N - 1];
			}
		}

		// processing odd pairs of registers
		// 0, [1, 2], [3, 4] ...
	sort_odd:
		for (unsigned j = 0; j < (N / 2 - 1); j++){
#pragma HLS UNROLL
			work_array[0] = work_array2[0];

			if (work_array2[2 * j + 1] < work_array2[2 * j + 2]) {
				work_array[2 * j + 1] = work_array2[2 * j + 2];
				work_array[2 * j + 2] = work_array2[2 * j + 1];
			}else{
				work_array[2 * j + 1] = work_array2[2 * j + 1];
				work_array[2 * j + 2] = work_array2[2 * j + 2];
			}

			if(N % 2 == 0){
				work_array[N - 1] = work_array2[N - 1];
			}
		}
	}

	//3. Write the result
write_res_loop:
	for (unsigned i = 0; i < N; i++) {
#pragma HLS PIPELINE
		out[i] = work_array[i];
	}
}
