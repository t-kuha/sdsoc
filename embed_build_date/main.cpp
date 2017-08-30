/*
 * main.cpp
 *
 *  Created on: Aug 28, 2017
 *      Author: kuriharat
 */


#include <stdio.h>

//#include "sds_utils.h"

void get_build_date(char str[21]){
#pragma HLS ARRAY_PARTITION variable=str complete
	// 20yy-mm-dd HH:MM:SS
	str[0] = __DATE__[7];
	str[1] = __DATE__[8];
	str[2] = __DATE__[9];
	str[3] = __DATE__[10];
	str[4] = ' ';
	str[5] = __DATE__[0];
	str[6] = __DATE__[1];
	str[7] = __DATE__[2];
	str[8] = ' ';
	str[9] = __DATE__[4];
	str[10] = __DATE__[5];
	str[11] = ' ';
	str[12] = __TIME__[0];
	str[13] = __TIME__[1];
	str[14] = __TIME__[2];
	str[15] = __TIME__[3];
	str[16] = __TIME__[4];
	str[17] = __TIME__[5];
	str[18] = __TIME__[6];
	str[19] = __TIME__[7];
	str[20] = '\0';
}

int main(int argc, char* argv[])
{

	char str[21] = "";
	/*sds_utils::*/get_build_date(str);

	printf("HW Build date: %s\n", str);

	return 0;
}
