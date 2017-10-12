/*
 * top.h
 *
 *  Created on: 2017/08/19
 *      Author: kuriharat
 */

#ifndef TOP_H_
#define TOP_H_

#include "bnn-library.h"
#include "config.h"


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo);

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val);

void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps);

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps);

#endif /* TOP_H_ */
