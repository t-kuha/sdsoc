
#include <iostream>

#include "hw.h"

#include "sds_lib.h"

void show_value(int* src, int num);


int main(int argc, char* argv[])
{
	std::cout << "-------------------------" << std::endl;

	// Need to use sds_alloc() for
	// Make sure all arrays passed to the zero_copy data mover are allocated with sds_alloc
	int* in = NULL;
	int* out1 = NULL;
	int* out2 = NULL;

	in = (int*) sds_alloc(NUM_ELEM*sizeof(int));
	out1 = (int*) sds_alloc(NUM_ELEM*sizeof(int));
	out2 = (int*) sds_alloc(NUM_ELEM*sizeof(int));

	// Initialize source array
	for(int i = 0; i < NUM_ELEM; i++){
		in[i] = 256 - i;
	}

	std::cout << "  Uninitialized Value" << std::endl;
	hw(out1, 1);
	show_value(out1, NUM_ELEM);

	std::cout << "  Initializing Local Memory..." << std::endl;
	hw(in, 0);
	show_value(in, NUM_ELEM);

	std::cout << "  Reading Back Initialized Value..." << std::endl;
	show_value(out2, NUM_ELEM);
	hw(out2, 1);
	show_value(out2, NUM_ELEM);


	if(in){
		sds_free(in);
	}
	if(out1){
		sds_free(out1);
	}
	if(out2){
		sds_free(out2);
	}

	std::cout << "-------------------------" << std::endl;

	return 0;
}

void show_value(int* src, int num)
{
	std::cout << "    ";
	for(int i = 0; i < num; i++){
		std::cout << src[i] << " ";
	}
	std::cout << std::endl;
}
