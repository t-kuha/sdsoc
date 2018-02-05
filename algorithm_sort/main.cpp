#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>	// std::time()

#include "ex_sort.h"

//#ifdef __SDSCC__
//#include "sds_lib.h"
//#endif

int main(int argc, char* argv[])
{
	std::cout << "--------------------------" << std::endl;

	std::vector<int> in;
	int* buf_in = NULL;
	int* buf_out = NULL;

	// For small data, ordinary malloc() will suffice
//#ifdef __SDSCC__
//	buf_in = (int*) sds_alloc(N*sizeof(int));
//	buf_out = (int*) sds_alloc(N*sizeof(int));
//#else
	buf_in = (int*) malloc(N*sizeof(int));
	buf_out = (int*) malloc(N*sizeof(int));
//#endif

	std::srand(std::time(0));
	for (int i = 0; i < N; i++) {
		//
		int x = std::rand();
		in.push_back(x);
		buf_in[i] = x;
	}

	// Sort
	std::sort(in.begin(), in.end(), std::greater<int>());
	ex_sort(buf_in, buf_out);

	// Show result
	for (int i = 0; i < N; i++) {
		if (in.at(i) == buf_out[i]) {
			std::cout << "O.K. " << std::endl;
		}
		else {
			std::cout << in.at(i) << " | " << buf_out[i] << std::endl;
		}
	}

//#ifdef __SDSCC__
//	sds_free(buf_in);
//	sds_free(buf_out);
//#else
	free(buf_in);
	free(buf_out);
//#endif

	std::cout << "--------------------------" << std::endl;
	return 0;
}
