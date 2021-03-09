CC=${CROSS_PREFIX}gcc
CXX=${CROSS_PREFIX}g++
AR=${CROSS_PREFIX}ar rcs
RANLIB=${CROSS_PREFIX}ranlib

mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
current_dir=$(shell dirname ${mkfile_path})
ZLIB_INCDIR=${current_dir}/../zlib-1.2.11/_install/include
ZLIB_LIBDIR=${current_dir}/../zlib-1.2.11/_install/lib

CFLAGS=-O3 -std=c++11 -I${XILINX_VIVADO}/include/ -DHLS_NO_XIL_FPO_LIB -I$(ZLIB_INCDIR)
LDFLAGS=-lminizip -laes -lz -L$(ZLIB_LIBDIR)
