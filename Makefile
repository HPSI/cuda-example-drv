#CUDA_PATH=/usr/local/cuda-5.0
CUDA_PATH=/usr
CFLAGS=-I$(CUDA_PATH)/include
LDFLAGS=-L$(CUDA_PATH)/lib -lcuda
NVCC=$(CUDA_PATH)/bin/nvcc

all: matSumKernel.ptx matSum

matSumKernel.ptx: matSumKernel.cu matSumKernel.h
	$(NVCC) -ptx matSumKernel.cu

matSum:
	$(CC) matSum.c -o matSum $(LDFLAGS)

clean:
	rm *.ptx matSum
