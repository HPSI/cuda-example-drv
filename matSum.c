/*
 * Vector addition (host code)
 * Andrei de A. Formiga, 2012-06-04
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>
#include <builtin_types.h>

#include "matSumKernel.h"
#include <string.h>

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err) {
		fprintf(stderr,
			"CUDA Driver API error = %04d from file <%s>, line %i.\n",
			err, file, line);
		exit(-1);
	}
}

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;
size_t totalGlobalMem;

char *module_file = (char *)"matSumKernel.ptx";
char *kernel_name = (char *)"matSum";

void initCUDA()
{
	int deviceCount = 0;
	CUresult err = cuInit(0);
	int major = 0, minor = 0;

	if (err == CUDA_SUCCESS)
		checkCudaErrors(cuDeviceGetCount(&deviceCount));

	if (deviceCount == 0) {
		fprintf(stderr, "Error: no devices supporting CUDA\n");
		exit(-1);
	}
	// get first CUDA device
	checkCudaErrors(cuDeviceGet(&device, 0));
#if 0
	char name[100];
	cuDeviceGetName(name, 100, device);
	printf("> Using device 0: %s\n", name);

	// get compute capabilities and the devicename
	checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));
	printf("  Total amount of global memory:   %llu bytes\n",
	       (unsigned long long)totalGlobalMem);
	printf("  64-bit Memory Address:           %s\n",
	       (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ?
	       "YES" : "NO");

#endif
	err = cuCtxCreate(&context, 0, device);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "* Error initializing the CUDA context.\n");
		cuCtxDetach(context);
		exit(-1);
	}

	err = cuModuleLoad(&module, module_file);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "* Error loading the module %s\n", module_file);
		cuCtxDetach(context);
		exit(-1);
	}

	err = cuModuleGetFunction(&function, module, kernel_name);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "* Error getting kernel function %s\n",
			kernel_name);
		cuCtxDetach(context);
		exit(-1);
	}
}

void finalizeCUDA()
{
	cuCtxDestroy(context);
}

void setupDeviceMemory(CUdeviceptr * d_a, CUdeviceptr * d_b, CUdeviceptr * d_c)
{
	checkCudaErrors(cuMemAlloc(d_a, sizeof(int) * N));
	checkCudaErrors(cuMemAlloc(d_b, sizeof(int) * N));
	checkCudaErrors(cuMemAlloc(d_c, sizeof(int) * N));
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
	checkCudaErrors(cuMemFree(d_a));
	checkCudaErrors(cuMemFree(d_b));
	checkCudaErrors(cuMemFree(d_c));
}

void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
	void *args;
	size_t arg_len = 3 * sizeof(CUdeviceptr);
	void *extra[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER,
		NULL,
		CU_LAUNCH_PARAM_BUFFER_SIZE,
		&arg_len,
		CU_LAUNCH_PARAM_END
	};
	printf("\n* Execute device kernel... *\n");

	args = malloc(arg_len);
	memcpy(args, &d_a, sizeof(d_a));
	memcpy(args + sizeof(d_a), &d_b, sizeof(d_b));
	memcpy(args + (2 * sizeof(d_a)), &d_c, sizeof(d_c));
	extra[1] = args;

	// grid for kernel: <<<N, 1>>>
	checkCudaErrors(cuLaunchKernel(function, N, 1, 1,	// Nx1x1 blocks
				       1, 1, 1,	// 1x1x1 threads
				       0, 0, 0, extra));
}

int main(int argc, char **argv)
{
	int a[N], b[N], c[N], i;
	CUdeviceptr d_a, d_b, d_c;

	// initialize host arrays
	for (i = 0; i < N; ++i) {
		a[i] = N - i;
		b[i] = i * i;
	}

	// initialize
	printf("- Initializing...\n");
	initCUDA();

	// allocate memory
	setupDeviceMemory(&d_a, &d_b, &d_c);

	// copy arrays to device
	checkCudaErrors(cuMemcpyHtoD(d_a, a, sizeof(int) * N));
	checkCudaErrors(cuMemcpyHtoD(d_b, b, sizeof(int) * N));

	// run
	printf("# Running the kernel...\n");
	int j = 10000;
	while (j--) {
		runKernel(d_a, d_b, d_c);
		printf("# Running the kernel...\n");

		// copy results to host and report
		checkCudaErrors(cuMemcpyDtoH(c, d_c, sizeof(int) * N));
		for (i = 0; i < N; ++i) {
			if (c[i] != a[i] + b[i])
				printf
				    ("* Error at array position %d: Expected %d, Got %d\n",
				     i, a[i] + b[i], c[i]);
		}
		printf("*** All checks complete.\n");
	}

	// finish
	printf("- Finalizing...\n");
	releaseDeviceMemory(d_a, d_b, d_c);
	finalizeCUDA();
	return 0;
}
