#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <string.h>

#include "cudss.h"

#define cuda_check(call) { \
		cudaError_t cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
                fprintf(stderr, "CUDA: something went wrong, error = %s\n", cudaGetErrorString(cuda_error)); \
                exit(-1); \
        } }

#define cudss_check(call) { \
		cudssStatus_t cuda_status = call; \
        if (cuda_status != CUDSS_STATUS_SUCCESS) { \
                fprintf(stderr, "CUDSS: something went wrong, status = %d\n", cuda_status); \
                exit(-1); \
        } }

#define cudss_check_and_go(call) { \
		cudssStatus_t cuda_status = call; \
		if (cuda_status != CUDSS_STATUS_SUCCESS) { \
				success = 0; \
				goto free; \
		} }

#include <lpsolver/ludec.hpp>

int ax_equals_b_solver(int n, int nnz, const int* offsets_h, const int* columns_h, const double* vals_h, int rhs_size, const double* b_h, double* x_h) {

	int* offsets_d = NULL;
	int* columns_d = NULL;
	double* vals_d = NULL;

    cuda_check(cudaMalloc(&offsets_d, (n + 1) * sizeof(int)));
    cuda_check(cudaMalloc(&columns_d, nnz * sizeof(int)));
    cuda_check(cudaMalloc(&vals_d, nnz * sizeof(double)));

	cuda_check(cudaMemcpy(offsets_d, offsets_h, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(columns_d, columns_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(vals_d, vals_h, nnz * sizeof(double), cudaMemcpyHostToDevice));

	double* b_d = NULL;
	double* x_d = NULL;

	cuda_check(cudaMalloc(&b_d, rhs_size * n * sizeof(double)));
	cuda_check(cudaMalloc(&x_d, rhs_size * n * sizeof(double)));

	cuda_check(cudaMemcpy(b_d, b_h, rhs_size * n * sizeof(double), cudaMemcpyHostToDevice));

	cudaStream_t stream = NULL;
    cuda_check(cudaStreamCreate(&stream));

    cudssHandle_t handle;
   	cudss_check(cudssCreate(&handle));

	cudss_check(cudssSetStream(handle, stream));

    cudssConfig_t solverConfig;
    cudssData_t solverData;

    cudss_check(cudssConfigCreate(&solverConfig));
    cudss_check(cudssDataCreate(handle, &solverData));

    cudssAlgType_t reorder_alg = CUDSS_ALG_2;
    cudss_check(cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG,
                &reorder_alg, sizeof(cudssAlgType_t)));

    cudssMatrix_t x, b;

    cudss_check(cudssMatrixCreateDn(&b, n, rhs_size, n, b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    cudss_check(cudssMatrixCreateDn(&x, n, rhs_size, n, x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

	cudssMatrix_t A;
    cudss_check(cudssMatrixCreateCsr(&A, n, n, nnz, offsets_d, NULL, columns_d, vals_d, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    int success = 1;
	cudss_check_and_go(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b));
	cudss_check_and_go(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b));
    cudss_check_and_go(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b));

free:
    cudss_check(cudssMatrixDestroy(A));
    cudss_check(cudssMatrixDestroy(b));
    cudss_check(cudssMatrixDestroy(x));
    cudss_check(cudssDataDestroy(handle, solverData));
    cudss_check(cudssConfigDestroy(solverConfig));
    cudss_check(cudssDestroy(handle));

	cuda_check(cudaStreamSynchronize(stream));

	cuda_check(cudaMemcpy(x_h, x_d, rhs_size * n * sizeof(double), cudaMemcpyDeviceToHost));
	
	cudaFree(offsets_d);
	cudaFree(columns_d);
	cudaFree(vals_d);

	cudaFree(b_d);
	cudaFree(x_d);

	return success;
}
