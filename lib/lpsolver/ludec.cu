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
				*success = 0; \
				goto free; \
		} }

#include <lpsolver/ludec.hpp>

mdata csc_to_csr(int n, int nnz, const int* rows, const int* cols, const double* vals) {
	mdata res;

    cuda_check(cudaMalloc(&res.offsets, (n + 1) * sizeof(int)));
    cuda_check(cudaMalloc(&res.columns, nnz * sizeof(int)));
    cuda_check(cudaMalloc(&res.vals, nnz * sizeof(double)));

	int *res_offsets = (int*)calloc(n + 1, sizeof(int));
	int *res_columns = (int*)malloc(nnz * sizeof(int));
	double *res_vals = (double*)malloc(nnz * sizeof(double));

	for (int i = 0; i < nnz; ++i) {
		res_offsets[rows[i] + 1]++;
	}
	for (int i = 1; i <= n; ++i) {
		res_offsets[i] += res_offsets[i - 1];
	}

	int* inds = (int*)malloc(n * sizeof(int));
	memcpy(inds, res_offsets, n * sizeof(int));

	int col = 0;
	for (int i = 0; i < nnz; ++i) {
		while (cols[col + 1] <= i) col++;
		res_columns[inds[rows[i]]] = col;
		res_vals[inds[rows[i]]] = vals[i];
		inds[rows[i]]++;
	}

	free(inds);

    cuda_check(cudaMemcpy(res.offsets, res_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(res.columns, res_columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(res.vals, res_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    free(res_offsets);
    free(res_columns);
    free(res_vals);
	return res;
		
}

double* ax_equals_b_solver(int n, int nnz, mdata A_data, int bsz, const double* b_h, int* success) {
	
	double* x_h = (double*)malloc(bsz * n * sizeof(double));

	int* offsets_d = A_data.offsets;
	int* columns_d = A_data.columns;
	double* vals_d = A_data.vals;

	double* b_d = NULL;
	double* x_d = NULL;

	cuda_check(cudaMalloc(&b_d, bsz * n * sizeof(double)));
	cuda_check(cudaMalloc(&x_d, bsz * n * sizeof(double)));

	cuda_check(cudaMemcpy(b_d, b_h, bsz * n * sizeof(double), cudaMemcpyHostToDevice));

	cudaStream_t stream = NULL;
    cuda_check(cudaStreamCreate(&stream));

    cudssHandle_t handle;
   	cudss_check(cudssCreate(&handle));

	cudss_check(cudssSetStream(handle, stream));

    cudssConfig_t solverConfig;
    cudssData_t solverData;

    cudss_check(cudssConfigCreate(&solverConfig));
    cudss_check(cudssDataCreate(handle, &solverData));

    cudssMatrix_t x, b;

    cudss_check(cudssMatrixCreateDn(&b, n, bsz, n, b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    cudss_check(cudssMatrixCreateDn(&x, n, bsz, n, x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

	cudssMatrix_t A;
    cudss_check(cudssMatrixCreateCsr(&A, n, n, nnz, offsets_d, NULL, columns_d, vals_d, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    *success = 1;
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

	cuda_check(cudaMemcpy(x_h, x_d, bsz * n * sizeof(double), cudaMemcpyDeviceToHost));
	
	cudaFree(offsets_d);

	cudaFree(columns_d);

	cudaFree(vals_d);

	cudaFree(b_d);
	cudaFree(x_d);

	return x_h;
}
