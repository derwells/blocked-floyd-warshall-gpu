#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// Custom library
#include "data.h"

#define N 100
#define BLOCKWIDTH 32
#define NUMTESTS 10
#define DO_CHECKS true

void rand_seed() {
    // Set randomizer seed
    int stime;
    long ltime;
    ltime = time(NULL);
    stime = (unsigned) ltime/2;
    srand(stime);
}

__device__ void device_print_square_matrix (int *M, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", M[i*size + j]);
        }
        printf("\n");
    }
}

void print_square_matrix(int *M, int size, bool enabled) {
    if (!enabled) return;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", M[i*size + j]);
        }
        printf("\n");
    }
}

void generate_square_matrix(int *M, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Random int from 0.0 -> 10.0
            M[i*size + j] = ((int)rand()/(int)(RAND_MAX/10));
        }
    }
}

void floyd_warshall_cpu(int *A, int size) {
	int B[N*N];
	for (int k = 0; k < size; k++) {
		for (int i = 0; i < size; i++) {
			for (int j = 0;  j < size; j++) {
				B[i*size + j] = min(A[i*size + j], A[i*size + k] + A[k*size + j]);
			}
		}
		for (int i = 0; i < size; i++) {
			for (int j = 0;  j < size; j++) {
				A[i*size + j] = B[i*size + j];
			}
		}
	}
}

void test_floyd_warshall_cpu(datawrite *writer) {
    double total_time_cpu = 0;
    for (int i = 0; i < NUMTESTS; i++) {
        // Clear device mem and cache
        cudaDeviceReset();

        // Generate random square matrix
        int A[N*N];
        generate_square_matrix(A, N);


        // Run and time FW CPU
        struct timeval start, end;
        gettimeofday(&start, NULL);

        floyd_warshall_cpu(A, N);

        gettimeofday(&end, NULL);
        double interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;

        printf("CPU: Test %d took %.7f seconds\n", i, interval);

        // For avg. calculation
        total_time_cpu += interval;

        // Record data
        csventry entry = { i + 1, N, "cpu", interval };
        writeCSVEntry(writer, &entry);
    }
    printf("CPU tests took %.7f seconds on average\n", total_time_cpu/NUMTESTS);
}

bool check(int *G, int *final_G, int n) {    
    double tol = 1e-6;
    double delta, err;  

    // Get correct version
    floyd_warshall_cpu(G, N);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Get squared err
            delta = G[i*n+j] - final_G[i*n+j];

            if (delta != 0) {
                printf("delta: %lf\n", delta);
                return false;
            }
        }
    }

    return true;
}

// Basic Floyd Warshall Algorithm running on the GPU instead of CPU. Uses global memory in the GPU.

__global__ void update_cells(int *d_in, int *d_out, int n, int k) {
    // Algorithm in the innermost loop of the FW algorithm. Updates cell values based on the current iteration.

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < n) && (col < n)) {
        int new_path = d_in[row * n + k] + d_in[k * n + col];
        int old_path = d_in[row * n + col];
        d_out[row * n + col] = min(new_path, old_path);
    }
}

void fw(
    int *d_A,
    int *d_B,
    int n
) {
    for (int k = 0; k < N; k++) {        
        dim3 dimGrid(ceil(N/BLOCKWIDTH), ceil(N/BLOCKWIDTH), 1);
        dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
        update_cells<<<dimGrid, dimBlock>>>(d_A, d_B, n, k);
        cudaDeviceSynchronize();
        cudaMemcpy(d_A, d_B, N*N*sizeof(int), cudaMemcpyDeviceToDevice);
    }
}

void test_floyd_warshall_gpu(datawrite *writer) {

    double total_time_gpu = 0;

    for (int i = 0; i < NUMTESTS; i++) {
        // Clear device mem and cache
        cudaDeviceReset();

        // Generate random graph
        int A[N * N];
        generate_square_matrix(A, N);

        // Init device memory
        // d_A -> input
        // d_B -> output
        int *d_A, *d_B;
        cudaMalloc((void **) &d_A, N*N*sizeof(int));
        cudaMemcpy(d_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_B, N*N*sizeof(int));

        // Start test proper
        struct timeval start, end;
        gettimeofday(&start, NULL);

        for (int k = 0; k < N; k++) {        
            dim3 dimGrid(ceil(N/BLOCKWIDTH), ceil(N/BLOCKWIDTH), 1);
			dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
			update_cells<<<dimGrid, dimBlock>>>(d_A, d_B, N, k);
			cudaDeviceSynchronize();
            cudaMemcpy(d_A, d_B, N*N*sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // Get running time
        gettimeofday(&end, NULL);
        double interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;
    
        // Check output
        if (DO_CHECKS) {
            int final_A[N*N];
            cudaMemcpy(final_A, d_A, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            if (check(A, final_A, N))
                printf("CORRECT!\n");
            cudaFree(d_A); cudaFree(d_B);
        }

        printf("GPU: Test %d took %.7f seconds\n", i, interval);
        total_time_gpu += interval;

        // Record data
        csventry entry = { i + 1, N, "fw", interval };
        writeCSVEntry(writer, &entry);
    }
    printf("GPU tests took %.7f seconds on average\n", total_time_gpu/NUMTESTS);
}

// Blocked Floyd Warshall Algorithm


__device__ void block_kernel(int *C, int *A, int *B, int row, int col) {
    for (int k = 0; k < BLOCKWIDTH; k++) {
        C[row * BLOCKWIDTH + col] = C[row * BLOCKWIDTH + col] > A[row * BLOCKWIDTH + k] + B[k * BLOCKWIDTH + col] ?
        A[row * BLOCKWIDTH + k] + B[k * BLOCKWIDTH + col] : C[row * BLOCKWIDTH + col];
    }
}

__global__ void blocked_floyd_warshall_phase_one (int k, int *G) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    __shared__ int C[BLOCKWIDTH * BLOCKWIDTH];
    __syncthreads();

    C[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    __syncthreads();

    block_kernel(C, C, C, ty, tx);
    __syncthreads();

    G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];
}

__global__ void blocked_floyd_warshall_phase_two (int k, int *G) {
    // Grid is one dimensional only
    int bx = blockIdx.x;
    int tx = threadIdx.x; int ty = threadIdx.y;
    if (bx == k) return;

    __shared__ int A[BLOCKWIDTH * BLOCKWIDTH];
    __shared__ int B[BLOCKWIDTH * BLOCKWIDTH];
    __shared__ int C[BLOCKWIDTH * BLOCKWIDTH];
    __syncthreads();

    // Calculate kth row

    C[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)];
    A[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    __syncthreads();
    block_kernel(C, A, C, ty, tx);
    __syncthreads();
    G[(k * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];

    // Calculate kth column

    C[ty * BLOCKWIDTH + tx] = G[(bx * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    B[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    __syncthreads();
    block_kernel(C, C, B, ty, tx);
    __syncthreads();
    G[(bx * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];
}

__global__ void blocked_floyd_warshall_phase_tree(int k, int *G) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    if (bx == k && by == k) return;

    __shared__ int A[BLOCKWIDTH * BLOCKWIDTH];  // block in col k, row by
    __shared__ int B[BLOCKWIDTH * BLOCKWIDTH];  // block in col bx, row k
    __shared__ int C[BLOCKWIDTH * BLOCKWIDTH];  // block in col bx, row by
    __syncthreads();

    C[ty * BLOCKWIDTH + tx] = G[(by * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)];
    A[ty * BLOCKWIDTH + tx] = G[(by * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    B[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)];
    __syncthreads();

    block_kernel(C, A, B, ty, tx);
    __syncthreads();

    G[(by * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];
}

void bfw(
    int *d_G,
    int n
) {
    int num_blocks = (N+BLOCKWIDTH-1) / (BLOCKWIDTH);
    dim3 dimGrid(num_blocks, num_blocks, 1);
    dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);

    for (int k = 0; k < num_blocks; k++) {
        blocked_floyd_warshall_phase_one<<<1, dimBlock>>>(k, d_G);
        blocked_floyd_warshall_phase_two<<<num_blocks, dimBlock>>>(k, d_G);
        blocked_floyd_warshall_phase_tree<<<dimGrid, dimBlock>>>(k, d_G);
    } 
}

void test_blocked_floyd_warshall(datawrite *writer) {

    double total_time_gpu = 0;

    for (int i = 0; i < NUMTESTS; i++) {
        // Clear device mem and cache
        cudaDeviceReset();

        // Generate random graph
        int G[N*N];
        generate_square_matrix(G, N);

        // Copy graph matrix to device
        int *d_G;
        cudaMalloc((void **) &d_G, N*N*sizeof(int));
        cudaMemcpy(d_G, G, N*N*sizeof(int), cudaMemcpyHostToDevice);

        struct timeval start, end;
        gettimeofday(&start, NULL);

        bfw(d_G, N);

        gettimeofday(&end, NULL);
        double interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;

        total_time_gpu += interval;

        if (DO_CHECKS) {
            int final_G[N*N];
            cudaMemcpy(final_G, d_G, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            if (check(G, final_G, N))
                printf("CORRECT!\n");
            cudaFree(d_G);
        }

        printf("Blocked: Test %d took %.7f seconds\n", i, interval);

        // Record data
        csventry entry = { i + 1, N, "bfw", interval };
        writeCSVEntry(writer, &entry);
    }
    printf("Blocked Floyd Warshall tests took %.7f seconds on average\n", total_time_gpu/NUMTESTS);

}


int main() {
    // Set randomizer seed
    rand_seed();
    
    datawrite *writer = (datawrite *) malloc(sizeof(writer));
    writer->path = "record.csv";
    openCSV(writer);
    writeCSVHeader(writer);

    test_floyd_warshall_cpu(writer);
    test_floyd_warshall_gpu(writer);
    test_blocked_floyd_warshall(writer);
    return 0;
}