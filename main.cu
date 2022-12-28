#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define N 800
#define BLOCKWIDTH 32
#define PRINTLOGS 0
#define NUMTESTS 10

void generate_square_matrix(float *M, int size) {
    int max = 9;
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            M[i*size + j] = rand() % max + 1.0;
        }
    }
}

__device__ void device_print_square_matrix (float *M, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", M[i*size + j]);
        }
        printf("\n");
    }
}

void print_square_matrix(float *M, int size) {
    if (!PRINTLOGS) return;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", M[i*size + j]);
        }
        printf("\n");
    }
}

void floyd_warshall_cpu(float *A, int size) {
	float B[N*N];
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

void test_floyd_warshall_cpu() {
    float total_time_cpu = 0;
    for (int i = 0; i < NUMTESTS; i++) {
        float A[N*N];
        generate_square_matrix(A, N);
        struct timeval start, end;
        gettimeofday(&start, NULL);
        floyd_warshall_cpu(A, N);
        gettimeofday(&end, NULL);
        float interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;
        printf("CPU: Test %d took %.7f seconds\n", i, interval);
        sleep(1.1);
        if(i == 0) continue;
        total_time_cpu += interval;
    }
    printf("CPU tests took %.7f seconds on average\n", total_time_cpu/NUMTESTS);
}

void check(float *G, float *final_G, int n)  {
    int correct = 1;
    float tol = 1e-6;
    floyd_warshall_cpu(G, N);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((G[i*n+j] - final_G[i*n+j])*(G[i*n+j] - final_G[i*n+j]) > tol) {
                correct = 0;
                break;
            }
        }
        if (!correct) break;
    }
    if (correct) printf("(CORRECT)\n");
    else printf("(INCORRECT)\n");
    
}

// Basic Floyd Warshall Algorithm running on the GPU instead of CPU. Uses global memory in the GPU.

__global__ void update_cells(float *d_in, float *d_out, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < n) && (col < n)) {
        float new_path = d_in[row * n + k] + d_in[k * n + col];
        float old_path = d_in[row * n + col];
        d_out[row * n + col] = min(new_path, old_path);
    }
}

void test_floyd_warshall_gpu() {
    float total_time_gpu = 0;
    for (int i = 0; i < NUMTESTS; i++) {
        float A[N * N];
        generate_square_matrix(A, N);
        struct timeval start, end;

        float *d_A, *d_B;
        cudaMalloc((void **) &d_A, N*N*sizeof(float));
        cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_B, N*N*sizeof(float));

        gettimeofday(&start, NULL);

        for (int k = 0; k < N; k++) {        
            dim3 dimGrid(ceil(N/BLOCKWIDTH), ceil(N/BLOCKWIDTH), 1);
			dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
			update_cells<<<dimGrid, dimBlock>>>(d_A, d_B, N, k);
			cudaDeviceSynchronize();
            cudaMemcpy(d_A, d_B, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        gettimeofday(&end, NULL);
        float final_A[N*N];
        cudaMemcpy(final_A, d_A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        check(A, final_A, N);
        cudaFree(d_A); cudaFree(d_B);

        float interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;
        sleep(1.1);
        printf("GPU: Test %d took %.7f seconds\n", i, interval);
        total_time_gpu += interval;
    }
    printf("GPU tests took %.7f seconds on average\n", total_time_gpu/NUMTESTS);
}

// Basic Floyd Warshall Algorithm that runs on the GPU employs tiling and shared memory

__global__ void update_cells_tiled(float *d_in, float *d_out, int n, int k) {
    __shared__ float shared_row[BLOCKWIDTH];
    __shared__ float shared_col[BLOCKWIDTH];
    __syncthreads();

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   

    if (tx == 0) {
        shared_col[ty] = d_in[row * n + k];      
    } 
    if (ty == 0) {
        shared_row[tx] = d_in[k * n + col];      
    }
    __syncthreads();

    if ((row < n) && (col < n)) {
        float new_path = shared_row[tx] + shared_col[ty];
        float old_path = d_in[row * n + col];
        d_out[row * n + col] = min(new_path, old_path);
    }
}

void test_tiled_floyd_warshall() {
    float total_time = 0;
    for (int i = 0; i < NUMTESTS; i++) {
        float A[N * N];
        generate_square_matrix(A, N);
        struct timeval start, end;

        float *d_A, *d_B;
        cudaMalloc((void **) &d_A, N*N*sizeof(float));
        cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_B, N*N*sizeof(float));

        gettimeofday(&start, NULL);

        for (int k = 0; k < N; k++) {        
            dim3 dimGrid(ceil(N/BLOCKWIDTH), ceil(N/BLOCKWIDTH), 1);
            dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
            update_cells_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, N, k);
            cudaDeviceSynchronize();
            cudaMemcpy(d_A, d_B, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        gettimeofday(&end, NULL);
        float final_A[N*N];
        cudaMemcpy(final_A, d_A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        check(A, final_A, N);
        cudaFree(d_A); cudaFree(d_B);

        float interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;
        sleep(1.1);
        printf("GPU (tiled): Test %d took %.7f seconds\n", i, interval);
        total_time += interval;
    }
    printf("GPU Tiled tests took %.7f seconds on average\n", total_time/NUMTESTS);
}

// Blocked Floyd Warshall Algorithm


__device__ void block_kernel(float *C, float *A, float *B, int row, int col) {
    for (int k = 0; k < BLOCKWIDTH; k++) {
        C[row * BLOCKWIDTH + col] = C[row * BLOCKWIDTH + col] > A[row * BLOCKWIDTH + k] + B[k * BLOCKWIDTH + col] ?
        A[row * BLOCKWIDTH + k] + B[k * BLOCKWIDTH + col] : C[row * BLOCKWIDTH + col];
    }
}

__global__ void blocked_floyd_warshall_phase_one (int k, float *G) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    __shared__ float C[BLOCKWIDTH * BLOCKWIDTH];
    __syncthreads();

    C[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    __syncthreads();

    block_kernel(C, C, C, ty, tx);
    __syncthreads();

    G[(k * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];
}

__global__ void blocked_floyd_warshall_phase_two (int k, float *G) {
    // Grid is one dimensional only
    int bx = blockIdx.x;
    int tx = threadIdx.x; int ty = threadIdx.y;
    if (bx == k) return;

    __shared__ float A[BLOCKWIDTH * BLOCKWIDTH];
    __shared__ float B[BLOCKWIDTH * BLOCKWIDTH];
    __shared__ float C[BLOCKWIDTH * BLOCKWIDTH];
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

__global__ void blocked_floyd_warshall_phase_tree(int k, float *G) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    if (bx == k && by == k) return;

    __shared__ float A[BLOCKWIDTH * BLOCKWIDTH];  // block in col k, row by
    __shared__ float B[BLOCKWIDTH * BLOCKWIDTH];  // block in col bx, row k
    __shared__ float C[BLOCKWIDTH * BLOCKWIDTH];  // block in col bx, row by
    __syncthreads();

    C[ty * BLOCKWIDTH + tx] = G[(by * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)];
    A[ty * BLOCKWIDTH + tx] = G[(by * BLOCKWIDTH + ty) * N + (k * BLOCKWIDTH + tx)];
    B[ty * BLOCKWIDTH + tx] = G[(k * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)];
    __syncthreads();

    block_kernel(C, A, B, ty, tx);
    __syncthreads();

    G[(by * BLOCKWIDTH + ty) * N + (bx * BLOCKWIDTH + tx)] = C[ty * BLOCKWIDTH + tx];
}

void test_blocked_floyd_warshall() {
    float total_time = 0;
    for (int i = 0; i < NUMTESTS; i++) {
        float G[N*N];
        generate_square_matrix(G, N);

        float *d_G;
        cudaMalloc((void **) &d_G, N*N*sizeof(float));
        cudaMemcpy(d_G, G, N*N*sizeof(float), cudaMemcpyHostToDevice);
        struct timeval start, end;
        gettimeofday(&start, NULL);

        int num_blocks = (N+BLOCKWIDTH-1) / (BLOCKWIDTH);
        dim3 dimGrid(num_blocks, num_blocks, 1);
        dim3 dimBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
        for (int k = 0; k < num_blocks; k++) {
            blocked_floyd_warshall_phase_one<<<1, dimBlock>>>(k, d_G);
            blocked_floyd_warshall_phase_two<<<num_blocks, dimBlock>>>(k, d_G);
            blocked_floyd_warshall_phase_tree<<<dimGrid, dimBlock>>>(k, d_G);
        }

        gettimeofday(&end, NULL);
        float interval = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1.0)/1000000;
        sleep(1.1);
        printf("Blocked: Test %d took %.7f seconds\n", i, interval);
        total_time += interval;
        float final_G[N*N];
        cudaMemcpy(final_G, d_G, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        check(G, final_G, N);
        cudaFree(d_G);
    }
    printf("Blocked Floyd Warshall tests took %.7f seconds on average\n", total_time/NUMTESTS);

}


int main() {
    test_floyd_warshall_gpu();
    test_tiled_floyd_warshall();
    test_blocked_floyd_warshall();
    return 0;
}