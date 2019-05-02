
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "math_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>

#define HEIGHT 4000
#define WIDTH (HEIGHT + 1)

typedef double *matrix;

matrix	A;		// matrix A				

const int maxnum = 15;	// max element size

// Gaussian elimination and backward substitution.
void eliminate_and_solve();
void eliminate_and_solve_with_cuda(int per_thread);

void Init_Matrix();

void Print_Matrix();

void Print_Solution();

double Avg_Diff(double CPU[], double GPU[]);

__global__
void subtract(double *d_A, int per_thread, int row);

__global__
void divide(double *d_A, int per_thread, int row);

__global__
void solve(double *d_A, int per_thread, int row);

__global__
void start();

int main()
{
	std::clock_t start_time;
	double time;

	std::ofstream out;
	std::string file = std::to_string(HEIGHT);
	file.append(".txt");
	out.open(file);
	out << HEIGHT << std::endl;
	std::cout << "Matrix size: " << HEIGHT << "x" << HEIGHT << std::endl;

	double *A_copy = new double[HEIGHT * WIDTH];

	double CPU_solution[HEIGHT];
	double GPU_solution[HEIGHT];

	// This kernel only exists to be the
	// first launched kernel. Due to the
	// first kernel launch sometimes taking
	// about 1.2 seconds extra.
	start<<<1,1>>>();
	cudaDeviceSynchronize();

	Init_Matrix();

	// Copying the matrix so the same matrix is worked
	// on by the GPU later.
	memcpy(A_copy, A, HEIGHT * WIDTH * sizeof(double));

	// Start the timer, eliminate, solve and stop the timer.
	start_time = clock();
	eliminate_and_solve();
	time = 1000.0 * (clock() - start_time) / CLOCKS_PER_SEC;
	out << "CPU\t" << time << std::endl;
	std::cout << "CPU time: " << time << " ms" << std::endl;
	for (int i = 0; i < HEIGHT; i++)
	{
		CPU_solution[i] = A[i*WIDTH + WIDTH - 1];
	}
	memcpy(A, A_copy, HEIGHT * WIDTH * sizeof(double));

	// Fetching device properties to know
	// how many threads are allowed in a block
	// on this device.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int max_threads = prop.maxThreadsPerBlock;
	
	// Running the same matrix through the kernel with different amounts of threads.
	for (int per_thread = 1; per_thread <= 5; per_thread++)
	{
		start_time = clock();
		eliminate_and_solve_with_cuda(per_thread);
		time = 1000.0 * (clock() - start_time) / CLOCKS_PER_SEC;
		std::cout << "GPU time: " << time << " ms (" << per_thread << " per thread)" << std::endl;

		for (int i = 0; i < HEIGHT; i++)
		{
			GPU_solution[i] = A[i*WIDTH + WIDTH - 1];
		}
		memcpy(A, A_copy, HEIGHT * WIDTH * sizeof(double));

		double diff = Avg_Diff(CPU_solution, GPU_solution);		
		if (diff > 0.001)
		{
			std::cout << "Failed." << std::endl;
			break;
		}
		else
		{
			out << "GPU" << per_thread << "\t" << time << std::endl;
			//out << diff << std::endl;
		}
	}
	out.close();

	delete[] A;
	delete[] A_copy;

	getchar();
	return 0;
}

__global__
void divide(double *d_A, int per_thread, int row)
{
	int col = row + threadIdx.x + blockIdx.x * blockDim.x;
	double div = d_A[row*WIDTH + row];

	for (int i = 0; i < per_thread; i++)
	{
		if (col < WIDTH)
		{
			d_A[row*WIDTH + col] /= div;
		}
		col += blockDim.x * gridDim.x;
	}
}

__global__
void subtract(double *d_A, int per_thread, int row)
{
	int curr = row + 1 + threadIdx.x + blockIdx.x * blockDim.x;

	// Reading the row to be subtracted into shared memory
	// as it will be used by every thread. Indexing by
	// threadId only as every block will need their own row.
	__shared__ double s_row[WIDTH];
	for (int i = 0; i < (int)ceil((double)WIDTH / (double)blockDim.x); i++)
	{
		int index = threadIdx.x + i * blockDim.x;
		if (index < WIDTH)
		{
			s_row[index] = d_A[row*WIDTH + index];
		}
	}
	__syncthreads();

	for (int i = 0; i < per_thread; i++)
	{
		if (curr < HEIGHT)
		{	
			double mul = d_A[curr*WIDTH + row];			
			for (int k = row; k < WIDTH; k++)
			{				
				d_A[curr*WIDTH + k] -= s_row[k] * mul;	
			}			
		}
		curr += blockDim.x * gridDim.x;
	}
}

__global__
void solve(double *d_A, int per_thread, int row)
{
	int curr = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < per_thread; i++)
	{
		if (curr < row)
		{
			d_A[curr*WIDTH + WIDTH - 1] -= d_A[row*WIDTH + WIDTH - 1] * d_A[curr*WIDTH + row];
		}
		curr += blockDim.x * gridDim.x;
	}
}

void eliminate_and_solve()
{
	for (int i = 0; i < HEIGHT; i++)
	{
		// For each row, divide the elements i,...,WIDTH
		// by (i,i).
		double div = A[i*WIDTH + i];
		for (int j = i; j < WIDTH; j++)
		{
			A[i*WIDTH + j] /= div;		
		}

		// After dividing, for each row i+1,...,HEIGHT
		// subtract A(j,i) * A(i,k) from each element
		// i,...,WIDTH in the row.
		for (int j = i + 1; j < HEIGHT; j++)
		{
			double mul = A[j*WIDTH + i];
			for (int k = i; k < WIDTH; k++)
			{
				A[j*WIDTH + k] -= A[i*WIDTH + k] * mul;
			}
		}
	}	
	// From each y value, remove the value 
	// of each variable times the quantity 
	// of the variable in that row.
	for (int i = HEIGHT - 2; i > -1; i--)
	{
		for (int j = i + 1; j < WIDTH - 1; j++)
		{			
			A[i*WIDTH + WIDTH - 1] -= A[i*WIDTH + j] * A[j*WIDTH + WIDTH - 1];	
		}
	}
}

void eliminate_and_solve_with_cuda(int per_thread)
{
	// Fetching device properties to know
	// how many threads are allowed in a block
	// on this device.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);	
	int max_threads = prop.maxThreadsPerBlock;
	
	// Creating a pointer and allocating memory
	// on the device for that pointer.
	double *d_A; 
	cudaMalloc(&d_A, WIDTH*HEIGHT * sizeof(double));
	
	// Copying the memory from the matrix over to the device.
	cudaMemcpy(d_A, A, WIDTH*HEIGHT * sizeof(double), cudaMemcpyHostToDevice);

	int total_threads, n_threads, n_blocks;

	// Gaussian elimination.
	for (int i = 0; i < HEIGHT; i++)
	{
		total_threads = std::ceil((double)(WIDTH-i) / (double)per_thread);
		n_threads = std::min(512, total_threads);
		n_blocks = std::ceil((double)total_threads / (double)n_threads);

		divide <<< n_blocks, n_threads >>> (d_A, per_thread, i); 
		subtract <<< n_blocks, n_threads >>> (d_A, per_thread, i);
	}

	// Backward substitution.
	for (int i = WIDTH - 2; i > -1; i--)
	{
		total_threads = std::ceil((double)i / (double)per_thread);
		n_threads = std::min(512, total_threads);
		n_blocks = std::ceil((double)total_threads / (double)n_threads);

		solve <<< n_blocks, n_threads >>> (d_A, per_thread, i);
	}

	// Copying the solved matrix back to the host, consisting of
	// the upper triangular matrix and the solution in the rightmost column.
	cudaMemcpy(A, d_A, WIDTH*HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

void Init_Matrix()
{
	A = new double[HEIGHT * WIDTH];

	for (int i = 0; i < HEIGHT; i++) 
	{
		for (int j = 0; j < WIDTH; j++) 
		{
			if (i == j) /* diagonal dominance */
			{
				A[i*WIDTH + j] = (double)(rand() % maxnum) + 5.0;
			}
			else
			{
				A[i*WIDTH+j] = (double)(rand() % maxnum) + 1.0;
				
			}
		}
	}

	// Ditched the y vector and placing the b vector at the end of A.
	for (int i = 0; i < HEIGHT; i++) 
	{
		A[(i * WIDTH + WIDTH - 1)] = (double)(rand() % maxnum) + 1.0;
	}
}

void Print_Matrix()
{
	int i, j;

	printf("Matrix A:\n");
	for (i = 0; i < HEIGHT; i++) 
	{
		printf("[");
		for (j = 0; j < WIDTH; j++)
			printf(" %5.2f,", A[i*WIDTH+j]);
		printf("]\n");
	}
}

void Print_Solution()
{
	for (int i = 0; i < HEIGHT; i++)
	{
		double d = A[i*WIDTH + WIDTH - 1];
		printf("x%i = %5.2f\n", i, d);
	}
}

double Avg_Diff(double CPU[], double GPU[])
{
	double avg = 0.0;
	for (int i = 0; i < HEIGHT; i++)
	{
		avg += std::abs(CPU[i] - GPU[i]);
	}
	avg /= HEIGHT;
	return avg;
}

__global__
void start()
{
	printf("starting...\n");
}