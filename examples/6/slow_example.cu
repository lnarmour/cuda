#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

/*

Stencil kernel example - better illustrates when using more than 1 thread per block is beneficial

  out[i] = in[i-3] + in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2] + in[i+3]

*/


#define N (2048*2048)
#define MAX_INT 10
#define THREADS_PER_BLOCK 512
#define RADIUS 3
#define BLOCK_SIZE (N / THREADS_PER_BLOCK)

__global__ void stencil(int *in, int *out) {
	int result = 0;
	for (int offset=-RADIUS; offset<=RADIUS; offset++) {
		result += in[blockIdx.x];
	}
	out[blockIdx.x] = result;
}


int main(void) {
	int *a, *b;             // host copies of a, b
	int *d_a, *d_b;         // device copies
	int size = sizeof(int) * N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);

	// set up input valies
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	for (int i=0; i<N; i++) {
		a[i] = rand() % MAX_INT + 1;
	}

	// timing
	struct timeval time;
	double elapsed_time;
	
	// call the main computation
	gettimeofday(&time, NULL);
	elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	// launch stencil() kernel on GPU
	stencil<<<N,1>>>(d_a, d_b);

	// copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	gettimeofday(&time, NULL);
	elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time;
	
	// timing information
	printf("Execution time : %lf sec.\n", elapsed_time);

	for (int i=0; i<N; i++) {
		//printf("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
		if (i == 50) {
			//printf("\n...only displaying first 50 indices\n\n");
			break;
		}
	}

	// cleanup
	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}

