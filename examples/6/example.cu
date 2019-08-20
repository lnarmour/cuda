#include <stdio.h>
#include <stdlib.h>

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

	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	int local_index = threadIdx.x + RADIUS;

	// read input elements in shared memory
	temp[local_index] = in[global_index];
	if (threadIdx.x < RADIUS) {
		temp[local_index - RADIUS] = in[global_index - RADIUS];
		temp[local_index + BLOCK_SIZE] = in[global_index + BLOCK_SIZE];
	}

	// apply the stencil
	int result = 0;
	for (int offset=-RADIUS; offset<=RADIUS; offset++) {
		result += temp[local_index + offset];
	}

	// store the result
	out[global_index] = result;

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

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	// launch stencil() kernel on GPU
	stencil<<<BLOCK_SIZE,THREADS_PER_BLOCK>>>(d_a, d_b);

	// copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		printf("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
		if (i == 20) {
			printf("\n...only displaying first 20 indices\n\n");
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

