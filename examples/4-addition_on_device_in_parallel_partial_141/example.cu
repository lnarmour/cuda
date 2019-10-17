#include <stdio.h>
#include <stdlib.h>

/*

Identical to example #4 except use fewer than N block when invoking add kernel 

*/

__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
#define MAX_INT 500
#define NUM_BLOCKS 141

int main(void) {
	int *a, *b, *c;             // host copies of a, b, c
	int *d_a, *d_b, *d_c;    // device copies
	int size = sizeof(int) * N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// set up input valies
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	for (int i=0; i<N; i++) {
		a[i] = rand() % MAX_INT + 1;
		b[i] = rand() % MAX_INT + 1;
	}

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// launch add() kernel on CPU
	add<<<NUM_BLOCKS,1>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);

	}

	// cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

