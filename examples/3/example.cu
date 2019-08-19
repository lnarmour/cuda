#include <stdio.h>
#include <stdlib.h>

/*

We can invoke add() on the device in parallel
Each parallel invocation is called a "block"
* the set of blocks in called a "grid"
* each invocation can refer to its block index using "blockIdx.x"

*/

__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
#define MAX_INT 500

int random_ints(int* array, int N) {
	for (int i=0; i<N; i++) {
		array[i] = rand() % MAX_INT + 1;
	}
	return 0;
}

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
	random_ints(a, N);
	random_ints(b, N);

	// copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// launch add() kernel on CPU
	add<<<1,1>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		if (i == 11) {
			printf("...\n");
		}
		if (10 < i < N-1) {			
			continue;
		}
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

