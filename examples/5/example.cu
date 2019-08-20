#include <stdio.h>
#include <stdlib.h>

/*

when invoking add, you use add<<<X,Y>>>
X is the number of "blocks"
Y is the number of "threads per block"

add<<<4,9>>> gives a configuration like this

    threadIdx.x   |   threadIdx.x   |   threadIdx.x   |   threadIdx.x 
|-----------------------------------------------------------------------|
|0|1|2|3|4|5|6|7|8|0|1|2|3|4|5|6|7|8|0|1|2|3|4|5|6|7|8|0|1|2|3|4|5|6|7|8|
|-----------------------------------------------------------------------|
  blockIdx.x = 0  | blockIdx.x = 1  | blockIdx.x = 2  | blockIdx.x = 3

Need to use a combination of threadIdx.x and blockIdx.x to index a thread uniquely.

  int index = threadIdx.x + blockIdx.x * 9;

Or more generally, you can use "blockDim.x" to give the number of threads per block

  int index = threadIdx.x + blockIdx.x * blockDim.x;

Why use threads AND blocks (and not just add<<<1,X>>> or add<<<X,1>>>)?
* threads share the same memory
* threads can efficiently communicate and synchronize

*/

__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

#define N (2048*2048)
#define MAX_INT 500
#define THREADS_PER_BLOCK 512

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
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++) {
		printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);
		if (i == 20) {
			printf("\n...only displaying first 20 indices\n\n");
			break;
		}
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

