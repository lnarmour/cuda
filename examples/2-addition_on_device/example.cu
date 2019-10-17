#include <stdio.h>

/*
add() is executed on the device
a, b, and c must point to device memory
need to allocate memory on the GPU
 
pointers can be passed from host to device or device to host
but device points cannot be dereferenced on host, and vice versa

cudaMalloc(), cudaFree(), cudaMemcpy()  <-->  malloc(), free(), memcpy()
*/

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int main(void) {
	int a, b, c;             // host copies of a, b, c
	int *d_a, *d_b, *d_c;    // device copies
	int size = sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// set up input valies
	a = 2;
	b = 7;

	// copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// launch add() kernel on CPU
	add<<<1,1>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	// cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("a=%d, b=%d, c=%d\n", a, b, c);

	return 0;
}

