all: example slow_example debug

example: example.cu
	nvcc -O3 -o pexample example.cu

slow_example: slow_example.cu
	nvcc -O3 -o slexample slow_example.cu

debug: example.cu
	nvcc -O0 -o dexample --debug --device-debug example.cu

clean:
	rm -rf *example
