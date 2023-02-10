targets=bin/matmul

.phony: all clean test
all: $(targets)

bin/matmul: build/main.o build/matrix.o
	nvcc -o $@ $^ 

build/main.o: src/main.cu
	nvcc -I include -o $@ -c src/main.cu

build/matrix.o: src/matrix.cu
	nvcc -I include -o $@ -c src/matrix.cu

clean:
	-rm bin/*

test: $(targets)
	bash scripts/test.bash

