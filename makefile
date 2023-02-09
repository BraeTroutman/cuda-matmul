targets=bin/matmul

.phony: all clean
all: $(targets)

bin/matmul: src/matmul.cu
	nvcc -o bin/matmul src/matmul.cu

clean:
	-rm bin/*

