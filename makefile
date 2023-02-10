targets=bin/matmul

.phony: all clean test
all: $(targets)

bin/matmul: src/matmul.cu
	nvcc -o bin/matmul src/matmul.cu

clean:
	-rm bin/*

test: $(targets)
	bash scripts/test.bash

