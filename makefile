targets=bin/matmul
sources=$(shell find src -type f -name *.cu)
objects=$(patsubst src/%,build/%,$(sources:.cu=.o))

.phony: all clean test run
all: $(targets)

bin/matmul: $(objects)
	nvcc -o $@ $^ 

build/%.o: src/%.cu
	nvcc -I include -o $@ -c $^

run: $(targets)
	bash scripts/run.sh

clean:
	-rm bin/* build/*

test: $(targets)
	bash scripts/test.bash

