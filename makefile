targets=bin/matmul
sources=$(shell find src -type f -name *.cu)
objects=$(patsubst src/%,build/%,$(sources:.cu=.o))
dbobjects=$(patsubst src/%,build/%,$(sources:.cu=.do))

.phony: all clean test run debug
all: $(targets)
debug: bin/debug

bin/matmul: $(objects)
	nvcc -o $@ $^

bin/debug: $(dbobjects)
	nvcc -g -o $@ $^

build/%.o: src/%.cu
	nvcc -I include -o $@ -c $^

build/%.do: src/%.cu
	nvcc -I include -g -o $@ -c $^

run: $(targets)
	bash scripts/run.sh

clean:
	-rm bin/* build/*

test: $(targets)
	bash scripts/test.bash

