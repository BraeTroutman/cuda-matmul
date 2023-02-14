#!/bin/bash

for i in {1..1000..10}
do
	./bin/matmul -t $i $i $i
done

