#!/bin/bash

for i in {1..100}
do
	M=$((RANDOM%99 + 1))
	N=$((RANDOM%99 + 1))
	K=$((RANDOM%99 + 1))
	RES=$(./bin/matmul -c $M $N $K)

	echo matmul $M $N $K $RES
done

