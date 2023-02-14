#!/bin/bash

for i in {1..100}
do
	M=$((RANDOM%99 + 1))
	N=$((RANDOM%99 + 1))
	K=$((RANDOM%99 + 1))
	RES=$(./bin/matmul $M $N $K)

	if [ -z $RES ]
	then
		echo test matmul $M $N $K success!
	else
		echo test matmul $M $N $K failed!
	fi
done

