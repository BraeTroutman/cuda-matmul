#!/bin/bash

for i in {1..100}
do
	M=$((RANDOM%100))
	N=$((RANDOM%100))
	K=$((RANDOM%100))
	RES=$(./bin/matmul $M $N $K)

	if [ -z $RES ]
	then
		echo test matmul $M $N $K success!
	else
		echo test matmul $M $N $K failed!
	fi
done

