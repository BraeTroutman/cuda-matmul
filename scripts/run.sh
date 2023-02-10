#!/bin/bash

time="/usr/bin/time"

for i in {1..1000..10}
do
	$time -f %E ./bin/matmul $i
done

