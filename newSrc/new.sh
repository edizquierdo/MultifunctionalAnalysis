#!/bin/bash

current_dir=$(pwd)

mkdir S1

for ((i=0; i<10; i+=1));
do
	qsub -m abe -N evo -o evo.qlog -l nodes=1:ppn=16,vmem=2gb,walltime=2:30:00 -F "`expr $i` `expr $i + 1` `expr $current_dir` $(date +%N)" runFromNtoM.sh
	sleep 1
done
