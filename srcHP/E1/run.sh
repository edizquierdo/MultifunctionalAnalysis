#!/bin/bash

for ((i = 20; i < 100; i += 1));
do
  sbatch --export=JB=$i main.sh
  sleep 7
done
