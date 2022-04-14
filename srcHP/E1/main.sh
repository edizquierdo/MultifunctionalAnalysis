#!/bin/bash
#SBATCH -J mpg
#SBATCH -p general
#SBATCH -o mpg_%j.txt
#SBATCH -e mpg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=2:00:00

mkdir $JB;
cd $JB;
echo "N1" $JB
time ../main $JB
cd ../;
