#!/bin/bash

cd $3
cd S1

for ((i = $1; i <=$2; i++));
do
        mkdir $i
        cd $i
        echo "Exp" $i $3
        time ../../main $4
        cd ..;
done
