#!/usr/bin/env bash

for period in 10 20 30 40 
do
    for lag in 5 10 15
    do
        for noise in 1 2 3
        do
        python exp.py $lag $noise $period
        done
    done
done




