#!/bin/bash

for ((size=10; size<=60; size=size+10)); do
  for ((i=0; i<5; i=i+1)); do
    echo "-----";
        time python3 model_test.py -s -n $size -f $i >> test_bag_10_60.csv
  done
done
