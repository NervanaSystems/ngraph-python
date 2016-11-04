#!/bin/bash

rm *.pb
rm *.pb.txt
rm checkpoint
rm *ckpt*
rm data/events.*
rm data/t10k-labels-idx1-ubyte.gz data/train-labels-idx1-ubyte.gz \
   data/t10k-images-idx3-ubyte.gz data/train-images-idx3-ubyte.gz
