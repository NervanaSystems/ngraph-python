## Pointer Networks
- Pointer Networks (Ptr-Nets) deal with sequence to sequence problems in which each token in the output sequence corresponds to a position in the input sequence. Sorting, convex hull and Traveling Salesman Problem are examples of applicable problems.
- Ptr-Nets use an attention mechanism to select a member of the input sequence at each step of the output. This example demonstrates an application of Ptr-Nets to approximate Planar Traveling Salesman Problem.

## How to run
1. Download TSP dataset `tsp_5_train.zip` and `tsp_10_train.zip` from [here](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU)
2. Unzip `tsp_5_train.zip` and `tsp_10_train.zip`
```
$ unzip '*.zip'
```
3. Run training script
```
$ python ptr-net.py --train_file tsp10.txt --test_file tsp10_test.txt -b gpu
```
