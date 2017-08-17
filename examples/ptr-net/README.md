## Pointer Networks
- Pointer Networks (Ptr-Nets) deal with sequence to sequence problems in which each token in the output sequence corresponds to a position in the input sequence. Sorting, convex hull and Traveling Salesman Problem are examples of applicable problems.
- Ptr-Nets use an attention mechanism to select a member of the input sequence at each step of the output. This example demonstrates an application of Ptr-Nets to approximate Planar Traveling Salesman Problem.

## How to run
```
$ python ptr-net.py -b gpu
```
