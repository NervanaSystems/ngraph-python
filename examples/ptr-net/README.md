## Pointer Networks
- Pointer Net deal with problems that each token in the output sequence is corresponding to positions in the input sequence. Sorting, convex hull and Traveling Salesman Problem fall under these kind of problems.
- Pointer Net uses attention as a pointer to select a member of the input sequence as the output.
- The repository shows ngraph implementation of Pointer Nets to approximate Planer Traveling Salesman Problem.
- Reference paper: https://arxiv.org/pdf/1506.03134.pdf

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
## Results
- Results training on tsp5.txt
>iteration = 1000, train loss = 2.32192277908
iteration = 2000, train loss = 0.46896520257
iteration = 3000, train loss = 0.217246204615
iteration = 4000, train loss = 0.246910408139
iteration = 5000, train loss = 0.104850962758
iteration = 6000, train loss = 0.0337684042752
iteration = 7000, train loss = 0.0199971366674
iteration = 8000, train loss = 0.0131694134325
iteration = 9000, train loss = 0.0102642579004
iteration = 10000, train loss = 0.0103495847434
iteration = 11000, train loss = 0.0107053946704
iteration = 12000, train loss = 0.00945739354938
iteration = 13000, train loss = 0.00575671391562
iteration = 14000, train loss = 0.0155959874392
iteration = 15000, train loss = 0.0326099693775
iteration = 16000, train loss = 0.0160880722106
iteration = 17000, train loss = 0.0100189754739
iteration = 18000, train loss = 0.00303012714721
iteration = 19000, train loss = 0.0117024183273
iteration = 20000, train loss = 0.00976016186178

- Results training on tsp10.txt
>iteration = 1000, train loss = 1.20918560028
iteration = 2000, train loss = 0.168795406818
iteration = 3000, train loss = 0.0863107442856
iteration = 4000, train loss = 0.0689085796475
iteration = 5000, train loss = 0.0350853949785
iteration = 6000, train loss = 0.0326116643846
iteration = 7000, train loss = 0.0325198583305
iteration = 8000, train loss = 0.0345276221633
iteration = 9000, train loss = 0.0271318014711
iteration = 10000, train loss = 0.0296661164612
iteration = 11000, train loss = 0.0212019626051
iteration = 12000, train loss = 0.0255753938109
iteration = 13000, train loss = 0.0725105628371
iteration = 14000, train loss = 0.0258009489626
iteration = 15000, train loss = 0.573940873146
iteration = 16000, train loss = 0.0236293375492
iteration = 17000, train loss = 0.0536655895412
iteration = 18000, train loss = 0.0177622660995
iteration = 19000, train loss = 0.00931741949171
iteration = 20000, train loss = 0.0163616128266

## To-dos
- [X] change decoder input to coordinates for teacher forcing  
- [X] ptr-net convergence issue (compare to [TF implementation](https://github.com/devsisters/pointer-network-tensorflow))
- [X] use LSTM as enc/dec rnn cell
- [X] function to calculate travel distance
- [ ] variable size sequence training
- [ ] inference code
