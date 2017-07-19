## Pointer Networks
- Approximate Planer Traveling Salesman Problem using Pointer Networks
- reference paper: https://arxiv.org/pdf/1506.03134.pdf

## How to run
1. Download TSP dataset `tsp_5_train.zip` and `tsp_10_train.zip` from [here](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU)
2. Unzip `tsp_5_train.zip` and `tsp_10_train.zip`
```
$ unzip '*.zip'
```
3. Run training script
```
$ python ptr-net.py --train_file tsp5.txt --test_file tsp5_test.txt
```
## Results
- prelim results on tsp5.txt

>iteration = 4000, train loss = 1.59555494785
iteration = 8000, train loss = 1.11453413963
iteration = 12000, train loss = 0.834193050861
iteration = 16000, train loss = 0.63794618845
iteration = 20000, train loss = 0.486799806356
iteration = 24000, train loss = 0.381220757961
iteration = 28000, train loss = 0.352742105722
iteration = 32000, train loss = 0.354856789112
iteration = 36000, train loss = 0.37286776304d2
iteration = 40000, train loss = 0.333687841892

- prelim results on tsp10.txt


## To-dos
- [X] change decoder input to coordinates for teacher forcing  
- [ ] ptr-net inference code
- [ ] ptr-net convergence issue (compare to [TF implementation](https://github.com/devsisters/pointer-network-tensorflow))
- [X] use LSTM as enc/dec rnn cell
- [ ] script to calculate travel distance and compare with original paper
- [ ] visualize TSP  
- [ ] python/shell script to download TSP data
