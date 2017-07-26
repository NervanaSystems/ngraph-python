## Pointer Networks
- Pointer Net deal with problems that each token in the output sequence is corresponding to positions in the input sequence. Sorting, convex hull and Traveling Salesman Problem fall under this kind of problems.
- Pointer Net uses attention as a pointer to select a member of the input sequence as the output.
- The repository shows ngraph implementation of Pointer Net to approximate Planer Traveling Salesman Problem.
- reference paper: https://arxiv.org/pdf/1506.03134.pdf

## How to run
1. Download TSP dataset `tsp_5_train.zip` and `tsp_10_train.zip` from [here](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU)
2. Unzip `tsp_5_train.zip` and `tsp_10_train.zip`
```
$ unzip '*.zip'
```
3. Run training script
```
$ python ptr-net.py --train_file tsp5.txt --test_file tsp5_test.txt -b gpu
```
## Results
- prelim results training on tsp5.txt

>iteration = 2000, train loss = 1.57829642296
iteration = 4000, train loss = 1.43903386593
iteration = 6000, train loss = 1.02534890175
iteration = 8000, train loss = 0.931654274464
iteration = 10000, train loss = 0.785848796368
iteration = 12000, train loss = 0.824071228504
iteration = 14000, train loss = 0.760764241219
iteration = 16000, train loss = 0.723568558693
iteration = 18000, train loss = 0.612337231636
iteration = 20000, train loss = 0.533858776093
iteration = 22000, train loss = 0.520471572876
iteration = 24000, train loss = 0.419541060925
iteration = 26000, train loss = 0.433132320642
iteration = 28000, train loss = 0.470048248768
iteration = 30000, train loss = 0.489885240793
iteration = 32000, train loss = 0.509543120861
iteration = 34000, train loss = 0.397153705359
iteration = 36000, train loss = 0.41118773818
iteration = 38000, train loss = 0.330927848816
iteration = 40000, train loss = 0.34331125021
iteration = 42000, train loss = 0.376413494349
iteration = 44000, train loss = 0.388570189476
iteration = 46000, train loss = 0.327951878309
iteration = 48000, train loss = 0.379947990179
iteration = 50000, train loss = 0.317634046078

- prelim results training on tsp10.txt
>iteration = 2000, train loss = 2.85296750069
iteration = 4000, train loss = 2.84532856941
iteration = 6000, train loss = 2.66449666023
iteration = 8000, train loss = 2.56625890732
iteration = 10000, train loss = 2.19565153122
iteration = 12000, train loss = 1.94617462158
iteration = 14000, train loss = 2.06753349304
iteration = 16000, train loss = 1.78943669796
iteration = 18000, train loss = 1.67663383484
iteration = 20000, train loss = 1.63309168816
iteration = 22000, train loss = 1.68619692326
iteration = 24000, train loss = 1.50074219704
iteration = 26000, train loss = 1.50416409969
iteration = 28000, train loss = 1.4880001545
iteration = 30000, train loss = 1.43878722191
iteration = 32000, train loss = 1.34956288338
iteration = 34000, train loss = 1.36554932594
iteration = 36000, train loss = 1.36947906017
iteration = 38000, train loss = 1.34887468815
iteration = 40000, train loss = 1.38094007969
iteration = 42000, train loss = 1.22148537636
iteration = 44000, train loss = 1.33542692661
iteration = 46000, train loss = 1.23805546761
iteration = 48000, train loss = 1.19426417351
iteration = 50000, train loss = 1.30143654346

## To-dos
- [X] change decoder input to coordinates for teacher forcing  
- [ ] ptr-net convergence issue (compare to [TF implementation](https://github.com/devsisters/pointer-network-tensorflow))
- [X] use LSTM as enc/dec rnn cell
- [ ] function to calculate travel distance
- [ ] ptr-net inference code
- [ ] visualize TSP  
