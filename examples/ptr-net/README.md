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
$ python ptr-net.py --train_file tsp5.txt --test_file tsp5_test.txt
```
## Results
- prelim results training on tsp5.txt

>iteration = 4000, train loss = 1.59592020512
iteration = 8000, train loss = 1.06542396545
iteration = 12000, train loss = 0.864562988281
iteration = 16000, train loss = 0.696883380413
iteration = 20000, train loss = 0.496299445629
iteration = 24000, train loss = 0.43381780386
iteration = 28000, train loss = 0.372637659311
iteration = 32000, train loss = 0.33082139492
iteration = 36000, train loss = 0.349985271692
iteration = 40000, train loss = 0.341779232025
iteration = 44000, train loss = 0.33200699091
iteration = 48000, train loss = 0.352579593658
iteration = 52000, train loss = 0.286086380482
iteration = 56000, train loss = 0.284330248833
iteration = 60000, train loss = 0.331149876118
iteration = 64000, train loss = 0.265735834837
iteration = 68000, train loss = 0.245906025171
iteration = 72000, train loss = 0.283038437366
iteration = 76000, train loss = 0.296033680439
iteration = 80000, train loss = 0.234311491251
iteration = 84000, train loss = 0.280319094658
iteration = 88000, train loss = 0.241472199559
iteration = 92000, train loss = 0.263256013393
iteration = 96000, train loss = 0.275329083204
iteration = 100000, train loss = 0.235205605626

- prelim results training on tsp10.txt
>iteration = 4000, train loss = 2.84994125366
iteration = 8000, train loss = 2.63330841064
iteration = 12000, train loss = 2.16516757011
iteration = 16000, train loss = 1.8092457056
iteration = 20000, train loss = 1.46874678135
iteration = 24000, train loss = 1.30825471878
iteration = 28000, train loss = 1.18671369553
iteration = 32000, train loss = 1.0799434185
iteration = 36000, train loss = 1.01472973824
iteration = 40000, train loss = 0.994896769524
iteration = 44000, train loss = 0.957517623901
iteration = 48000, train loss = 0.952610969543
iteration = 52000, train loss = 0.927261054516
iteration = 56000, train loss = 0.892623126507
iteration = 60000, train loss = 0.782183647156
iteration = 64000, train loss = 0.802037835121
iteration = 68000, train loss = 0.839537620544
iteration = 72000, train loss = 0.769252955914
iteration = 76000, train loss = 0.874385476112
iteration = 80000, train loss = 0.818211376667
iteration = 84000, train loss = 0.815545678139
iteration = 88000, train loss = 0.720605194569
iteration = 92000, train loss = 0.729014635086
iteration = 96000, train loss = 0.736841499805
iteration = 100000, train loss = 0.716338455677

## To-dos
- [X] change decoder input to coordinates for teacher forcing  
- [ ] ptr-net inference code
- [ ] ptr-net convergence issue (compare to [TF implementation](https://github.com/devsisters/pointer-network-tensorflow))
- [X] use LSTM as enc/dec rnn cell
- [ ] script to calculate travel distance and compare with original paper
- [ ] visualize TSP  
- [ ] python/shell script to download TSP data
