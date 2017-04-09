# Implementation of Deep Speech 2 in Nervana Graph

A barebones implementation of Baidu SVAIL's [deep speech 2](https://arxiv.org/abs/1512.02595) model in Nervana Graph that's mostly self-contained with the exception of the CTC cost function which is provided by wrapping Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc).
  
## Getting Started
1. Within an ngraph virtualenv, run ```pip install cffi```.
2. Build Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc) and set the environment variable WARP_CTC_PATH to point to the location of the built library (typically the location of the ``build`` folder containing ``libwarpctc.so``).
3. To train a model with the default parameters, run 

```
python train.py -t <num_iterations> -z <batch_size> [-b <backend>] 
```
