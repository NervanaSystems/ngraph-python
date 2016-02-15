# CIFAR 10 Conv MGPU Benchmark Neon vs. MXNet

## Results Summary
### Time
|       | Titan + Titan + 980 | Titan + Titan | Titan   | 980     |
|-------|---------------------|---------------|---------|---------|
| Neon  | N/A                 | 43.526s       | 20.616s | 20.054s |
| MXNet | 34.269s             | 33.149s       | 38.159s | 40.120s |

### Memory
|       | Titan + Titan + 980 | Titan + Titan | Titan  | 980    |
|-------|---------------------|---------------|--------|--------|
| Neon  | N/A                 | 900 + 172 MB  | 903 MB | 869 MB |
| MXNet | 185 + 185 + 152 MB  | 185 + 190 MB  | 212 MB | 180 MB |

### Some Findings
* Neon faster than MXNet in single GPU (1.5X)
* MXNet are memory eficient compared to Neon (4X)
* Neon's multi gpu slower than single under this setting
* MXNet's speed 2GPU > 3GPU > 1GPU

### Settings
* A MXNet implementation of `neon/examples/cifar10_conv.py` equivalent
* 10 epochs
* 128 batch size
* Tested 3, 2, 1 GPUs

### Devices
* Device 0: Titan X (idle memory 138MiB at test)
* Device 1: Titan X (idle memory 29MiB at test)
* Device 2: GTX 980 (idle memory 15MiB at test)
* Tested on `max4`


## Neon
### Titan X + Titan X + GTX 980
```
time python cifar10_conv.py -b mgpu -m 3 -e 10 -z 128
(crashed)
Boost.Python.ArgumentError: Python argument types in
    pycuda._driver.memset_d32_async(NoneType, int, int, Stream)
did not match C++ signature:
    memset_d32_async(unsigned long long dest, unsigned int data, unsigned int size, pycudaboost::pytho
n::api::object stream=None)
```

### Titan X + Titan X
```
time python cifar10_conv.py -b mgpu -m 3 -e 10 -z 128
real    0m43.526s
user    0m40.118s
sys     0m3.797s
peak memory    (1038 - 138)MiB  + (201 - 29)MiB
```

### Titan X
```
$ time python cifar10_conv.py -b gpu -e 10 -z 128 -i 0
real    0m20.616s
user    0m17.219s
sys     0m3.303s
peak memory    (1041 - 138)MiB
```

### GTX 980
```
$ time python cifar10_conv.py -b gpu -e 10 -z 128 -i 2
real    0m20.054s
user    0m17.349s
sys     0m3.114s
peak memory    (884 - 15)MiB
```


## MXNet
### Titan X + Titan X + GTX 980
```
$ time python cifar10_mxnet.py --batch-size 128 --lr 0.1 --num-epoch 10 --gpus 0,1,2
real    0m34.269s
user    2m27.192s
sys     0m19.635s
peak memory    (323 - 138)MiB + (214 - 29)MiB + (167 - 15)MiB
```

### Titan X + Titan X
```
$ time python cifar10_mxnet.py --batch-size 128 --lr 0.1 --num-epoch 10 --gpus 0,1
real    0m33.149s
user    2m11.367s
sys     0m18.270s
peak memory    (328 - 138)MiB + (219 - 29)MiB
```

### Titan X
```
$ time python cifar10_mxnet.py --batch-size 128 --lr 0.1 --num-epoch 10 --gpus 0
real    0m38.159s
user    1m42.768s
sys     0m12.782s
peak memory    (350 - 138)MiB
```

### GTX 980
```
$ time python cifar10_mxnet.py --batch-size 128 --lr 0.1 --num-epoch 10 --gpus 2
real    0m40.120s
user    1m47.269s
sys     0m12.934s
peak memory    (195 - 15)MiB
```


