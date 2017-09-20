# Deep Residual Networks
This example demonstrates training a deep residual network as first described in [He et. al.][msra1]. It can handle CIFAR10 and Imagenet datasets
## Files
data.py - Loads CIFAR10 or imagenet dataset and creates aeon objects
resnet.py - Defines object for Residual network
train_resnet.py - Trains the resnet depending on dataset and size choosen
## Usage
```python train_resnet.py -b <cpu,gpu> --size <20,56> -t 64000 -z <64,128>```
## Citation
```
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
```

[msra1]: <http://arxiv.org/abs/1512.03385>`