"""
implemnts equivalent neon/examples/cifar10_conv.py model for mxnet
"""

import find_mxnet
import mxnet as mx

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0),
                act_type="relu"):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter,
                                 kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act

def get_symbol(num_classes = 10):
    data = mx.symbol.Variable(name="data")

    # conv, bn, relu
    conv1 = ConvFactory(data=data, num_filter=16, kernel=(5,5), act_type="relu")
    # pool
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(2,2))
    # conv, bn, relu
    conv2 = ConvFactory(data=pool1, num_filter=32, kernel=(5,5), act_type="relu")
    # pool
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(2,2))
    # flatten
    flatten = mx.symbol.Flatten(data=pool2)
    # full, bn, relu
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    fc1_bn =  mx.symbol.BatchNorm(data=fc1)
    fc1_act = mx.symbol.Activation(data=conv2, act_type="relu")
    # softmax
    fc2 = mx.symbol.FullyConnected(data=fc1_bn, num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    return softmax