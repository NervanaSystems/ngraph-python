#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Sequential
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pooling, Activation
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax
from ngraph.frontends.neon.model import ResidualModule
from ngraph.frontends.neon import ax


# Helpers
def cifar10_mean_subtract(x):
    bgr_mean = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([113.9, 123.0, 125.3]))
    bgr_std = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([66.7, 62.1, 63.0]))
    return (x - bgr_mean) / bgr_std


def i1k_mean_subtract(x):
    bgr_mean = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([127.0, 119.0, 104.0]))
    return (x - bgr_mean)


# Number of residual modules at each node in imagenet(need a better name than node)
def num_i1k_resmods(size):
    # Num Residual modules
    if(size == 18):
        num_resnet_mods = [2, 2, 2, 2]
    elif(size == 34) or (size == 50):
        num_resnet_mods = [3, 4, 6, 3]
    elif(size == 101):
        num_resnet_mods = [3, 4, 23, 3]
    elif(size == 152):
        num_resnet_mods = [3, 8, 36, 3]
    else:
        raise ValueError("This should be caught before coming this far.")
    return num_resnet_mods


# Returns dict of convolution layer parameters
def conv_params(fil_size, num_fils, strides=1, batch_norm=True, activation=Rectlin()):
    return dict(filter_shape=(fil_size, fil_size, num_fils),
                filter_init=KaimingInit(),
                strides=strides,
                padding=(1 if fil_size > 1 else 0),
                batch_norm=batch_norm,
                activation=activation)


# Class for constructing the network as described in paper below
# Deep Residual Learning for Image Recognition
# http://arxiv.org/abs/1512.03385
class BuildResnet(Sequential):
    def __init__(self, net_type, resnet_size, bottleneck, num_resnet_mods, batch_norm=True):
        # For CIFAR10 dataset
        if (net_type in ('cifar10', 'cifar100')):
            # Number of Filters
            num_fils = [16, 32, 64]
            # Network Layers
            layers = [
                # Subtracting mean as suggested in paper
                Preprocess(functor=cifar10_mean_subtract),
                # First Conv with 3x3 and stride=1
                Convolution(**conv_params(3, 16, batch_norm=batch_norm))]
            first_resmod = True  # Indicates the first residual module
            # Loop 3 times for each filter.
            for fil in range(3):
                # Lay out n residual modules so that we have 2n layers.
                for resmods in range(num_resnet_mods):
                    if(resmods == 0):
                        if(first_resmod):
                            # Strides=1 and Convolution side path
                            main_path, side_path = self.get_mp_sp(num_fils[fil],
                                                                  net_type, direct=False,
                                                                  batch_norm=batch_norm)
                            layers.append(ResidualModule(main_path, side_path))
                            layers.append(Activation(Rectlin()))
                            first_resmod = False
                        else:
                            # Strides=2 and Convolution side path
                            main_path, side_path = self.get_mp_sp(num_fils[fil], net_type,
                                                                  direct=False, strides=2,
                                                                  batch_norm=batch_norm)
                            layers.append(ResidualModule(main_path, side_path))
                            layers.append(Activation(Rectlin()))
                    else:
                        # Strides=1 and direct connection
                        main_path, side_path = self.get_mp_sp(num_fils[fil], net_type,
                                                              batch_norm=batch_norm)
                        layers.append(ResidualModule(main_path, side_path))
                        layers.append(Activation(Rectlin()))
            # Do average pooling --> fully connected--> softmax.
            layers.append(Pooling([8, 8], pool_type='avg'))
            layers.append(Affine(axes=ax.Y, weight_init=KaimingInit(), batch_norm=batch_norm))
            layers.append(Activation(Softmax()))
        # For I1K dataset
        elif (net_type in ('i1k', 'i1k100')):
            # Number of Filters
            num_fils = [64, 128, 256, 512]
            # Number of residual modules we need to instantiate at each level
            num_resnet_mods = num_i1k_resmods(resnet_size)
            # Network layers
            layers = [
                # Subtracting mean
                Preprocess(functor=i1k_mean_subtract),
                # First Conv layer
                Convolution((7, 7, 64), strides=2, padding=3,
                            batch_norm=batch_norm, activation=Rectlin(),
                            filter_init=KaimingInit()),
                # Max Pooling
                Pooling([3, 3], strides=2, pool_type='max', padding=1)]
            first_resmod = True  # Indicates the first residual module for which strides are 1
            # Loop 4 times for each filter
            for fil in range(4):
                # Lay out residual modules as in num_resnet_mods list
                for resmods in range(num_resnet_mods[fil]):
                    if(resmods == 0):
                        if(first_resmod):
                            # Strides=1 and Convolution Side path
                            main_path, side_path = self.get_mp_sp(num_fils[fil],
                                                                  net_type,
                                                                  direct=False,
                                                                  bottleneck=bottleneck,
                                                                  batch_norm=batch_norm)
                            layers.append(ResidualModule(main_path, side_path))
                            layers.append(Activation(Rectlin()))
                            first_resmod = False
                        else:
                            # Strides=2 and Convolution side path
                            main_path, side_path = self.get_mp_sp(num_fils[fil],
                                                                  net_type,
                                                                  direct=False,
                                                                  bottleneck=bottleneck,
                                                                  strides=2,
                                                                  batch_norm=batch_norm)
                            layers.append(ResidualModule(main_path, side_path))
                            layers.append(Activation(Rectlin()))
                    else:
                        # Strides=1 and direct connection
                        main_path, side_path = self.get_mp_sp(num_fils[fil],
                                                              net_type,
                                                              bottleneck=bottleneck,
                                                              batch_norm=batch_norm)
                        layers.append(ResidualModule(main_path, side_path))
                        layers.append(Activation(Rectlin()))
            # Do average pooling --> fully connected--> softmax.
            layers.append(Pooling([7, 7], pool_type='avg'))
            layers.append(Affine(axes=ax.Y, weight_init=KaimingInit(),
                                 batch_norm=batch_norm))
            layers.append(Activation(Softmax()))
        else:
            raise NameError("Incorrect dataset. Should be --dataset cifar10 or --dataset i1k")
        super(BuildResnet, self).__init__(layers=layers)

    # This methods takes dataset type and returns main path and side path
    def get_mp_sp(self, num_fils, net_type, direct=True, bottleneck=False, strides=1,
                  batch_norm=True):
        if(net_type in ('cifar10', 'cifar100')):
            # Mainpath for CIFAR10 is fixed
            main_path = Sequential([
                Convolution(**conv_params(3, num_fils, strides=strides, batch_norm=batch_norm)),
                Convolution(**conv_params(3, num_fils, activation=None, batch_norm=batch_norm))])
            # Side Path
            if(direct):
                side_path = None
            else:
                side_path = Convolution(**conv_params(1, num_fils,
                                                      strides=strides, activation=None,
                                                      batch_norm=batch_norm))
        elif(net_type in ('i1k', 'i1k100')):
            # Mainpath for i1k is depends if bottleneck is enabled or not
            if(bottleneck):
                main_path = Sequential([
                    Convolution(**conv_params(1, num_fils, strides=strides,
                                              batch_norm=batch_norm)),
                    Convolution(**conv_params(3, num_fils, batch_norm=batch_norm)),
                    Convolution(**conv_params(1, num_fils * 4, activation=None,
                                              batch_norm=batch_norm))])
            else:
                main_path = Sequential([
                    Convolution(**conv_params(3, num_fils, strides=strides,
                                              batch_norm=batch_norm)),
                    Convolution(**conv_params(3, num_fils, activation=None,
                                              batch_norm=batch_norm))])
            # Side Path
            if(direct):
                side_path = None
            else:
                if(bottleneck):
                    side_path = Convolution(**conv_params(1, num_fils * 4,
                                                          strides=strides, activation=None,
                                                          batch_norm=batch_norm))
                else:
                    side_path = Convolution(**conv_params(1, num_fils,
                                                          strides=strides, activation=None,
                                                          batch_norm=batch_norm))
        else:
            raise NameError("Incorrect dataset. Should be --dataset cifar10 or --dataset i1k")
        return main_path, side_path
