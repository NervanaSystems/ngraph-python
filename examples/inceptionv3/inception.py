#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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

"""
    This file contains the building blocks of the inceptionv3 network
    Network architecture follows from https://arxiv.org/pdf/1512.00567.pdf
    and https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
"""
from ngraph.frontends.neon import UniformInit, BatchNorm
from ngraph.frontends.neon import Convolution, Pooling, Sequential, Parallel
from ngraph.frontends.neon import Rectlin, Softmax, Dropout


def conv_params(filter_shape, strides=1, batch_norm=BatchNorm(rho=0.999), activation=Rectlin(),
                bias_init=None,
                filter_init=UniformInit(), padding=0):
    return dict(filter_shape=filter_shape,
                strides=strides,
                padding=padding,
                batch_norm=batch_norm,
                activation=activation,
                filter_init=filter_init,
                bias_init=bias_init)


class Inceptionv3_b1(Parallel):

    def __init__(self, branch_units=[(64,), (48, 64), (64, 96, 96), (64,)], name=None):
        """
        First inception block with four branches, concatenated in the end
            1. 1x1 conv
            2. 1x1 conv, 5x5 conv
            3. 1x1 conv, 3x3conv, 3x3 conv
            4. 3x3 pool, 1x1 conv
        Convolution(H, W, K) : height, width, number of filters
        Mixed_5b, Mixed_5c, Mixed_5d layers
        """
        (p1, p2, p3, p4) = branch_units

        branch1 = Convolution(name=name + '_br1_1x1conv', **
                              conv_params(filter_shape=(1, 1, p1[0])))

        branch2 = Sequential([Convolution(name=name + '_br2_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p2[0]))),
                              Convolution(name=name + '_br2_5x5conv',
                                          **conv_params(filter_shape=(5, 5, p2[1]), padding=2))])

        branch3 = Sequential([Convolution(name=name + '_br3_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p3[0]))),
                              Convolution(name=name + '_br3_3x3conv1',
                                          **conv_params(filter_shape=(3, 3, p3[1]), padding=1)),
                              Convolution(name=name + '_br3_3x3conv2',
                                          **conv_params(filter_shape=(3, 3, p3[2]), padding=1))])

        branch4 = Sequential([Pooling(name=name + '_br4_avgpool', pool_shape=(3, 3),
                                      padding=1, strides=1, pool_type="avg"),
                              Convolution(name=name + '_br4_conv1x1',
                                          **conv_params(filter_shape=(1, 1, p4[0])))])

        branches = [branch1, branch2, branch3, branch4]
        super(Inceptionv3_b1, self).__init__(name=name, branches=branches, mode='concat')


class Inceptionv3_b2(Parallel):

    def __init__(self, branch_units=[(384,), (64, 96, 96)], name=None):
        """
        Second inception block with three branches, concatenated in the end
            1. 3x3 conv (stride = 2, valid)
            2. 1x1 conv, 3x3 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride = 2, valid)
        Convolution(H, W, K) : height, width, number of filters
        Mixed_6a layer
        """
        (p1, p2) = branch_units

        branch1 = Convolution(name=name + '_br1_3x3conv',
                              **conv_params(filter_shape=(3, 3, p1[0]),
                                            strides=2, padding=0))

        branch2 = Sequential([Convolution(name=name + '_br2_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p2[0]))),
                              Convolution(name=name + '_br2_3x3conv1',
                                          **conv_params(filter_shape=(3, 3, p2[1]), padding=1)),
                              Convolution(name=name + '_br2_3x3conv2',
                                          **conv_params(filter_shape=(3, 3, p2[2]),
                                                        strides=2, padding=0))])

        branch3 = Pooling(pool_shape=(3, 3), padding=0, strides=2, pool_type="max",
                          name=name + '_br3_maxpool')

        branches = [branch1, branch2, branch3]
        super(Inceptionv3_b2, self).__init__(name=name, branches=branches, mode='concat')


class Inceptionv3_b3(Parallel):

    def __init__(self, branch_units=[(192), (160, 160, 192),
                                     (160, 160, 160, 160, 192), (192,)], name=None):
        """
        Third inception block with four branches, concatenated in the end
            1. 1x1 conv
            2. 1x1 conv, 1x7 conv, 7x1 conv
            3. 1x1 conv, 7x1 conv, 1x7 conv, 7x1 conv, 1x7 conv
            4. 3x3 pool, 1x1 conv
            Convolution(H, W, K) : height, width, number of filters
        Mixed_6b, Mixed_6c, Mixed_6c, Mixed_6d, Mixed_6e layers
        """
        (p1, p2, p3, p4) = branch_units
        branch1 = Convolution(name=name + '_br1_1x1conv',
                              **conv_params(filter_shape=(1, 1, p1[0])))

        branch2 = Sequential([Convolution(name=name + '_br2_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p2[0]))),
                              Convolution(name=name + '_br2_1x7conv',
                                          **conv_params(filter_shape=(1, 7, p2[1]),
                                                        padding={'H': 0,
                                                                 'W': 3,
                                                                 'D': 0})),
                              Convolution(name=name + '_br27x1conv',
                                          **conv_params(filter_shape=(7, 1, p2[2]),
                                                        padding={'H': 3,
                                                                 'W': 0,
                                                                 'D': 0}))])

        branch3 = Sequential([Convolution(name=name + '_br3_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p3[0]))),
                              Convolution(name=name + '_br3_7x1conv1',
                                          **conv_params(filter_shape=(7, 1, p3[1]),
                                                        padding={'H': 3, 'W': 0, 'D': 0})),
                              Convolution(name=name + '_br3_1x7conv1',
                                          **conv_params(filter_shape=(1, 7, p3[2]),
                                                        padding={'H': 0, 'W': 3, 'D': 0})),
                              Convolution(name=name + '_br3_7x1conv2',
                                          **conv_params(filter_shape=(7, 1, p3[3]),
                                                        padding={'H': 3, 'W': 0, 'D': 0})),
                              Convolution(name=name + '_br3_1x7conv2',
                                          **conv_params(filter_shape=(1, 7, p3[4]),
                                                        padding={'H': 0, 'W': 3, 'D': 0}))])

        branch4 = Sequential([Pooling(name=name + '_br4_avgpool',
                                      pool_shape=(3, 3), padding=1, strides=1, pool_type="avg"),
                              Convolution(name=name + '_br4_1x1conv',
                                          **conv_params(filter_shape=(1, 1, p4[0])))])
        branches = [branch1, branch2, branch3, branch4]
        super(Inceptionv3_b3, self).__init__(name=name, branches=branches, mode='concat')


class Inceptionv3_b4(Parallel):

    def __init__(self, branch_units=[(192, 320), (192, 192, 192, 192)], name=None):
        """
        Fourth inception block with three branches, concatenated in the end
            1. 1x1 conv, 3x3 conv (stride=2, valid)
            2. 1x1 conv, 1x7 conv, 7x1 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride=2, valid)
            Convolution(H, W, K) : height, width, number of filters
        Mixed_7a layer
        """
        (p1, p2) = branch_units
        branch1 = Sequential([Convolution(name=name + '_br1_conv1x1',
                                          **conv_params(filter_shape=(1, 1, p1[0]))),
                              Convolution(name=name + '_br1_conv3x3',
                                          **conv_params(filter_shape=(3, 3, p1[1]),
                                                        strides=2, padding=0))])

        branch2 = Sequential([Convolution(name=name + '_br2_conv1x1',
                                          **conv_params(filter_shape=(1, 1, p2[0]))),
                              Convolution(name=name + '_br2_conv1x7',
                                          **conv_params(filter_shape=(1, 7, p2[1]),
                                                        padding={'H': 0, 'W': 3, 'D': 0})),
                              Convolution(name=name + '_br2_conv7x1',
                                          **conv_params(filter_shape=(7, 1, p2[2]),
                                                        padding={'H': 3, 'W': 0, 'D': 0})),
                              Convolution(name=name + '_br2_conv3x3',
                                          **conv_params(filter_shape=(3, 3, p2[3]),
                                                        strides=2, padding=0))])

        branch3 = Pooling(name=name + '_br3_maxpool', pool_shape=(3, 3),
                          padding=0, strides=2, pool_type="max")
        branches = [branch1, branch2, branch3]
        super(Inceptionv3_b4, self).__init__(name=name, branches=branches, mode='concat')


class Inceptionv3_b5(Parallel):

    def __init__(self, branch_units=[(320,), (384, 384, 384),
                                     (448, 384, 384, 384), (192,)], name=None):
        """
        Fifth inception block with four branches, concatenated in the end
            1. 1x1 conv
            2. 1x1 conv, followed by two sub-branches [1x3 conv, 3x1 conv]
            3. 1x1 conv, 3x3 conv, followed by two sub-branches [1x3 conv, 3x1 conv]
            4. 3x3 pool, 1x1 conv
            Convolution(H, W, K) : height, width, number of filters
        Mixed_7b, Mixed_7c layers
        """
        (p1, p2, p3, p4) = branch_units

        # Branch 1
        branch1 = Convolution(name=name + '_br1_conv1x1', **
                              conv_params(filter_shape=(1, 1, p1[0])))

        # Branch 2
        branch2a1_params = conv_params(filter_shape=(1, 3, p2[1]),
                                       padding={'H': 0, 'W': 1, 'D': 0})
        branch2a2_params = conv_params(filter_shape=(3, 1, p2[2]),
                                       padding={'H': 1, 'W': 0, 'D': 0})
        # This is the sub-branch with two parallel branches of 1x3 and 3x1
        branch2a = Parallel([Convolution(name=name + '_br2a_conv1x3', **branch2a1_params),
                             Convolution(name=name + '_br2a_conv3x1', **branch2a2_params)])
        branch2 = Sequential([Convolution(name=name + '_br2_conv1x1', **
                                          conv_params(filter_shape=(1, 1, p2[0]))), branch2a])

        # Branch 3
        branch3a1_params = conv_params(filter_shape=(1, 3, p3[2]),
                                       padding={'H': 0, 'W': 1, 'D': 0})
        branch3a2_params = conv_params(filter_shape=(3, 1, p3[3]),
                                       padding={'H': 1, 'W': 0, 'D': 0})
        branch3a = Parallel([Convolution(name=name + '_br3_conv1x3', **branch3a1_params),
                             Convolution(name=name + '_br3_conv3x1', **branch3a2_params)])
        branch3 = Sequential([Convolution(name=name + '_br3_conv1x1',
                                          **conv_params(filter_shape=(1, 1, p3[0]))),
                              Convolution(name=name + '_br3_conv3x3',
                                          **conv_params(filter_shape=(3, 3, p3[1]),
                                                        padding=1)),
                              branch3a])

        # Branch 4
        branch4 = Sequential([Pooling(name=name + '_br4_avgpool',
                                      pool_shape=(3, 3),
                                      padding=1,
                                      strides=1,
                                      pool_type="avg"),
                              Convolution(name=name + '_br4_conv1x1',
                                          **conv_params(filter_shape=(1, 1, p4[0])))])

        # Combine branches
        branches = [branch1, branch2, branch3, branch4]
        super(Inceptionv3_b5, self).__init__(name=name, branches=branches, mode='concat')


class Inception(object):

    def __init__(self, mini=False):
        """
        Builds Inception model based on:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
        """
        # Input size is 299 x 299 x 3
        if mini:
            """
            This is the mini model with reduced number of filters in each layer
            """
            # Root branch of the tree
            seq1 = Sequential([Convolution(name='conv_1a_3x3',
                                           **conv_params(filter_shape=(3, 3, 32),
                                                         padding=0, strides=2)),
                               # conv2d_1a_3x3
                               Convolution(name='conv_2a_3x3', **
                                           conv_params(filter_shape=(3, 3, 16), padding=0)),
                               # conv2d_2a_3x3
                               Convolution(name='conv_2b_3x3', **
                                           conv_params(filter_shape=(3, 3, 16), padding=1)),
                               # conv2d_2b_3x3
                               Pooling(name='pool_1_3x3', pool_shape=(3, 3), padding=0,
                                       strides=2, pool_type='max'),  # maxpool_3a_3x3
                               Convolution(name='conv_3b_1x1', **
                                           conv_params(filter_shape=(1, 1, 16))),
                               # conv2d_3b_1x1
                               Convolution(name='conv_4a_3x3', **
                                           conv_params(filter_shape=(3, 3, 32), padding=1)),
                               # conv2d_4a_3x3
                               Pooling(name='pool_2_3x3', pool_shape=(3, 3), padding=0,
                                       strides=2, pool_type='max'),  # maxpool_5a_3x3
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32),
                                               (32, )], name='mixed_5b'),
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32),
                                               (32, )], name='mixed_5c'),
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32),
                                               (32, )], name=' mixed_5d'),
                               Inceptionv3_b2([(32,), (32, 32, 32)], name=' mixed_6a'),
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)], name='mixed_6b'),
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)], name='mixed_6c'),
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)], name='mixed_6d'),
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)], name='mixed_6e')])

            # Branch of main classifier
            seq2 = Sequential([Inceptionv3_b4([(32, 32), (32, 32, 32, 32)], name='mixed_7a'),
                               Inceptionv3_b5([(32,), (32, 32, 32),
                                               (32, 32, 32, 32), (32,)], name='mixed_7b'),
                               Inceptionv3_b5([(32,), (32, 32, 32),
                                               (32, 32, 32, 32), (32,)], name='mixed_7c'),
                               Pooling(pool_shape=(8, 8), padding=0, strides=2, pool_type='avg'),
                               # Last Avg Pool
                               Dropout(keep=0.8),
                               Convolution(name='main_final_conv1x1',
                                           **conv_params(filter_shape=(1, 1, 1000),
                                                         activation=Softmax(),
                                                         batch_norm=False))])

            # Auxiliary classifier
            seq_aux = Sequential([Pooling(pool_shape=(5, 5),
                                          padding=0, strides=3, pool_type='avg'),
                                  Convolution(name='aux_conv1x1_v1',
                                              **conv_params(filter_shape=(1, 1, 32))),
                                  Convolution(name='aux_conv5x5',
                                              **conv_params(filter_shape=(5, 5, 32))),
                                  Convolution(name='aux_conv1x1_v2',
                                              **conv_params(filter_shape=(1, 1, 1000),
                                                            activation=Softmax(),
                                                            batch_norm=False))])

        else:
            # Root branch of the tree
            seq1 = Sequential([Convolution(name='conv_1a_3x3',
                                           **conv_params(filter_shape=(3, 3, 32),
                                                         padding=0, strides=2)),
                               # conv2d_1a_3x3
                               Convolution(name='conv_2a_3x3', **
                                           conv_params(filter_shape=(3, 3, 32), padding=0)),
                               # conv2d_2a_3x3
                               Convolution(name='conv_2b_3x3', **
                                           conv_params(filter_shape=(3, 3, 64), padding=1)),
                               # conv2d_2b_3x3
                               Pooling(name='pool_1_3x3', pool_shape=(3, 3), padding=0,
                                       strides=2, pool_type='max'),  # maxpool_3a_3x3
                               Convolution(name='conv_3b_1x1', **
                                           conv_params(filter_shape=(1, 1, 80))),
                               # conv2d_3b_1x1
                               Convolution(name='conv_4a_3x3', **
                                           conv_params(filter_shape=(3, 3, 192), padding=1)),
                               # conv2d_4a_3x3
                               Pooling(name='pool_2_3x3', pool_shape=(3, 3), padding=0,
                                       strides=2, pool_type='max'),  # maxpool_5a_3x3
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96),
                                               (32, )], name='mixed_5b'),
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96),
                                               (64, )], name='mixed_5c'),
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96),
                                               (64, )], name=' mixed_5d'),
                               Inceptionv3_b2([(384,), (64, 96, 96)],
                                              name=' mixed_6a'),
                               Inceptionv3_b3([(192,), (128, 128, 192),
                                               (128, 128, 128, 128, 192), (192,)],
                                              name='mixed_6b'),
                               Inceptionv3_b3([(192,), (160, 160, 192),
                                               (160, 160, 160, 160, 192), (192,)],
                                              name='mixed_6c'),
                               Inceptionv3_b3([(192,), (160, 160, 192),
                                               (160, 160, 160, 160, 192), (192,)],
                                              name='mixed_6d'),
                               Inceptionv3_b3([(192,), (192, 192, 192),
                                               (192, 192, 192, 192, 192), (192,)],
                                              name='mixed_6e')])

            # Branch of main classifier
            seq2 = [Inceptionv3_b4([(192, 320), (192, 192, 192, 192)], name='mixed_7a'),
                    Inceptionv3_b5([(320,), (384, 384, 384),
                                    (448, 384, 384, 384), (192,)], name='mixed_7b'),
                    Inceptionv3_b5([(320,), (384, 384, 384),
                                    (448, 384, 384, 384), (192,)], name='mixed_7c'),
                    Pooling(pool_shape=(8, 8), padding=0, strides=2, pool_type='avg'),
                    # Last Avg Pool
                    Dropout(keep=0.8),
                    Convolution(name='main_final_conv1x1',
                                **conv_params(filter_shape=(1, 1, 1000),
                                              activation=Softmax(),
                                              batch_norm=False))]
            seq2 = Sequential(seq2)

            # Auxiliary classifier
            my_seq = [Pooling(pool_shape=(5, 5), padding=0, strides=3, pool_type='avg'),
                      Convolution(name='aux_conv1x1_v1', **conv_params(filter_shape=(1, 1, 128))),
                      Convolution(name='aux_conv5x5', **conv_params(filter_shape=(5, 5, 768))),
                      Convolution(name='aux_conv1x1_v2',
                                  **conv_params(filter_shape=(1, 1, 1000),
                                                activation=Softmax(),
                                                batch_norm=False))]
            seq_aux = Sequential(my_seq)

        self.seq1 = seq1
        self.seq2 = seq2
        self.seq_aux = seq_aux
