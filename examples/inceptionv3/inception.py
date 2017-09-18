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
from ngraph.frontends.neon import XavierInit, UniformInit
from ngraph.frontends.neon import Convolution, Pool2D, Sequential, Parallel
from ngraph.frontends.neon import Rectlin, Softmax, Dropout, Explin


def conv_params(filt_params, strides=1, batch_norm=True, activation=Rectlin(),
                bias_init=UniformInit(low=-0.3, high=0.3),
                filter_init=UniformInit(low=-0.3, high=0.3), padding=0):
    return dict(fshape=filt_params,
                strides=strides,
                padding=padding,
                batch_norm=batch_norm,
                activation=activation,
                filter_init=filter_init,
                bias_init=bias_init)


class Inceptionv3_b1(object):

    def __init__(self, branch_units=[(64,), (48, 64), (64, 96, 96), (64,)]):

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

        branch1 = Convolution(**conv_params(filt_params=(1, 1, p1[0])))
        branch2 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p2[0]))),
                              Convolution(**conv_params(filt_params=(5, 5, p2[1]),
                                                        padding=2))])
        branch3 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p3[0]))),
                              Convolution(**conv_params(filt_params=(3, 3, p3[1]),
                                                        padding=1)),
                              Convolution(**conv_params(filt_params=(3, 3, p3[2]),
                                                        padding=1))])
        branch4 = Sequential([Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                              Convolution(**conv_params(filt_params=(1, 1, p4[0])))])

        self.model = Parallel([branch1, branch2, branch3, branch4])

    def __call__(self, in_obj):
        return self.model(in_obj)


class Inceptionv3_b2(object):

    def __init__(self, branch_units=[(384,), (64, 96, 96)]):

        """
        Second inception block with three branches, concatenated in the end
            1. 3x3 conv (stride = 2, valid)
            2. 1x1 conv, 3x3 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride = 2, valid)
        Convolution(H, W, K) : height, width, number of filters
        Mixed_6a layer
        """
        (p1, p2) = branch_units

        branch1 = Convolution(**conv_params(filt_params=(3, 3, p1[0]),
                                            strides=2, padding=0))

        branch2 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p2[0]))),
                              Convolution(**conv_params(filt_params=(3, 3, p2[1]),
                                                        padding=1)),
                              Convolution(**conv_params(filt_params=(3, 3, p2[2]),
                                                        strides=2, padding=0))])

        branch3 = Pool2D(fshape=3, padding=0, strides=2, op="max")

        self.model = Parallel([branch1, branch2, branch3])

    def __call__(self, in_obj):
        return self.model(in_obj)


class Inceptionv3_b3(object):

    def __init__(self, branch_units=[(192), (160, 160, 192), (160, 160, 160, 160, 192), (192,)]):

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
        branch1 = Convolution(**conv_params(filt_params=(1, 1, p1[0])))
        branch2 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p2[0]))),
                              Convolution(**conv_params(filt_params=(1, 7, p2[1]),
                                                        padding={'pad_h': 0,
                                                                 'pad_w': 3,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(7, 1, p2[2]),
                                                        padding={'pad_h': 3,
                                                                 'pad_w': 0,
                                                                 'pad_d': 0}))])
        branch3 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p3[0]))),
                              Convolution(**conv_params(filt_params=(7, 1, p3[1]),
                                                        padding={'pad_h': 3,
                                                                 'pad_w': 0,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(1, 7, p3[2]),
                                                        padding={'pad_h': 0,
                                                                 'pad_w': 3,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(7, 1, p3[3]),
                                                        padding={'pad_h': 3,
                                                                 'pad_w': 0,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(1, 7, p3[4]),
                                                        padding={'pad_h': 0,
                                                                 'pad_w': 3,
                                                                 'pad_d': 0}))])
        branch4 = Sequential([Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                              Convolution(**conv_params(filt_params=(1, 1, p4[0])))])

        self.model = Parallel([branch1, branch2, branch3, branch4])

    def __call__(self, in_obj):
        return self.model(in_obj)


class Inceptionv3_b4(object):

    def __init__(self, branch_units=[(192, 320), (192, 192, 192, 192)]):

        """
        Fourth inception block with three branches, concatenated in the end
            1. 1x1 conv, 3x3 conv (stride=2, valid)
            2. 1x1 conv, 1x7 conv, 7x1 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride=2, valid)
            Convolution(H, W, K) : height, width, number of filters
        Mixed_7a layer
        """
        (p1, p2) = branch_units
        branch1 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p1[0]))),
                              Convolution(**conv_params(filt_params=(3, 3, p1[1]),
                                                        strides=2, padding=0))])
        branch2 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p2[0]))),
                              Convolution(**conv_params(filt_params=(1, 7, p2[1]),
                                                        padding={'pad_h': 0,
                                                                 'pad_w': 3,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(7, 1, p2[2]),
                                                        padding={'pad_h': 3,
                                                                 'pad_w': 0,
                                                                 'pad_d': 0})),
                              Convolution(**conv_params(filt_params=(3, 3, p2[3]),
                                                        strides=2, padding=0))])
        branch3 = Pool2D(fshape=3, padding=0, strides=2, op="max")
        self.model = Parallel([branch1, branch2, branch3])

    def __call__(self, in_obj):
        return self.model(in_obj)


class Inceptionv3_b5(object):

    def __init__(self, branch_units=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)]):

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
        branch1 = Convolution(**conv_params(filt_params=(1, 1, p1[0])))

        # Branch 2
        branch2a1_params = conv_params(filt_params=(1, 3, p2[1]),
                                       padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        branch2a2_params = conv_params(filt_params=(3, 1, p2[2]),
                                       padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})
        # This is the sub-branch with two parallel branches of 1x3 and 3x1
        branch2a = Parallel([Convolution(**branch2a1_params),
                             Convolution(**branch2a2_params)])
        branch2 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p2[0]))),
                              branch2a])

        # Branch 3
        branch3a1_params = conv_params(filt_params=(1, 3, p3[2]),
                                       padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        branch3a2_params = conv_params(filt_params=(3, 1, p3[3]),
                                       padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})
        branch3a = Parallel([Convolution(**branch3a1_params),
                             Convolution(**branch3a2_params)])
        branch3 = Sequential([Convolution(**conv_params(filt_params=(1, 1, p3[0]))),
                              Convolution(**conv_params(filt_params=(3, 3, p3[1]),
                                                        padding=1)),
                              branch3a])

        # Branch 4
        branch4 = Sequential([Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                              Convolution(**conv_params(filt_params=(1, 1, p4[0])))])

        # Combine branches
        self.model = Parallel([branch1, branch2, branch3, branch4])

    def __call__(self, in_obj):
        return self.model(in_obj)


class Inception(object):
    def __init__(self, mini=False):

        """
        Builds an Inception model
        """

        # Input size is 299 x 299 x 3
        # weight initialization
        if(mini is True):
            """
            This is the mini model with reduced number of filters in each layer
            """
            # Root branch of the tree
            seq1 = Sequential([Convolution(**conv_params(filt_params=(3, 3, 32),
                               #Convolution(**conv_params(filt_params=(3, 3, 16), 
                                                         padding=0, strides=2)),
                               # conv2d_1a_3x3
                               Convolution(**conv_params(filt_params=(3, 3, 16), padding=0)),
                               # conv2d_2a_3x3
                               Convolution(**conv_params(filt_params=(3, 3, 16), padding=1)),
                               # conv2d_2b_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_3a_3x3
                               Convolution(**conv_params(filt_params=(1, 1, 16))),
                               # conv2d_3b_1x1
                               #Convolution(**conv_params(filt_params=(3, 3, 32), padding=1)),
                               Convolution(**conv_params(filt_params=(3, 3, 16), padding=1)),
                               # conv2d_4a_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_5a_3x3
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5b
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5c
                               Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5d
                               Inceptionv3_b2([(32,), (32, 32, 32)]),  # mixed_6a
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)]),  # mixed_6b
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)]),  # mixed_6c
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)]),  # mixed_6d
                               Inceptionv3_b3([(32,), (32, 32, 32),
                                               (32, 32, 32, 32, 32), (32,)])])  # mixed_6e

            # Branch of main classifier
            seq2 = Sequential([Inceptionv3_b4([(32, 32), (32, 32, 32, 32)]),  # mixed_7a
                               Inceptionv3_b5([(32,), (32, 32, 32),
                                               (32, 32, 32, 32), (32,)]),  # mixed_7b
                               Inceptionv3_b5([(32,), (32, 32, 32),
                                               (32, 32, 32, 32), (32,)]),  # mixed_7c
                               Pool2D(fshape=8, padding=0, strides=2, op='avg'),  # Last Avg Pool
                               Dropout(keep=0.8),
                               Convolution(**conv_params(filt_params=(1, 1, 1000),
                                                         activation=Softmax()))])

            # Auxiliary classifier
            seq_aux = Sequential([Pool2D(fshape=5, padding=0, strides=3, op='avg'),
                                  Convolution(**conv_params(filt_params=(1, 1, 32))),
                                  Convolution(**conv_params(filt_params=(5, 5, 32))),
                                  Convolution(**conv_params(filt_params=(1, 1, 1000),
                                                            activation=Softmax()))])

        else:
            seq1 = Sequential([Convolution(**conv_params(filt_params=(3, 3, 32),
                                                         padding=0, strides=2)),
                               # conv2d_1a_3x3
                               Convolution(**conv_params(filt_params=(3, 3, 32), padding=0)),
                               # conv2d_2a_3x3
                               Convolution(**conv_params(filt_params=(3, 3, 64), padding=1)),
                               # conv2d_2b_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_3a_3x3
                               Convolution(**conv_params(filt_params=(1, 1, 80))),
                               # conv2d_3b_1x1
                               Convolution(**conv_params(filt_params=(3, 3, 192), padding=1)),
                               # conv2d_4a_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_5a_3x3
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96), (32, )]),  # mixed_5b
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96), (64, )]),  # mixed_5c
                               Inceptionv3_b1([(64,), (48, 64), (64, 96, 96), (64, )]),  # mixed_5d
                               Inceptionv3_b2([(384,), (64, 96, 96)]),  # mixed_6a
                               Inceptionv3_b3([(192,), (128, 128, 192),
                                               (128, 128, 128, 128, 192), (192,)]),  # mixed_6b
                               Inceptionv3_b3([(192,), (160, 160, 192),
                                               (160, 160, 160, 160, 192), (192,)]),  # mixed_6c
                               Inceptionv3_b3([(192,), (160, 160, 192),
                                               (160, 160, 160, 160, 192), (192,)]),  # mixed_6d
                               Inceptionv3_b3([(192,), (192, 192, 192),
                                               (192, 192, 192, 192, 192), (192,)])])  # mixed_6e

            # Branch of main classifier
            seq2 = Sequential([Inceptionv3_b4([(192, 320), (192, 192, 192, 192)]),  # mixed_7a
                               Inceptionv3_b5([(320,), (384, 384, 384),
                                               (448, 384, 384, 384), (192,)]),  # mixed_7b
                               Inceptionv3_b5([(320,), (384, 384, 384),
                                               (448, 384, 384, 384), (192,)]),  # mixed_7c
                               Pool2D(fshape=8, padding=0, strides=2, op='avg'),  # Last Avg Pool
                               Dropout(keep=0.8),
                               Convolution(**conv_params(filt_params=(1, 1, 1000),
                                                         activation=Softmax()))])

            # Auxiliary classifier
            seq_aux = Sequential([Pool2D(fshape=5, padding=0, strides=3, op='avg'),
                                  Convolution(**conv_params(filt_params=(1, 1, 128))),
                                  Convolution(**conv_params(filt_params=(5, 5, 768))),
                                  Convolution(**conv_params(filt_params=(1, 1, 1000),
                                                            activation=Softmax()))])

        self.seq1 = seq1
        self.seq2 = seq2
        self.seq_aux = seq_aux
