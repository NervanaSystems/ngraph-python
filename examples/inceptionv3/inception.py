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
import ngraph as ng
from ngraph.frontends.neon import XavierInit, UniformInit
from ngraph.frontends.neon import Convolution, Pool2D, Sequential, Dropout
from ngraph.frontends.neon import Rectlin, Softmax


class Inceptionv3_b1(Sequential):

    def __init__(self, branch_units=[(64,), (48, 64), (64, 96, 96), (64,)], activation=Rectlin(),
                 bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

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

        self.branch_1 = Convolution((1, 1, p1[0]), activation=activation,
                                    bias_init=bias_init, batch_norm=True,
                                    filter_init=filter_init)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((5, 5, p2[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init, padding=2)]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((3, 3, p3[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init, padding=1),
                         Convolution((3, 3, p3[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init, padding=1)]
        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init)]

np.seterr(all='raise')

parser = NgraphArgparser(description=__doc__)
# Default batch_size for convnet-googlenet is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 224
X_train = np.random.uniform(-1, 1, (args.batch_size, 3, image_size, image_size))
y_train = np.ones(shape=(args.batch_size), dtype=np.int32)
train_data = {'image': {'data': X_train,
                        'axes': ('batch', 'C', 'height', 'width')},
              'label': {'data': y_train,
                        'axes': ('batch',)}}
train_set = ArrayIterator(train_data,
                          batch_size=args.batch_size,
                          total_iterations=args.num_iterations)
inputs = train_set.make_placeholders(include_iteration=True)
ax.Y.length = 1000  # number of outputs of last layer.

# weight initialization
bias_init = UniformInit(low=-0.08, high=0.08)


class Inception(Sequential):

    def __init__(self, branch_units, activation=Rectlin(),
                 bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

        """
        Second inception block with three branches, concatenated in the end
            1. 3x3 conv (stride = 2, valid)
            2. 1x1 conv, 3x3 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride = 2, valid)
        Convolution(H, W, K) : height, width, number of filters
        Mixed_6a layer
        """
        (p1, p2) = branch_units

        self.branch_1 = Convolution((3, 3, p1[0]), activation=activation,
                                    bias_init=bias_init, strides=2, batch_norm=True,
                                    filter_init=filter_init, padding=0)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((3, 3, p2[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init, padding=1),
                         Convolution((3, 3, p2[2]), activation=activation,
                                     bias_init=bias_init, strides=2, batch_norm=True,
                                     filter_init=filter_init, padding=0)]
        self.branch_3 = [Pool2D(fshape=3, padding=0, strides=2, op="max")]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)

        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)
        branch_2_output = self.branch_2[2](branch_2_output)

        branch_3_output = self.branch_3[0](in_obj)

        outputs = [branch_1_output, branch_2_output, branch_3_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


class Inceptionv3_b3(Sequential):

    def __init__(self, branch_units=[(192), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
                 activation=Rectlin(), bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

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

        self.branch_1 = Convolution((1, 1, p1[0]), activation=activation,
                                    bias_init=bias_init,
                                    filter_init=filter_init)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((1, 7, p2[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p2[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0})]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((7, 1, p3[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((1, 7, p3[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p3[3]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((1, 7, p3[4]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0})]
        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)

        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)
        branch_2_output = self.branch_2[2](branch_2_output)

        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_3_output = self.branch_3[2](branch_3_output)
        branch_3_output = self.branch_3[3](branch_3_output)
        branch_3_output = self.branch_3[4](branch_3_output)

        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


class Inceptionv3_b4(Sequential):

    def __init__(self, branch_units=[(192, 320), (192, 192, 192, 192)],
                 activation=Rectlin(), bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

        """
        Fourth inception block with three branches, concatenated in the end
            1. 1x1 conv, 3x3 conv (stride=2, valid)
            2. 1x1 conv, 1x7 conv, 7x1 conv, 3x3 conv (stride=2, valid)
            3. 3x3 pool (stride=2, valid)
            Convolution(H, W, K) : height, width, number of filters
        Mixed_7a layer
        """
        (p1, p2) = branch_units

        self.branch_1 = [Convolution((1, 1, p1[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((3, 3, p1[1]), activation=activation,
                                     bias_init=bias_init, strides=2,
                                     filter_init=filter_init, padding=0)]
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((1, 7, p2[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p2[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((3, 3, p2[3]), activation=activation,
                                     bias_init=bias_init, strides=2, batch_norm=True,
                                     filter_init=filter_init, padding=0)]
        self.branch_3 = [Pool2D(fshape=3, padding=0, strides=2, op="max")]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1[0](in_obj)
        branch_1_output = self.branch_1[1](branch_1_output)

        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)
        branch_2_output = self.branch_2[2](branch_2_output)
        branch_2_output = self.branch_2[3](branch_2_output)

        branch_3_output = self.branch_3[0](in_obj)

        outputs = [branch_1_output, branch_2_output, branch_3_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


class Inceptionv3_b5(Sequential):

    def __init__(self, branch_units=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)],
                 activation=Rectlin(), bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

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

        self.branch_1 = Convolution((1, 1, p1[0]), activation=activation,
                                    bias_init=bias_init, batch_norm=True,
                                    filter_init=filter_init)

        self.branch_2 = Convolution((1, 1, p2[0]), activation=activation,
                                    bias_init=bias_init, batch_norm=True,
                                    filter_init=filter_init)
        self.branch_2a = Convolution((1, 3, p2[1]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        self.branch_2b = Convolution((3, 1, p2[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})

        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init),
                         Convolution((3, 3, p3[1]), activation=activation,
                                     bias_init=bias_init, padding=1, batch_norm=True,
                                     filter_init=filter_init)]
        self.branch_3a = Convolution((1, 3, p3[2]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        self.branch_3b = Convolution((3, 1, p3[3]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init,
                                     padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})

        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                     bias_init=bias_init, batch_norm=True,
                                     filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)
        branch_2_output = self.branch_2(in_obj)
        branch_2a_output = self.branch_2a(branch_2_output)
        branch_2b_output = self.branch_2b(branch_2_output)
        branch_2_outputs = [branch_2a_output, branch_2b_output]
        branch_2_output = ng.concat_along_axis(branch_2_outputs,
                                               branch_1_output.axes.channel_axis())

        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_3a_output = self.branch_3a(branch_3_output)
        branch_3b_output = self.branch_3b(branch_3_output)
        branch_3_outputs = [branch_3a_output, branch_3b_output]
        branch_3_output = ng.concat_along_axis(branch_3_outputs,
                                               branch_1_output.axes.channel_axis())

        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


class Inception(object):
    def __init__(self, mini=False):

        """
        Builds an Inception model
        """

        # Input size is 299 x 299 x 3
        # weight initialization
        bias_init = UniformInit(low=-0.08, high=0.08)
        if(mini is True):
            """
            This is the mini model with reduced number of filters in each layer
            """
            # Root branch of the tree
            seq1 = Sequential([Convolution((3, 3, 32), padding=0, strides=2, batch_norm=True,
                                           activation=Rectlin(), bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_1a_3x3
                               Convolution((3, 3, 16), activation=Rectlin(), padding=0,
                                           batch_norm=True, bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_2a_3x3
                               Convolution((3, 3, 16), activation=Rectlin(), padding=1,
                                           batch_norm=True, bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_2b_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_3a_3x3
                               Convolution((1, 1, 16), activation=Rectlin(), batch_norm=True,
                                           bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_3b_1x1
                               Convolution((3, 3, 32), activation=Rectlin(), padding=1,
                                           batch_norm=True, bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_4a_3x3
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
                               Convolution((1, 1, 1000), activation=Softmax(),
                                           bias_init=bias_init, filter_init=XavierInit())])

            # Auxiliary classifier
            seq_aux = Sequential([Pool2D(fshape=5, padding=0, strides=3, op='avg'),
                                  Convolution((1, 1, 32), activation=Rectlin(), batch_norm=True,
                                              bias_init=bias_init, filter_init=XavierInit()),
                                  Convolution((5, 5, 32), padding=0, activation=Rectlin(),
                                              batch_norm=True,
                                              bias_init=bias_init, filter_init=XavierInit()),
                                  Convolution((1, 1, 1000), activation=Softmax(),
                                              bias_init=bias_init, filter_init=XavierInit())])

        else:
            seq1 = Sequential([Convolution((3, 3, 32), padding=0, strides=2, batch_norm=True,
                                           activation=Rectlin(), bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_1a_3x3
                               Convolution((3, 3, 32), activation=Rectlin(), padding=0,
                                           batch_norm=True,
                                           bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_2a_3x3
                               Convolution((3, 3, 64), activation=Rectlin(), padding=1,
                                           batch_norm=True,
                                           bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_2b_3x3
                               Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_3a_3x3
                               Convolution((1, 1, 80), activation=Rectlin(), batch_norm=True,
                                           bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_3b_1x1
                               Convolution((3, 3, 192), activation=Rectlin(),
                                           padding=1, batch_norm=True, bias_init=bias_init,
                                           filter_init=XavierInit()),  # conv2d_4a_3x3
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
                               Convolution((1, 1, 1000), activation=Softmax(),
                                           bias_init=bias_init, filter_init=XavierInit())])

            # Auxiliary classifier
            seq_aux = Sequential([Pool2D(fshape=5, padding=0, strides=3, op='avg'),
                                  Convolution((1, 1, 128), activation=Rectlin(), batch_norm=True,
                                              bias_init=bias_init, filter_init=XavierInit()),
                                  Convolution((5, 5, 768), padding=0, activation=Rectlin(),
                                              batch_norm=True,
                                              bias_init=bias_init, filter_init=XavierInit()),
                                  Convolution((1, 1, 1000), activation=Softmax(),
                                              bias_init=bias_init, filter_init=XavierInit())])

        self.seq1 = seq1
        self.seq2 = seq2
        self.seq_aux = seq_aux
