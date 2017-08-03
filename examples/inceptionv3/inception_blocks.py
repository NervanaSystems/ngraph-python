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
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from tqdm import tqdm
from contextlib import closing
from ngraph.frontends.neon import NgraphArgparser, ArrayIterator
from ngraph.frontends.neon import XavierInit, UniformInit
from ngraph.frontends.neon import Affine, Convolution, Pool2D, Sequential
from ngraph.frontends.neon import Rectlin, Softmax, Identity, GradientDescentMomentum
from ngraph.frontends.neon import ax
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
                                    bias_init=bias_init,
                                    filter_init=filter_init)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((5, 5, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=2)]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((3, 3, p3[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=1),
                         Convolution((3, 3, p3[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=1)]
        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)
        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)

        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_3_output = self.branch_3[2](branch_3_output)

        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


class Inceptionv3_b2(Sequential):

    def __init__(self, branch_units=[(384,), (64, 96, 96)], activation=Rectlin(),
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
                                    bias_init=bias_init, strides=2,
                                    filter_init=filter_init, padding=0)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((3, 3, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=1),
                         Convolution((3, 3, p2[2]), activation=activation,
                                     bias_init=bias_init, strides=2,
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
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p2[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0})]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((7, 1, p3[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((1, 7, p3[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p3[3]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((1, 7, p3[4]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0})]
        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                     bias_init=bias_init,
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
                                    bias_init=bias_init,
                                    filter_init=filter_init),
                         Convolution((3, 3, p1[1]), activation=activation,
                                     bias_init=bias_init, strides=2,
                                     filter_init=filter_init, padding=0)]
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((1, 7, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 3, 'pad_d': 0}),
                         Convolution((7, 1, p2[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 3, 'pad_w': 0, 'pad_d': 0}),
                         Convolution((3, 3, p2[3]), activation=activation,
                                     bias_init=bias_init, strides=2,
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
                                    bias_init=bias_init,
                                    filter_init=filter_init)

        self.branch_2 = Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init)
        self.branch_2a = Convolution((1, 3, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        self.branch_2b = Convolution((3, 1, p2[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})

        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((3, 3, p3[1]), activation=activation,
                                     bias_init=bias_init, padding=1,
                                     filter_init=filter_init)]
        self.branch_3a = Convolution((1, 3, p3[2]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 0, 'pad_w': 1, 'pad_d': 0})
        self.branch_3b = Convolution((3, 1, p3[3]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding={'pad_h': 1, 'pad_w': 0, 'pad_d': 0})

        self.branch_4 = [Pool2D(fshape=3, padding=1, strides=1, op="avg"),
                         Convolution((1, 1, p4[0]), activation=activation,
                                    bias_init=bias_init,
                                    filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)

        branch_2_output = self.branch_2(in_obj)
        branch_2a_output = self.branch_2a(branch_2_output)
        branch_2b_output = self.branch_2b(branch_2_output)
        branch_2_outputs = [branch_2a_output, branch_2b_output]
        branch_2_output = ng.concat_along_axis(branch_2_outputs, branch_1_output.axes.channel_axis())

        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_3a_output = self.branch_3a(branch_3_output)
        branch_3b_output = self.branch_3b(branch_3_output)
        branch_3_outputs = [branch_3a_output, branch_3b_output]
        branch_3_output = ng.concat_along_axis(branch_3_outputs, branch_1_output.axes.channel_axis())

        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


