# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from builtins import object
import ngraph as ng
import collections
from ngraph.frontends.neon.axis import ar, ax


class Layer(object):
    """TODO."""

    def __init__(
            self,
            name=None,
            graph=None,
            axes=None,
            parallelism="Unknown",
            **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.name = name
        self.axes = axes

    def configure(self, in_obj):
        """
        Add to computation graph for the layer.

        Arguments:
          in_obj: The input for the layer

        Returns:
          The output of the layer
        """
        return in_obj


class ParameterLayer(Layer):
    """TODO."""

    def __init__(self, init=None, name=None, parallelism='Unknown', **kwargs):
        super(ParameterLayer, self).__init__(name=name, parallelism=parallelism,
                                             **kwargs)
        self.has_params = True
        self.init = init
        self.W = None
        self.dW = None
        self.batch_sum = None


class nnLayer(object):
    def __init__(self, name=None, inputs=None, outputs=None, axes=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.axes = axes

    def train_outputs(self, in_obj):
        raise NotImplementedError()

    def inf_outputs(self, in_obj):
        return self.train_outputs(in_obj)


class nnPreprocess(nnLayer):
    def __init__(self, functor, **kwargs):
        super(nnPreprocess, self).__init__(**kwargs)
        self.functor = functor

    def train_outputs(self, in_obj):
        return self.functor(in_obj)


class nnAffine(nnLayer):
    def __init__(self, init, nout=None, activation=(lambda x: x), bias_init=None, **kwargs):
        super(nnAffine, self).__init__(**kwargs)
        if self.axes is None:
            assert(nout is not None), "Must provide either axes or nout to Affine"
        self.nout = nout
        self.init = init
        self.activation = activation
        self.b = 0
        self.bias_init = bias_init

    def train_outputs(self, in_obj):
        out_axes = ng.make_axes(self.axes or [ng.make_axis(self.nout, name='Hidden')])
        in_axes = in_obj.axes.sample_axes()
        in_axes = in_axes - in_axes.recurrent_axes()
        w_axes = out_axes - out_axes.recurrent_axes() + in_axes.get_dual()
        self.W = ng.Variable(axes=w_axes, initial_value=self.init(w_axes.lengths))
        return self.activation(ng.dot(self.W, in_obj, use_dual=True) + self.b)


class nnConv1d(nnLayer):
    def __init__(self, fshape, init, strides, padding, activation=(lambda x: x), bias_init=None):

        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'S': 1, 'W': 1, 'T': 1, 'D': 1}  # 2D & 3D parameters

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fshape = {'R': fshape[0], 'K': fshape[1]}
        if isinstance(strides, int):
            strides = {'str_h': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

        self.init = init
        self.activation = activation
        self.b = 0
        self.bias_init = bias_init

    def train_outputs(self, in_obj):
        # if self.bias_init:
        #     b_axes = ng.Axes([ng.Axis(self.convparams['K'], name='K')])
        #     self.b = ng.Variable(axes=b_axes, initial_value=self.bias_init(b_axes.lengths))

        # Need to expand dims out if we are less than CDHWN

        # if len(in_obj.axes.role_axes(ar.D)):
        #     ng.ExpandDims(in_obj.


        # if
        # TODO:  Careful about the conv state that gets tied to op vs. layer
        convparams = self.convparams.copy()
        convparams['C'] = in_obj.axes.role_axes('channel').lengths[0]
        w_axes = ng.Axes([ng.Axis(convparams[ax], name=ax) for ax in ('C', 'T', 'R', 'S', 'K')])
        self.W = ng.Variable(axes=w_axes, initial_value=self.init(w_axes.lengths))
        return self.activation(ng.convolution(convparams, in_obj, self.W) + self.b)



class nnConv(nnLayer):
    def __init__(self, fshape, init, strides, padding, activation=(lambda x: x), bias_init=None):

        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1}  # 3D parameters
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

        self.init = init
        self.activation = activation
        self.b = 0
        self.bias_init = bias_init

    def train_outputs(self, in_obj):
        # if self.bias_init:
        #     b_axes = ng.Axes([ng.Axis(self.convparams['K'], name='K')])
        #     self.b = ng.Variable(axes=b_axes, initial_value=self.bias_init(b_axes.lengths))

        # TODO:  Careful about the conv state that gets tied to op vs. layer
        convparams = self.convparams.copy()
        convparams['C'] = in_obj.axes.role_axes('channel').lengths[0]
        w_axes = ng.Axes([ng.Axis(convparams[ax], name=ax) for ax in ('C', 'T', 'R', 'S', 'K')])
        self.W = ng.Variable(axes=w_axes, initial_value=self.init(w_axes.lengths))
        return self.activation(ng.convolution(convparams, in_obj, self.W) + self.b)


class nnPool(nnLayer):

    """
    Pooling layer implementation.

    Arguments:
        fshape (int, tuple(int, int)): one or two dimensional shape
            of pooling window
        op (str, optional): pooling operation in [max, avg]. Defaults to "max"
        strides (int, dict, optional): strides to apply pooling window
            over. An int applies to both dimensions, or a dict with str_h
            and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        name (str, optional): layer name. Defaults to "PoolingLayer"
    """

    def __init__(self, fshape, op="max", strides={}, padding={}):
        self.poolparams = {'str_h': None, 'str_w': None, 'str_d': None, 'str_c': None,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0, 'pad_c': 0,
                           'J': 1, 'T': 1, 'D': 1, 'op': op}  # 3D paramaters

        # keep args around in __dict__ for get_description
        self.op = op
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fkeys = ('R', 'S') if len(fshape) == 2 else ('T', 'R', 'S')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        elif fshape == 'all':
            fshape = dict(R=None, S=None)
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.poolparams.update(d)

    def train_outputs(self, in_obj):
        shapedict = in_obj.shape_dict()
        self.poolparams.update(shapedict)
        if self.poolparams['R'] is None:
            self.poolparams['R'] = shapedict['H']
            self.poolparams['S'] = shapedict['W']
        return ng.pooling(self.poolparams, in_obj)

    def inf_outputs(self, in_obj):
        return self.train_outputs(in_obj)

