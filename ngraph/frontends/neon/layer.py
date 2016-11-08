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
from ngraph.frontends.neon.axis import ar, ax
import numpy as np
from operator import itemgetter


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

    def inference_outputs(self, in_obj):
        return self.train_outputs(in_obj)


class nnPreprocess(nnLayer):
    def __init__(self, functor, **kwargs):
        super(nnPreprocess, self).__init__(**kwargs)
        self.functor = functor

    def train_outputs(self, in_obj):
        return self.functor(in_obj)


class nnAffine(nnLayer):
    def __init__(self, init, nout=None, activation=(lambda x: x), bias=None, **kwargs):
        super(nnAffine, self).__init__(**kwargs)
        if self.axes is None:
            assert(nout is not None), "Must provide either axes or nout to Affine"

        self.nout = nout
        self.init = init
        self.activation = activation
        self.bias = bias
        self.W = None
        self.b = 0 if self.bias is None else None

    def train_outputs(self, in_obj):
        out_axes = ng.make_axes(self.axes or [ng.make_axis(self.nout, name='Hidden')])
        in_axes = in_obj.axes.sample_axes()
        in_axes = in_axes - in_axes.recurrent_axes()
        w_axes = out_axes - out_axes.recurrent_axes() + in_axes.get_dual()
        b_axes = in_obj.axes.sample_axes()
        if self.W is None:
            self.W = ng.variable(axes=w_axes, initial_value=self.init(w_axes.lengths))
        if self.b is None:
            self.b = ng.variable(axes=b_axes, initial_value=self.bias(b_axes.lengths))

        return self.activation(ng.dot(self.W, in_obj, use_dual=True))# + self.b)


class nnConvBase(nnLayer):
    """
    Convolutional layer that requires explicit binding of all spatial roles

    Args:
        fshape (dict): filter shape -- must contain keys 'T', 'R', 'S', 'K'
        init (function): function for later initializing filters
        strides (dict): stride specification -- must contain keys 'str_d', 'str_h', 'str_w'
        padding (dict): pad specification -- must contain keys 'pad_d', 'pad_h', 'pad_w'

    """
    def __init__(self, fshape, init, strides, padding, **kwargs):
        super(nnConvBase, self).__init__(**kwargs)
        self.convparams = dict(T=None, R=None, S=None, K=None,
                               pad_h=None, pad_w=None, pad_d=None,
                               str_h=None, str_w=None, str_d=None)

        for d in [fshape, strides, padding]:
            self.convparams.update(d)

        missing_keys = [k for k, v in self.convparams.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError("Missing conv keys: {}".format(missing_keys))

        self.init = init
        self.f_axes = None
        self.o_axes = None
        self.W = None

    def train_outputs(self, in_obj):
        cpm = self.convparams.copy()
        in_axes = in_obj.axes
        if self.f_axes is None:
            self.f_axes = in_axes.role_axes(ar.Channel)
            for _ax in (ax.T, ax.R, ax.S, ax.K):
                self.f_axes += ng.make_axis(name=_ax.short_name, roles=_ax.roles)
            self.f_axes[1:].set_shape(itemgetter(*'TRSK')(cpm))

            self.W = ng.variable(axes=self.f_axes, initial_value=self.init(self.f_axes.lengths))

        # TODO: clean this up
        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.make_axis(self.f_axes[4].length, name='C', roles=[ar.Channel]),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_d'], cpm['str_d'], role=ar.Depth),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_h'], cpm['str_h'], role=ar.Height),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_w'], cpm['str_w'], role=ar.Width),
                ax.N
            ])

        return ng.convolution(cpm, in_obj, self.W, axes=self.o_axes)


class nnConv2D(nnConvBase):
    def __init__(self, fshape, init, strides, padding, **kwargs):
        if isinstance(fshape, tuple) or isinstance(fshape, list):
            if len(fshape) == 2:
                fshape = (1, fshape[0], fshape[0], fshape[1])
            elif len(fshape) == 3:
                fshape = (1, fshape[0], fshape[1], fshape[2])
            fshape = {k: x for k, x in zip('TRSK', fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides, 'str_d': 1}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding, 'pad_d': 0}

        super(nnConv2D, self).__init__(fshape, init, strides, padding, **kwargs)


class nnConvolution(nnConv2D):
    def __init__(self, fshape, init, strides=1, padding=0, activation=(lambda x: x), **kwargs):
        self.activation = activation
        super(nnConvolution, self).__init__(fshape, init, strides, padding, **kwargs)

    def train_outputs(self, in_obj):
        return self.activation(super(nnConvolution, self).train_outputs(in_obj))

class nnActivation(nnLayer):
    def __init__(self, transform, **kwargs):
        self.transform = transform
        super(nnActivation, self).__init__(**kwargs)

    def train_outputs(self, in_obj):
        return self.transform(in_obj)

class nnPoolBase(nnLayer):
    """
    Pooling layer that requires explicit binding of all spatial roles

    Args:
        fshape (dict): filter shape -- must contain keys 'J', 'T', 'R', 'S',
        init (function): function for later initializing filters
        strides (dict): stride specification -- must contain keys 'str_c', str_d', 'str_h', 'str_w'
        padding (dict): pad specification -- must contain keys 'pad_c', pad_d', 'pad_h', 'pad_w'

    """
    def __init__(self, fshape, strides, padding, op='max', **kwargs):
        super(nnPoolBase, self).__init__(**kwargs)
        self.poolparams = dict(J=None, T=None, R=None, S=None,
                               pad_h=None, pad_w=None, pad_d=None, pad_c=None,
                               str_h=None, str_w=None, str_d=None, str_c=None,
                               op=op)

        for d in [fshape, strides, padding]:
            self.poolparams.update(d)

        missing_keys = [k for k, v in self.poolparams.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError("Missing pooling keys: {}".format(missing_keys))

        self.o_axes = None

    def train_outputs(self, in_obj):
        ppm = self.poolparams.copy()
        in_axes = in_obj.axes
        # TODO: clean this up
        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.spatial_axis(in_axes, ppm['J'], ppm['pad_c'], ppm['str_c'], role=ar.Channel),
                ng.spatial_axis(in_axes, ppm['T'], ppm['pad_d'], ppm['str_d'], role=ar.Depth),
                ng.spatial_axis(in_axes, ppm['R'], ppm['pad_h'], ppm['str_h'], role=ar.Height),
                ng.spatial_axis(in_axes, ppm['S'], ppm['pad_w'], ppm['str_w'], role=ar.Width),
                ax.N
            ])

        return ng.pooling(ppm, in_obj, axes=self.o_axes)


class nnPool2D(nnPoolBase):
    def __init__(self, fshape, strides=1, padding=0, **kwargs):

        if isinstance(fshape, int):
            fshape = (1, 1, fshape, fshape)
        if isinstance(fshape, tuple) or isinstance(fshape, list):
            if len(fshape) == 2:
                fshape = (1, 1, fshape[0], fshape[1])
            if len(fshape) != 4:
                raise ValueError("Incorrect filter specification: {}".format(fshape))
            fshape = {k: x for k, x in zip('JTRS', fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides, 'str_d': 1, 'str_c': 1}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding, 'pad_d': 0, 'pad_c': 0}
        super(nnPool2D, self).__init__(fshape, strides, padding, **kwargs)


class nnRecurrent(nnLayer):
    """
    Basic recurrent layer.
    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None, **kwargs):
        super(nnRecurrent, self).__init__(**kwargs)
        self.nout = output_size
        self.activation = activation
        self.init = init
        self.init_inner = init_inner or init

    def train_outputs(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
           in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer

        Returns:
           (Tensor): output

        """
        in_axes = in_obj.axes
        self.time_axis = in_axes.recurrent_axes()[0]

        def get_steps(x, time_axis):
            return [ng.slice_along_axis(x, time_axis, i) for i in range(time_axis.length)]

        if self.axes is not None:
            hidden_axes = self.axes - self.axes.recurrent_axes()
        else:
            hidden_axes = ng.make_axes([ng.make_axis(self.nout, name='Hidden_in')])

        w_in_axes = hidden_axes + (in_axes.sample_axes() - in_axes.recurrent_axes()).get_dual()
        w_re_axes = hidden_axes + hidden_axes.get_dual()

        self.W_input = ng.variable(axes=w_in_axes,
                                   initial_value=self.init(w_in_axes.lengths), name="W_in")
        self.W_recur = ng.variable(axes=w_re_axes,
                                   initial_value=self.init_inner(w_re_axes.lengths), name="W_re")
        self.b = ng.variable(axes=hidden_axes, initial_value=0, name="bias")

        h_ff_buf = ng.dot(self.W_input, in_obj, use_dual=True, name="W_in_dot_in")
        h_ff_s = get_steps(h_ff_buf, self.time_axis)
        self.h_init = ng.constant(np.zeros(h_ff_s[0].axes.lengths),
                                  axes=h_ff_s[0].axes,
                                  name="h_init")
        hprev = [self.h_init]

        for i in range(self.time_axis.length):
            d = ng.dot(self.W_recur, hprev[i], use_dual=True, name="W_rec_dot_h{}".format(i))
            h = self.activation(d + h_ff_s[i] + self.b)
            h.name = "activ{}".format(i)
            hprev.append(h)

        rnn_out = ng.Stack(hprev[1:], self.time_axis, pos=1)
        return rnn_out
