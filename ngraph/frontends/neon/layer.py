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
    def __init__(self, name=None, inputs=None, outputs=None, axes=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.axes = axes

    def train_outputs(self, in_obj):
        raise NotImplementedError()

    def inference_outputs(self, in_obj):
        return self.train_outputs(in_obj)


class Preprocess(Layer):
    def __init__(self, functor, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        self.functor = functor

    def train_outputs(self, in_obj):
        return self.functor(in_obj)


class Linear(Layer):
    metadata = {'layer_type': 'linear'}

    def __init__(self, init, nout=None, **kwargs):
        super(Linear, self).__init__(**kwargs)
        if self.axes is None:
            assert(nout is not None), "Must provide either axes or nout to Linear"

        self.nout = nout
        self.init = init
        self.W = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        out_axes = ng.make_axes(self.axes or [ng.make_axis(self.nout).named('Hidden')])
        in_axes = in_obj.axes.sample_axes()
        in_axes = in_axes - in_axes.recurrent_axes()
        w_axes = out_axes - out_axes.recurrent_axes() + [axis - 1 for axis in in_axes]
        if self.W is None:
            self.W = ng.variable(axes=w_axes, initial_value=self.init(w_axes.lengths))

        return ng.dot(self.W, in_obj)


class ConvBase(Layer):
    """
    Convolutional layer that requires explicit binding of all spatial roles

    Args:
        fshape (dict): filter shape -- must contain keys 'T', 'R', 'S', 'K'
        init (function): function for later initializing filters
        strides (dict): stride specification -- must contain keys 'str_d', 'str_h', 'str_w'
        padding (dict): pad specification -- must contain keys 'pad_d', 'pad_h', 'pad_w'

    """
    metadata = {'layer_type': 'convolution'}

    def __init__(self, fshape, init, strides, padding, **kwargs):
        super(ConvBase, self).__init__(**kwargs)
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

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        cpm = self.convparams.copy()
        in_axes = in_obj.axes
        if self.f_axes is None:
            self.f_axes = in_axes.role_axes(ar.Channel)
            for _ax in (ax.T, ax.R, ax.S, ax.K):
                self.f_axes += ng.make_axis(roles=_ax.roles).named(_ax.short_name)
            self.f_axes[1:].set_shape(itemgetter(*'TRSK')(cpm))

            self.W = ng.variable(axes=self.f_axes, initial_value=self.init(self.f_axes.lengths))

        # TODO: clean this up
        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.make_axis(self.f_axes[4].length, roles=[ar.Channel]).named('C'),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_d'], cpm['str_d'], role=ar.Depth),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_h'], cpm['str_h'], role=ar.Height),
                ng.spatial_axis(in_axes, self.f_axes, cpm['pad_w'], cpm['str_w'], role=ar.Width),
                ax.N
            ])

        return ng.convolution(cpm, in_obj, self.W, axes=self.o_axes)


class Conv2D(ConvBase):
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

        super(Conv2D, self).__init__(fshape, init, strides, padding, **kwargs)


class Activation(Layer):
    metadata = {'layer_type': 'activation'}

    def __init__(self, transform, **kwargs):
        self.transform = transform
        super(Activation, self).__init__(**kwargs)

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        # An activation layer with no transform defaults to identity
        if self.transform:
            return self.transform(in_obj)
        else:
            return in_obj


class PoolBase(Layer):
    """
    Pooling layer that requires explicit binding of all spatial roles

    Args:
        fshape (dict): filter shape -- must contain keys 'J', 'T', 'R', 'S',
        init (function): function for later initializing filters
        strides (dict): stride specification -- must contain keys 'str_c', str_d', 'str_h', 'str_w'
        padding (dict): pad specification -- must contain keys 'pad_c', pad_d', 'pad_h', 'pad_w'

    """
    metadata = {'layer_type': 'pooling'}

    def __init__(self, fshape, strides, padding, op='max', **kwargs):
        super(PoolBase, self).__init__(**kwargs)
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

    @ng.with_op_metadata
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


class Pool2D(PoolBase):
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
        super(Pool2D, self).__init__(fshape, strides, padding, **kwargs)


class Bias(Layer):
    """
    Bias layer, common to linear and convolutional layers

    Args:
        init (function): function for later initializing bias values
        shared (bool): applies only to convolutional biases.  Whether to use same bias for
                       entire feature map.  Default true.
    """
    metadata = {'layer_type': 'bias'}

    def __init__(self, init, shared=True, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.W = None
        self.init = init
        self.shared = shared

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        if self.init:
            w_axes = in_obj.axes.sample_axes()
            if self.shared and len(in_obj.axes.role_axes(ar.Channel)) != 0:
                w_axes = in_obj.axes.role_axes(ar.Channel)

            self.W = self.W or ng.variable(axes=w_axes, initial_value=self.init(w_axes.lengths))
            return in_obj + self.W
        else:
            return in_obj


class Affine(Layer):
    def __init__(self, w_init, nout=None, b_init=None, activation=None,
                 batch_norm=False, **kwargs):
        self.linear = Linear(init=w_init, nout=nout, **kwargs)
        self.bias = Bias(init=b_init)
        self.batch_norm = BatchNorm() if batch_norm else None
        self.activation = Activation(transform=activation)

    def train_outputs(self, in_obj):
        l_out = self.linear.train_outputs(in_obj)
        b_out = self.bias.train_outputs(l_out)
        bn_out = self.batch_norm.train_outputs(b_out) if self.batch_norm else b_out
        a_out = self.activation.train_outputs(bn_out)
        return a_out

    def inference_outputs(self, in_obj):
        l_out = self.linear.inference_outputs(in_obj)
        b_out = self.bias.inference_outputs(l_out)
        bn_out = self.batch_norm.inference_outputs(b_out) if self.batch_norm else b_out
        a_out = self.activation.inference_outputs(bn_out)
        return a_out


class Convolution(Layer):
    def __init__(self, fshape, w_init, strides=1, padding=0, b_init=None, activation=None,
                 batch_norm=False, **kwargs):
        self.conv = Conv2D(fshape, w_init, strides, padding, **kwargs)
        self.bias = Bias(init=b_init)
        self.batch_norm = BatchNorm() if batch_norm else None
        self.activation = Activation(transform=activation)

    def train_outputs(self, in_obj):
        l_out = self.conv.train_outputs(in_obj)
        b_out = self.bias.train_outputs(l_out)
        bn_out = self.batch_norm.train_outputs(b_out) if self.batch_norm else b_out
        a_out = self.activation.train_outputs(bn_out)
        return a_out

    def inference_outputs(self, in_obj):
        l_out = self.conv.inference_outputs(in_obj)
        b_out = self.bias.inference_outputs(l_out)
        bn_out = self.batch_norm.inference_outputs(b_out) if self.batch_norm else b_out
        a_out = self.activation.inference_outputs(bn_out)
        return a_out


class BatchNorm(Layer):
    """
    A batch normalization layer as described in [Ioffe2015]_.

    Normalizes a batch worth of inputs by subtracting batch mean and
    dividing by batch variance.  Then scales by learned factor gamma and
    shifts by learned bias beta.

    Notes:

    .. [Ioffe2015] http://arxiv.org/abs/1502.03167
    """
    metadata = {'layer_type': 'batch_norm'}

    def __init__(self, rho=0.9, eps=1e-3, **kwargs):
        self.rho = rho
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.gmean = None
        self.gvar = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        in_axes = in_obj.axes.sample_axes()
        red_axes = ng.make_axes()
        if len(in_axes.role_axes(ar.Channel)) != 0:
            red_axes += in_axes.sample_axes() - in_axes.role_axes(ar.Channel)
        red_axes += in_obj.axes.batch_axes()
        out_axes = in_axes - red_axes

        self.gamma = self.gamma or ng.variable(axes=out_axes, initial_value=1.0).named('gamma')
        self.beta = self.beta or ng.variable(axes=out_axes, initial_value=0.0).named('beta')
        self.gvar = self.gvar or ng.persistent_tensor(axes=out_axes, initial_value=1.0)
        self.gmean = self.gmean or ng.persistent_tensor(axes=out_axes, initial_value=1.0)

        xmean = ng.mean(in_obj, reduction_axes=red_axes)
        xvar = ng.variance(in_obj, reduction_axes=red_axes)
        ng.assign(self.gmean, self.gmean * self.rho + xmean * (1.0 - self.rho))
        ng.assign(self.gvar, self.gvar * self.rho + xvar * (1.0 - self.rho))

        return self.gamma * (in_obj - xmean) / ng.sqrt(xvar + self.eps) + self.beta

    def inference_outputs(self, in_obj):
        return self.gamma * (in_obj - self.gmean) / ng.sqrt(self.gvar + self.eps) + self.beta


class Dropout(Layer):
    """
    Layer for stochastically droping activations to prevent overfitting
    Args:
        keep (float):  Number between 0 and 1 that indicates probability of any particular
                       activation being dropped.  Default 0.5.
    """

    metadata = {'layer_type': 'dropout'}

    def __init__(self, keep=0.5, **kwargs):
        self.keep = keep
        self.mask = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        in_axes = in_obj.axes.sample_axes()
        self.mask = self.mask or ng.persistent_tensor(axes=in_axes).named('mask')
        self.mask = ng.uniform(self.mask, low=0.0, high=1.0) <= self.keep
        return self.mask * in_obj

    def inference_outputs(self, in_obj):
        return self.keep * in_obj


class Recurrent(Layer):
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
    metadata = {'layer_type': 'recurrent'}

    def __init__(self, output_size, init, init_inner=None, activation=None, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.nout = output_size
        self.activation = activation
        self.init = init
        self.init_inner = init_inner or init

    @ng.with_op_metadata
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
            hidden_axes = ng.make_axes([ng.make_axis(self.nout).named('Hidden_in')])

        w_in_axes = hidden_axes + [axis - 1 for axis in in_axes.sample_axes() -
                                   in_axes.recurrent_axes()]
        w_re_axes = hidden_axes + [axis - 1 for axis in hidden_axes]

        self.W_input = ng.variable(axes=w_in_axes,
                                   initial_value=self.init(w_in_axes.lengths)
                                   ).named("W_in")
        self.W_recur = ng.variable(axes=w_re_axes,
                                   initial_value=self.init_inner(w_re_axes.lengths)
                                   ).named("W_re")
        self.b = ng.variable(axes=hidden_axes, initial_value=0).named("bias")

        h_ff_buf = ng.dot(self.W_input, in_obj).named("W_in_dot_in")
        h_ff_s = get_steps(h_ff_buf, self.time_axis)
        self.h_init = ng.constant(np.zeros(h_ff_s[0].axes.lengths),
                                  axes=h_ff_s[0].axes).named('h_init')

        hprev = [self.h_init]

        for i in range(self.time_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                d = ng.dot(self.W_recur, hprev[i]).named("W_rec_dot_h{}".format(i))
                h = self.activation(d + h_ff_s[i] + self.b)
                h.name = "activ{}".format(i)
                hprev.append(h)

        rnn_out = ng.Stack(hprev[1:], self.time_axis, pos=1)
        return rnn_out
