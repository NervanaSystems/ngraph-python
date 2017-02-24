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

import collections
from contextlib import contextmanager
from cachetools import cached, keys
import ngraph as ng
from ngraph.frontends.neon.axis import ar


def output_dim(X, S, padding, strides, pooling=False, dilation=1):
    """
    Compute along 1 dimension, with these sizes, what will be the output dimension.

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        strides (int): striding
        pooling (bool): flag for setting pooling layer size
        dilation (int): dilation of filter
    """

    S = dilation * (S - 1) + 1
    size = ((X - S + 2 * padding) // strides) + 1

    if pooling and padding >= S:
        raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

    return size


def inference_mode(*args, **kwargs):
    """
    cachetools.cached key function to ensure that caching takes into account the current value of
    Layer.inference_mode.
    """

    # If the value is provided, just use that instead of the global flag.
    if "inference_mode" not in kwargs:
        kwargs["inference_mode"] = Layer.inference_mode

    return keys.hashkey(*args, **kwargs)


class Layer(object):
    inference_mode = False

    def __init__(self, name=None):
        self.name = name

    def __call__(self, in_obj, inference):
        raise NotImplementedError()

    @staticmethod
    @contextmanager
    def inference_mode_on():
        Layer.inference_mode = True
        yield Layer.inference_mode
        Layer.inference_mode = False


class Preprocess(Layer):

    def __init__(self, functor, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        self.functor = functor

    @cached({})
    def __call__(self, in_obj):
        return self.functor(in_obj)


class Linear(Layer):
    metadata = {'layer_type': 'linear'}

    def __init__(self, init, nout=None, axes=None, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.axes = axes
        self.nout = nout
        self.init = init
        if self.axes is None:
            assert(self.nout is not None), "Must provide either axes or nout to Linear"
        self.W = None

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj):
        if self.W is None:
            out_axes = ng.make_axes(self.axes or [ng.make_axis(self.nout).named('Hidden')])
            in_axes = in_obj.axes.sample_axes() - in_obj.axes.recurrent_axes()
            out_axes -= out_axes.recurrent_axes()

            w_axes = out_axes + [axis - 1 for axis in in_axes]
            self.W = ng.variable(axes=w_axes, initial_value=self.init).named('LinW')

        return ng.dot(self.W, in_obj)


class LookupTable(Layer):
    """
    Lookup table layer that often is used as word embedding layer

    Args:
        vocab_size (int): the vocabulary size
        embed_dim (int): the size of embedding vector
        init (Initializor): initialization function
        update (bool): if the word vectors get updated through training
        pad_idx (int): by knowing the pad value, the update will make sure always
                       have the vector representing pad value to be 0s.

    """
    metadata = {'layer_type': 'lookuptable'}

    def __init__(self, vocab_size, embed_dim, init, update=True, pad_idx=None, **kwargs):
        super(LookupTable, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init = init
        self.update = update
        self.pad_idx = pad_idx
        self.role_order = (ar.time, ar.batch)
        self.W = None

    def lut_init(self, axes, pad_word_axis, pad_idx):
        """
        Initialization function for the lut.
        After using the initialization to fill the whole array, set the part that represents
        padding to be 0.
        """
        init_w = self.init(axes)
        if pad_word_axis is 0:
            init_w[pad_idx] = 0
        else:
            init_w[:, pad_idx] = 0
        return init_w

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj):
        """
        Arguments:
            in_obj (Tensor): object that provides the lookup indices
        """
        in_obj.axes.find_by_short_name('time')[0].add_role(ar.time)
        in_obj.axes.find_by_short_name('time')[0].is_recurrent = True
        in_obj = ng.axes_with_role_order(in_obj, self.role_order)
        in_obj = ng.flatten(in_obj)
        in_axes = in_obj.axes

        self.lut_v_axis = ng.make_axis(self.vocab_size).named('V')
        self.lut_f_axis = ng.make_axis(self.embed_dim).named('F')

        self.w_axes = ng.make_axes([self.lut_v_axis, self.lut_f_axis])
        self.lut_o_axes = in_axes + ng.make_axes([self.lut_f_axis])
        self.o_axes = ng.make_axes([self.lut_f_axis]) + in_axes[0].axes

        if self.W is None:
            self.W = ng.variable(axes=self.w_axes,
                                 initial_value=self.lut_init(
                                     self.w_axes, self.lut_v_axis, self.pad_idx)
                                 ).named('LutW')

        lut_result = ng.lookuptable(self.W, in_obj, self.lut_o_axes, update=self.update,
                                    pad_idx=self.pad_idx)
        return ng.axes_with_order(ng.unflatten(lut_result), self.o_axes)


class ConvBase(Layer):
    """
    Convolutional layer that requires explicit binding of all spatial roles

    Args:
        fshape (dict): filter shape -- must contain keys 'T', 'R', 'S', 'K'
        init (function): function for later initializing filters
        strides (dict): stride specification -- must contain keys 'str_d', 'str_h', 'str_w'
        padding (dict): pad specification -- must contain keys 'pad_d', 'pad_h', 'pad_w'
        dilation (dict): dilation specification -- must contain keys 'dil_d', 'dil_h', 'dil_w'

    """
    metadata = {'layer_type': 'convolution'}

    def __init__(self, fshape, init, strides, padding, dilation, **kwargs):
        super(ConvBase, self).__init__(**kwargs)
        self.convparams = dict(T=None, R=None, S=None, K=None,
                               pad_h=None, pad_w=None, pad_d=None,
                               str_h=None, str_w=None, str_d=None,
                               dil_h=None, dil_w=None, dil_d=None)

        for d in [fshape, strides, padding, dilation]:
            self.convparams.update(d)

        missing_keys = [k for k, v in self.convparams.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError("Missing conv keys: {}".format(missing_keys))

        self.role_order = (ar.features_input, ar.features_0,
                           ar.features_1, ar.features_2, ar.batch)
        self.filter_roles = self.role_order[:-1] + (ar.features_output,)

        self.init = init
        self.f_axes = None
        self.o_axes = None
        self.W = None

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj):
        cpm = self.convparams.copy()
        in_obj = ng.axes_with_role_order(in_obj, self.role_order)
        in_axes = in_obj.axes

        if self.f_axes is None:
            self.f_axes = ng.make_axes([in_axes[0]])
            for nm, role in zip('TRSK', self.filter_roles[1:]):
                self.f_axes += ng.make_axis(roles=[role], length=cpm[nm]).named(nm)
            self.W = ng.variable(axes=self.f_axes, initial_value=self.init).named('convwt')

        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.make_axis(roles=a.roles).named(a.short_name) for a in in_axes if not a.is_batch
            ])
            # set lengths
            out_shape = [
                self.f_axes[-1].length,
                output_dim(in_axes[1].length, cpm['T'], cpm['pad_d'], cpm['str_d'], False,
                           cpm['dil_d']),
                output_dim(in_axes[2].length, cpm['R'], cpm['pad_h'], cpm['str_h'], False,
                           cpm['dil_h']),
                output_dim(in_axes[3].length, cpm['S'], cpm['pad_w'], cpm['str_w'], False,
                           cpm['dil_w'])
            ]
            self.o_axes.set_shape(out_shape)
            self.o_axes += in_axes.batch_axes()

        return ng.convolution(cpm, in_obj, self.W, axes=self.o_axes)


class Conv2D(ConvBase):

    def __init__(self, fshape, init, strides, padding, dilation, **kwargs):
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
        if isinstance(dilation, int):
            dilation = {'dil_h': dilation, 'dil_w': dilation, 'dil_d': 1}

        super(Conv2D, self).__init__(fshape, init, strides, padding, dilation, **kwargs)


class Activation(Layer):
    metadata = {'layer_type': 'activation'}

    def __init__(self, transform, **kwargs):
        self.transform = transform
        super(Activation, self).__init__(**kwargs)

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj):
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

        self.role_order = (ar.features_input, ar.features_0,
                           ar.features_1, ar.features_2, ar.batch)
        self.o_axes = None

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj):
        ppm = self.poolparams.copy()
        in_obj = ng.axes_with_role_order(in_obj, self.role_order)
        in_axes = in_obj.axes

        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.make_axis(roles=a.roles).named(a.short_name) for a in in_axes if not a.is_batch
            ])
            # set lengths
            out_shape = [
                output_dim(in_axes[0].length, ppm['J'], ppm['pad_d'], ppm['str_d']),
                output_dim(in_axes[1].length, ppm['T'], ppm['pad_d'], ppm['str_d']),
                output_dim(in_axes[2].length, ppm['R'], ppm['pad_h'], ppm['str_h']),
                output_dim(in_axes[3].length, ppm['S'], ppm['pad_w'], ppm['str_w'])
            ]
            self.o_axes.set_shape(out_shape)
            self.o_axes += in_axes.batch_axes()
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
    @cached({})
    def __call__(self, in_obj):
        if self.init:
            w_axes = in_obj.axes.sample_axes()
            if self.shared and len(in_obj.axes.role_axes(ar.features_input)) != 0:
                w_axes = in_obj.axes.role_axes(ar.features_input)

            self.W = self.W or ng.variable(axes=w_axes, initial_value=self.init)
            return in_obj + self.W
        else:
            return in_obj


class Affine(Layer):

    def __init__(self, weight_init, nout=None, bias_init=None, activation=None,
                 batch_norm=False, **kwargs):
        self.linear = Linear(init=weight_init, nout=nout, **kwargs)
        self.bias = Bias(init=bias_init)
        self.batch_norm = BatchNorm() if batch_norm else None
        self.activation = Activation(transform=activation)

    def __call__(self, in_obj):
        l_out = self.linear(in_obj)
        b_out = self.bias(l_out)
        bn_out = self.batch_norm(b_out) if self.batch_norm else b_out
        return self.activation(bn_out)


class Convolution(Layer):

    def __init__(self, fshape, filter_init, strides=1, padding=0, dilation=1, bias_init=None,
                 activation=None, batch_norm=False, **kwargs):
        self.conv = Conv2D(fshape, filter_init, strides, padding, dilation, **kwargs)
        self.bias = Bias(init=bias_init)
        self.batch_norm = BatchNorm() if batch_norm else None
        self.activation = Activation(transform=activation)

    def __call__(self, in_obj):
        l_out = self.conv(in_obj)
        b_out = self.bias(l_out)
        bn_out = self.batch_norm(b_out) if self.batch_norm else b_out
        return self.activation(bn_out)


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
        # rho needs to be allocated storage because it will be changed dynamically during tuning
        self.rho = ng.persistent_tensor(axes=(), initial_value=rho).named('rho')
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.gmean = None
        self.gvar = None

    @ng.with_op_metadata
    @cached({}, key=inference_mode)
    def __call__(self, in_obj):

        in_axes = in_obj.axes.sample_axes()
        red_axes = ng.make_axes()
        if len(in_axes.role_axes(ar.features_input)) != 0:
            red_axes += in_axes.sample_axes() - in_axes.role_axes(ar.features_input)
        red_axes += in_obj.axes.batch_axes()
        out_axes = in_axes - red_axes

        self.gamma = self.gamma or ng.variable(axes=out_axes, initial_value=1.0).named('gamma')
        self.beta = self.beta or ng.variable(axes=out_axes, initial_value=0.0).named('beta')
        self.gvar = self.gvar or ng.persistent_tensor(axes=out_axes, initial_value=1.0)
        self.gmean = self.gmean or ng.persistent_tensor(axes=out_axes, initial_value=0.0)

        xmean = ng.mean(in_obj, reduction_axes=red_axes)
        xvar = ng.variance(in_obj, reduction_axes=red_axes)
        if Layer.inference_mode:
            return self.gamma * (in_obj - self.gmean) / ng.sqrt(self.gvar + self.eps) + self.beta
        else:
            return ng.sequential([
                ng.assign(self.gmean, self.gmean * self.rho + xmean * (1.0 - self.rho)),
                ng.assign(self.gvar, self.gvar * self.rho + xvar * (1.0 - self.rho)),
                self.gamma * (in_obj - xmean) / ng.sqrt(xvar + self.eps) + self.beta
            ])

    def set_tuning_iteration(self, batch_index):
        # Following tuning, one must divide self.gvar by rho in order to debias
        self.rho.value[()] = float(batch_index) / (batch_index + 1.0)


class Dropout(Layer):
    """
    Layer for stochastically dropping activations to prevent overfitting
    Args:
        keep (float):  Number between 0 and 1 that indicates probability of any particular
                       activation being dropped.  Default 0.5.
    """

    metadata = {'layer_type': 'dropout'}

    def __init__(self, keep=0.5, **kwargs):
        self.keep = keep
        self.mask = None

    @ng.with_op_metadata
    @cached({}, key=inference_mode)
    def __call__(self, in_obj):
        if Layer.inference_mode:
            return self.keep * in_obj
        else:
            if self.mask is None:
                in_axes = in_obj.axes.sample_axes()
                self.mask = ng.persistent_tensor(axes=in_axes).named('mask')
            self.mask = ng.uniform(self.mask, low=0.0, high=1.0) <= self.keep
            return self.mask * in_obj


def get_steps(x, time_axis, backward=False):
    time_iter = list(range(time_axis.length))
    if backward:
        time_iter = reversed(time_iter)
    return [ng.slice_along_axis(x, time_axis, i) for i in time_iter]


class Recurrent(Layer):
    """
    Basic recurrent layer.
    Arguments:
        nout (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be True to make the layer stateless,
                            set to False to be stateful.
        return_sequence (bool): default to be True to return the whole sequence output.
        backward (bool): default to be False to process the sequence left to right
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """
    metadata = {'layer_type': 'recurrent'}

    def __init__(self, nout, init, init_inner=None, activation=None,
                 reset_cells=True, return_sequence=True, backward=False, **kwargs):
        super(Recurrent, self).__init__(**kwargs)

        self.nout = nout
        self.activation = activation
        self.init = init
        self.init_inner = init_inner if init_inner is not None else init
        self.reset_cells = reset_cells
        self.return_sequence = return_sequence
        self.backward = backward

    def interpret_axes(self, in_obj, init_state):
        in_axes = in_obj.axes

        self.recurrent_axis = in_axes.recurrent_axes()[0]

        # if init state is given, use that as hidden axes
        if init_state:
            self.hidden_axes = init_state.axes.sample_axes() - init_state.axes.recurrent_axes()
            if sum(self.hidden_axes.full_lengths) != self.nout:
                raise ValueError("Length of init_state must be the same as nout: " +
                                 "{} != {}".format(sum(self.hidden_axes.full_lengths),
                                                   self.nout))
        else:
            self.hidden_axes = ng.make_axes([ng.make_axis(self.nout).named('Hidden')])

        self.hidden_state_axes = self.hidden_axes + in_axes.batch_axes()
        self.recurrent_axis_idx = len(self.hidden_axes)

        # using the axes to create weight matrices
        self.w_in_axes = self.hidden_axes + [axis - 1 for axis in (in_axes.sample_axes()
                                                                   - self.recurrent_axis)]

        self.w_re_axes = self.hidden_axes + [axis - 1 for axis in self.hidden_axes]

    def _step(self, inp, states):
        h_ff = ng.dot(self.W_input, inp)
        h_rec = ng.dot(self.W_recur, states)
        h = self.activation(h_rec + h_ff + self.b)
        return h

    @ng.with_op_metadata
    @cached({}, key=inference_mode)
    def __call__(self, in_obj, init_state=None):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (Tensor): object that provides initial state

        Returns:
            rnn_out (Tensor): output

        """
        # try to understand the axes from the input
        self.interpret_axes(in_obj, init_state)

        # initialize the hidden states
        if init_state is not None:
            self.h_init = init_state
        else:
            if self.reset_cells:
                self.h_init = ng.constant(
                    const=0, axes=self.hidden_state_axes).named('h_init')
            else:
                self.h_init = ng.variable(
                    initial_value=0, axes=self.hidden_state_axes).named('h_init')

        self.W_input = ng.variable(axes=self.w_in_axes,
                                   initial_value=self.init).named("W_in")
        self.W_recur = ng.variable(axes=self.w_re_axes,
                                   initial_value=self.init_inner).named("W_re")
        self.b = ng.variable(axes=self.hidden_axes, initial_value=0).named("bias")

        h = self.h_init
        h_list = []

        # slice the inputs into time slices
        in_s = get_steps(in_obj, self.recurrent_axis, self.backward)

        # unrolling computations
        for i in range(self.recurrent_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                h = self._step(in_s[i], h)
                h_list.append(h)

        if self.return_sequence is True:
            # only when returning a sequence, need to reverse the output
            h_list = h_list[::-1] if self.backward else h_list
            rnn_out = ng.stack(h_list, self.recurrent_axis, pos=self.recurrent_axis_idx)
        else:
            rnn_out = h_list[-1]

        return rnn_out


class BiRNN(Layer):
    """
    Bi-directional recurrent layer.
    Arguments:
        nout (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be True to make the layer stateless,
                            set to False to be stateful.
        return_sequence (bool): default to be True to return the whole sequence output.
        sum_out (bool): default to be False. When True, sum the outputs from both directions
        concat_out (bool): default to False. When True, concatenate the outputs from both
                           directions. If concat_out and sum_out are both False, output will be a
                           list.
        name (str, optional): name to refer to this layer as.
    """
    metadata = {'layer_type': 'birnn'}

    def __init__(self, nout, init, init_inner=None, activation=None,
                 reset_cells=False, return_sequence=True, sum_out=False,
                 concat_out=False, **kwargs):
        if sum_out and concat_out:
            raise ValueError("sum_out and concat_out cannot both be True")

        super(BiRNN, self).__init__(**kwargs)
        self.sum_out = sum_out
        self.concat_out = concat_out
        self.nout = nout
        self.fwd_rnn = Recurrent(nout, init, init_inner, activation=activation,
                                 reset_cells=reset_cells, return_sequence=return_sequence)
        self.bwd_rnn = Recurrent(nout, init, init_inner, activation=activation,
                                 reset_cells=reset_cells, return_sequence=return_sequence,
                                 backward=True)

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj, init_state=None):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (Tensor or list): object that provides initial state

        Returns:
            if sum_out or concat_out - rnn_out (Tensor): output
            otherwise - rnn_out (list of Tensors): list of length 2

        """
        if isinstance(in_obj, collections.Sequence):
            if len(in_obj) != 2:
                raise ValueError("If in_obj is a sequence, it must have length 2")
            if in_obj[0].axes != in_obj[1].axes:
                raise ValueError("If in_obj is a sequence, each element must have the same axes")
            fwd_in = in_obj[0]
            bwd_in = in_obj[1]
        else:
            fwd_in = in_obj
            bwd_in = in_obj

        if isinstance(init_state, collections.Sequence):
            if len(init_state) != 2:
                raise ValueError("If init_state is a sequence, it must have length 2")
            if init_state[0].axes != init_state[1].axes:
                raise ValueError("If init_state is a sequence, " +
                                 "each element must have the same axes")
            fwd_init = init_state[0]
            bwd_init = init_state[1]
        else:
            fwd_init = init_state
            bwd_init = init_state

        with ng.metadata(direction="fwd"):
            fwd_out = self.fwd_rnn(fwd_in, fwd_init)
        with ng.metadata(direction="bwd"):
            bwd_out = self.bwd_rnn(bwd_in, bwd_init)

        if self.sum_out:
            return fwd_out + bwd_out
        elif self.concat_out:
            ax_list = list()
            for out in [fwd_out, bwd_out]:
                axes = out.axes.sample_axes() - out.axes.recurrent_axes()
                if len(axes) == 1:
                    ax_list.append(axes[0])
                else:
                    raise ValueError("Multiple hidden axes. Unable to concatenate automatically")
            return ng.ConcatOp([fwd_out, bwd_out], ax_list)
        else:
            return [fwd_out, bwd_out]


class LSTM(Recurrent):
    """
    Long Short-Term Memory (LSTM) layer based on
    Hochreiter and Schmidhuber, Neural Computation 9(8): 1735-80 (1997).

    Arguments:
        nout (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be True to make the layer stateless,
                            set to False to be stateful.
        return_sequence (bool): default to be True to return the whole sequence output.
        backward (bool): default to be False to process the sequence left to right
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (output_size, input_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)

    Gates: i - input gate, f - forget gate, o - output gate, g - input modulation

    """
    metadata = {'layer_type': 'LSTM',
                'gates': ['i', 'f', 'o', 'g']}

    def __init__(self, nout, init, init_inner=None, activation=None, gate_activation=None,
                 reset_cells=True, return_sequence=True, backward=False, **kwargs):
        super(LSTM, self).__init__(nout, init, init_inner, activation, reset_cells,
                                   return_sequence, backward, **kwargs)

        self.gate_activation = gate_activation

    def _step(self, inp, states):
        h_state = states[0]
        c_state = states[1]

        ifog = {k: ng.dot(self.W_input[k], inp) + ng.dot(self.W_recur[k], h_state)
                + self.b[k] for k in self.metadata['gates']}

        ifog_act = {k: self.activation(ifog[k]) if k is 'g'
                    else self.gate_activation(ifog[k]) for k in self.metadata['gates']}

        c = ifog_act['f'] * c_state + ifog_act['i'] * ifog_act['g']
        # c_prev is the state before applying activation
        h = ifog_act['o'] * self.activation(c)
        return [h, c]

    @ng.with_op_metadata
    @cached({})
    def __call__(self, in_obj, init_state=None):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (tuple of Tensor): object that provides initial state, and in LSTM,
                                          it includes hidden state, and cell states

        Returns:
            rnn_out (Tensor): output

        """
        # try to understand the axes from the input
        if init_state is not None:
            assert len(init_state) == 2 and init_state[0].axes == init_state[1].axes
            self.interpret_axes(in_obj, init_state[0])
        else:
            self.interpret_axes(in_obj, init_state)

        # initialize the hidden states
        if init_state is not None:
            self.h_init = init_state[0]
            self.c_init = init_state[1]
        else:
            if self.reset_cells:
                self.h_init = ng.temporary(initial_value=0,
                                           axes=self.hidden_state_axes).named('h_init')
                self.c_init = ng.temporary(initial_value=0,
                                           axes=self.hidden_state_axes).named('c_init')
            else:
                self.h_init = ng.variable(initial_value=0,
                                          axes=self.hidden_state_axes).named('h_init')
                self.c_init = ng.variable(initial_value=0,
                                          axes=self.hidden_state_axes).named('c_init')

        # params are dictionary for i, f, o, g
        self.W_input = {k: ng.variable(axes=self.w_in_axes,
                                       initial_value=self.init).
                        named("W_in_{}".format(k)) for k in self.metadata['gates']}

        self.W_recur = {k: ng.variable(axes=self.w_re_axes,
                                       initial_value=self.init_inner).
                        named("W_re_{}".format(k)) for k in self.metadata['gates']}

        self.b = {k: ng.variable(axes=self.hidden_axes,
                                 initial_value=0).
                  named("bias_{}".format(k)) for k in self.metadata['gates']}

        h = self.h_init
        c = self.c_init

        h_list = []
        c_list = []

        # feedforward computation
        in_s = get_steps(in_obj, self.recurrent_axis, self.backward)

        # recurrent computation
        for i in range(self.recurrent_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                [h, c] = self._step(in_s[i], [h, c])
                h_list.append(h)
                c_list.append(c)

        if self.return_sequence is True:
            if self.backward:
                h_list = h_list[::-1]
                c_list = c_list[::-1]
            lstm_out = ng.stack(h_list, self.recurrent_axis, pos=self.recurrent_axis_idx)
        else:
            lstm_out = h_list[-1]

        if self.reset_cells is True:
            return lstm_out
        else:
            return ng.sequential([
                ng.doall([
                    ng.assign(self.h_init, h_list[-1]),
                    ng.assign(self.c_init, c_list[-1])
                ]),
                lstm_out
            ])
