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
from ngraph.frontends.neon.axis import ar


def output_dim(X, S, padding, strides, pooling=False):
    """
    Compute along 1 dimension, with these sizes, what will be the output dimension.

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        strides (int): striding
        pooling (bool): flag for setting pooling layer size
    """
    size = ((X - S + 2 * padding) // strides) + 1

    if pooling and padding >= S:
        raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

    return size


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
        in_axes = in_obj.axes.sample_axes() - in_obj.axes.recurrent_axes()
        out_axes -= out_axes.recurrent_axes()

        w_axes = out_axes + [axis - 1 for axis in in_axes]
        if self.W is None:
            self.W = ng.variable(axes=w_axes, initial_value=self.init(w_axes))

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
    def train_outputs(self, in_obj):
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

        self.W = ng.variable(axes=self.w_axes,
                             initial_value=self.lut_init(
                                 self.w_axes, self.lut_v_axis, self.pad_idx)
                             ).named('W')

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

        self.role_order = (ar.features_input, ar.features_0,
                           ar.features_1, ar.features_2, ar.batch)
        self.filter_roles = self.role_order[:-1] + (ar.features_output,)

        self.init = init
        self.f_axes = None
        self.o_axes = None
        self.W = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
        cpm = self.convparams.copy()
        in_obj = ng.axes_with_role_order(in_obj, self.role_order)
        in_axes = in_obj.axes

        if self.f_axes is None:
            self.f_axes = ng.make_axes([in_axes[0]])
            for nm, role in zip('TRSK', self.filter_roles[1:]):
                self.f_axes += ng.make_axis(roles=[role], length=cpm[nm]).named(nm)
            self.W = ng.variable(axes=self.f_axes, initial_value=self.init(self.f_axes))

        if self.o_axes is None:
            self.o_axes = ng.make_axes([
                ng.make_axis(roles=a.roles).named(a.short_name) for a in in_axes if not a.is_batch
            ])
            # set lengths
            out_shape = [
                self.f_axes[-1].length,
                output_dim(in_axes[1].length, cpm['T'], cpm['pad_d'], cpm['str_d']),
                output_dim(in_axes[2].length, cpm['R'], cpm['pad_h'], cpm['str_h']),
                output_dim(in_axes[3].length, cpm['S'], cpm['pad_w'], cpm['str_w'])
            ]
            self.o_axes.set_shape(out_shape)
            self.o_axes += in_axes.batch_axes()

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

        self.role_order = (ar.features_input, ar.features_0,
                           ar.features_1, ar.features_2, ar.batch)
        self.o_axes = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
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
    def train_outputs(self, in_obj):
        if self.init:
            w_axes = in_obj.axes.sample_axes()
            if self.shared and len(in_obj.axes.role_axes(ar.features_input)) != 0:
                w_axes = in_obj.axes.role_axes(ar.features_input)

            self.W = self.W or ng.variable(axes=w_axes, initial_value=self.init(w_axes))
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

    def __init__(self, fshape, filter_init, strides=1, padding=0, bias_init=None, activation=None,
                 batch_norm=False, **kwargs):
        self.conv = Conv2D(fshape, filter_init, strides, padding, **kwargs)
        self.bias = Bias(init=bias_init)
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
        # rho needs to be allocated storage because it will be changed dynamically during tuning
        self.rho = ng.persistent_tensor(axes=(), initial_value=rho).named('rho')
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.gmean = None
        self.gvar = None

    @ng.with_op_metadata
    def train_outputs(self, in_obj):
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

        with ng.sequential_op_factory() as pf:
            xmean = ng.mean(in_obj, reduction_axes=red_axes)
            xvar = ng.variance(in_obj, reduction_axes=red_axes)
            ng.assign(self.gmean, self.gmean * self.rho + xmean * (1.0 - self.rho))
            ng.assign(self.gvar, self.gvar * self.rho + xvar * (1.0 - self.rho))
            self.gamma * (in_obj - xmean) / ng.sqrt(xvar + self.eps) + self.beta
        return pf()

    def set_tuning_iteration(self, batch_index):
        # Following tuning, one must divide self.gvar by rho in order to debias
        self.rho.value[()] = float(batch_index) / (batch_index + 1.0)

    def inference_outputs(self, in_obj):
        return self.gamma * (in_obj - self.gmean) / ng.sqrt(self.gvar + self.eps) + self.beta


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
    def train_outputs(self, in_obj):
        in_axes = in_obj.axes.sample_axes()
        self.mask = self.mask or ng.persistent_tensor(axes=in_axes).named('mask')
        self.mask = ng.uniform(self.mask, low=0.0, high=1.0) <= self.keep
        return self.mask * in_obj

    def inference_outputs(self, in_obj):
        return self.keep * in_obj


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
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
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
                 reset_cells=False, return_sequence=True, backward=False, **kwargs):
        super(Recurrent, self).__init__(**kwargs)

        self.nout = nout
        self.activation = activation
        self.init = init
        self.init_inner = init_inner or init
        self.reset_cells = reset_cells
        self.return_sequence = return_sequence
        self.backward = backward

    @ng.with_op_metadata
    def train_outputs(self, in_obj, init_state=None):
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
        in_axes = in_obj.axes

        self.time_axis = in_axes.recurrent_axes()[0]
        self.time_axis_idx = in_axes.index(self.time_axis)

        # if init state is given, use that as hidden axes
        if init_state:
            hidden_axes = init_state.axes.sample_axes() - init_state.axes.recurrent_axes()
        else:
            if self.axes is not None:
                hidden_axes = self.axes - self.axes.recurrent_axes()
            else:
                hidden_axes = ng.make_axes([ng.make_axis(self.nout).named('Hidden')])

        # using the axes to create weight matrices
        w_in_axes = hidden_axes + [axis - 1 for axis in in_axes.sample_axes() -
                                   in_axes.recurrent_axes()]
        w_re_axes = hidden_axes + [axis - 1 for axis in hidden_axes]

        hidden_state_axes = hidden_axes + in_axes.batch_axes()

        self.W_input = ng.variable(axes=w_in_axes,
                                   initial_value=self.init(w_in_axes)
                                   ).named("W_in")
        self.W_recur = ng.variable(axes=w_re_axes,
                                   initial_value=self.init_inner(w_re_axes)
                                   ).named("W_re")
        self.b = ng.variable(axes=hidden_axes, initial_value=0).named("bias")

        # initialize the hidden states
        if init_state is not None:
            self.h_init = init_state
        else:
            if self.reset_cells:
                self.h_init = ng.constant(const=0, axes=hidden_state_axes).named('h_init')
            else:
                self.h_init = ng.variable(initial_value=0, axes=hidden_state_axes).named('h_init')

        h_list = [self.h_init]

        # feedforward computation
        in_s = get_steps(in_obj, self.time_axis, self.backward)

        # recurrent computation
        for i in range(self.time_axis.length):
            with ng.metadata(recurrent_step=str(i)):
                h_ff = ng.dot(self.W_input, in_s[i]).named("W_in_dot_in_{}".format(i))
                h_rec = ng.dot(self.W_recur, h_list[i]).named("W_rec_dot_h_{}".format(i))
                h = self.activation(h_rec + h_ff + self.b).named("h_{}".format(i))
                h_list.append(h)

        if self.return_sequence is True:
            h_list = h_list[1:][::-1] if self.backward else h_list[1:]
            rnn_out = ng.stack(h_list, self.time_axis, pos=self.time_axis_idx)
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
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        return_sequence (bool): default to be True to return the whole sequence output.
        sum_out (bool): default to be False to return both directional outputs in a list.
                        When True, sum the outputs from both directions, so it can go to
                        following fully connected layers.
        name (str, optional): name to refer to this layer as.
    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """
    metadata = {'layer_type': 'birnn'}

    def __init__(self, nout, init, init_inner=None, activation=None,
                 reset_cells=False, return_sequence=True, sum_out=False, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.sum_out = sum_out
        self.nout = nout
        self.fwd_rnn = Recurrent(nout, init, init_inner, activation=activation,
                                 reset_cells=reset_cells, return_sequence=return_sequence)
        self.bwd_rnn = Recurrent(nout, init, init_inner, activation=activation,
                                 reset_cells=reset_cells, return_sequence=return_sequence,
                                 backward=True)

    @ng.with_op_metadata
    def train_outputs(self, in_obj, init_state=None):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                 information for layer
            init_state (Tensor or list): object that provides initial state

        Returns:
            rnn_out (Tensor): output

        """
        if isinstance(in_obj, list) and len(in_obj) == 2:
            # make sure these 2 streams of inputs share axes
            assert in_obj[0].axes == in_obj[1].axes
            fwd_in = in_obj[0]
            bwd_in = in_obj[1]
            in_axes = in_obj[0].axes
        else:
            fwd_in = in_obj
            bwd_in = in_obj
            in_axes = in_obj.axes

        if isinstance(init_state, list) and len(init_state) == 2:
            assert init_state[0].axes == init_state[1].axes
            fwd_init = init_state[0]
            bwd_init = init_state[1]
        else:
            fwd_init = init_state
            bwd_init = init_state

        # create the hidden axes here and set for both directions
        rnn_axes = ng.make_axes([ng.make_axis(self.nout).named('Hidden'),
                                 in_axes.recurrent_axes()[0]])

        self.fwd_rnn.axes = self.bwd_rnn.axes = rnn_axes
        fwd_out = self.fwd_rnn.train_outputs(fwd_in, fwd_init)
        bwd_out = self.bwd_rnn.train_outputs(bwd_in, bwd_init)

        if self.sum_out:
            return fwd_out + bwd_out
        else:
            return [fwd_out, bwd_out]
