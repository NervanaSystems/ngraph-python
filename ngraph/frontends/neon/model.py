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
from __future__ import division

from operator import itemgetter
from ngraph.frontends.neon.graph import SubGraph
import ngraph as ng


class Parallel(SubGraph):
    """
    Parallel is a container of layers. each of which operates on the same input
    Output of each branch is concatenated to form a larger tensor

    Arguments:
        branches: List of layers that will operate on the same input
        name: name to be used with selector

    Example:
    .. code-block:: python
        branches = [Convolution(name='br1', filter_shape=(3, 3, 16)),
                    Convolution(name='br2', filter_shape=(3, 3, 32))]
        par1 = Parallel(branches, concat_axis=branches[0].axes.channel_axis())
        output = par1(input)

    The above code is equivalent of doing
    .. code-block:: python
        br1 = Convolution(name='br1', filter_shape=(3, 3, 16)
        br2 = Convolution(name='br2', filter_shape=(3, 3, 32)
        br1_out = br1(input)
        br2_out = br2(input)
        output = [br1_out, br2_out]
        output = ng.concat_along_axis(output, br1_out.axes.channel_axis())

    """
    # TODO: option to sum the outputs of branches rather than concatenate
    def __init__(self, branches, name=None, mode='concat', **kwargs):
        super(Parallel, self).__init__(name=name, **kwargs)
        self.branches = branches
        self.mode = mode
        if not ((self.mode is None) or (self.mode == 'concat')):
            raise NotImplementedError("Unrecognized mode:{}".format(str(self.mode)))

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, merge_axis=None):
        outputs = [branch(in_obj) for branch in self.branches]
        if (type(merge_axis) == str):
            merge_axis = ng.make_axis(name=merge_axis)

        if self.mode == 'concat':
            # Concatenate along the given axis
            if merge_axis is None:
                merge_axis = outputs[0].axes.channel_axis()
            outputs = ng.concat_along_axis(outputs, merge_axis)
        elif self.mode is None:
            # Return the output list directly
            pass
        else:
            pass
        return outputs


class Sequential(SubGraph):
    """
    Sequential is a container of layers that passes data through the layers in series.

    Arguments:
        layers: List of different layers in the network
        name: name to be used with selector

    Example:
    .. code-block:: python
        layers = [
                Preprocess(functor=cifar10_mean_subtract),
                Convolution((7, 7, 64), activation=Rectlin(), filter_init=KaimingInit())]
        seq1 = Sequential(layers)
        output = seq1(input)

    The above code is equivalent of doing
        preprocess = Preprocess(functor=cifar10_mean_subtract)
        conv = Convolution((7, 7, 64), activation=Rectlin(), filter_init=KaimingInit())
        x = preprocess(input)
        output = conv(x)
    """
    def __init__(self, layers, name=None, **kwargs):
        super(Sequential, self).__init__(name=name, **kwargs)
        self.layers = layers

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, **kwargs):
        for l in self.layers:
            in_obj = l(in_obj, **kwargs)
        return in_obj


class Container(object):
    """
    POC code only

    Two string->`Op` dictionaries representing a container of op_graphs
    """
    def __init__(self, inputs=dict(), outputs=dict()):
        self.inputs = inputs
        self.outputs = outputs

    def add(self, rhs):
        new_inputs = self.inputs.copy()
        new_outputs = self.outputs.copy()
        # these label -> Op mappings are
        # still pointing to the same ops
        new_inputs.update(rhs.inputs)
        new_outputs.update(rhs.outputs)
        return Container(new_inputs, new_outputs)

    def subset(self, inputs=None, outputs=None):
        """
        Eventually, a user should be able to subset using op_graph `Selectors`
        similar to XPath or Jquery selectors. Boolean combinations of op type,
        op name regex, and upstream, downstream from given ops.
        Here we just have selection by exact name.
        """
        return Container({k: v for k, v in self.inputs.items() if k in inputs},
                         {k: v for k, v in self.outputs.items() if k in outputs})


class BoundComputation(object):
    """
    Callable object that has inputs and outputs of a computation bound via names

    This class allows to define the computation using named inputs and outputs.
    Inputs are then provided as a dictionary and outputs are returned as a
    dictionary.

    Arguments:
        transformer (object): Transformer object defined in the model
        named_outputs (dict): Output entities wanted for the computation
        named_inputs (dict): Input entities needed for the computation
    """
    def __init__(self, transformer, named_outputs, named_inputs):
        self.input_keys = tuple(sorted(named_inputs.keys()))

        self.output_keys = tuple(sorted(named_outputs.keys()))

        outputs = itemgetter(*self.output_keys)(named_outputs)
        outputs = [outputs] if len(self.output_keys) == 1 else list(outputs)

        inputs = itemgetter(*self.input_keys)(named_inputs)
        inputs = [inputs] if len(self.input_keys) == 1 else list(inputs)
        self.num_outputs = len(outputs)
        self.comp_func = transformer.computation(outputs, *inputs)

    def __call__(self, named_buffers):
        inputs = itemgetter(*self.input_keys)(named_buffers)
        inputs = [inputs] if len(self.input_keys) == 1 else list(inputs)
        result_tuple = self.comp_func(*inputs)
        result_dict = {k: v for k, v in zip(self.output_keys, result_tuple)}
        return result_dict


def make_bound_computation(transformer, named_outputs, named_inputs):
    """
    Creates a `BoundComputation` instance that takes named input arrays
    and returns named output arrays.

    Arguments:
        transformer (object): Transformer object defined in the model
        named_outputs (dict): Output entities wanted for the computation
        named_inputs (dict): Input entities needed for the computation

    Example:
        .. code-block:: python
        # inputs is a dictionary of placeholders
        train_outputs = {'batch_cost': mean_cost, 'updates': updates}
        train_inputs = {'mini_batch': inputs['input_data']}
        train_computation = make_bound_compuation(transformer,
                                                  train_outputs,
                                                  train_inputs)
        output_dict = train_computation({"minibatch": 0})
    """

    return BoundComputation(transformer, named_outputs, named_inputs)


class ResidualModule(object):
    """
    Creates a Residual object which takes in two parallel paths and returns their
    element-wise sum.
    It assumes that both the parallel paths have same dimensions.
    If side_path is None then the original input is added to main_path output

    Arguments:
        main_path: This path typically contains Conv layers
        side_path: This path implements the skip connections which can be direct mapping or
                    1x1 convs for matching dimensions
    Example:
    .. code-block:: python
        layers = [
                Preprocess(functor=cifar10_mean_subtract),
                Convolution((7, 7, 64), activation=Rectlin(), filter_init=KaimingInit())
                ResidualModule(main_path, side_path)]

    TODO:
    When Gokce Keskin merges inception model, inherit from Parallel class and add "sum" mode with
    2 branches being main_path and side_path
    https://github.com/NervanaSystems/private-ngraph/issues/2176
    """
    def __init__(self, main_path, side_path=None):
        self.main_path = main_path
        self.side_path = side_path

    def __call__(self, in_obj):
        # Computes the output of main path. Parallel path 1
        mp = self.main_path(in_obj)
        # Computes the output of side path. Parallel path 2
        sp = in_obj if (self.side_path is None) else self.side_path(in_obj)
        # Check if their dimensions match
        if(mp.axes == sp.axes) and (mp.axes.lengths == sp.axes.lengths):
            # Sum both and return
            return mp + sp
        else:
            raise ValueError("Dimensions mismatch. " + str(mp.axes) + " VS " + str(sp.axes))
