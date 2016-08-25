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

from contextlib import contextmanager

import cachetools
import numpy as np
from builtins import object, str

from geon.op_graph.arrayaxes import TensorDescription, \
    AxisIDTuple, Axes, FlattenedAxis, PaddedAxis, Axis, SlicedAxis
from geon.op_graph.nodes import Node
from geon.util.threadstate import get_thread_state
from geon.util.generics import generic_method


def tensor_descriptions(args):
    """
    A list of tensor descriptions for Ops.

    Arguments:
      args: A list of Ops.

    Returns:
      A list of the Op's tensor descriptions.
    """
    return (arg.tensor_description() for arg in args)


class Op(Node):
    """
    Any operation that can be in an AST

    Arguments:
        initializers: List of one-time initializations to run before the op.
        kwargs: Args defined in related classes.

    Attributes:
        schemas: Information about how the Op was generated.
        initializers: A list of additional Ops to run before this Op is run the first time.
    """

    # Default is to not collect Ops as they are created
    get_thread_state().ops = [None]

    @staticmethod
    def _get_thread_ops():
        """
        :return: The stack of Ops being collected.
        """
        return get_thread_state().ops

    @staticmethod
    @contextmanager
    def captured_ops(ops=None):
        """
        Capture all Ops created within the context.

        Arguments:
            ops: List for collecting created ops.

        """
        try:
            Op._get_thread_ops().append(ops)
            yield (ops)
        finally:
            Op._get_thread_ops().pop()

    def __init__(self, initializers=None, **kwargs):
        super(Op, self).__init__(**kwargs)
        self.schemas = []
        self._adjoints = None
        self.initializers = initializers or []

        ops = Op._get_thread_ops()[-1]
        if ops is not None:
            ops.append(self)

    def add_schema(self, schema, set_generate_adjoints=True):
        """
        Adds a description of some op substructure.

        When a function generates a groups of nodes, it can add a schema
        describing the roles of these nodes.  The schema may include its
        own generate_adjoints.

        Arguments:
          schema: param set_generate_adjoints: Whether to override the node's generate_adjoints
        with the version from the schema.
          set_generate_adjoints: TODO

        Returns:
          TODO
        """
        self.schemas.insert(0, schema)
        if set_generate_adjoints:
            # generate_adjoints is normally called with *args, but for a
            # schema we call it with the associated node.
            def generate_adjoints(adjoints, adjoint, *ignore):
                """
                TODO.

                Arguments:
                  adjoints: TODO
                  adjoint: TODO
                  *ignore: TODO
                """
                schema.generate_adjoints(adjoints, adjoint, self)
            # Replace generate_adjoints for self
            self.generate_adjoints = generate_adjoints

    def find_schema(self, t):
        """
        Find a schema of particular type.

        Searches added schema for one of type t.

        Arguments:
          t: The type of schema desired.

        Returns:
          A schema of type t, or None.
        """
        for schema in self.schemas:
            if isinstance(schema, t):
                return schema
        return None

    @property
    def defs(self):
        """TODO."""
        return {}

    def variables(self, trainable=True, filter=None):
        """
        Return all parameters used in computing this node.

        Arguments:
          trainable: TODO
          filter: TODO

        Returns:
          TODO
        """
        params = set()

        if filter is None:
            filter = lambda node: ('trainable' in node.tags) is trainable

        def visitor(node):
            """
            TODO.

            Arguments:
              node: TODO
            """
            if isinstance(node, Variable) and filter(node):
                params.add(node)

        Node.visit_input_closure([self], visitor)

        return set(params)

    @property
    @cachetools.cached({})
    def initial_adjoint(self):
        if len(self.axes) == 0:
            return Constant(1)
        else:
            return placeholder(axes=self.axes)

    @cachetools.cached({})
    def adjoints(self):
        """
        Returns a map containing the adjoints of this op with respect to other
        ops. Creates the map if it does not already exist.  Most models only
        require the adjoints map for their scalar loss functions, in which case
        the adjoint is initialized to a scalar 1.  Some autodiff tests
        calculate the derivative of a tensor by initializing all but one
        elements of a tensor to zero and the remaining element to one.  To
        allow this, we create a placeholder for the initial adjoint and allow
        it to be accessed by the _initial_adjoint field.

        Returns:
          TODO
        """
        adjoints = {
            self: self.initial_adjoint,
        }

        # visit ops in reverse depth first post-order. it is important that
        # ordered_ops returns a copy of this traversal order since the graph
        # may change as we generate adjoints and we don't want to visit those
        # new ops.
        for o in reversed(Op.ordered_ops([self])):
            if o in adjoints:
                adjoint = adjoints[o]
                if o.scale is not None:
                    adjoint = adjoint * o.scale

                o.generate_adjoints(adjoints, adjoint, *o.args)

        return adjoints

    @staticmethod
    def ordered_ops(results):
        """
        depth-first, post-order "Bottom Up" traversal of Ops in results.

        Ops will only appear once in result.

        Arguments:
          results: a list of ops which are the roots of the graph traversal

        Returns:
          list of Ops in depth-first, post-order
        """
        ordered_ops = []
        Node.visit_input_closure(results, lambda o: ordered_ops.append(o))
        return ordered_ops

    def as_node(self, x):
        """
        Overrides a method of the Node superclass.

        Arguments:
          x: TODO

        Returns:
          TODO
        """
        return Op.as_op(x)

    @staticmethod
    def as_op(x):
        """
        Used to cast python values that are captured in the op
        tree so that they can be properly evaluated.

        Arguments:
          x: TODO

        Returns:
          TODO
        """
        if isinstance(x, Op):
            return x

        return Constant(x)

    @staticmethod
    def as_ops(xs):
        """
        TODO.

        Arguments:
          xs: TODO

        Returns:
          TODO
        """
        return tuple(Op.as_op(x) for x in xs)

    @property
    def ops(self):
        """TODO."""
        return []

    @staticmethod
    def simple_prune(results):
        """
        TODO.

        Arguments:
          results: TODO

        Returns:
          TODO
        """
        SimplePrune(results)

    def tensor_description(self):
        return None

    @cachetools.cached({})
    def call_info(self):
        """
        Creates the tensor descriptions (of this op or its arguments)
        required to evaluate it.
        The list is used to allocate buffers (in the transformers)
        and supply values to the transform method
        (in the transform_call_info) method.

        Arguments:

        Returns:
          TODO
        """
        return list(tensor_descriptions(self.args))

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class InitTensor(Op):
    """
    Initializes a device tensor from a CPU tensor.

    Arguments:
        tensor: Tensor to be intialized.
        valfun: Function that performs initialization
        kwargs: Other op args.

    Attributes:
        valfun: A CPU function that produces the initial value for the tensor.

    """
    def __init__(self, tensor, valfun, **kwargs):
        super(InitTensor, self).__init__(args=(tensor,), **kwargs)
        self.valfun = valfun


class SetItem(Op):
    """TODO."""

    def __init__(self, tensor, item, val, **kwargs):
        super(SetItem, self).__init__(args=(tensor, val), **kwargs)
        self.item = item

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        tensor, val = tensor_descriptions(self.args)
        return [tensor, val.reaxe(tensor.axes)]


class doall(Op):
    """TODO."""

    def __init__(self, all, **kwargs):
        super(doall, self).__init__(args=all, **kwargs)

    def call_info(self):
        return []


class Fill(Op):
    """TODO."""

    def __init__(self, tensor, const, **kwargs):
        super(Fill, self).__init__(args=(tensor,), **kwargs)
        self.const = const


class TensorOp(Op):
    """
    Super class for all Ops whose value is a Tensor.

    :ivar dtype: The dtype of the tensor elements.
    :ivar scale: If not None, scale grad by this amount.
    """

    def __init__(self, dtype=None, axes=None, scale=None, **kwds):
        """

        :param dtype: The dtype of the tensor; defaults to np.float32
        :param axes: The tensor axes.
        :param scale: If specified, scales the gradient by this amount.
        :param kwds: Op keywords.
        """
        super(TensorOp, self).__init__(**kwds)
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.dtype = dtype
        if axes is not None:
            axes = Axes(axes)
        self.__axes = axes

        # Derivative will be scaled by this
        self.scale = scale

    def generate_add_delta(self, adjoints, delta):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO

        Returns:
          TODO
        """
        delta = sum(delta, reduction_axes=delta.axes - self.axes)
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta + adjoints[self]

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __add__(self, val):
        return add(self, val)

    def __radd__(self, val):
        return add(val, self)

    def __sub__(self, val):
        return subtract(self, val)

    def __rsub__(self, val):
        return subtract(val, self)

    def __mul__(self, val):
        return multiply(self, val)

    def __rmul__(self, val):
        return multiply(val, self)

    def __truediv__(self, val):
        return divide(self, val)

    def __rtruediv__(self, val):
        return divide(val, self)

    def __rdiv__(self, val):
        return divide(val, self)

    def __pow__(self, val):
        return power(self, val)

    def __rpow__(self, val):
        return power(val, self)

    # Python always uses eq for comparing keys, so if we override __eq__ we
    # cannot have sets of tensors, or using them as dictionary keys.  So,
    # we must use Equal explicitly in transform.  defmod and define __eq__
    # if it can ensure that its nodes do not need to be used as keys.
    # def __eq__(self, val):
    #    return equal(self, val)

    # def __ne__(self, val):
    #    return not_equal(self, val)

    def __lt__(self, val):
        return less(self, val)

    def __gt__(self, val):
        return greater(self, val)

    def __le__(self, val):
        return less_equal(self, val)

    def __ge__(self, val):
        return greater_equal(self, val)

    # Only works when capturing ops
    def __setitem__(self, key, val):
        return SetItem(self, key, val)

    # Only works when capturing ops
    def __iadd__(self, val):
        return SetItem(self, slice(None, None, None), self + val)

    # Only works when capturing ops
    def __isub__(self, val):
        return SetItem(self, slice(None, None, None), self - val)

    # Only works when capturing ops
    def __imul__(self, val):
        return SetItem(self, slice(None, None, None), self * val)

    # Only works when capturing ops
    def __idiv__(self, val):
        return SetItem(self, slice(None, None, None), self / val)

    def __getitem__(self, item):
        return Slice(self, item)

    def with_axes(self, axes):
        return AxesCastOp(self, axes=axes)

    def __axes__(self):
        return self.axes

    @cachetools.cached({})
    def tensor_description(self):
        """
        Returns a TensorDescription describing the output of this TensorOp

        Returns:
          TensorDescription for this op.
        """
        return TensorDescription(self.axes, dtype=self.dtype)

    @property
    def axes(self):
        """TODO."""
        if self.__axes is not None:
            return self.__axes
        else:
            raise NotImplementedError

    def generate_adjoints(self, *args, **kwargs):
        """
        TODO.

        Arguments:
          *args: TODO
          **kwargs: TODO
        """
        pass

    @property
    def shape(self):
        """ returns self.axes

        This is required for parameter initializers in legacy neon code.  It
        expects layers to implement a shape that it can use to pass through
        layers
        """
        return self.axes

    def mean(self, out_axes=(), **kwargs):
        """
        TODO.

        Arguments:
          out_axes: TODO

        Returns:
          TODO
        """
        return mean(self, out_axes=out_axes, **kwargs)

    @property
    def value(self):
        """
        Returns a handle to the device tensor.

        The transformer must have been initialized.

        :return: A handle to the device tensor.
        """
        return self.tensor_description().value


class AxesCastOp(TensorOp):
    """Used to label a tensor with known axes, without altering its value"""
    def __init__(self, x, axes, **kwargs):
        super(AxesCastOp, self).__init__(args=(x,), axes=axes, **kwargs)

    @cachetools.cached({})
    def tensor_description(self):
        return self.args[0].tensor_description().cast(self.axes)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, AxesCastOp(
            Broadcast(delta, axes=self.axes),
            axes=x.axes
        ))


class Broadcast(TensorOp):
    """Used to add additional axes for a returned derivative."""

    def __init__(self, x, **kwargs):
        super(Broadcast, self).__init__(args=(x,), **kwargs)

    @cachetools.cached({})
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        td, = tensor_descriptions(self.args)
        return td.reaxe(self.axes)


class ExpandDims(TensorOp):
    """TODO."""

    def __init__(self, x, axis, dim, **kwargs):
        axes = x.axes[:dim].concat(Axes(axis,)).concat(x.axes[dim:])
        super(ExpandDims, self).__init__(args=(x,), axes=axes, **kwargs)
        self.axis = axis
        self.dim = dim

    @cachetools.cached({})
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return x.reaxe_with_dummy_axis(self.axis, self.dim)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(
            adjoints,
            sum(delta, reduction_axes=Axes(self.axis,))
        )


class Slice(TensorOp):
    """TODO."""

    def __init__(self, x, slices, axes=None, **kwargs):
        if axes is None:
            axes = []
            for axis, s in zip(x.axes, slices):
                # if s is an int, we are doing a getitem, for example y = x[1]
                # and so this axis will no longer exist in the result.
                if not isinstance(s, int):
                    axes.append(SlicedAxis(axis, s))

            axes = Axes(axes)

        super(Slice, self).__init__(
            args=(x,),
            axes=axes,
            **kwargs
        )

        self.slices = slices

    @cachetools.cached({})
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return x.slice(self.slices, self.axes)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(
            adjoints,
            Unslice(delta, self.slices, axes=x.axes)
        )


class AllocationOp(TensorOp):
    """TODO."""

    def __init__(
            self,
            init=None,
            initial_value=None,
            **kwargs):
        super(AllocationOp, self).__init__(**kwargs)
        if init is not None:
            with Op.captured_ops(self.initializers):
                init.fill(self)
        elif callable(initial_value):
            self.initializers.append(assign(self, initial_value()))
        elif initial_value is not None:
            self.initializers.append(assign(self, initial_value))


class ComputationOp(TensorOp):
    """
    A ComputationOp is a Tensor result of some sort of operation.
    """

    def __init__(self, **kwargs):
        super(ComputationOp, self).__init__(**kwargs)

        for arg in self.args:
            arg.users.add(self)

    @property
    def defs(self):
        """TODO."""
        return {self}

    @property
    def graph_label(self):
        """TODO."""
        return self.__class__.__name__ + '[' + self.name + ']'


class Stack(ComputationOp):
    def __init__(self, x_list, axis, pos=0, **kwargs):
        self.pos = pos
        x_axes = x_list[0].axes
        axes = Axes(tuple(x_axes[:pos]) + (axis,) + tuple(x_axes[pos:]))
        super(Stack, self).__init__(args=tuple(x_list), axes=axes)

    def generate_adjoints(self, adjoints, delta, *x_list):
        delta = Broadcast(delta, axes=self.axes)
        s = [slice(None)] * len(self.axes)
        for i, x in enumerate(x_list):
            s[self.pos] = i
            x.generate_add_delta(
                adjoints,
                Slice(delta, tuple(s), axes=x.axes)
            )


# Currently implemented using unrolling
# f_in_to_h should be python functions that take and return tensors
# The length of time_axis should not be changed after this function
# is called, else the stacking will be invalid.
def recurrent(x, f_in_to_h, f_h_to_h, time_axis, stack_pos=-1):
    h = f_in_to_h(x)
    hs = [h]
    for i in range(time_axis.length - 1):
        h = f_h_to_h(h)
        hs.append(h)
    return Stack(hs, time_axis, pos=stack_pos)


class Unslice(ComputationOp):
    """TODO."""

    def __init__(self, x, slices, **kwargs):
        super(Unslice, self).__init__(args=(x,), **kwargs)
        self.slices = slices
        self.input_axes = x.axes

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = super(Unslice, self).call_info()
        return [self.tensor_description().slice(self.slices, self.input_axes), x]

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
        """
        x.generate_add_delta(adjoints, Slice(delta, self.slices, axes=x.axes))


class RNG(object):
    """TODO."""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def uniform(self, low=0.0, high=1.0, size=None, **kwargs):
        """
        TODO.

        Arguments:
          low: TODO
          high: TODO
          size: TODO
          **kwargs: TODO

        Returns:
          TODO
        """

        def value_fun(tensor_description):
            return self.rng.uniform(low, high, tensor_description.sizes).astype(
                tensor_description.dtype)

        val = constant_storage(axes=size, **kwargs)
        val.initializers.append(InitTensor(val, value_fun))
        return val

    def normal(self, loc, scale, size, **kwargs):
        """
        TODO.

        Arguments:
          loc: TODO
          scale: TODO
          size: TODO
          **kwargs: TODO

        Returns:
          TODO
        """

        def value_fun(tensor_description):
            return self.rng.normal(
                loc, scale, tensor_description.sizes).astype(
                tensor_description.dtype)

        val = constant_storage(axes=size, **kwargs)
        val.initializers.append(InitTensor(val, value_fun))
        return val


class ElementWise(ComputationOp):
    """TODO."""

    def __init__(self, args, **kwargs):
        args = Op.as_ops(args)
        axis_ids = AxisIDTuple()
        for arg in args:
            axis_ids += arg.axes.as_axis_ids()
        axes = axis_ids.as_axes()
        super(ElementWise, self).__init__(
            args=args,
            axes=axes,
            **kwargs
        )

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        return [arg.reaxe(self.axes) for arg in tensor_descriptions(self.args)]


class AllReduce(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(AllReduce, self).__init__(args=(x,), **kwargs)


class placeholder(AllocationOp):
    """Can be set externally."""

    def __init__(self, tags=None, **kwargs):
        if tags is None:
            tags = set()
        tags.add('persistent')
        super(placeholder, self).__init__(tags=tags, **kwargs)

    def generate_adjoints(self, tape, delta):
        """
        TODO.

        Arguments:
          tape: TODO
          delta: TODO
        """
        pass

    @property
    def graph_label(self):
        return self.__class__.__name__ + '[' + self.name + ']'


class Constant(AllocationOp):
    """
    A scalar constant that appears in a graph.

    if you want a constant tensor and a numpy array to initialize it, use
    NumPyTensor for now.

    Arguments:

    Returns:
      TODO
    """

    def __init__(self, const, axes=Axes(), **kwargs):
        self.const = const
        super(Constant, self).__init__(
            axes=axes, dtype=np.dtype(np.float32), **kwargs)
        self.initializers.append(Fill(self, const))
        self.tags.add('persistent')

    def generate_adjoints(self, adjoints, delta):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO

        Returns:
          TODO
        """
        pass

    @property
    def graph_label(self):
        """TODO."""
        shapes = self.axes.lengths
        if not shapes or max(shapes) <= 2:
            return str(self.const)
        if self.name == self.id:
            return 'Constant'
        return self.name

    def __str__(self):
        return '<{cl} ({const})>'.format(
            cl=self.__class__.__name__, const=self.const)


class absolute(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(absolute, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, sign(x) * delta)


class add(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(add, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class argmax(ComputationOp):
    """TODO."""

    def __init__(self, x, max_axes=None, **kwargs):
        if max_axes is None:
            max_axes = x.axes.sample_axes()
            axes = x.axes.batch_axes()
        else:
            axes = x.axes - max_axes
        self.max_axes = max_axes
        super(argmax, self).__init__(
            args=(x,), axes=axes, dtype=np.dtype(np.int64), **kwargs
        )

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return [x.reaxe(self.max_axes + self.axes)]


class argmin(ComputationOp):
    """TODO."""

    def __init__(self, x, min_axes=None, **kwargs):
        if min_axes is None:
            min_axes = x.axes.sample_axes()
            axes = x.axes.batch_axes()
        else:
            axes = x.axes - min_axes
        self.min_axes = min_axes
        super(argmax, self).__init__(
            args=(x,), axes=axes, dtype=np.dtype(np.int64), **kwargs
        )

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return [x.reaxe(self.min_axes + self.axes)]


class cos(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(cos, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * sin(x))


class divide(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(divide, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * self / x)
        y.generate_add_delta(adjoints, -delta * self / y)


class dot(ComputationOp):
    """TODO."""

    def __init__(self, x, y,
                 reduction_axes=None, out_axes=None,
                 numpy_matching=False,
                 forward_dot=None,
                 **kwargs):
        self.axis_id_info = self.compute_axis_id_info(
            x, y, reduction_axes, out_axes,
            forward_dot, numpy_matching
        )
        self.out_axes = out_axes
        self.reduction_axes = reduction_axes
        axes = self.axis_id_info[0].as_axes()
        super(dot, self).__init__(
            args=(x, y), axes=axes, **kwargs
        )

    def compute_axis_id_info(self, x, y,
                             reduction_axes, out_axes,
                             forward_dot, use_numpy_matching):
        """
        TODO.

        Arguments:
          x: TODO
          y: TODO
          reduction_axes: TODO
          out_axes: TODO
          forward_dot: TODO
          use_numpy_matching: TODO

        Returns:
          TODO
        """
        x_axis_ids = x.axes.as_axis_ids()
        y_axis_ids = y.axes.as_axis_ids()

        if forward_dot is not None:
            y_axis_ids = forward_dot.axis_id_info[0]
            forward_axis_ids = forward_dot.axis_id_info[0]
        else:
            forward_axis_ids = None

        if use_numpy_matching:
            out_axis_ids = x_axis_ids[:-1]\
                + y_axis_ids[:-2]\
                + AxisIDTuple(y_axis_ids[-1],)
            x_red_axis_ids = AxisIDTuple(x_axis_ids[-1])
            y_red_axis_ids = AxisIDTuple(y_axis_ids[-2])
            return (out_axis_ids, x_red_axis_ids, y_red_axis_ids,
                    None, forward_axis_ids)
        else:
            dummy = None
            if reduction_axes is None:
                red_axis_ids = AxisIDTuple.intersect(
                    x_axis_ids,
                    y_axis_ids
                )
            else:
                red_axis_ids = reduction_axes.as_axis_ids()

            if out_axes is not None:
                out_axis_ids = out_axes.as_axis_ids()
            else:
                out_axis_ids = (
                    (x_axis_ids - red_axis_ids) +
                    (y_axis_ids - red_axis_ids)
                )
            red_axis_ids -= out_axis_ids

            if len(red_axis_ids) == 0:
                dummy = Axis(1)
                red_axis_ids = AxisIDTuple(dummy.axis_id(0),)

            return (out_axis_ids, red_axis_ids, red_axis_ids,
                    dummy, forward_axis_ids)

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, y = tensor_descriptions(self.args)
        out_axis_ids, x_red_axis_ids, y_red_axis_ids, dummy, forward_axis_ids\
            = self.axis_id_info

        if dummy is not None:
            x = x.reaxe_with_dummy_axis(dummy)
            y = y.reaxe_with_dummy_axis(dummy)

        a = x.dot_reaxe_left(x_red_axis_ids)
        b = y.dot_reaxe_right(
            y_red_axis_ids,
            forward_axis_ids=forward_axis_ids
        )
        a_axes, b_axes = a.axes, b.axes
        o = self.tensor_description().reaxe(a_axes[:-1].concat(b_axes[1:]))

        return [o, a, b]

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        # The delta must be passed in as the second argument
        # to preserve the forward axes mapping.
        x.generate_add_delta(
            adjoints,
            dot(y, delta, out_axes=x.axes, forward_dot=self)
        )
        y.generate_add_delta(
            adjoints,
            dot(x, delta, out_axes=y.axes, forward_dot=self)
        )


class ElementWiseBoolean(ElementWise):
    """TODO."""

    def __init__(self, x, y, dtype=np.dtype(bool), **kwargs):
        super(ElementWiseBoolean, self).__init__(
            args=(x, y), dtype=dtype, **kwargs)


class equal(ElementWiseBoolean):
    """TODO."""


class not_equal(ElementWiseBoolean):
    """TODO."""


class greater(ElementWiseBoolean):
    """TODO."""


class less(ElementWiseBoolean):
    """TODO."""


class greater_equal(ElementWiseBoolean):
    """TODO."""


class less_equal(ElementWiseBoolean):
    """TODO."""


class Softmax(object):
    """TODO."""

    def __init__(self, x, exps, Z):
        self.x = x
        self.exps = exps
        self.Z = Z

    def generate_adjoints(self, adjoints, delta, op):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          op: TODO

        Returns:
          TODO
        """
        z = delta * op
        zs = sum(z, reduction_axes=self.x.axes.sample_axes())
        self.x.generate_add_delta(adjoints, (z - zs * op))


def softmax(x, softmax_axes=None, **kwargs):
    """
    TODO.

    Arguments:
      x: TODO
      softmax_axes: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    if softmax_axes is None:
        softmax_axes = x.axes.sample_axes()
    x = x - max(x, reduction_axes=softmax_axes)
    exps = exp(x)
    Z = sum(exps, reduction_axes=softmax_axes)
    result = exps / Z
    result.add_schema(Softmax(x=x, exps=exps, Z=Z))
    return result


class ReductionOp(ComputationOp):
    """TODO."""

    def __init__(self, x, reduction_axes=None, out_axes=None, **kwargs):
        self.out_axes, self.reduction_axes\
            = self.compute_axes(x, reduction_axes, out_axes)
        self.mode = None
        super(ReductionOp, self).__init__(
            args=(x,), axes=self.out_axes, **kwargs
        )

    def compute_axes(self, x, reduction_axes, out_axes):
        """
        TODO.

        Arguments:
          x: TODO
          reduction_axes: TODO
          out_axes: TODO

        Returns:
          TODO
        """
        if reduction_axes is None:
            if out_axes is None:
                reduction_axes = x.axes.sample_axes()
            else:
                reduction_axes = x.axes - Axes(out_axes)
        else:
            reduction_axes = Axes(reduction_axes)

        if out_axes is None:
            out_axes = x.axes - reduction_axes
        else:
            out_axes = Axes(out_axes)

        return out_axes, reduction_axes

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)

        if len(self.reduction_axes) == 0:
            # TODO do this as a reaxe to 1d or something
            xr = x.reaxe(self.axes)
            self.mode = 'copy'
            return [xr]
        else:
            red_axes = [FlattenedAxis(self.reduction_axes)]
            red_axes.extend(self.axes)
            red_axes = Axes(red_axes)
            self.mode = 0
            return [x.reaxe(red_axes)]


class max(ReductionOp):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(max, self).__init__(x, **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, equal(x, self) * delta)


class min(ReductionOp):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(min, self).__init__(x, **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, equal(x, self) * delta)


class sum(ReductionOp):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sum, self).__init__(x, **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta)


def assign(lvalue, rvalue):
    """
    TODO.

    Arguments:
      lvalue: TODO
      rvalue: TODO

    Returns:
      TODO
    """
    return SetItem(lvalue, (), rvalue)


class tensor_size(ComputationOp):
    """TODO."""

    def __init__(self, x, reduction_axes=None, out_axes=None, **kwargs):
        if reduction_axes is None:
            if out_axes is None:
                reduction_axes = x.axes.sample_axes()
            else:
                reduction_axes = x.axes - Axes(out_axes)
        else:
            reduction_axes = Axes(reduction_axes)
        self.reduction_axes = reduction_axes
        super(tensor_size, self).__init__(axes=Axes())


def pad(x, paddings, axes=None, **kwargs):
    """
    Pads a tensor with zeroes along each of its dimensions.

    Arguments:
      x: the tensor to be padded
      paddings: the length of the padding along each dimension.
        should be an array with the same length as x.axes.
        Each element of the array should be either an integer,
        in which case the padding will be symmetrical, or a tuple
        of the form (before, after)
      axes: the axes to be given to the padded tensor.
        If unsupplied, we create anonymous axes of the correct lengths.
      **kwargs: Additional args for the created Op.

    Returns:
        TensorOp: symbolic expression for the padded tensor
    """
    def pad_to_tuple(pad):
        if isinstance(pad, int):
            pad = (pad, pad)
        return pad

    paddings = tuple(pad_to_tuple(pad) for pad in paddings)
    if axes is None:
        axes = Axes(
            PaddedAxis(axis, pad) for axis, pad in zip(x.axes, paddings)
        )

    def to_slice(pad):
        """
        TODO.

        Arguments:
          pad: TODO

        Returns:
          TODO
        """
        s = (pad[0], -pad[1])
        s = tuple(None if p == 0 else p for p in s)
        return slice(s[0], s[1], 1)
    slices = tuple(to_slice(p) for p in paddings)
    return Unslice(x, axes=axes, slices=slices, **kwargs)


class Variable(AllocationOp):
    """TODO."""

    def __init__(self, tags=None, trainable=True, persistent=True, constant=False, **kwargs):
        if tags is None:
            tags = set()
        else:
            tags = set(tags)
        if trainable:
            tags.add('trainable')
        if persistent:
            tags.add('persistent')
        if constant:
            tags.add('constant')
        super(Variable, self).__init__(tags=tags, **kwargs)


class NumPyTensor(Variable):
    """
    NumPyTensor is effectively a non-trainable, persistent, constant tensor
    """

    def __init__(self, nptensor, axes, **kwargs):
        super(NumPyTensor, self).__init__(trainable=False, persistent=True,
                                          constant=True, dtype=nptensor.dtype,
                                          axes=axes, **kwargs)
        self.base_tensor = nptensor
        axes = Axes(axes)
        if nptensor.shape != axes.lengths:
            raise ValueError((
                "Tried to initialize NumPyTensor with numpy array of "
                "shape {np_shape} though gave axes with a different "
                "shape {axes_shape}."
            ).format(
                np_shape=nptensor.shape,
                axes_shape=axes.lengths,
            ))

        def value_fun(tensor):
            return nptensor

        self.initializers.append(InitTensor(self, value_fun))

    @property
    def nptensor(self):
        return self.base_tensor


def temporary(**kwargs):
    """
    TODO.

    Arguments:
      **kwargs: TODO

    Returns:
      TODO
    """
    return Variable(trainable=False, persistent=True, constant=False, **kwargs)


def constant_storage(**kwargs):
    return Variable(trainable=False, persistent=True, constant=True, **kwargs)


class exp(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(exp, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * self)


class log(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(log, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        def do_adjoints(delta, x):
            """
            TODO.

            Arguments:
              delta: TODO
              x: TODO

            Returns:
              TODO
            """

            if False and isinstance(x, softmax):
                x, = x.args

            if isinstance(x, divide):
                a, b = x.args
                do_adjoints(delta, a)
                do_adjoints(-delta, b)

            elif isinstance(x, exp):
                x.args[0].generate_add_delta(adjoints, delta)

            else:
                x.generate_add_delta(adjoints, delta / x)

        do_adjoints(delta, x)


def safelog(x, limit=np.exp(-50)):
    return log(maximum(x, limit))


class maximum(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(maximum, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, equal(self, x) * delta)
        y.generate_add_delta(adjoints, equal(self, y) * delta)


class minimum(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(minimum, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, equal(self, x) * delta)
        y.generate_add_delta(adjoints, equal(self, y) * delta)


class multiply(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(multiply, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * y)
        y.generate_add_delta(adjoints, x * delta)


class negative(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(negative, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, -delta)


class onehot(ComputationOp):
    """TODO."""

    def __init__(self, x, axis=None, axes=None, **kwargs):
        if axis is None:
            if axes is None:
                raise ValueError('Cannot determine one-hot axis.')
            axis = (axes - x.axes)[0]
        else:
            if axes is None:
                x_sample = x.axes.sample_axes()
                x_batch = x.axes.batch_axes()
                axes = Axes(axis) + x_sample + x_batch
        self.axis = axis
        super(onehot, self).__init__(args=(x,), axes=axes, **kwargs)

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        axis, axes = self.axis, self.axes
        reaxes = Axes([axis, AxisIDTuple.sub(axes, Axes(axis,)).as_axes()])
        return [
            self.tensor_description().reaxe(reaxes),
            x.reaxe(Axes(FlattenedAxis(x.axes)))
        ]


class power(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(power, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * y * self / x)
        y.generate_add_delta(adjoints, delta * self * log(x))


class reciprocal(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(reciprocal, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, -self * self * delta)


class sign(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sign, self).__init__(args=(x,), **kwargs)


class Sigmoid(object):
    """Sigmoid"""

    def __init__(self, x):
        self.x = x

    def generate_adjoints(self, adjoints, delta, op):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          op: TODO

        Returns:
          TODO
        """
        self.x.generate_add_delta(adjoints, delta * op * (1.0 - op))


def sigmoid(x, **kwargs):
    """
    TODO.

    Arguments:
      x: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    result = reciprocal(exp(-x) + 1)
    result.add_schema(Sigmoid(x=x))
    return result


class sin(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sin, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * cos(x))


class sqrt(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(sqrt, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, .5 * delta * self)


class square(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(square, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, 2.0 * delta * x)


class subtract(ElementWise):
    """TODO."""

    def __init__(self, x, y, **kwargs):
        super(subtract, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO
          y: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)


class tanh(ElementWise):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(tanh, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          x: TODO

        Returns:
          TODO
        """
        x.generate_add_delta(adjoints, delta * (1.0 - self * self))


class Function(Node):
    """TODO."""

    def __init__(self, ops):
        super(Function, self).__init__()
        from geon.analysis import Digraph
        self.ops = Digraph(ops)
        self.instructions = self.ops.topsort()
        args, defs = set(), set()
        for op in self.instructions:
            # Kernel defines the def of each operation
            defs |= set([op])
            # Kernel uses the args of each operation
            # except whatever is being defined
            args |= set(op.args) - defs
        self.args = args
        self.defs = defs
        self.initializers = [x for x in op.initializers
                             for op in self.instructions]

    @property
    def inputs(self):
        """TODO."""
        return self.use


class Buffer(object):
    """TODO."""

    def __init__(self, color, size):
        self.color = color
        self.size = size
        self.data = None
        self.views = set()


def mean(x, **kwargs):
    """
    TODO.

    Arguments:
      x: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    return sum(x, **kwargs) / tensor_size(x, **kwargs)


def deriv(dep, indep):
    """
    TODO.

    Arguments:
      dep: TODO
      indep: TODO

    Returns:
      TODO
    """
    Op.simple_prune([dep, indep])
    adjoint = dep.adjoints()[indep]
    return Broadcast(adjoint, axes=indep.axes)


class CrossEntropyMultiInner(object):
    """TODO."""

    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def generate_adjoints(self, adjoints, delta, op):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          op: TODO

        Returns:
          TODO
        """
        self.s.generate_add_delta(adjoints, delta)
        self.x.generate_add_delta(adjoints, self.y * delta)


def cross_entropy_multi(y, t, usebits=False, out_axes=None,
                        enable_softmax_opt=True,
                        enable_diff_opt=True, **kwargs):
    """
    TODO.

    Arguments:
      y: TODO
      t: TODO
      usebits: TODO
      out_axes: TODO
      enable_softmax_opt: TODO
      enable_diff_opt: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    smy = y.find_schema(Softmax)
    if enable_softmax_opt and smy is not None:
        # This depends on sum(t) being 1
        x = smy.x
        Z = smy.Z
        s = -sum(x * t, out_axes=out_axes)
        result = s + safelog(Z)
        if enable_diff_opt:
            result.add_schema(CrossEntropyMultiInner(x=x, y=y, s=s))
    else:
        result = -sum(safelog(y) * t, out_axes=out_axes)
    if usebits:
        result = result * np.float(1. / np.log(2.0))
    return result


class CrossEntropyBinaryInner(object):
    """TODO."""

    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def generate_adjoints(self, adjoints, delta, op):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          op: TODO

        Returns:
      TODO
        """
        self.x.generate_add_delta(adjoints, (self.y - self.t) * delta)
        self.t.generate_add_delta(adjoints, self.x * delta)


def cross_entropy_binary_inner(y, t, enable_sig_opt=True,
                               enable_diff_opt=True, **kwargs):
    """
    TODO.

    Arguments:
      y: TODO
      t: TODO
      enable_sig_opt: TODO
      enable_diff_opt: TODO
      **kwargs: TODO

    Returns:
      TODO
    """
    sigy = y.find_schema(Sigmoid)
    if enable_sig_opt and sigy is not None:
        # Simpler equivalent
        x = sigy.x
        result = (1 - t) * x - safelog(y)
        if enable_diff_opt:
            result.add_schema(CrossEntropyBinaryInner(x=x, y=y, t=t))
    else:
        result = -(safelog(y) * t + safelog(1 - y) * (1 - t))

    return result


def cross_entropy_binary(y, t, out_axes=None):
    """
    TODO.

    Arguments:
      y: TODO
      t: TODO
      out_axes: TODO

    Returns:
      TODO
    """
    return sum(cross_entropy_binary_inner(y, t), out_axes=out_axes)


class SimplePrune(object):
    """TODO."""

    def __init__(self, results):
        self.results = results
        self.reps = []

        has_work = True
        while has_work:
            self.init()
            for op in Op.ordered_ops(self.results):
                self.visit(op)
            has_work = self.do_replacements()

    def init(self):
        """TODO."""
        self.reps = []

    def add_rep(self, op, replacement):
        """
        TODO.

        Arguments:
          op: TODO
          replacement: TODO

        Returns:
          TODO
        """
        # Can't replace op if its being returned
        if op not in self.results:
            self.reps.append((op, replacement))

    def do_replacements(self):
        """TODO."""
        for old, rep in self.reps:
            old_users = set(old.users)
            for user in old_users:
                user.replace_arg(old, rep)
        return len(self.reps) > 0

    @generic_method
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO
        """
        pass

    @visit.on_type(negative)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if isinstance(x, Constant):
            self.add_rep(op, Constant(-x.const))

    @visit.on_type(multiply)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, y = op.args
        rep = None
        if isinstance(x, Constant):
            if x.const == 0:
                rep = x
            elif x.const == 1:
                rep = y
            elif x.const == -1:
                rep = negative(y)
        elif isinstance(y, Constant):
            if y.const == 0:
                rep = y
            elif y.const == 1:
                rep = x
            elif y.const == -1:
                rep = negative(x)
        if rep is not None:
            self.add_rep(op, rep)

    @visit.on_type(add)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, y = op.args
        rep = None
        if isinstance(x, Constant):
            if x.const == 0:
                rep = y
        elif isinstance(y, Constant):
            if y.const == 0:
                rep = x
        if rep is not None:
            self.add_rep(op, rep)

    @visit.on_type(sum)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if isinstance(x, Constant):
            val = x.const * op.reduction_axes.size
            self.add_rep(op, Constant(val))

    @visit.on_type(log)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, = op.args
        if isinstance(x, divide):
            num, denom = x.args
            if isinstance(num, exp):
                exp_x, = num.args
                self.add_rep(op, exp_x - type(op)(denom))
        elif isinstance(x, exp):
            exp_x, = x.args
            self.add_rep(op, exp_x)
