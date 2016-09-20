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
from builtins import object

from ngraph.op_graph.arrayaxes import TensorDescription, \
    AxisIDTuple, Axes, FlattenedAxis, PaddedAxis, Axis, SlicedAxis
from ngraph.util.generics import generic_method
from ngraph.util.nodes import Node
from ngraph.util.threadstate import get_thread_state


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
    Any operation that can be in an AST.

    Arguments:
        const: The value of a constant Op, or None,
        constant (bool): The Op is constant.  Default False.
        graph_label_type: A label that should be used when drawing the graph.  Defaults to
            the class name.
        initializers: List of one-time initializations to run before the op.
        persistent (bool): The value will be retained from computation to computation and
            not shared.  Default False.
        reference (bool): The storage is accessed via a reference.  Default False.
        trainable (bool): The value is trainable.  Default False.
        kwargs: Args defined in related classes.

    Attributes:
        const: The value of a constant.
        constant (bool): The value is constant.
        graph_label_type: A label that should be used when drawing the graph.
        initializers (list): Additional Ops to run before this Op is run the first time.
        other_deps (set): Ops in addtion to args that must run before this op.
        persistent (bool): The value will be retained from computation to computation and
            not shared.  Always True if reference is set.
        reference (bool): The storage is accessed via a reference.  Implies persistent.
        trainable: The value is trainable.
        schemas: Information about how the Op was generated.
        user_deps (set): Ops that must run before using this op.  Set SetItem.
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

    def __init__(self,
                 const=None,
                 constant=False,
                 initializers=None,
                 persistent=False,
                 reference=False,
                 trainable=False,
                 graph_label_type=None,
                 **kwargs):
        super(Op, self).__init__(**kwargs)

        if graph_label_type is None:
            graph_label_type = self.__class__.__name__
        self.graph_label_type = graph_label_type

        self.other_deps = set()
        self.user_deps = set()
        for arg in self.args:
            self.other_deps.update(arg.user_deps)

        self.schemas = []
        self._adjoints = None
        self.const = const
        self.constant = constant
        self.initializers = initializers or []
        self.__persistent = persistent
        self.reference = reference
        self.trainable = trainable

        ops = Op._get_thread_ops()[-1]
        if ops is not None:
            ops.append(self)
        self.style = {}
        self.ops = []

    @property
    def assignable(self):
        """

        Returns: True if the tensor can be assigned to.

        """
        return not self.constant

    @property
    def graph_label(self):
        """The label used for drawings of the graph."""
        return "{}[{}]".format(self.graph_label_type, self.name)

    @property
    def scalar(self):
        return 0 == len(self.axes)

    @property
    def persistent(self):
        """

        Returns: True if value is not shared and is retained through computation.  Always true
        for a reference.

        """
        return self.__persistent or self.reference

    @property
    def device_op(self):
        """

        Returns:
            True if the Op executes on the device.
        """
        return True

    @persistent.setter
    def persistent(self, value):
        self.__persistent = value

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

    @property
    def defs(self):
        """
        Returns:
            For liveness analysis.  The storage associated with everything
            in the returned list is modified when the Op is executed.

        """
        return [self]

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

    def variables(self, filter=None):
        """
        Return all trainable Ops used in computing this node.

        Arguments:
            filter: Boolean filter of op, defaults to trainable.

        Returns:
            Set of trainable Ops.
        """
        params = set()

        if filter is None:
            filter = lambda op: op.trainable

        def visitor(node):
            """
            TODO.

            Arguments:
              node: TODO
            """
            if filter(node):
                params.add(node)

        Node.visit_input_closure([self], visitor)

        return set(params)

    @property
    @cachetools.cached({})
    def initial_adjoint(self):
        """
        Most models only require the adjoints map for their scalar loss
        functions, in which case the adjoint is initialized to a scalar 1.
        Some autodiff tests calculate the derivative of a tensor by
        initializing all but one elements of a tensor to zero and the remaining
        element to one.  To allow this, we create a placeholder for the initial
        adjoint and allow it to be accessed by the _initial_adjoint field.
        """
        if len(self.axes) == 0:
            return Constant(1)
        else:
            return placeholder(axes=self.axes)

    @cachetools.cached({})
    def adjoints(self):
        """
        Returns a map containing the adjoints of this op with respect to other
        ops.

        Creates the map if it does not already exist.  Most models only
        require the adjoints map for their scalar loss functions, in which case
        the adjoint is initialized to a scalar 1.  Some autodiff tests
        calculate the derivative of a tensor by initializing all but one
        elements of a tensor to zero and the remaining element to one.  To
        allow this, we create a placeholder for the initial adjoint and allow
        it to be accessed by the _initial_adjoint field.

        Returns:
            Map from Op to dSelf/dOp.
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
        Creates the TensorDescriptions (of this op or its arguments)
        required to evaluate it.

        The list is used to allocate buffers (in the transformers) and supply
        values to the transform method (in the transform_call_info) method.

        Only TensorDescriptions of the arguments are necessary.  A
        TensorDescription of the output is generate by calling
        self.tensor_description()
        """
        return list(tensor_descriptions(self.args))

    def __repr__(self):
        return '<{cl}({gl}):{id}>'.format(
            cl=self.__class__.__name__,
            gl=self.graph_label_type,
            id=id(self)
        )


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

    @property
    def device_op(self):
        """

        Returns:
            False, because this is run from the CPU.
        """
        return False


class SetItem(Op):
    """
    tensor[item] = val.

    Arguments:
        tensor: An assignable TensorOp.
        item: The index.
        val: The value to assign.
        force (bool): Override constant check on tensor.
        **kwargs: Args for related classes.
    """

    def __init__(self, tensor, item, val, force=False, **kwargs):
        if not force and not tensor.assignable:
            raise ValueError("{} is not assignable.".format(tensor))
        super(SetItem, self).__init__(args=(tensor, val), **kwargs)
        self.item = item
        self.input = None
        tensor.user_deps = {self}

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

    @property
    def defs(self):
        """

        Returns:
            SetItem modifies the variable being set.

        """
        return [self.args[0]]


class doall(Op):
    """
    Compute every Op in all.

    Arguments:
        all: Ops to be computed.
        **kwargs: Args for related classes.
    """

    def __init__(self, all, **kwargs):
        super(doall, self).__init__(args=all, **kwargs)

    def call_info(self):
        return []

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.
        """
        return False


class Fill(Op):
    """
    Fill a tensor with a scalar value.

    Arguments:
        tensor: An assignable TensorOp.
        scalar: A scalar value.
        force (bool): Disable constant check on tensor.
    """

    def __init__(self, tensor, scalar, force=False, **kwargs):
        super(Fill, self).__init__(args=(tensor,), **kwargs)
        if not force and not tensor.assignable:
            raise ValueError("{} is not assignable.".format(tensor))
        if isinstance(scalar, TensorOp):
            if scalar.constant:
                scalar = scalar.const
            else:
                raise ValueError("{} is not a scalar constant".format(scalar))
        else:
            npscalar = np.asarray(scalar, dtype=tensor.dtype)
            if 0 != len(npscalar.shape):
                raise ValueError("{} is not a scalar".format(scalar))
            scalar = npscalar[()]

        self.scalar = scalar


class TensorOp(Op):
    """
    Super class for all Ops whose value is a Tensor.

    Arguments:
        axes: The axes of the tensor.
        dtype: The element type of the tensor.
        scale: If specified, a scaling factor applied during updates.
        **kwargs: Arguments for related classes.
    """

    def __init__(self, dtype=None, axes=None, scale=None, **kwargs):
        super(TensorOp, self).__init__(**kwargs)
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.dtype = dtype
        if axes is not None:
            axes = Axes(axes)
        self.__axes = axes

        self.scale = scale

    def generate_add_delta(self, adjoints, delta):
        """
        Adds delta to the backprop contribution..

        Arguments:
            adjoints: dy/dOp for all Ops used to compute y.
            delta: Backprop contribute.
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

    def __div__(self, val):
        return divide(self, val)

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
        """

        Returns: The axes of the tensor.

        """
        if self.__axes is not None:
            return self.__axes
        else:
            raise NotImplementedError

    @axes.setter
    def axes(self, value):
        if self.__axes is not None:
            raise ValueError()
        self.__axes = value

    @property
    def has_axes(self):
        """

        Returns: True if axes have been set.

        """
        return self.__axes is not None

    def generate_adjoints(self, adjoints, delta, *args):
        """
        With delta as the computation for the adjoint of this Op, incorporates delta into the
        adjoints for thr args.

        Args:
            adjoints: dy/dOp for all ops involved in computing y.
            delta: Backprop amount for this Op.
            *args: The args of this Op.
        """
        pass

    @property
    def shape(self):
        """
        This is required for parameter initializers in legacy neon code.  It
        expects layers to implement a shape that it can use to pass through
        layers.

        Returns: self.axes
        """
        return self.axes

    def mean(self, **kwargs):
        """
        Used in Neon front end.

        Returns: mean(self)

        """
        return mean(self, **kwargs)

    @property
    def value(self):
        """
        Returns a handle to the device tensor.

        The transformer must have been initialized.

        :return: A handle to the device tensor.
        """
        return self.tensor_description().value


class AxesCastOp(TensorOp):
    """
    Used to label a tensor with known axes, without altering its value

    Arguments:
        x: A tensor.
        axes: The new axes.

    """
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

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.
        """
        return False


class Broadcast(TensorOp):
    """
    Used to add additional axes for a returned derivative.

    Arguments:
        x: The tensor to broadcast.
        axes: The new axes.
    """

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

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.

        """
        return False


class ExpandDims(TensorOp):
    """
    Adds additional axes into a tensor.

    Arguments:
        x: The tensor.
        axis: The additional axis.
        dim: The position to add the axes.
    """

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

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.

        """
        return False


class Slice(TensorOp):
    """
    Creates a sliced version of a tensor.

    Arguments:
        x: The tensor.
        slices: One slice for each dimension in x.
        axes: Axes for the result.  If not specified, axes will be generated.
    """

    def __init__(self, x, slices, axes=None, **kwargs):
        if len(slices) != len(x.shape):
            raise ValueError((
                'There should be one slice in slices for each dimension in '
                'input tensor.  Input tensor had {tensor_dim} dimensions, '
                'but slices has length {slices_len}.'
            ).format(
                tensor_dim=len(x.shape),
                slices_len=len(slices),
            ))

        if axes is None:
            axes = []
            for axis, s in zip(x.axes, slices):
                # if s is an int, we are doing a getitem, for example y = x[1]
                # and so this axis will no longer exist in the result.
                if not isinstance(s, int):
                    # if nop slice, don't slice the axis
                    if s == slice(None, None, None):
                        axes.append(axis)
                    else:
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

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.

        """
        return False


def slice_along_axis(x, axis, idx):
    """
    Returns a slice of a tensor constructed by indexing into a single axis
    at a single position. If the axis occurs multiple times in the dimensions
    of the input tensor, we select only on the first occurrence.

    Arguments:
        x: input tensor
        axis: axis along which to slice
        idx: index to select from the axis
    Returns:
        y: a slice of x
    """
    pos = x.axes.index(axis)
    ss = tuple(idx if i == pos else slice(None) for i in range(len(x.axes)))
    axes = x.axes[:pos].concat(x.axes[pos + 1:])
    return Slice(x, ss, axes=axes)


class AllocationOp(TensorOp):
    """
    Value comes directly from storage.

    Arguments:
        input: The storage is used as an input from the CPU. Implies persistent.
        init: A Neon initialization function with a fill method that takes the tensor
            as an argument.
        initial_value: If callable, a function that generates an Op whose tensor should be
            used as the initial value.  Otherwise an Op that should be used as the initial
            value.

    Attributes:
        input (bool): The storage is used as an input.
    """

    def __init__(
            self,
            init=None,
            initial_value=None,
            input=False,
            persistent=False,
            **kwargs):
        if input:
            persistent = True
        super(AllocationOp, self).__init__(persistent=persistent, **kwargs)
        self.input = input

        if init is not None:
            with Op.captured_ops(self.initializers):
                init.fill(self)
        elif callable(initial_value):
            self.initializers.append(assign(self, initial_value()))
        elif initial_value is not None:
            self.initializers.append(assign(self, initial_value))

    @property
    def defs(self):
        """

        Returns:
            AllocationOp is not executed, so its appearance in the instruction stream does
            not affect liveness of its value.

        """
        return []

    @property
    def device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.

        """
        return False


def Constant(const, axes=None, constant=True, trainable=False, graph_label_type=None, **kwargs):
    """
    Makes a constant scalar/tensor.  For a tensor, Constant provides the opportunity
        to supply axes.  Scalar/NumPytensor arguments are usually automatically converted to
        tensors, but Constant may be used to supply axes or in the rare cases where Constant
        is not automatically provided.

    Args:
        const: The constant, a scalar or a NumPy array.
        axes: The axes for the constant.
        constant (:obj:`bool`, optional): True; this value should not be writable.
        trainable (:obj:`bool`, optional): False; this value should not be trained.
        graph_label_type (:obj:`str`, optional): Label for drawn graphs, defaults to <Const...>
        **kwargs: Other parameters for the op.

    Returns:
        An AllocationOp for the constant.
    """
    if graph_label_type is None:
        graph_label_type = "<Const({})>".format(const)
    val = AllocationOp(axes=axes, constant=constant, trainable=trainable,
                       graph_label_type=graph_label_type, **kwargs)
    nptensor = np.asarray(const, dtype=val.dtype)

    if not val.has_axes:
        val.axes = Axes(nptensor.shape)

    if nptensor.shape != val.axes.lengths:
        raise ValueError((
            "Tried to initialize constant with numpy array of "
            "shape {np_shape} though gave axes with a different "
            "shape {axes_shape}."
        ).format(
            np_shape=nptensor.shape,
            axes_shape=val.axes.lengths,
        ))

    val_tensor = nptensor
    if len(val.axes) == 0:
        val_tensor = nptensor[()]
    val.const = val_tensor

    def value_fun(tensor):
        return val_tensor

    val.initializers.append(InitTensor(val, value_fun))

    return val


def is_constant(value):
    """
    Test an Op to see if it is a constant.

    Args:
        value: An Op

    Returns: True if value is a constant.

    """
    return isinstance(value, AllocationOp) and value.constant


def is_constant_scalar(value):
    """
    Tests an Op to see if it is a constant scalar.

    Args:
        value: An Op.

    Returns: True if value is a constant scalar.

    """
    return isinstance(value, AllocationOp) and value.constant and len(value.axes) == 0


def constant_value(value):
    """
    Returns the constant value of an Op.

    Args:
        value: A constant op.

    Returns: The constant value.

    """
    if not is_constant(value):
        raise ValueError()
    return value.const


def constant_storage(graph_label_type="Constant", **kwargs):
    """
    A tensor that is supposed to remain constant.

    Args:
        graph_label_type: Label for drawing graphs.
        **kwargs: Other args for AllocationOp.

    Returns:

    """

    return AllocationOp(graph_label_type=graph_label_type,
                        constant=True, persistent=True,
                        trainable=False, **kwargs)


def placeholder(constant=False, trainable=False, input=True, graph_label_type="placeholder",
                **kwargs):
    """
    A persistent tensor to be initialized from the CPU.

    Args:
        constant (:obj:`bool`, optional): False.
        trainable (:obj:`bool`, optional): False.
        input (:obj:`bool`, optional): Allow value to be passed in computation args.  Default True.
        graph_label_type (:obj:`str`, optional): Label used for drawing graphs.
            Defaults to placeholder.
        **kwargs: Other args for AllocationOp.

    Returns: An AllocationOp.

    """
    return AllocationOp(graph_label_type=graph_label_type,
                        constant=constant, persistent=True, trainable=trainable,
                        input=input, **kwargs)


def temporary(graph_label_type="Temp", **kwargs):
    """
    Temporary storage.

    Args:
        graph_label_type (:obj:`str`, optional): Used for drawing graphs.
        **kwargs: Other args for AllocationOp.

    Returns: An AllocationOp.

    """
    return AllocationOp(graph_label_type=graph_label_type,
                        constant=False, persistent=False,
                        trainable=False, **kwargs)


def Variable(trainable=True, graph_label_type="Variable", **kwargs):
    """
    A trainable tensor.

    Args:
        trainable (:obj:`bool`, optional): Is in lists of trainable variables.  Default True.
        graph_label_type: Used for drawing graphs, defaults to Variable.
        **kwargs: Other args for AllocationOp.

    Returns: An AllocationOp.

    """
    return AllocationOp(graph_label_type=graph_label_type,
                        constant=False, persistent=True,
                        trainable=trainable, **kwargs)


class Stack(TensorOp):
    """ TODO."""
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


class Unslice(TensorOp):
    """
    A computation to reverse a slicing operation.
    Primarily used internally to implement expansions of tensors
    such as the derivative of a slice and a padding function.
    However, there is no reason why this operation should not be used
    by a higher-level module or the end user.

    Arguments:
        x: The tensor.
        slices: slices to be unsliced.
        axes: axes of result.

    Attributes:
        slices: The slices.
        input_axes: The axes of the input x.
    """

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


class ElementWise(TensorOp):
    """TODO."""

    def __init__(self, args, **kwargs):
        args = Op.as_ops(args)
        arg_axess = [arg.axes.squeeze() for arg in args]

        # We do this to have the same output shape as numpy
        if len(args) == 2\
                and args[0].axes.are_numeric()\
                and args[1].axes.are_numeric():
            arg_axess = sorted(arg_axess, key=len, reverse=True)

        axis_ids = AxisIDTuple()
        for arg_axes in arg_axess:
            axis_ids += arg_axes.as_axis_ids()
        axes = axis_ids.as_axes()

        super(ElementWise, self).__init__(args=args, axes=axes, **kwargs)

    @cachetools.cached({})
    def call_info(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        return [
            arg.squeeze().reaxe(self.axes)
            for arg in tensor_descriptions(self.args)
        ]


class AllReduce(Op):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(AllReduce, self).__init__(args=(x,), **kwargs)


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
    """
    Computation :math:`x+y`, element-wise addition.

    Arguments:
        x: First argument.
        y: Second argument.
        axes (optional): Axes for the result.

    Returns:
        The computation.
    """

    def __init__(self, x, y, **kwargs):
        super(add, self).__init__(args=(x, y), **kwargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        Contributes to the backprop computation:

        | :math:`z = x + y`
        | :math:`dz = dx + dy`
        | so
        | :math:`\overline{x} \leftarrow \overline{x} + \overline{z}`
        | :math:`\overline{y} \leftarrow \overline{y} + \overline{z}`.
        """
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class argmax(TensorOp):
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


class argmin(TensorOp):
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
    """
    Element wise :math:`cos(x)`.

    Arguments:
        x: The input.
        **kwargs: Related class arguments.
    """

    def __init__(self, x, **kwargs):
        super(cos, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        | :math:`z = \cos(x)`
        | :math:`dz = -\sin(x) dx`
        | so
        | :math:`\overline{x} \leftarrow \overline{x} - \overline{z}\sin(x)`
        """
        x.generate_add_delta(adjoints, -delta * sin(x))


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


class dot(TensorOp):
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

        # We ensure that the axes of the output view given to the transformer
        # (o.axes) are in the same order as the underlying tensor (self.axes)
        if o.axes != self.axes:
            o, a, b = o.transpose(), b.transpose(), a.transpose()

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
    """
    A schema to use to shortcut formula for the softmax derivative.
    """

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
        zs = sum(z)
        self.x.generate_add_delta(adjoints, (z - zs * op))


def softmax(x, normalization_axes=None, **kwargs):
    """
    The softmax activation function.

    Arguments:
      x: input
      normalization_axes: dimensions over which we normalize
      **kwargs: options

    Returns:
        y: output of softmax function
    """
    if normalization_axes is None:
        normalization_axes = x.axes.sample_axes()\
            - x.axes.recurrent_axes()
    x = x - max(x, reduction_axes=normalization_axes)
    exps = exp(x)
    Z = sum(exps, reduction_axes=normalization_axes)
    result = exps / Z
    result.add_schema(Softmax(x=x, exps=exps, Z=Z))
    return result


class ReductionOp(TensorOp):
    """
    Handles shared behaviour of computations that aggregate their inputs
    over some dimensions, e.g. sum, max.
    """

    def __init__(self, x, reduction_axes=None, out_axes=None, **kwargs):
        self.out_axes, self.reduction_axes\
            = self._compute_axes(x, reduction_axes, out_axes)
        self.mode = None
        super(ReductionOp, self).__init__(
            args=(x,), axes=self.out_axes, **kwargs
        )

    def _compute_axes(self, x, reduction_axes, out_axes):
        """
        A helper function to choose compatible reduction and output
        axes from the arguments and defaults.

        Arguments:
          x: input
          reduction_axes: may contain reduction axes specified by the user
          out_axes: may contain output axes specified by ther user

        Returns:
            (reduction_axes, output_axes): computed axes
        """
        if reduction_axes is None:
            if out_axes is None:
                reduction_axes = x.axes.sample_axes()\
                    - x.axes.recurrent_axes()
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
        x.generate_add_delta(adjoints, Broadcast(delta, axes=x.axes))


def assign(lvalue, rvalue, **kwargs):
    """
    Assignment; lvalue <= rvalue

    Arguments:
        lvalue: Tensor to assign to.
        rvalue: Value to be assigned.
        kwargs: options, including name
    """
    return SetItem(lvalue, (), rvalue, **kwargs)


class tensor_size(TensorOp):
    """
    A scalar returning the total size of a tensor.
    Arguments:
        x: The tensor whose axes we are measuring.
        reduction_axes: if supplied, return the size
            of these axes instead.
        kwargs: options, including name
    """
    def __init__(self, x, reduction_axes=None, **kwargs):
        if reduction_axes is None:
            reduction_axes = x.axes
        self.reduction_axes = reduction_axes
        super(tensor_size, self).__init__(axes=Axes())


class batch_size(tensor_size):
    """
    A scalar returning the total size of the batch axes of
    a tensor.
    Arguments:
        x: The tensor whose axes we are measuring.
        kwargs: options, including name
    """
    def __init__(self, x, **kwargs):
        super(batch_size, self).__init__(
            x=x,
            reduction_axes=x.axes.batch_axes(),
            **kwargs
        )


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
    if len(x.axes) != len(paddings):
        raise ValueError((
            "pad's paddings has length {pad} which needs to be the same "
            "as the number of axes in x ({x})"
        ).format(
            pad=len(paddings),
            x=len(x.axes),
        ))

    def pad_to_tuple(pad):
        if isinstance(pad, int):
            pad = (pad, pad)
        return pad

    paddings = tuple(pad_to_tuple(pad) for pad in paddings)
    if axes is None:
        axes = Axes(
            PaddedAxis(axis, pad) if pad != (0, 0) else axis
            for axis, pad in zip(x.axes, paddings)
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


class onehot(TensorOp):
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


class Function(Op):
    """TODO."""

    def __init__(self, ops):
        super(Function, self).__init__()
        from ngraph.analysis import Digraph
        self.ops = Digraph(ops)
        self.instructions = self.ops.topsort()
        args, defs = set(), set()
        for op in self.instructions:
            # Kernel defines the def of each operation
            defs.add(op)
            # Kernel uses the args of each operation
            # except whatever is being defined
            args |= set(op.args) - defs
        self.args = args
        self.__defs = defs
        self.initializers = [x for x in op.initializers
                             for op in self.instructions]

    @property
    def defs(self):
        """

        Returns:
            The cumulative invalidated storage for the op sequence.

        """
        return self.__defs

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


def deriv(dependent_op, independent_op):
    """
    TODO.

    Arguments:
      dependent_op: TODO
      independent_op: TODO

    Returns:
      TODO
    """
    Op.simple_prune([dependent_op, independent_op])
    adjoints = dependent_op.adjoints()

    if independent_op not in adjoints:
        # TODO: check to see if independent_op is even used to compute
        # dependent_op.  If so, pinpoint which Op isn't defining the necessary
        # adjoints.  If it isn't used, give that more specific error to the
        # user.
        raise ValueError((
            "Attempted to take the derivative of {dependent_op} with respect "
            "to {independent_op}, but {independent_op} was not present in "
            "{dependent_op}'s adjoints.  This is most likely because "
            "{independent_op} isn't used to compute {dependent_op} or one of "
            "the ops used to compute {independent_op} hasn't defined the "
            "necessary adjoints."
        ).format(
            dependent_op=dependent_op,
            independent_op=independent_op,
        ))

    adjoint = adjoints[independent_op]
    return Broadcast(adjoint, axes=independent_op.axes)


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
    if out_axes is None:
        out_axes = y.axes.recurrent_axes() + y.axes.batch_axes()
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
        if x.scalar and x.constant:
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
        if x.scalar and x.constant:
            if x.const == 0:
                rep = x
            elif x.const == 1:
                rep = y
            elif x.const == -1:
                rep = negative(y)
        elif y.scalar and y.constant:
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
        if x.scalar and x.constant:
            if x.const == 0:
                rep = y
        elif y.scalar and y.constant:
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
        if x.scalar and x.constant:
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
