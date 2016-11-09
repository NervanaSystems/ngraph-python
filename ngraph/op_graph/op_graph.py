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
# ----------------------------------------------------------------------------
from __future__ import division

from contextlib import contextmanager

import collections
import inspect
import cachetools
import numpy as np
from builtins import object, str

from ngraph.op_graph.axes import TensorDescription, \
    make_axis, make_axes, Axes, FlattenedAxis, PaddedAxis, SlicedAxis, default_dtype, \
    default_int_dtype
from ngraph.util.names import NameableValue
from ngraph.util.threadstate import get_thread_state
from ngraph.util.ordered import OrderedSet


def tensor_descriptions(args):
    """
    A list of tensor descriptions for Ops.

    Arguments:
      args: A list of Ops.

    Returns:
      A list of the Op's tensor descriptions.
    """
    return (arg.tensor_description() for arg in args)


def tdcache():
    """
    Decorator to mark tensor description method as cached.

    Returns:
        Cache decorator set to use a particular cache.

    """
    return cachetools.cached(cache=tdcache.tensor_description_cache)

tdcache.tensor_description_cache = {}


class DebugInfo(object):
    """Mixin that captures file/line location of an object's creation."""

    def __init__(self, **kwargs):
        # TODO This is a good first cut for debugging info, but it would be nice to
        # TODO be able to reliably walk the stack back to user code rather than just
        # TODO back past this constructor
        super(DebugInfo, self).__init__(**kwargs)
        frame = None
        try:
            frame = inspect.currentframe()
            while frame.f_locals.get('self', None) is self:
                frame = frame.f_back
            while frame:
                filename, lineno, function, code_context, index = inspect.getframeinfo(
                    frame)
                if -1 == filename.find('ngraph/op_graph'):
                    break
                frame = frame.f_back

            self.filename = filename
            self.lineno = lineno
            self.code_context = code_context
        finally:
            del frame

    @property
    def file_info(self):
        """
        Return file location that created the node.

        Returns:
            String with file location that created the node.

        """
        return 'File "{filename}", line {lineno}'.format(
            filename=self.filename, lineno=self.lineno)


class Op(NameableValue, DebugInfo):
    """
    Any operation that can be in an AST.

    Arguments:
        args: Values used by this node.
        const: The value of a constant Op, or None,
        constant (bool): The Op is constant.  Default False.
        forward: If not None, the node to use instead of this node.
        initializers: List of one-time initializations to run before the op.
        persistent (bool): The value will be retained from computation to computation and
            not shared.  Default False.
        reference (bool): The storage is accessed via a reference.  Default False.
        tags: String or a set of strings used for filtering in searches.
        trainable (bool): The value is trainable.  Default False.
        kwargs: Args defined in related classes.

    Attributes:
        const: The value of a constant.
        constant (bool): The value is constant.
        initializers (list): Additional Ops to run before this Op is run the first time.
        other_deps (OrderedSet): Ops in addtion to args that must run before this op.
        persistent (bool): The value will be retained from computation to computation and
            not shared.  Always True if reference is set.
        reference (bool): The storage is accessed via a reference.  Implies persistent.
        schemas: Information about how the Op was generated.
        tags: Set of strings used for filtering in searches.
        trainable: The value is trainable.
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

    # The thread's global user_deps map
    get_thread_state().user_deps = [dict()]

    @staticmethod
    def _get_thread_user_deps():
        return get_thread_state().user_deps

    @staticmethod
    @contextmanager
    def saved_user_deps(user_deps_map=None):
        """
        Switches the user_deps map within a context.

        The user_deps of an Op are Ops that must run before the Op is used. When Ops are
        generated outside of the normal stream, such as initializions that run once before any
        computation, they must be isolated from the normal tracking of variable pre-dependencies.

        Arguments:
            user_deps_map:  The new user deps map to use. If not provided, one is created
            and returned.

        """
        if user_deps_map is None:
            user_deps_map = dict()
        try:
            Op._get_thread_user_deps().append(user_deps_map)
            yield (user_deps_map)
        finally:
            Op._get_thread_user_deps().pop()

    @property
    def user_deps(self):
        """

        :return:
            Set of Ops the must come before this Op is used.  See SetItem.
        """
        return Op._get_thread_user_deps()[-1].get(self, OrderedSet())

    @user_deps.setter
    def user_deps(self, value):
        Op._get_thread_user_deps()[-1][self] = value

    def __init__(self,
                 args=(),
                 tags=None,
                 const=None,
                 constant=False,
                 initializers=None,
                 persistent=True,
                 reference=False,
                 trainable=False,
                 **kwargs):
        super(Op, self).__init__(**kwargs)
        self.__args = ()
        self.tags = set()
        self.args = args
        # TODO: is this ok?  __repr__ wants a .name
        if self.name is None:
            self.name = 'empty_name'

        if tags is not None:
            if isinstance(tags, collections.Iterable) and \
                    not isinstance(tags, (bytes, str)):
                self.tags.update(tags)
            else:
                self.tags.add(tags)

        # List to keep generation deterministic
        self.other_deps = OrderedSet()
        for arg in self.args:
            for dep in arg.user_deps:
                self.add_other_dep(dep)
        self.schemas = []
        self._adjoints = None
        self.const = const
        self.is_constant = constant
        self.initializers = OrderedSet()
        if initializers is not None:
            for initializer in initializers:
                self.add_initializer(initializer)
        self.__persistent = persistent
        self.reference = reference
        self.trainable = trainable

        ops = Op._get_thread_ops()[-1]
        if ops is not None:
            ops.append(self)
        self.style = {}
        self.ops = []
        self.__forward = None

    @property
    def args(self):
        """All the inputs to this node."""
        return self.__args

    @args.setter
    def args(self, args):
        """
        Replace old inputs with new inputs.

        Arguments:
            args: New arguments
        """
        self.__args = tuple(args)

    @staticmethod
    def visit_input_closure(roots, fun):
        """
        "Bottom-up" post-order traversal of root and their inputs.

        Nodes will only be visited once, even if there are multiple routes to the
        same Node.

        Arguments:
            roots: root set of nodes to visit
            fun: Function to call on each visited node

        Returns:
            None
        """
        visited = set()

        def visit(node):
            """
            Recursively visit all nodes used to compute this node.

            Arguments:
                node: the node to visit

            Returns:
                None
            """
            node = node.forwarded
            node.update_forwards()
            if node not in visited:
                for n in node.other_deps + list(node.args):
                    visit(n)
                fun(node)
                visited.add(node)

        for node in roots:
            visit(node)

    @property
    def forward(self):
        """
        If not None, self has been replaced with forward.

        When set, invalidates cached tensor descriptions.

        Returns:
             None or the replacement.
        """
        return self.__forward

    @forward.setter
    def forward(self, value):
        self.__forward = value
        # Transfer the other_deps to value. Initializations have already been captured.
        for dep in self.other_deps:
            value.add_other_dep(dep)
        tdcache.tensor_description_cache.clear()

    @property
    def forwarded(self):
        """
        Finds the op that handles this op.

        Returns:
             Follows forwarding to the op that shoud handle this op.
        """
        result = self
        while True:
            if not result.__forward:
                return result
            result = result.__forward

    def add_other_dep(self, dep):
        # Add the dep to the op that actually does the work.
        self.device_op.other_deps.add(dep.forwarded)

    def add_initializer(self, init):
        self.initializers.add(init)

    def update_forwards(self):
        """
        Updates internal op references with their forwarded versions.
        """

        self.args = tuple(arg.forwarded for arg in self.args)
        other_deps = self.other_deps
        self.other_deps = OrderedSet()
        for op in other_deps:
            self.add_other_dep(op)
        self.initializers = [op.forwarded for op in self.initializers]

    def replace_self(self, rep):
        self.forward = rep

    @property
    def assignable(self):
        """

        Returns: True if the tensor can be assigned to.

        """
        return not self.is_constant

    @property
    def is_scalar(self):
        return 0 == len(self.axes)

    @property
    def scalar_op(self):
        if not self.is_scalar:
            raise ValueError()
        return self

    @property
    def persistent(self):
        """

        Returns: True if value is not shared and is retained through computation.  Always true
        for a reference.

        """
        return self.__persistent or self.reference

    @property
    def is_device_op(self):
        """

        Returns:
            True if the Op executes on the device.
        """
        return True

    @property
    def device_op(self):
        """
        Returns the op that performs the operations on the device.

        Returns: self
        """
        return self

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
        params = OrderedSet()

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

        Op.visit_input_closure([self], visitor)

        return params

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
            return constant(1)
        else:
            return placeholder(self.axes)

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
        Op.visit_input_closure(results, lambda o: ordered_ops.append(o))
        return ordered_ops

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

    def __str__(self):
        return self.graph_label

    def __repr__(self):
        return '<{cl}({gl}):{id}>'.format(
            cl=self.__class__.__name__,
            gl=self.graph_label_type,
            id=id(self)
        )


def as_op(x):
    """
    Finds an Op appropriate for x.

    If x is an Op, it returns x. Otherwise, constant(x) is returned.

    Arguments:
      x: Some value.

    Returns:
      Op:
    """
    if isinstance(x, Op):
        return x

    return constant(x)


def as_ops(xs):
    """
    Converts an iterable of values to a tuple of Ops using as_op.

    Arguments:
        xs: An iterable of values.

    Returns:
        A tuple of Ops.
    """
    return tuple(as_op(x) for x in xs)


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
    def is_device_op(self):
        """

        Returns:
            False, because this is run from the CPU.
        """
        return False


class SetItem(Op):
    """
    tensor[item] = val.

    Arguments:
        tensor (AssignableTensorOp): An assignable TensorOp.
        item: The index.
        val: The value to assign.
        force (bool): Override constant check on tensor.
        **kwargs: Args for related classes.
    """

    def __init__(self, tensor, item, val, force=False, **kwargs):
        tensor, val = as_ops((tensor, val))
        if not force and not tensor.assignable:
            raise ValueError("{} is not assignable.".format(tensor))
        val = broadcast(val, tensor.axes)
        super(SetItem, self).__init__(args=(tensor, val), **kwargs)
        self.item = item
        tensor.user_deps = OrderedSet([self])
        self.force = force


class SetItemOneDim(Op):
    def __init__(self, tensor, item, val, force=False, **kwargs):
        if val.is_scalar:
            val = val.scalar_op
        super(SetItemOneDim, self).__init__(args=(tensor, val), **kwargs)
        self.item = item
        self.force = force


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
    def is_device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.
        """
        return False


class Fill(Op):
    """
    Fill a tensor with a scalar value.

    Arguments:
        tensor (AssignableTensorOp): An assignable TensorOp.
        scalar: A scalar value.
        force (bool): Disable constant check on tensor.
    """

    def __init__(self, tensor, scalar, force=False, **kwargs):
        super(Fill, self).__init__(args=(tensor,), **kwargs)
        if not force and not tensor.assignable:
            raise ValueError("{} is not assignable.".format(tensor))
        if isinstance(scalar, TensorOp):
            if scalar.is_constant:
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
        self.dtype = default_dtype(dtype)
        if axes is not None:
            axes = make_axes(axes)
        self.__axes = axes

        self.scale = scale

    def generate_add_delta(self, adjoints, delta):
        """
        Adds delta to the backprop contribution..

        Arguments:
            adjoints: dy/dOp for all Ops used to compute y.
            delta: Backprop contribute.
        """
        if not Axes.same_elems(self.axes, delta.axes):
            raise ValueError(
                'A tensor and its adjoint must have the same axes.'
            )
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

    def __mod__(self, val):
        return mod(self, val)

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

    def __axes__(self):
        return self.axes

    @tdcache()
    def tensor_description(self):
        """
        Returns a TensorDescription describing the output of this TensorOp

        Returns:
          TensorDescription for this op.
        """
        return TensorDescription(self.axes, dtype=self.dtype, name=self.name)

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

    def insert_axis(self, index, axis):
        """
        Inserts an axis
        Arguments:
            index   : Index to insert at
            axis    : The Axis object to insert
        """
        if self.__axes is None:
            raise ValueError()
        self.__axes.insert(index, axis)

    def append_axis(self, axis):
        if self.__axes is None:
            raise ValueError()
        self.__axes.append(axis)

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

    def shape_dict(self):
        """
        Retuns: shape of this tensor as a dictionary
        """
        return self.axes.shape_dict()

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
        return self.forwarded.tensor_description().value


class ReshapeOp(TensorOp):
    def __init__(self, x, **kwargs):
        super(ReshapeOp, self).__init__(
            args=(x,),
            dtype=x.dtype,
            **kwargs
        )

    @property
    def is_scalar(self):
        """
        Reshape adds shape information, but we retain being a scalar.

        Returns:
            True if the value comes from a scalar.

        """
        return self.args[0].is_scalar

    @property
    def scalar_op(self):
        return self.args[0].scalar_op

    @property
    def is_device_op(self):
        """
        Returns:
            False, because this is handled by the transformer.
        """
        return False

    @property
    def device_op(self):
        """
        Returns the op that performs the operations on the device.

        Returns: Argument's device_op

        """
        return self.args[0].device_op


class Transpose(ReshapeOp):
    """
    Used to reverse the axes of a tensor.

    Arguments:
        x: A tensor.
    """
    def __init__(self, x, **kwargs):
        super(Transpose, self).__init__(
            x,
            axes=reversed(x.axes),
            **kwargs
        )

    @tdcache()
    def tensor_description(self):
        return self.args[0].tensor_description().transpose(self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, Transpose(delta))


class AxesCastOp(ReshapeOp):
    """
    Used to label a tensor with known axes, without altering its value

    Arguments:
        x: A tensor.
        axes: The new axes.

    """
    def __init__(self, x, axes, **kwargs):
        super(AxesCastOp, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        return self.args[0].tensor_description().cast(self.axes, self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, cast_axes(delta, x.axes))


def cast_axes(tensor, axes, name=None):
    """
    Cast the axes of a tensor to new axes.

    Args:
        tensor (TensorOp): The tensor.
        axes (Axes): The new axes.
        name (String, optional): The name of the result.

    Returns:
        TensorOp: The tensor with new axes.
    """
    return AxesCastOp(tensor, axes, name=name)


class ExpandDims(ReshapeOp):
    """
    Adds additional axes into a tensor.
    Arguments:
        x: The tensor.
        axis: The additional axis.
        dim: The position to add the axes.
    """

    def __init__(self, x, axis, dim, **kwargs):
        axes = []
        axes.extend(x.axes[:dim])
        axes.append(axis)
        axes.extend(x.axes[dim:])
        axes = make_axes(axes)
        super(ExpandDims, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        """
        TODO.
        Arguments:
        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return x.broadcast(self.axes)

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
            sum(delta, reduction_axes=delta.axes - x.axes)
        )


class ResultHandle(ReshapeOp):
    def __init__(self, x, **kwargs):
        super(ResultHandle, self).__init__(
            x, **kwargs
        )

    @property
    def device_op(self):
        return self

    @tdcache()
    def tensor_description(self):
        td, = tensor_descriptions(self.args)
        return td.broadcast(td.axes, self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(delta)


class BroadcastOp(ReshapeOp):
    """
    Used to add additional axes for a returned derivative.

    Arguments:
        x: The tensor to broadcast.
        axes: The new axes.
    """

    def __init__(self, x, axes, **kwargs):
        assert Axes.check_broadcast(x.axes, axes)
        super(BroadcastOp, self).__init__(
            x, axes=axes, **kwargs
        )

    @tdcache()
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        td, = tensor_descriptions(self.args)
        return td.broadcast(self.axes, self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sum(
            delta,
            reduction_axes=delta.axes - x.axes,
            out_axes=x.axes
        ))


def broadcast(x, axes):
    """
    Broadcast the axes of x.

    Args:
        x (TensorOp): The tensor.
        axes: New axes.

    Returns:
        TensorOp: Tensor with additional axes.
    """
    axes = make_axes(axes)
    if x.axes == axes:
        return x
    return BroadcastOp(x, axes)


def axes_with_order(x, axes):
    """
    Return a tensor with a different axes order.

    Args:
        x (TensorOp): The tensor.
        axes (Axes): A permutation of the axes of the tensor.

    Returns:
        TensorOp: The new tensor.

    """
    if x.axes == axes:
        return x
    return ReorderAxes(x, axes)


class ReorderAxes(ReshapeOp):
    """
    Reorders the axes of a tensor, without making a copy.

    Arguments:
        x: The tensor whose axes to reorder.
        axes: The new axes.
    """
    def __init__(self, x, axes, **kwargs):
        if not Axes.same_elems(x.axes, axes):
            raise ValueError(
                'The input and output axes must have the same elements.'
            )
        super(ReorderAxes, self).__init__(
            x, axes=axes, **kwargs
        )

    @tdcache()
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        td, = tensor_descriptions(self.args)
        return td.reorder(self.axes, self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, axes_with_order(
            delta,
            x.axes
        ))


class Slice(ReshapeOp):
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

            axes = make_axes(axes)

        super(Slice, self).__init__(
            x,
            axes=axes,
            **kwargs
        )

        self.slices = slices

    @tdcache()
    def tensor_description(self):
        """
        TODO.

        Arguments:

        Returns:
          TODO
        """
        x, = tensor_descriptions(self.args)
        return x.slice(self.slices, self.axes, self.name)

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
    axes = x.axes[:pos] + x.axes[pos + 1:]
    return Slice(x, ss, axes=axes)


class Flatten(ReshapeOp):
    def __init__(self, x, axes, **kwargs):
        if isinstance(x, ReshapeOp):
            x = Dimshuffle(x, axes=x.axes)
        assert Axes.check_flatten(x.axes, axes)
        super(Flatten, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        x, = tensor_descriptions(self.args)
        return x.flatten(self.axes, name=self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, unflatten(
            delta,
            axes=x.axes
        ))


def flatten(x, axes=None, **kwargs):
    if axes is None:
        if len(x.axes) == 1:
            return x
        else:
            axes = Axes((FlattenedAxis(x.axes),))

    if x.is_scalar:
        return x

    if isinstance(x, Flatten) and x.axes == axes:
        return x
    return Flatten(x, axes=axes, **kwargs)


def flatten_at(x, idx):
    if idx == 0 or idx == len(x.axes):
        return flatten(x)
    else:
        return flatten(x, make_axes((
            make_axes(x.axes[:idx]).flatten(),
            make_axes(x.axes[idx:]).flatten()
        )))


class Unflatten(ReshapeOp):
    def __init__(self, x, axes=None, **kwargs):
        if axes is None:
            axes = []
            for axis in x.axes:
                axes.extend(axis.axes)
        axes = make_axes(axes)
        assert Axes.check_unflatten(x.axes, axes)
        super(Unflatten, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        x, = tensor_descriptions(self.args)
        return x.unflatten(self.axes, self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, flatten(
            delta,
            axes=x.axes
        ))


def unflatten(x, axes=None, **kwargs):
    if axes is None:
        axes = []
        for axis in x.axes:
            axes.extend(axis.axes)
    axes = Axes(axes)
    if axes == x.axes:
        return x
    return Unflatten(x, axes=axes, **kwargs)


class AssignableTensorOp(TensorOp):
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
        super(AssignableTensorOp, self).__init__(persistent=persistent, **kwargs)
        self.input = input

        with Op.saved_user_deps():
            # Run initializations in a clean context so their SetItems don't modify user_deps
            # for the main computations.
            # TODO Maybe we want to use a single context for all of initialization.  We would
            # need to do the following in a separate method called during transformation.
            if init is not None:
                capture = []
                with Op.captured_ops(capture):
                    init.fill(self)
                for c in capture:
                    self.add_initializer(c)
            elif callable(initial_value):
                self.add_initializer(assign(self, initial_value()))
            elif initial_value is not None:
                self.add_initializer(assign(self, initial_value))

    @property
    def defs(self):
        """

        Returns:
            AssignableTensorOp is not executed, so its appearance in the instruction stream does
            not affect liveness of its value.

        """
        return []

    @property
    def is_device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.

        """
        return False


def constant(const, axes=None, dtype=None, name=None):
    """
    Makes a constant scalar/tensor.  For a tensor, constant provides the opportunity
        to supply axes.  Scalar/NumPytensor arguments are usually automatically converted to
        tensors, but constant may be used to supply axes or in the rare cases where constant
        is not automatically provided.

    Args:
        const: The constant, a scalar or a NumPy array.
        axes: The axes for the constant.
        dtype (optional): The dtype to use.
    Returns:
        An AssignableTensorOp for the constant.
    """
    graph_label_type = "<Const({})>".format(const)
    val = AssignableTensorOp(axes=axes, constant=True, persistent=True, trainable=False,
                             graph_label_type=graph_label_type, dtype=dtype, name=name)
    nptensor = np.asarray(const, dtype=val.dtype)

    if not val.has_axes:
        val.axes = make_axes([make_axis(x, match_on_length=True) for x in nptensor.shape])

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

    val.add_initializer(InitTensor(val, value_fun))

    return val


def is_constant(value):
    """
    Test an Op to see if it is a constant.

    Args:
        value: An Op

    Returns: True if value is a constant.

    """
    return isinstance(value, AssignableTensorOp) and value.is_constant


def is_constant_scalar(value):
    """
    Tests an Op to see if it is a constant scalar.

    Args:
        value: An Op.

    Returns: True if value is a constant scalar.

    """
    return value.is_constant and value.is_scalar


def constant_value(value):
    """
    Returns the constant value of an Op.

    Args:
        value (TensorOp): A constant op.

    Returns: The constant value.

    """
    if not is_constant(value):
        raise ValueError()
    return value.const


def constant_storage(axes, dtype=None, name=None, initial_value=None):
    """
    A tensor that is supposed to remain constant.

    Args:
        axes (Axes): The axes of the constant storage.
        dtype (optional): The dtype of the storage.
        name (String, optional): A name for the storage.
        initial_value: A host constant or callable. If a callable, will be called
            to produce the value.


    Returns:
        AssignableTensorOp: The constant storage.
    """

    return AssignableTensorOp(graph_label_type="constant",
                              constant=True, persistent=True,
                              trainable=False,
                              axes=axes, dtype=dtype, name=name,
                              initial_value=initial_value)


def placeholder(axes, dtype=None, initial_value=None, name=None):
    """
    A persistent tensor to be initialized from the CPU.

    Args:
        axes (Axes): The axes of the placeholder.
        dtype (optional): The dtype of the placeholder.
        name (String, optional): The name of the placeholder.
        initial_value (optional): A host constant or callable. If callable, will
            be called to generate an initial value.

    Returns:
        AssignableTensorOp: The placeholder.

    """
    return AssignableTensorOp(graph_label_type="placeholder",
                              constant=False, persistent=True, trainable=False,
                              input=True,
                              axes=axes, dtype=dtype, name=name,
                              initial_value=initial_value)


def temporary(axes, dtype=None, name=None, init=None):
    """
    Temporary storage.

    Statically allocates storage that may be reused outside of the scope of the values.

    Args:
        axes (Axes): The axes of the storage.
        dtype (optional): The dtype of the storage.
        name (String, optional): A name for the storage.
        init (optional): Neon-style init.

    Returns:
        AssignableTensorOp: The placeholder.

    """
    return AssignableTensorOp(graph_label_type="Temp",
                              constant=False, persistent=True,
                              trainable=False,
                              init=init,
                              axes=axes, dtype=dtype, name=name)


def persistent_tensor(axes, dtype=None, initial_value=None, name=None, init=None):
    """
    Persistent storage.

    Storage that will retain its value from computation to computation.

    Args:
        axes (Axes): The axes of the persistent storage.
        dtype (optional): The dtype of the persistent storage.
        initial_value (optional): A host constant or callable. If callable, will
            be called to generate an initial value.
        name (String, optional): The name of the persistent storage.
        init (optional): Neon init.

    Returns:
        AssignableTensorOp: The persistent storage.

    """
    return AssignableTensorOp(graph_label_type="Persistent",
                              constant=False, persistent=True,
                              trainable=False,
                              axes=axes, dtype=dtype, name=name,
                              initial_value=initial_value,
                              init=init)


def variable(axes, dtype=None, name=None, initial_value=None, init=None):
    """
    A trainable tensor.

    Args:
        axes (Axes): Axes for the variable.
        dtype (optional): The dtype for the tensor.
        initial_value: A constant or callable. If a callable, the callable
            will be called to provide an initial value.
        init: For neon backwards-compatibility.

    Returns:
        AssignableTensorOp: The variable.

    """
    return AssignableTensorOp(graph_label_type="Variable",
                              constant=False, persistent=True,
                              trainable=True, axes=axes, name=name,
                              dtype=dtype,
                              initial_value=initial_value, init=init)


class Stack(TensorOp):
    """
    Joins a list of identically-axed tensors along a new axis.

    Arguments:
        x_list: A list of identically-axed tensors to join.
        axis: The axis to select joined tensors.
        pos: The position within the axes of the x_list tensors to insert axis in the result.
        **kwargs: Other args for TensorOp.

    Parameters:
        pos: The position of the join axis.
    """
    def __init__(self, x_list, axis, pos=0, **kwargs):
        self.pos = pos
        x_axes = x_list[0].axes
        axes = make_axes(tuple(x_axes[:pos]) + (axis,) + tuple(x_axes[pos:]))
        super(Stack, self).__init__(args=tuple(x_list), axes=axes)

    def generate_adjoints(self, adjoints, delta, *x_list):
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
        return [self.tensor_description().slice(self.slices, self.input_axes, self.name), x]

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
        val.add_initializer(InitTensor(val, value_fun))
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
        val.add_initializer(InitTensor(val, value_fun))
        return val


class AllReduce(Op):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(AllReduce, self).__init__(args=(x,), **kwargs)


class ElementWise(TensorOp):
    pass


class UnaryElementwiseOp(ElementWise):
    def __init__(self, x, **kwargs):
        super(UnaryElementwiseOp, self).__init__(
            args=(x,),
            axes=x.axes,
            **kwargs
        )


def create_unary_elementwise(cls_name, generate_adjoints=None):
    d = {}
    if generate_adjoints is not None:
        d['generate_adjoints'] = generate_adjoints
    return type(cls_name, (UnaryElementwiseOp,), d)


def neg_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, -delta)


negative = create_unary_elementwise('negative', neg_adjoints)


def abs_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, sign(x) * delta)


absolute = create_unary_elementwise('absolute', abs_adjoints)


def sin_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, delta * cos(x))


sin = create_unary_elementwise('sin', sin_adjoints)


def cos_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, -delta * sin(x))


cos = create_unary_elementwise('cos', cos_adjoints)


def tanh_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, delta * (1.0 - self * self))


tanh = create_unary_elementwise('tanh', tanh_adjoints)


def exp_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, delta * self)


exp = create_unary_elementwise('exp', exp_adjoints)


def log_adjoints(self, adjoints, delta, x):
    def do_adjoints(delta, x):
        if isinstance(x, Divide):
            a, b = x.args
            do_adjoints(delta, a)
            do_adjoints(-delta, b)
        elif isinstance(x, exp):
            x.args[0].generate_add_delta(adjoints, delta)
        else:
            x.generate_add_delta(adjoints, delta / x)
    do_adjoints(delta, x)


log = create_unary_elementwise('log', log_adjoints)

safelog_cutoff = 50.0


def safelog(x, limit=np.exp(-safelog_cutoff)):
    return log(maximum(x, limit))


def reci_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, -self * self * delta)


reciprocal = create_unary_elementwise('reciprocal', reci_adjoints)


sign = create_unary_elementwise('sign')


def square_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, 2.0 * delta * x)


square = create_unary_elementwise('square', square_adjoints)


def sqrt_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, .5 * delta * self)


sqrt = create_unary_elementwise('sqrt', sqrt_adjoints)


class BinaryElementWiseAxesOp(ElementWise):
    def __init__(self, x, y, **kwargs):
        self.kwargs = kwargs
        x, y = as_ops((x, y))
        axes = x.axes + y.axes
        x = broadcast(x, axes)
        y = broadcast(y, axes)

        super(BinaryElementWiseAxesOp, self).__init__(
            args=(x, y),
            axes=axes,
            **kwargs
        )

    @property
    def one_dimensional(self):
        x, y = self.args
        return len(x.axes) <= 1 and len(y.axes) <= 1

    @property
    def zero_dimensional(self):
        x, y = self.args
        return len(x.axes) == 0 and len(y.axes) == 0


class BinaryElementWiseLowDOp(ElementWise):
    def __init__(self, x, y, **kwargs):
        self.kwargs = kwargs

        if x.is_scalar:
            x = x.scalar_op
        if y.is_scalar:
            y = y.scalar_op

        super(BinaryElementWiseLowDOp, self).__init__(
            args=(x, y),
            **kwargs
        )

    @property
    def one_dimensional(self):
        x, y = self.args
        return len(x.axes) <= 1 and len(y.axes) <= 1

    @property
    def zero_dimensional(self):
        x, y = self.args
        return len(x.axes) == 0 and len(y.axes) == 0


def create_binary_elementwise(name,
                              one_dim_name,
                              zero_dim_name,
                              func_name=None,
                              generate_adjoints=None,
                              one_dim_generate_adjoints=None,
                              zero_dim_generate_adjoints=None):
    d = {}
    if generate_adjoints is not None:
        d['generate_adjoints'] = generate_adjoints
    BinClass = type(name, (BinaryElementWiseAxesOp,), d)

    d = {}
    if one_dim_generate_adjoints is not None:
        d['generate_adjoints'] = one_dim_generate_adjoints
    OneDimBinClass = type(one_dim_name, (BinaryElementWiseLowDOp,), d)

    d = {}
    if zero_dim_generate_adjoints is not None:
        d['generate_adjoints'] = zero_dim_generate_adjoints
    ZeroDimBinClass = type(zero_dim_name, (BinaryElementWiseLowDOp,), d)

    def reduce_to_oned(self):
        x, y = self.args
        if x.is_scalar and y.is_scalar:
            return ZeroDimBinClass(x.scalar_op, y.scalar_op, axes=self.axes, **self.kwargs)
        else:
            x, y = flatten(x), flatten(y)
            return unflatten(OneDimBinClass(x, y, axes=FlattenedAxis(self.axes), **self.kwargs))
    BinClass.reduce_to_oned = reduce_to_oned

    if func_name is None:
        return BinClass, OneDimBinClass, ZeroDimBinClass
    else:
        def func(*args, **kwargs):
            return BinClass(*args, **kwargs)
        func.__name__ = func_name
        return BinClass, OneDimBinClass, ZeroDimBinClass, func


def add_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta)
    y.generate_add_delta(adjoints, delta)


Add, AddOneDim, AddZeroDim, add = create_binary_elementwise(
    'AddOp', 'AddOneDim', 'AddZeroDim', 'addX', add_adjoints
)


def add(x, y, name=None):
    """
    Returns a TensorOp for the sum of x and y.

    Args:
        x (TensorOp): The first input.
        y (TensorOp):  The second input.
        name (String, optional): A name for the sum.

    Returns:
        TensorOp: x + y

    """
    return Add(x, y)


def subtract_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta)
    y.generate_add_delta(adjoints, -delta)


Subtract, SubtractOneDim, SubtractZeroDim, subtract = create_binary_elementwise(
    'Subtract', 'SubtractOneDim', 'SubtractZeroDim',
    'subtract', subtract_adjoints
)


def multiply_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * y)
    y.generate_add_delta(adjoints, x * delta)


Multiply, MultiplyOneDim, MultiplyZeroDim, multiply = create_binary_elementwise(
    'Multiply', 'MultiplyOneDim', 'MultiplyZeroDim',
    'multiply', multiply_adjoints
)


def divide_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * self / x)
    y.generate_add_delta(adjoints, -delta * self / y)


Divide, DivideOneDim, DivideZeroDim, divide = create_binary_elementwise(
    'Divide', 'DivideOneDim', 'DivideZeroDim',
    'divide', divide_adjoints
)

Mod, ModOneDim, ModZeroDim, mod = create_binary_elementwise(
    'Mod', 'ModOneDim', 'ModZeroDim',
    'mod', None
)


def maximum_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, greater(x, y) * delta)
    y.generate_add_delta(adjoints, greater(y, x) * delta)


Maximum, MaximumOneDim, MaximumZeroDim, maximum = create_binary_elementwise(
    'Maximum', 'MaximumOneDim', 'MaximumZeroDim', 'maximum', maximum_adjoints
)


def minimum_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, less(x, y) * delta)
    y.generate_add_delta(adjoints, less(y, x) * delta)


Minimum, MinimumOneDim, MinimumZeroDim, minimum = create_binary_elementwise(
    'Minimum', 'MinimumOneDim', 'MinimumZeroDim', 'minimum', minimum_adjoints
)


def power_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * y * self / x)
    y.generate_add_delta(adjoints, delta * self * log(x))


Power, PowerOneDim, PowerZeroDim, power = create_binary_elementwise(
    'Power', 'PowerOneDim', 'PowerZeroDim', 'power', power_adjoints
)


Equal, EqualOneDim, EqualZeroDim, equal\
    = create_binary_elementwise('Equal', 'EqualOneDim', 'EqualZeroDim', 'equal')


NotEqual, NotEqualOneDim, NotEqualZeroDim, not_equal\
    = create_binary_elementwise('NotEqual', 'NotEqualOneDim', 'NotEqualZeroDim', 'not_equal')


Greater, GreaterOneDim, GreaterZeroDim, greater\
    = create_binary_elementwise('Greater', 'GreaterOneDim', 'GreaterZeroDim', 'greater')


Less, LessOneDim, LessZeroDim, less\
    = create_binary_elementwise('Less', 'LessOneDim', 'LessZeroDim', 'less')


GreaterEqual, GreaterEqualOneDim, GreaterEqualZeroDim, greater_equal\
    = create_binary_elementwise(
        'GreaterEqual', 'GreaterEqualOneDim',
        'GreaterEqualZeroDim', 'greater_equal'
    )


LessEqual, LessEqualOneDim, LessEqualZeroDim, less_equal\
    = create_binary_elementwise('LessEqual', 'LessEqualOneDim', 'LessEqualZeroDim', 'less_equal')


class Dimshuffle(TensorOp):
    def __init__(self, x, axes, **kwargs):
        if not Axes.same_elems(x.axes, axes):
            raise ValueError(
                'The input and output axes must have the same elements.'
            )
        old_poss = []
        for axis in axes:
            old_pos = Axes.find_axis(x.axes, axis)
            old_poss.append(old_pos)
        self.old_axis_positions = tuple(old_poss)

        super(Dimshuffle, self).__init__(
            args=(x,),
            axes=axes
        )

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(
            adjoints,
            delta
        )


class DotOp(TensorOp):
    def __init__(self, x, y, **kwargs):
        self.x_reduction_axes = x.axes.intersect(y.axes.get_dual())
        self.y_reduction_axes = self.x_reduction_axes.get_dual(1)
        self.x_out_axes = x.axes - self.x_reduction_axes
        self.y_out_axes = y.axes - self.y_reduction_axes

        if len(self.x_out_axes.intersect(self.y_out_axes)):
            raise ValueError("Intersection in out axes for dot.")
        axes = self.x_out_axes + self.y_out_axes

        super(DotOp, self).__init__(
            args=(x, y), axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        Generates the adjoint contributions for x and y.

        On input, x axes can be grouped as IJ* and y axes as JK where
        J* is predecessor of J.

        Axes will be:
            Delta: IK.
            x adj: IJ*
            y adj: JK

        For x adj, we have IK and JK, so we dual K for delta and J for y
        to get IK* and J*K for a product of IJ*.

        For y adj, we have IJ* and IK, to get JK, so we dual I and undual
        J* in x, to get I*J and IK for a product of JK.

        Args:
            adjoints: The adjoints for the deriv being computed.
            delta (TensorOp): The backprop op.
            x (TensorOp): The x argument.
            y (TensorOp): The y argument.

        """
        x.generate_add_delta(
            adjoints,
            dot(dualed_axes(delta, self.y_out_axes, -1, 0),
                dualed_axes(y, self.y_reduction_axes, -1, 0))
        )
        y.generate_add_delta(
            adjoints,
            dot(dualed_axes(x, self.x_out_axes, -1, +1), delta)
        )


def dualed_axes(x, filter, in_dual_offset, out_dual_offset):
    """
    Cast axes to a dual offset of axes depending on membership in dual_axes.

    In a dot(a, b), each pair of axes (a_i, b_j) between a and b where
    a_i = b_j - 1
    will be paired for multiplication and then summing.

    Args:
        x (TensorOp): A tensor.
        filter: A collection of axes.
        in_dual_offset: Dual shift amount for axes in filter.
        out_dual_offset: Dual shift amount for axes not in filter.

    Returns:
        TesnsorOp: x with axes cast.

    """
    def dualed(axis):
        if axis in filter:
            return axis + in_dual_offset
        else:
            return axis + out_dual_offset
    return cast_axes(x, (dualed(axis) for axis in x.axes))


def dot(x, y, name=None):
    """
    The dot product of x and y.

    Reduction axes in x are those whose dual offset is one less than an axis in y.

    Args:
        x (TensorOp): First argument.
        y (TensorOp): Second argumnent.
        name (String, optional): Name for the TensorOp.

    Returns:
        TensorOp: The dot product.

    """
    return DotOp(x, y, name=name)


def squared_L2(x):
    """
    Returns the dot of x and y, with the axes of x set to their dual offset.

    Args:
        x (TensorOp): The first value, axes shifted down by 1.
        y (TensorOp): The second value.

    Returns:
        TensorOp: The result.

    """
    return dot(dualed_axes(x, x.axes, -1, 0), x)


class LowDimensionalDot(TensorOp):
    def __init__(self, x, y, axes, **kwargs):
        super(LowDimensionalDot, self).__init__(args=(x, y), axes=axes, **kwargs)


class DotOneDimensional(LowDimensionalDot):
    def __init__(self, x, y, axes, **kwargs):
        assert len(x.axes) == 1 and len(y.axes) == 1
        super(DotOneDimensional, self).__init__(
            x, y, axes, **kwargs
        )


class DotTwoDimensional(LowDimensionalDot):
    def __init__(self, x, y, axes, **kwargs):
        assert len(x.axes) == 2 and len(y.axes) == 2
        super(DotTwoDimensional, self).__init__(
            x, y, axes, **kwargs
        )


class DotTwoByOne(LowDimensionalDot):
    def __init__(self, x, y, axes, **kwargs):
        assert len(x.axes) == 2 and len(y.axes) == 1
        super(DotTwoByOne, self).__init__(
            x, y, axes, **kwargs
        )


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
    must_reduce = True

    def __init__(self, x, reduction_axes=None, out_axes=None, dtype=None, **kwargs):
        if reduction_axes is None and out_axes is None:
            reduction_axes = x.axes.sample_axes() - x.axes.recurrent_axes()
            out_axes = x.axes - reduction_axes
        elif reduction_axes is None:
            out_axes = make_axes(out_axes)
            reduction_axes = x.axes - out_axes
        elif out_axes is None:
            reduction_axes = make_axes(reduction_axes)
            out_axes = x.axes - reduction_axes
        else:
            out_axes = make_axes(out_axes)
            reduction_axes = make_axes(reduction_axes)
        assert reduction_axes.intersect(out_axes) == make_axes(())

        self.reduction_axes = reduction_axes
        self.kwargs = kwargs

        super(ReductionOp, self).__init__(
            args=(x,),
            axes=out_axes,
            dtype=dtype
        )
        assert self.valid

    @property
    def valid(self):
        return True


def create_twod_reduction_op(name,
                             red_cls,
                             two_dim_generate_adjoints=None):
    def valid_two(self):
        x, = self.args
        return len(x.axes) == 2\
            and self.reduction_axes == x.axes[:1]\
            and self.out_axes == x.axes[1:]
    d = {'valid': valid_two, 'must_reduce': False}

    if two_dim_generate_adjoints is not None:
        d['generate_adjoints'] = two_dim_generate_adjoints

    RedTwoDimClass = type(name, (red_cls,), d)
    return RedTwoDimClass


def create_oned_reduction_op(name,
                             red_cls,
                             one_dim_generate_adjoints=None):
    def valid_one(self):
        x, = self.args
        return len(x.axes) == 1\
            and self.reduction_axes == x.axes

    d = {'valid': valid_one, 'must_reduce': False}
    if one_dim_generate_adjoints is not None:
        d['generate_adjoints'] = one_dim_generate_adjoints

    RedOneDimClass = type(name, (red_cls,), d)
    return RedOneDimClass


def create_reduction_op(name,
                        two_dim_name,
                        one_dim_name,
                        func_name=None,
                        generate_adjoints=None,
                        two_dim_generate_adjoints=None,
                        one_dim_generate_adjoints=None):
    d = {}
    if generate_adjoints is not None:
        d['generate_adjoints'] = generate_adjoints
    RedClass = type(name, (ReductionOp,), d)

    RedTwoDimClass = create_twod_reduction_op(
        two_dim_name,
        RedClass,
        two_dim_generate_adjoints
    )

    RedOneDimClass = create_oned_reduction_op(
        one_dim_name,
        RedClass,
        one_dim_generate_adjoints
    )

    def reduce_to_twod(self):
        x, = self.args
        reduction_axes = self.reduction_axes
        out_axes = self.axes

        if len(reduction_axes) == 0:
            return broadcast(x, axes=out_axes)
        elif len(x.axes) == 0:
            return broadcast(x, axes=out_axes)

        if len(out_axes) == 0:
            x = flatten(x)
            return RedOneDimClass(
                x,
                reduction_axes=x.axes,
                out_axes=make_axes(()),
                dtype=self.dtype,
                **self.kwargs
            )
        else:
            x = broadcast(x, axes=reduction_axes + out_axes)
            x = flatten_at(x, len(reduction_axes))

            out = RedTwoDimClass(
                x,
                reduction_axes=make_axes((x.axes[0],)),
                out_axes=make_axes((x.axes[1],)),
                dtype=self.dtype,
                **self.kwargs
            )
            out = unflatten(out)
            return broadcast(out, axes=out_axes)
    RedClass.reduce_to_twod = reduce_to_twod

    if func_name is None:
        return RedClass, RedTwoDimClass, RedOneDimClass
    else:
        def func(*args, **kwargs):
            return RedClass(*args, **kwargs)
        func.__name__ = func_name
        return RedClass, RedTwoDimClass, RedOneDimClass, func


def max_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, equal(x, self) * delta)


Max, MaxTwoDim, MaxOneDim, max = create_reduction_op(
    'Max', 'MaxTwoDim', 'MaxOneDim', 'max', max_adjoints
)


def min_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, equal(x, self) * delta)


Min, MinTwoDim, MinOneDim, min = create_reduction_op(
    'Min', 'MinTwoDim', 'MinOneDim', 'min', min_adjoints
)


def sum_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(
        adjoints,
        broadcast(delta, x.axes)
    )


Sum, SumTwoDim, SumOneDim, sum = create_reduction_op(
    'Sum', 'SumTwoDim', 'SumOneDim', 'sum', sum_adjoints
)


Argmax, ArgmaxTwoDim, ArgmaxOneDim = create_reduction_op(
    'Argmax', 'ArgmaxTwoDim', 'ArgmaxOneDim'
)


def argmax(x, dtype=None, **kwargs):
    return Argmax(x, dtype=default_int_dtype(dtype), **kwargs)


Argmin, ArgminTwoDim, ArgminOneDim = create_reduction_op(
    'Argmin', 'ArgminTwoDim', 'ArgminOneDim'
)


def argmin(x, dtype=None, **kwargs):
    return Argmin(x, dtype=default_int_dtype(dtype), **kwargs)


def assign(lvalue, rvalue, **kwargs):
    """
    Assignment; lvalue <= rvalue

    Arguments:
        lvalue: Tensor to assign to.
        rvalue: Value to be assigned.
        kwargs: options, including name
    """
    return SetItem(lvalue, (), rvalue, **kwargs)


def variance(x, out_axes=None, reduction_axes=None):
    return mean(square(x - mean(x, out_axes=out_axes, reduction_axes=reduction_axes)),
                out_axes=out_axes, reduction_axes=reduction_axes)


class tensor_size(TensorOp):
    """
    A scalar returning the total size of a tensor.
    Arguments:
        x: The tensor whose axes we are measuring.
        reduction_axes: if supplied, return the size
            of these axes instead.
        kwargs: options, including name
    """
    def __init__(self, x, reduction_axes=None, out_axes=None, **kwargs):
        if reduction_axes is None and out_axes is None:
            reduction_axes = x.axes.sample_axes()
        elif reduction_axes is None:
            reduction_axes = x.axes - out_axes
        self.reduction_axes = reduction_axes
        super(tensor_size, self).__init__(axes=())


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
        axes = make_axes(
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


class Onehot(TensorOp):
    """
    Converts a tensor containing class indices to a onehot representation.
    For example, if x is a one-dimesnional tensor with value [0, 1], and the
    number of classes is 2, we convert x to a onehot representation by replacing
    0 and 1 with vectors: 0 -> [1, 0] and 1 -> [0, 1].

    We add the added dimension in the leftmost place.

    Arguments:
        x: The tensor to convert to a onehot form.
        axis: The axis along which to construct the onehot form. It should not be
        in x and should have length equal to the number of classes.
    """
    def __init__(self, x, axis, **kwargs):
        self.axis = axis
        super(Onehot, self).__init__(
            args=(x,),
            axes=make_axes((axis,)) + x.axes,
            **kwargs
        )

    def as_two_dim(self):
        """
        Constructs a subgraph that is equivalent to this op and can be evaluated
        by a transformer that only handles two dimensions.

        Returns:
            A subgraph equivalent to this op.
        """
        x, = self.args
        if len(x.axes) > 1:
            x = flatten(x)
            out = OnehotTwoDim(x, self.axis)
            out = unflatten(
                out,
                [out.axes[0]] + list(out.axes[1].axes)
            )
            return out
        else:
            return OnehotTwoDim(x, self.axis)


def onehot(*args, **kwargs):
    return Onehot(*args, **kwargs)


class OnehotTwoDim(Onehot):
    """
    Handles conversion from one-dimensional vector of class labels
    to a two-dimensional onehot representation.

    Arguments:
        x: The tensor to convert to a onehot form.
        axis: The axis along which to construct the onehot form. It should not be
        in x and should have length equal to the number of classes.
    """
    def __init__(self, x, axis, **kwargs):
        assert len(x.axes) == 1
        super(OnehotTwoDim, self).__init__(x, axis, **kwargs)


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
        self.views = OrderedSet()


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
    adjoints = dependent_op.forwarded.adjoints()

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

    adjoint = adjoints[independent_op.forwarded]
    return broadcast(adjoint.forwarded, axes=independent_op.axes)


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
                        enable_diff_opt=True):
    """
    Computes the cross-entropy of two distributions.

    Arguments:
        y: The output of the model; each sample is a PDF.
        t: The true values; each sample is PDF.
        usebits: Use binary log.
        out_axes: Axes in result.  Default batch and reduction axes.
        enable_softmax_opt: Use optimization when y is softmax. Default True.
        enable_diff_opt: User derivative optimization when y is softmax.  Default True.

    Returns:
        The cross-entropy.
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


def cross_entropy_binary_inner(y, t, enable_sig_opt=True, enable_diff_opt=True):
    """
    Computes cross-entropy of individual samples.

    Arguments:
        y: Output of model, in range [0, 1].
        t: True values, in [0, 1].
        enable_sig_opt: Enable optimization when y is sigmoid.  Default True.
        enable_diff_opt: Enable optimization of derivative when y is sigmoid.  Default True.

    Returns:
        Cross entropy of individual samples.
    """
    result = -(safelog(y) * t + safelog(1 - y) * (1 - t))
    sigy = y.find_schema(Sigmoid)
    if sigy is not None:
        x = sigy.x
        if enable_sig_opt:
            # Simpler equivalent
            result = (1 - t) * maximum(x, -safelog_cutoff) - safelog(y)
        if enable_diff_opt:
            result.add_schema(CrossEntropyBinaryInner(x=x, y=y, t=t))

    return result


def cross_entropy_binary(y, t, usebits=False, out_axes=None,
                         enable_sig_opt=True, enable_diff_opt=True):
    """
    Computes cross-entropy.

    Arguments:
        y: Output of model, in range [0, 1]
        t: True values, in [0, 1].
        use_bits: Use binary log.
        out_axes: Axes of result; default is batch and recurrent axis.
        enable_sig_opt: Enable optimization when y is sigmoid. Default True.
        enable_diff_opt: Enable optimization of derivative when y is sigmoid. Default True.

    Returns:
        Cross entropy.
    """
    result = sum(cross_entropy_binary_inner(y, t), out_axes=out_axes,
                 enable_sig_opt=enable_sig_opt, enable_diff_opt=enable_diff_opt)

    if usebits:
        result = result * np.float(1. / np.log(2.0))
    return result
