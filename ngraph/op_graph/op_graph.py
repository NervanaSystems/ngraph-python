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
from builtins import object
from functools import wraps
from collections import defaultdict

from ngraph.op_graph.axes import TensorDescription, \
    make_axis, make_axes, Axes, FlattenedAxis, slice_axis, default_dtype, \
    default_int_dtype, AxesMap
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


@contextmanager
def metadata(**metadata):
    """
    Capture all Ops created within the context. Hides ops created in this
    context from parent contexts.
    """
    with Op.all_ops() as ops:
        yield
    for op in ops:
        if isinstance(op, TensorValueOp):
            # make sure tensorvalue op matches thing it reads from
            op.metadata.update(op.states_read[0].metadata)
        else:
            op.metadata.update(metadata)


def with_op_metadata(f, metadata=None):
    """
    Decorator to add metadata to all ops created inside the decorated function.
    If this decorator is applied to a method of a class with a class
    variable `metadata` defined as a dictionary then we add that to the
    op metadata to attach.
    """
    metadata = metadata or dict()
    assert isinstance(metadata, dict), "Metadata must be dict, not {}".format(type(metadata))

    @wraps(f)
    def wrapper(*args, **kwargs):
        with Op.all_ops() as ops:
            result = f(*args, **kwargs)
        # If this decorator is applied to a method of a class with a class
        # variable called `metadata` then we add that to the
        if len(args) > 0 and hasattr(type(args[0]), 'metadata'):
            metadata.update(type(args[0]).metadata)
        for op in ops:
            op.metadata.update(metadata)
        return result
    return wrapper


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
        metadata: String key value dictionary for frontend metadata.
        kwargs: Args defined in related classes.

    Attributes:
        const: The value of a constant.
        constant (bool): The value is constant.
        control_deps (OrderedSet): Ops in addtion to args that must run before this op.
        persistent (bool): The value will be retained from computation to computation and
            not shared.  Always True if reference is set.
        metadata: Dictionary with of string keys and values used for attaching
            arbitrary metadata to nodes.
        trainable: The value is trainable.
    """

    # Default is to not collect Ops as they are created
    @staticmethod
    def _get_thread_ops():
        """
        :return: The stack of Ops being collected.
        """
        try:
            ops = get_thread_state().ops
        except AttributeError:
            ops = [None]
            get_thread_state().ops = ops
        return ops

    @staticmethod
    def get_all_ops():
        try:
            all_ops = get_thread_state().all_ops
        except AttributeError:
            all_ops = [None]
            get_thread_state().all_ops = all_ops
        return all_ops

    # We need to create another stack here because all_ops and captured_ops
    # have different semantics that don't work with a shared stack
    @staticmethod
    @contextmanager
    def all_ops(ops=None, isolate=False):
        """
        Collects all Ops created within the context. Does not hide ops created
        in this context from parent contexts unless isolate is True.
        """
        if ops is None:
            ops = []
        try:
            all_ops = Op.get_all_ops()
            all_ops.append(ops)
            yield (ops)
        finally:
            all_ops.pop()
            parent = all_ops[-1]
            if not isolate and parent is not None:
                parent.extend(ops)

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

    @staticmethod
    def visit_input_closure(roots, fun):
        """
        Topological sort order traversal of root and their inputs.

        Nodes will only be visited once, even if there are multiple routes to the
        same Node.

        Arguments:
            roots: root set of nodes to visit
            fun: Function to call on each visited node

        Returns:
            None
        """
        available = OrderedSet()
        counts = dict()
        parents = defaultdict(OrderedSet)
        ready = OrderedSet()

        available.update(root.forwarded for root in roots)
        while available:
            node = available.pop()
            node.update_forwards()

            if node in counts:
                continue

            children = OrderedSet((child.forwarded for child in node.control_deps))
            if children:
                counts[node] = len(children)
                for child in children:
                    parents[child].add(node)
                available.update(children)
            else:
                ready.add(node)

        while ready:
            node = ready.pop()
            fun(node)
            for p in parents.get(node, []):
                count = counts[p] - 1
                if count == 0:
                    ready.add(p)
                    del counts[p]
                else:
                    counts[p] = count
        if len(counts) > 0:
            raise ValueError("Graph not a DAG")

    def __init__(self,
                 args=(),
                 metadata=None,
                 const=None,
                 constant=False,
                 persistent=False,
                 trainable=False,
                 **kwargs):
        super(Op, self).__init__(**kwargs)
        self.__args = tuple(as_op(arg) for arg in args)
        self.metadata = dict()

        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be of type dict,"
                                 "not {} of {}".format(type(metadata), metadata))
            self.metadata.update(metadata)

        # List to keep generation deterministic
        self.__control_deps = OrderedSet()
        self.__deriv_handler = None
        self._const = const
        self._is_constant = constant
        self._is_persistent = persistent
        self._is_trainable = trainable

        # Add this op to the all op accounting lists
        ops = Op._get_thread_ops()[-1]
        if ops is not None:
            ops.append(self)
        all_ops = Op.get_all_ops()[-1]
        if all_ops is not None:
            all_ops.append(self)

        self.style = {}
        self.__forward = None

    @property
    def tensor(self):
        """

        Returns: The op providing the value.

        """
        return self

    @property
    def states_read(self):
        """

        Returns: All state read by this op.

        """
        return OrderedSet()

    @property
    def states_written(self):
        """

        Returns: All state written by this op.

        """
        return OrderedSet()

    def __str__(self):
        return self.graph_label

    def __repr__(self):
        return '<{cl}({gl}):{id}>'.format(
            cl=self.__class__.__name__,
            gl=self.graph_label_type,
            id=id(self)
        )

    @property
    def is_constant(self):
        """

        Returns: True if this op is a constant tensor.

        """
        return False

    @property
    def const(self):
        """

        Returns: For a constant, returns the constant value.

        """
        return None

    @property
    def is_input(self):
        """

        Returns: True if this op is a tensor that the host can write to.

        """
        return False

    @property
    def is_persistent(self):
        """

        Returns: True if this op is a tensor whose value is preserved from computation
            to computation.

        """
        return False

    @property
    def is_trainable(self):
        """

        Returns: True if this op is a tensor that is trainable, i.e. is Op.variables
            will return it.

        """
        return False

    @property
    def is_placeholder(self):
        """

        Returns: True if this op is a placeholder, i.e. a place to attach a tensor.

        """
        return False

    @property
    def is_tensor_op(self):
        """

        Returns: True if this op is a tensor.

        """
        return False

    @property
    def is_scalar(self):
        """

        Returns: True if this op is a scalar.

        """
        return 0 == len(self.axes)

    @property
    def is_device_op(self):
        """

        Returns:
            True if the Op executes on the device.
        """
        return True

    @property
    def scalar_op(self):
        """
        Returns the scalar op version of this op.  Will be overridden by subclasses
        """
        if not self.is_scalar:
            raise ValueError()
        return self

    @property
    def args(self):
        """All the inputs to this node."""
        return self.__args

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
        self.update_forwards()
        value.update_forwards()

        # Make sure everything that is supposed to happen
        # before this op still happens
        for dep in self.__control_deps:
            value.add_control_dep(dep)
        self.__forward = value
        tdcache.tensor_description_cache.clear()
        value.metadata.update(self.metadata)

    @property
    def forwarded(self):
        """
        Finds the op that handles this op.

        Returns:
             Follows forwarding to the op that shoud handle this op.
        """
        result = self
        while True:
            if result.__forward is None:
                return result
            result = result.__forward

    @property
    def control_deps(self):
        """

        Returns:
            Ops that must execute before this one can.
        """
        return self.__control_deps + self.args

    def add_control_dep(self, dep):
        """
        Add an op that needs to run before this op.

        Args:
            dep: The op.

        """
        dep = dep.forwarded
        if dep is not self and dep not in self.control_deps:
            self.__control_deps.add(dep)

    def remove_control_dep(self, dep):
        """
        Remove an op from the list of ops that need to run before this op.

        Args:
            dep: The op.

        """
        self.update_forwards()
        self.__control_deps.remove(dep.forwarded)

    def update_forwards(self):
        """
        Replaces internal op references with their forwarded versions.

        Any subclass that uses ops stored outside of args and control_deps
        needs to override this method to update those additional ops.

        This is mainly to reduce the number of places that need to explicitly check
        for forwarding.

        """

        for op in self.control_deps:
            if op.forward is not None:
                self.__args = tuple(arg.forwarded for arg in self.__args)
                control_deps = self.__control_deps
                self.__control_deps = OrderedSet()
                for op in control_deps:
                    self.add_control_dep(op.forwarded)
                break

    def replace_self(self, rep):
        self.forward = as_op(rep)

    @property
    def deriv_handler(self):
        """
        Overrides processing of this op for this derivative.

        Returns:
            The op that should be used to process this op. If no deriv_handler has been set,
            self is returned.

        """
        if self.__deriv_handler is None:
            return self
        else:
            return self.__deriv_handler

    @deriv_handler.setter
    def deriv_handler(self, deriv_handler):
        if deriv_handler is self:
            deriv_handler = None
        self.__deriv_handler = deriv_handler

    @property
    def defs(self):
        """
        Returns:
            For liveness analysis.  The storage associated with everything
            in the returned list is modified when the Op is executed.

        """
        return [self]

    def variables(self):
        """
        Return all trainable Ops used in computing this node.

        Returns:
            Set of trainable Ops.
        """
        params = OrderedSet()

        def visitor(node):
            """
            TODO.

            Arguments:
              node: TODO
            """
            if node.tensor.is_trainable:
                params.add(node.tensor)

        Op.visit_input_closure([self], visitor)

        return params

    def placeholders(self):
        """
        Return all placeholder Ops used in computing this node.

        Returns:
            Set of placeholder Ops.
        """
        params = OrderedSet()

        def visitor(node):
            """
            TODO.

            Arguments:
              node: TODO
            """
            if node.tensor.is_placeholder:
                params.add(node.tensor)

        Op.visit_input_closure([self], visitor)

        return params

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


def as_op(x):
    """
    Finds an Op appropriate for x.

    If x is an Op, it returns x. Otherwise, constant(x) is returned.

    Arguments:
      x: Some value.

    Returns:
      Op:
    """
    if isinstance(x, AssignableTensorOp):
        return TensorValueOp(x)

    if isinstance(x, Op):
        return x

    return TensorValueOp(constant(x))


def as_ops(xs):
    """
    Converts an iterable of values to a tuple of Ops using as_op.

    Arguments:
        xs: An iterable of values.

    Returns:
        A tuple of Ops.
    """
    return tuple(as_op(x) for x in xs)


class AssignOp(Op):
    """
    tensor[...] = val.

    Arguments:
        tensor (AssignableTensorOp): An assignable TensorOp.
        val: The value to assign.
        **kwargs: Args for related classes.
    """

    def __init__(self, tensor, val, **kwargs):
        # convert val to op
        # TODO: requires explicit broadcast in future
        if not isinstance(val, Op):
            val = as_op(val)
            if len(val.axes) == len(tensor.axes):
                val = cast_axes(val, tensor.axes)

        # automatic broadcast
        # currently requires val's axes to be a subset of tensor's axes
        # TODO: requires explicit broadcast in future
        if len(val.axes - tensor.axes) > 0:
            raise ValueError(
                "tensor(LHS) has axes %s, val(RHS) has axes %s,"
                "val's axes should be subset of tensor's axes" %
                (val.axes, tensor.axes))
        val = broadcast(val, tensor.axes)

        super(AssignOp, self).__init__(args=(tensor, val), **kwargs)

    @property
    def states_written(self):
        return self.args[0].states_read


class AssignOneDOp(Op):
    """
    Assign a value to a 1d tensor.

    Arguments:
        tensor (AssignableTensorOp): The value to assign to.
        value (TensorOp): The value.
    """

    def __init__(self, tensor, val, **kwargs):
        if val.is_scalar:
            val = val.scalar_op
        super(AssignOneDOp, self).__init__(args=(tensor, val), **kwargs)

    @property
    def states_written(self):
        return self.args[0].states_read


def assign(lvalue, rvalue):
    """
    Assignment; lvalue <= rvalue

    Arguments:
        lvalue: Tensor to assign to.
        rvalue: Value to be assigned.
        item (optional):
    """
    return AssignOp(lvalue, rvalue)


class SetItemOp(Op):
    """
    tensor[item] = val

    This is a stub and has no frontend support at this time.

    Arguments:
        tensor (AssignableTensorOp): An assignable tensor.
        item: An index into the tensor.
        val (TensorOp): A value to assign.

    """

    def __init__(self, tensor, item, val, **kwargs):
        super(SetItemOp, self).__init__(args=(tensor, val), **kwargs)
        self.item = tuple(item)

    @property
    def states_written(self):
        return self.args[0].states_read


class ControlBlockOp(Op):
    """
    An Op that affects execution sequencing.
    """
    def __init__(self, **kwargs):
        super(ControlBlockOp, self).__init__(**kwargs)

    @property
    def is_device_op(self):
        """

        Returns:
            False, because this is handled by the transformer.
        """
        return False


class ParallelOp(ControlBlockOp):
    """
    Compute every Op in all in any order compatible with existing dependencies.

    Arguments:
        all: Ops to be computed.
        **kwargs: Args for related classes.
    """
    def __init__(self, all, **kwargs):
        super(ParallelOp, self).__init__(**kwargs)
        for op in all:
            self.add_control_dep(op)


def doall(all):
    return ParallelOp(all)


class ComputationOp(ParallelOp):
    """
    Represents a host-callable graph computation.

    Arguments:
        returns: Values returned by the computation. A list, set, or op.
        *args: Inputs to the computation. Must be placeholders or variables.

    Parameters:
        returns: Ops returned.
        parameters: Parameter ops.
    """
    def __init__(self, returns, *args, **kwargs):
        if isinstance(returns, collections.Container):
            all = type(returns)(as_op(ret) for ret in returns)
        elif isinstance(returns, Op):
            all = [as_op(returns)]
        elif returns is not None:
            raise ValueError()
        else:
            all = []

        self.values = all
        self.returns = returns
        super(ComputationOp, self).__init__(all=all, **kwargs)

        def is_input(arg):
            return arg.tensor.is_input

        placeholders = self.placeholders()
        if len(args) == 1 and args[0] == 'all':
            args = placeholders

        args = tuple(as_op(arg) for arg in args)
        arg_tensors = set(arg.tensor for arg in args)
        missing_tensors = [t for t in placeholders.difference(arg_tensors)]
        if len(missing_tensors) > 0:
            raise ValueError("All used placeholders must be supplied to a computation.")

        for arg in args:
            if not (arg.tensor.is_input):
                raise ValueError((
                    'The arguments to a computation must all be Ops with property '
                    'is_input=True, but the op passed had is_input=False.'
                    'In most cases you want to pass placeholder ops in as arguments.  '
                    '{op} was passed in, of type {op_type}.'
                ).format(
                    op=arg,
                    op_type=arg.__class__.__name__,
                ))

        self.parameters = args
        for arg in args:
            self.add_control_dep(arg)


def computation(returns, *args):
    """
    Defines a host-callable graph computation.

    Arguments:
        returns: Values returned by the computation. A list, set, or op.
        *args: Inputs to the computation.

    Returns:
        A computation op.
    """

    return ComputationOp(returns, *args)


class Fill(Op):
    """
    Fill a tensor with a scalar value.

    Arguments:
        tensor (AssignableTensorOp): An assignable TensorOp.
        scalar: A scalar value.
    """

    def __init__(self, tensor, scalar, **kwargs):
        super(Fill, self).__init__(args=(tensor,), **kwargs)
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

    @property
    def states_written(self):
        return self.args[0].states_read


class TensorOp(Op):
    """
    Super class for all Ops whose value is a Tensor.

    Arguments:
        axes: The axes of the tensor.
        dtype: The element type of the tensor.
        scale: If specified, a scaling factor applied during updates.
        is_value_op: If specified, the normal dtype/axes/scale defaulting is disabled
          since those values will be supplied by a subclass, such as ValueOp.
        **kwargs: Arguments for related classes.
    """

    def __init__(self, dtype=None, axes=None, scale=None, is_value_op=None, **kwargs):
        super(TensorOp, self).__init__(**kwargs)
        if not is_value_op:
            self.dtype = default_dtype(dtype)
            if axes is not None:
                axes = make_axes(axes)
            self.__axes = axes
            self.scale = scale

    @property
    def is_tensor_op(self):
        return True

    @property
    @cachetools.cached(cache=dict())
    def one(self):
        """
        Returns a singleton constant 1 for this Op. Used by DerivOp to ensure that
         we don't build unique backprop graphs for every variable.

        Returns:
            A unique constant 1 associated with this TensorOp.

        """
        return as_op(1)

    @cachetools.cached({})
    def adjoints(self, error):
        """
        Returns a map containing the adjoints of this op with respect to other
        ops.

        Creates the map if it does not already exist.

        Arguments:
            error (TensorOp, optional): The tensor holding the error value
                the derivative will be computed at. Must have the same axes as dependent.


        Returns:
            Map from Op to dSelf/dOp.
        """
        adjoints = {
            self.tensor: error,
        }

        # visit ops in reverse depth first post-order. it is important that
        # ordered_ops returns a copy of this traversal order since the graph
        # may change as we generate adjoints and we don't want to visit those
        # new ops. Some ops may be containers for other ops, so we create an
        # ordered set to ensure we don't do multiple backprops.
        processed = set()
        for o in reversed(Op.ordered_ops([self])):
            if o.tensor in processed:
                continue
            if o.tensor in adjoints:
                adjoint = adjoints[o.tensor]
                if o.scale is not None:
                    adjoint = adjoint * o.scale

                deriv_handler = o.deriv_handler
                deriv_handler.generate_adjoints(adjoints, adjoint, *deriv_handler.args)
                processed.add(o.tensor)

        return adjoints

    def generate_add_delta(self, adjoints, delta):
        """
        Adds delta to the backprop contribution..

        Arguments:
            adjoints: dy/dOp for all Ops used to compute y.
            delta: Backprop contribute.
        """
        if not self.axes.is_equal_set(delta.axes):
            raise ValueError(
                'delta axes {} do not match adjoint axes {}'
                .format(delta.axes, self.axes)
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

    def __setitem__(self, key, val):
        if key == slice(None) or key is Ellipsis:
            return assign(self, val)
        raise ValueError("Setting {} is not supported yet".format(key))

    # Only works when capturing ops
    def __iadd__(self, val):
        return assign(self, self + val)

    # Only works when capturing ops
    def __isub__(self, val):
        return assign(self, self - val)

    # Only works when capturing ops
    def __imul__(self, val):
        return assign(self, self * val)

    # Only works when capturing ops
    def __idiv__(self, val):
        return assign(self, self / val)

    def __getitem__(self, item):
        if isinstance(item, slice) and len(self.axes) > 1:
            item = (item,)
        item += tuple(slice(None) for _ in range(len(self.axes) - len(item)))
        return tensor_slice(self, item)

    def __axes__(self):
        return self.axes

    @tdcache()
    def tensor_description(self):
        """
        Returns a TensorDescription describing the output of this TensorOp

        Returns:
          TensorDescription for this op.
        """
        return TensorDescription(self.axes, dtype=self.dtype, name=self.name,
                                 is_persistent=self.is_persistent,
                                 is_input=self.is_input,
                                 is_placeholder=self.is_placeholder)

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

    def mean(self, reduction_axes=None, out_axes=None):
        """
        Used in Neon front end.

        Returns: mean(self)

        """
        return mean(self, reduction_axes=reduction_axes, out_axes=out_axes)

    @property
    def value(self):
        """
        Returns a handle to the device tensor.

        The transformer must have been initialized.

        :return: A handle to the device tensor.
        """
        return self.forwarded.tensor_description().value


class ValueOp(TensorOp, ControlBlockOp):
    """
    Mixin class for ops whose value is another op.

    Arguments:
        tensor: The tensor supplying the value for this op.

    """
    def __init__(self, tensor=None, **kwargs):
        super(ValueOp, self).__init__(args=(), is_value_op=True, **kwargs)
        self.__tensor = tensor

    def tensor_description(self):
        return self.tensor.tensor_description()

    @property
    def tensor(self):
        """
        The op that ultimately supplies the value. See value_tensor.

        Returns:
            The op that supplies the value.

        """
        return self.__tensor.tensor

    @property
    def value_tensor(self):
        """
        The op whose value is returned by this op.

        Returns:
            The immediate value returned by this op; see tensor for the closure.

        """
        return self.__tensor

    @value_tensor.setter
    def value_tensor(self, tensor):
        self.__tensor = tensor

    @property
    def control_deps(self):
        base_deps = super(ValueOp, self).control_deps
        if self.value_tensor is not None and self.value_tensor.is_device_op:
            # Add value_tensor if it is a real op
            return base_deps + [self.value_tensor]
        else:
            return base_deps

    @property
    def is_tensor_op(self):
        return self.tensor.is_tensor_op

    @property
    def value(self):
        return self.tensor.value

    @property
    def axes(self):
        return self.tensor.axes

    @property
    def dtype(self):
        return self.tensor.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.tensor.dtype = dtype

    @property
    def is_constant(self):
        return self.tensor.is_constant

    @property
    def const(self):
        return self.tensor.const

    @property
    def scale(self):
        return self.tensor.scale

    @property
    def states_read(self):
        return self.value_tensor.states_read

    @property
    def states_written(self):
        return self.value_tensor.states_written

    def generate_add_delta(self, adjoints, delta):
        self.tensor.generate_add_delta(adjoints, delta)


class SequentialOp(ValueOp):
    """
    Given a list of ops, ensure that every op that has not already been executed is executed in
    the given order. The value of the last op is the value of this op.

    Ops will only be executed once, so to return the value of an earlier op, just add it again at
    the end of the list.

    Control dependencies are not computed until after the graph is computed, i.e. after derivatives
    are expanded.

    Arguments:
        ops: Sequence of ops to compute. If not specified, set the attribute ops when known. This
            is useful for subclassing.

    Attributes:
        ops: The list of ops to be computed. The last op is the returned value.
    """
    def __init__(self, ops=None, **kwargs):
        super(SequentialOp, self).__init__(**kwargs)
        self.value_tensor = None
        self.__ops = None
        if ops is not None:
            self.ops = ops

    @property
    def ops(self):
        return self.__ops

    @ops.setter
    def ops(self, ops):
        self.__ops = list(as_op(op).forwarded for op in ops)

        for op in self.__ops:
            self.add_control_dep(op)
        self.value_tensor = self.__ops[-1]

        # Ops that have already executed.
        done_ops = set()

        # State => op_tops that have written state
        writers = defaultdict(OrderedSet)
        # State => op_tops that have read state
        readers = defaultdict(OrderedSet)
        for op_top in self.__ops:
            ordered_ops = Op.ordered_ops([op_top])
            # Make ops that read/write state execute after the op_tops that last read/wrote
            # the state.
            for op in ordered_ops:
                if op in done_ops:
                    # The op already ran, so it doesn't run here
                    continue
                for state in op.states_read:
                    for write_op in writers[state]:
                        op.add_control_dep(write_op)
                for state in op.states_written:
                    for read_op in readers[state]:
                        op.add_control_dep(read_op)
            # Register this op_top with each state it read/wrote.
            for op in ordered_ops:
                if op in done_ops:
                    # The op already ran, so it doesn't run here
                    continue
                for state in op.states_written:
                    writers[state].add(op_top)
                for state in op.states_read:
                    readers[state].add(op_top)
            done_ops.update(ordered_ops)


def sequential(ops=None):
    """
    Compute every op in order, compatible with existing dependencies, returning last value.

    Ops will only be executed once, so to return the value of an earlier op, just add it again at
    the end of the list.

    Arguments:
        ops: Sequence of ops to compute.

    """
    sequential_op = SequentialOp(ops)
    sequential_op.deriv_handler = sequential_op.value_tensor
    # Note: Can't return value_tensor here because we may need some ops to execute
    # after it. For example,
    # op_1, op_2, op_3, op_1 has value of op_1, but op_1 won't force op_2 and op_3 to run.
    return sequential_op


class TensorValueOp(ValueOp):
    """
    A read of an AssignableTensorOp.

    This provides a way to maintain different control information on different
    versions of state.

    """
    def __init__(self, tensor, **kwargs):
        super(TensorValueOp, self).__init__(tensor=tensor, **kwargs)

        for key in ['device', 'device_id', 'parallel']:
            if key in tensor.metadata:
                self.metadata[key] = tensor.metadata[key]

    @property
    def states_read(self):
        return OrderedSet([self.tensor])


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
        return self.args[0].tensor_description().transpose().named(self.name)

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
        axes = make_axes(axes)
        self._check_valid_axes(x, axes)
        super(AxesCastOp, self).__init__(x, axes=axes, **kwargs)

    def _check_valid_axes(self, x, axes):
        if not x.is_scalar and x.axes.lengths != axes.lengths:
            raise ValueError("casting axes {} must have the same length as original axes {}"
                             .format(axes, x.axes))

    @tdcache()
    def tensor_description(self):
        return self.args[0].tensor_description().cast(self.axes).named(self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, cast_axes(delta, x.axes))


class RoleCastOp(AxesCastOp):
    """
    Used to set the names of the axes of a tensor, without altering its value.
    If the names of the new axes are the same as the incoming tensor's axes,
    leave the original axis alone.  Otherwise, create a new axis with the
    length of the original and the name of the new.
    Arguments:
        x: A tensor.
        axes: The new axes.
    """

    def __init__(self, x, axes, **kwargs):
        axes = make_axes([
            old_axis if old_axis == new_axis else make_axis(old_axis.length, new_axis.name)
            for old_axis, new_axis in zip(x.axes, axes)
        ])
        self._check_valid_axes(x, axes)

        super(RoleCastOp, self).__init__(x, axes=axes, **kwargs)

    def _check_valid_axes(self, x, axes):
        if len(x.axes) != len(axes):
            raise ValueError(
                "casting axes {} must have the same number of axes as original axes {}"
                .format(axes, x.axes)
            )

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, cast_role(delta, x.axes))


class MapRolesOp(AxesCastOp):
    """
    Used to set the names of the axes of a tensor, without altering its value.

    If the names of the new axes are the same as the incoming tensor's axes,
    leave the original axis alone.  Otherwise, create a new axis with the
    length of the original and the name of the new.

    Arguments:
        x: A tensor.
        axes_map: An AxesMap object describing the mapping from axis_name ->
        axis_name that should be performed.  Axis whose names don't appear in
        the axes_map won't be changed.
    """

    def __init__(self, x, axes_map, **kwargs):
        self.axes_map = AxesMap(axes_map)

        super(MapRolesOp, self).__init__(x, axes=self.axes_map.map_axes(x.axes), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, MapRolesOp(delta, self.axes_map.invert()))


def cast_axes(tensor, axes):
    """
    Cast the axes of a tensor to new axes.

    Args:
        tensor (TensorOp): The tensor.
        axes (Axes): The new axes.

    Returns:
        TensorOp: The tensor with new axes.
    """
    axes = make_axes(axes)
    if tensor.axes.lengths != axes.lengths:
        raise ValueError("casting axes {} must have the same length as original axes {}"
                         .format(axes, tensor.axes))
    if len(axes.lengths) == 0:
        return tensor

    return AxesCastOp(tensor, axes)


def map_roles(tensor, axes_map):
    """
    Cast the axes' roles of a tensor to new roles.

    Args:
        tensor (TensorOp): The tensor.
        axes_map ({name: name}:  AxesMap from name to name

    Returns:
        TensorOp: The tensor with new axes.
    """
    return MapRolesOp(tensor, axes_map)


def cast_role(tensor, axes):
    """
    Cast the axes' roles of a tensor to new roles.

    Args:
        tensor (TensorOp): The tensor.
        axes (Axes): The new axes.

    Returns:
        TensorOp: The tensor with new axes.
    """
    axes = make_axes(axes)
    if len(tensor.axes) != len(axes):
        raise ValueError(
            'Tried to cast Axes {} to have the roles from {}.  Both Axes '
            'must have the same number of Axes.'
            .format(tensor.axes, axes)
        )
    return RoleCastOp(tensor, axes)


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


def expand_dims(x, axis, dim):
    """
    Adds additional axes into a tensor.
    Arguments:
        x: The tensor.
        axis: The additional axis.
        dim: The position to add the axes.
    """
    if axis in x.axes:
        return x
    return ExpandDims(x, axis, dim)


class BroadcastOp(ReshapeOp):
    """
    Used to add additional axes for a returned derivative.

    Arguments:
        x: The tensor to broadcast.
        axes: The new axes.
    """

    def __init__(self, x, axes, **kwargs):
        Axes.assert_valid_broadcast(x.axes, axes)
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
        return td.broadcast(self.axes).named(self.name)

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


def axes_with_role_order(x, roles):
    """
    Return a tensor with a different axes order according to
    specified roles.  Will expand dims as necessary with inferred
    axes for missing roles

    Args:
        x (TensorOp): The tensor.
        roles (sequence, AxisRoles): A permutation of the roles
                                     of axes of the tensor.

    Returns:
        TensorOp: The new tensor.

    """
    reordered_axes = make_axes()
    y = x
    for r in roles:
        ax_i = y.axes.role_axes(r)
        if len(ax_i) == 0:
            ax_i = make_axis(length=1, roles=[r])
        elif len(ax_i) == 1:
            ax_i = ax_i[0]
        else:
            raise ValueError("Unable to handle multiple axes with role {}".format(r.name))
        reordered_axes |= ax_i
        # This will only add the missing axes to the front
        y = expand_dims(y, ax_i, 0)

    # Ensure that axes of x are a subset of y
    if not (x.axes & y.axes).is_equal_set(x.axes):
        raise ValueError("Input axes contain roles not encompassed by role list: {}".format(
            x.axes - (x.axes & y.axes)
        ))

    return axes_with_order(y, reordered_axes)


def axes_with_order(x, axes):
    """
    Return a tensor with a different axes order.

    Args:
        x (TensorOp): The tensor.
        axes (Axes): A permutation of the axes of the tensor.

    Returns:
        TensorOp: The new tensor.

    """
    axes = make_axes(axes)
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
        if not x.axes.is_equal_set(axes):
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
        return td.reorder(self.axes).named(self.name)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, axes_with_order(
            delta,
            x.axes
        ))


def tensor_slice(x, slices, axes=None):
    """
    Creates a sliced version of a tensor.

    Args:
        x: The tensor.
        slices: One slice for each dimension in x.
        axes: Axes for the result.  If not specified, axes will be generated.

    Returns:
        A sliced view of the tensor.
    """
    return TensorSliceOp(x, slices, axes)


class TensorSliceOp(ReshapeOp):
    """
    Creates a sliced version of a tensor.

    Arguments:
        x: The tensor.
        slices: One slice for each dimension in x.
        axes: Axes for the result.  If not specified, axes will be generated.
    """

    def __init__(self, x, slices, axes=None, **kwargs):
        slices = tuple(slices)
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
                        axes.append(slice_axis(axis, s))

            axes = make_axes(axes)

        super(TensorSliceOp, self).__init__(
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
        return x.slice(self.slices, self.axes).named(self.name)

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
            _unslice(delta, self.slices, x.axes)
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
    return tensor_slice(x, ss, axes=axes)


class Flatten(ReshapeOp):

    def __init__(self, x, axes, **kwargs):
        x = ContiguousOp(axes_with_order(x, x.axes))
        super(Flatten, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        x, = tensor_descriptions(self.args)
        return x.flatten(self.axes).named(self.name)

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
            axes = make_axes((FlattenedAxis(x.axes),))

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
        Axes.assert_valid_unflatten(x.axes, axes)
        super(Unflatten, self).__init__(x, axes=axes, **kwargs)

    @tdcache()
    def tensor_description(self):
        x, = tensor_descriptions(self.args)
        return x.unflatten(self.axes).named(self.name)

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
        is_input: The storage is used as an input from the CPU. Implies persistent.
        is_persistent: The storage value persists form computation to computation.
        is_constant: The storage value does not change once initialized.
        is_placeholder: This is a placeholder.
        const: The host value of the constant for constant storage.
        initial_value: If callable, a function that generates an Op whose tensor should be
            used as the initial value.  Otherwise an Op that should be used as the initial
            value.

    Attributes:
        input (bool): The storage is used as an input.
    """

    def __init__(
            self,
            initial_value=None,
            is_constant=False,
            is_input=False,
            is_persistent=False,
            is_trainable=False,
            is_placeholder=False,
            const=None,
            **kwargs):
        super(AssignableTensorOp, self).__init__(**kwargs)
        self._is_input = is_input
        self._is_persistent = is_persistent
        self._is_trainable = is_trainable
        self._is_constant = is_constant
        self._is_placeholder = is_placeholder
        self._const = const
        self.initial_value = None

        if initial_value is not None:
            # convert callable initial value
            if callable(initial_value):
                initial_value = initial_value(self.axes)
            if isinstance(initial_value, TensorOp):
                # Caffe2 currently wraps the initial value in a constant (Issue #1138)
                tensor = initial_value.tensor
                if tensor.is_constant:
                    initial_value = tensor.const
                else:
                    raise ValueError("initial_value must be convertible to a NumPy tensor")
            initial_value = np.asarray(initial_value, dtype=self.dtype)
            self.initial_value = initial_value

    @property
    def is_constant(self):
        return self._is_constant

    @property
    def const(self):
        return self._const

    @const.setter
    def const(self, value):
        if self._const is not None:
            raise ValueError("Cannot change const value")
        self._const = value

    @property
    def is_input(self):
        return self._is_input

    @property
    def is_persistent(self):
        return self._is_persistent

    @property
    def is_trainable(self):
        return self._is_trainable

    @property
    def is_placeholder(self):
        return self._is_placeholder

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

    def add_control_dep(self, op):
        """
        Allocations happen before executed ops, so control_deps are ignored.

        Args:
            op:

        Returns:

        """
        pass


def value_of(tensor):
    """
    Capture the value of a tensor.

    Args:
        tensor: The value to be captured.

    Returns:
        A copy of the value.

    """
    if tensor.is_constant:
        return tensor
    temp = temporary(axes=tensor.axes, dtype=tensor.dtype)
    return sequential([
        AssignOp(temp, tensor),
        temp
    ])


def constant(const, axes=None, dtype=None):
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

    nptensor = np.asarray(const, dtype=dtype)
    if axes and len(axes) == len(nptensor.shape):
        nptensor_axes = axes
    else:
        nptensor_axes = make_axes([make_axis(l) for l in nptensor.shape])
    graph_label_type = "<Const({})>".format(const)
    val = AssignableTensorOp(axes=nptensor_axes,
                             is_constant=True,
                             is_persistent=True,
                             graph_label_type=graph_label_type,
                             initial_value=nptensor,
                             const=nptensor,
                             dtype=dtype)

    if axes and len(axes) > 0 and val.is_scalar:
        val = broadcast(val, axes)
    return val


def placeholder(axes, dtype=None, initial_value=None):
    """
    A place for a tensor to be supplied; typically used for computation arguments.

    Args:
        axes (Axes): The axes of the placeholder.
        dtype (optional): The dtype of the placeholder.
        initial_value (optional): Deprecated. A host constant or callable. If callable, will
            be called to generate an initial value.

    Returns:
        AssignableTensorOp: The placeholder.

    """
    return AssignableTensorOp(graph_label_type="placeholder",
                              is_persistent=True,
                              is_input=True,
                              is_placeholder=True,
                              axes=axes, dtype=dtype,
                              initial_value=initial_value)


def temporary(axes, dtype=None, initial_value=None):
    """
    Temporary storage.

    Statically allocates storage that may be reused outside of the scope of the values.

    Args:
        axes (Axes): The axes of the storage.
        dtype (optional): The dtype of the storage.
        initial_value (optional): A host constant or callable. If callable, will
            be called to generate an initial value.
        constant (optional): Once initialization is complete, this tensor should not change.

    Returns:
        AssignableTensorOp: The placeholder.

    """
    return AssignableTensorOp(graph_label_type="Temp",
                              axes=axes, dtype=dtype,
                              initial_value=initial_value)


def persistent_tensor(axes, dtype=None, initial_value=None):
    """
    Persistent storage, not trainable.

    Storage that will retain its value from computation to computation.

    Args:
        axes (Axes): The axes of the persistent storage.
        dtype (optional): The dtype of the persistent storage.
        initial_value (optional): A host constant or callable. If callable, will
            be called to generate an initial value.

    Returns:
        AssignableTensorOp: The persistent storage.

    """
    return AssignableTensorOp(graph_label_type="Persistent",
                              is_persistent=True,
                              is_input=True,
                              axes=axes, dtype=dtype,
                              initial_value=initial_value)


def variable(axes, dtype=None, initial_value=None):
    """
    A trainable tensor.

    Args:
        axes (Axes): Axes for the variable.
        dtype (optional): The dtype for the tensor.
        initial_value: A constant or callable. If a callable, the callable
            will be called to provide an initial value.

    Returns:
        AssignableTensorOp: The variable.

    """
    return AssignableTensorOp(graph_label_type="Variable",
                              is_input=True,
                              is_persistent=True,
                              is_trainable=True,
                              axes=axes, dtype=dtype,
                              initial_value=initial_value)


class StackOp(SequentialOp):
    """
    Joins a list of identically-axed tensors along a new axis.

    Assign each argument into the appropriate slice of the storage associated
    with this op.

    Arguments:
        x_list: A list of identically-axed tensors to join.
        axis: The axis to select joined tensors.
        pos: The position within the axes of the x_list tensors to insert axis in the result.
        **kwargs: Other args for TensorOp.

    Parameters:
        pos: The position of the join axis.
    """

    def __init__(self, x_list, axis, pos=0, **kwargs):
        super(StackOp, self).__init__(**kwargs)
        self.pos = pos
        self.x_list = tuple(as_op(arg) for arg in x_list)
        if axis.length != len(x_list):
            raise ValueError("Axis must have the same length as x_list")
        arg_axes = self.x_list[0].axes
        axes_0 = arg_axes[:pos]
        axes_1 = arg_axes[pos:]
        # Axis layout for the result
        result_axes = axes_0 + axis + axes_1

        # With axes, we should be able to just setitem into a tensor shaped like the
        # result, but things don't quite work that way so we use a temp that would have
        # each arg in its own contiguous section, setitem into that, and reshape the result.
        storage_axes = make_axes((axis,) + tuple(arg_axes))
        self.storage = temporary(axes=storage_axes, dtype=self.x_list[0].dtype)
        slices = [slice(None)] * len(arg_axes)
        self.ops = [
            doall([SetItemOp(self.storage, [i] + slices, arg)
                   for i, arg in enumerate(self.x_list)
                   ]),
            axes_with_order(self.storage, result_axes)
        ]

        # Handle adjoint generation for the result
        self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
        s = [slice(None)] * len(self.storage.axes)
        for i, x in enumerate(self.x_list):
            s[self.pos] = i
            x.generate_add_delta(
                adjoints,
                axes_with_order(tensor_slice(delta, tuple(s)), x.axes)
            )


def stack(x_list, axis, pos=0):
    """

    Args:
        x_list: A list of identically-axed tensors to join.
        axis: The axis to select joined tensors.
        pos: The position within the axes of the x_list tensors to insert axis in the result.

    Returns:
        TensorOp: The joined tensors.

    """
    return StackOp(x_list, axis, pos)


class ConcatOp(SequentialOp):
    """
    Concatenates a list of tensors along specific axis. The axis can be different among each
    tensor, but must have a common role. All other axes should be identical.

    Args:
        x_list (list of TensorOps): A list of nearly identically-axed tensors to concatenate.
                                    They can have at most one axis that is different, and it must
                                    have a common role.
        axis_list (list of Axis): A list of Axis objects that will be concatenated along, one for
                                  each tensor in x_list.
    """

    def __init__(self, x_list, axis_list, **kwargs):
        super(ConcatOp, self).__init__(**kwargs)
        self.x_list = tuple(as_op(arg) for arg in x_list)
        self.axis_list = axis_list
        # Get common axes from first tensor in list
        arg_axes = self.x_list[0].axes
        ax = axis_list[0]
        common_axes = arg_axes - ax

        # Create long axis for concatenated tens1or
        concat_axis = make_axis(name=ax.name,
                                roles=ax.roles)

        # Store the axes order equivalent to the first tensor
        ind = arg_axes.index(ax)
        axes_0 = arg_axes[:ind]
        axes_1 = arg_axes[ind + 1:]
        result_axes = axes_0 + concat_axis + axes_1

        # With axes, we should be able to just setitem into a tensor shaped like the
        # result, but things don't quite work that way so we use a temp that would have
        # each arg in its own contiguous section, setitem into that, and reshape the result.
        storage_axes = make_axes([concat_axis] + list(axes_0) + list(axes_1))
        self.storage = temporary(axes=storage_axes, dtype=self.x_list[0].dtype)

        slices = [slice(None)] * (len(storage_axes) - 1)
        start = 0
        ops = []
        for ii, (x, ax) in enumerate(zip(self.x_list, axis_list)):
            if len(x.axes - common_axes) > 1:
                raise RuntimeError("Tensor {} has more than 1 axis not in common with"
                                   " other tensors".format(ii))
            if ax.length is None:
                raise RuntimeError("Tensor {} axis must have a specified length".format(ii))
            ops.append(SetItemOp(self.storage,
                                 [slice(start, start + ax.length)] + slices,
                                 axes_with_order(x, [ax] + list(storage_axes[1:]))))
            start += ax.length
        concat_axis.length = start
        self.ops = [
            doall(ops),
            axes_with_order(self.storage, result_axes)
        ]

        # Handle adjoint generation for the result
        self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
        slices = [slice(None)] * (len(self.storage.axes) - 1)
        storage_delta = axes_with_order(delta, self.storage.axes)
        start = 0
        for x, ax in zip(self.x_list, self.axis_list):
            delta_slice = tensor_slice(storage_delta,
                                       [slice(start, start + ax.length)] + slices)
            x.generate_add_delta(adjoints,
                                 axes_with_order(delta_slice,
                                                 x.axes))
            start += ax.length


def concat_along_axis(x_list, axis):
    """
    Concatenates a list of tensors along specific axis. The axis must appear in every tensor in the
    list.

    Args:
        x_list (list of TensorOps): A list of identically-axed tensors to concatenate
        axis (Axis): Axis to concatenate along

    Returns:
        The concatenated tensor op. Axes are ordered the same as in the first tensor in x_list.

    Examples:
        H = ng.make_axis(length=5)
        W = ng.make_axis(length=4)
        axes = ng.make_axes([H, W])
        x = ng.constant(np.ones(axes.full_lengths), axes=axes)
        y = ng.constant(np.ones(axes.full_lengths), axes=axes)
        c = ng.concat_along_axis([x, y], H)
    """

    if len(x_list) < 1:
        return x_list

    return ConcatOp(x_list, [axis for _ in range(len(x_list))])


def concat_role_axis(x_list, role):
    """
    Concatenates a list of tensors along an axis with the specified role. All other axes in each
    tensor should be identical.

    Args:
        x_list (list of TensorOps): A list of identically-axed tensors to concatenate
        role (AxisRole): Axis role common to every tensor in x_list

    Returns:
        The concatenated tensor op. Axes are ordered the same as in the first tensor in x_list.

    Examples:
        role = ng.make_axis_role("Concat")
        H1 = ng.make_axis(length=5, roles=[role])
        H2 = ng.make_axis(length=8, roles=[role])
        W = ng.make_axis(length=4)
        x = ng.constant(np.ones((H1.length, W.length)), axes=[H1, W])
        y = ng.constant(np.ones((H2.length, W.length)), axes=[H2, W])
        c = ng.concat_role_axis([x, y], role)
    """
    if len(x_list) < 1:
        return x_list

    def get_role_axis(axes, role):
        ax = axes.role_axes(role)
        if len(ax) > 1:
            raise RuntimeError("Multiple axes have role {}".format(role.name))
        elif len(ax) == 0:
            raise RuntimeError("No axis with role {}".format(role.name))
        else:
            return ax[0]

    return ConcatOp(x_list, [get_role_axis(x.axes, role) for x in x_list])


class UnsliceOp(SequentialOp):
    def __init__(self, x, slices, axes, **kwargs):
        super(UnsliceOp, self).__init__(**kwargs)
        self.x = x
        self.slices = slices
        temp = temporary(axes=axes, dtype=x.dtype).named('unslice')
        self.ops = [
            Fill(temp, 0),
            SetItemOp(temp, slices, x),
            temp
        ]

        # Handle adjoint generation for the result
        self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
        self.x.generate_add_delta(adjoints, tensor_slice(delta, self.slices, axes=self.x.axes))


def _unslice(x, slices, axes):
    """
    A computation to reverse a slicing operation.
    Used internally to implement expansions of tensors
    such as the derivative of a slice and a padding function.

    Arguments:
        x: The tensor.
        slices: slices to be unsliced.
        axes: axes of result.

    Attributes:
        slices: The slices.
        input_axes: The axes of the input x.
    """
    return UnsliceOp(x, slices, axes).value_tensor


class RngOp(TensorOp):

    def __init__(self, distribution, params, x, *args, **kwargs):
        """
        Arguments:
            x  : input tensor.
            distribution : either 'uniform' or 'normal'
            params: dict for specifying parameters of distribution
        Return:
        """
        if distribution not in ('uniform', 'normal'):
            raise ValueError((
                'unsupported distribution: {}'
            ).format(distribution))

        self.distribution = distribution
        self.params = params

        super(RngOp, self).__init__(
            args=(x,), axes=x.axes, *args, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


def uniform(x, low=0.0, high=1.0):
    """
    Fills x with uniform distribution between low and high.

    Args:
        x (TensorOp): A tensor.
        low (float): lower limit of distribution range
        high (float): upper limit of distribution range

    Returns:
        TensorOp: The  value of x.

    """
    return RngOp(distribution='uniform', params=dict(low=low, high=high), x=x)


def normal(x, loc=0.0, scale=1.0):
    """
    Fills x with normal distribution centered around loc and scaled by scale

    Args:
        x (TensorOp): A tensor.
        loc (float): mean of distribution
        scale (float): standard deviation of distribution

    Returns:
        TensorOp: The  value of x.

    """
    return RngOp(distribution='normal', params=dict(loc=loc, scale=scale), x=x)


class AllReduce(Op):
    """TODO."""

    def __init__(self, x, **kwargs):
        super(AllReduce, self).__init__(args=(x,), **kwargs)


class ElementWiseOp(TensorOp):
    pass


class UnaryElementWiseOp(ElementWiseOp):

    def __init__(self, x):
        super(UnaryElementWiseOp, self).__init__(args=(x,), axes=x.axes)


class StopGradient(UnaryElementWiseOp):
    """ TODO """

    @tdcache()
    def tensor_description(self):
        return self.tensor.tensor_description()

    @property
    def is_tensor_op(self):
        return False

    @property
    def value(self):
        return self.tensor.value

    @property
    def axes(self):
        return self.tensor.axes

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 0.)


def stop_gradient(x):
    """ TODO """
    return StopGradient(x)


class NegativeOp(UnaryElementWiseOp):
    """
    Negative of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)


def negative(x):
    """
    Returns the negative of x.

    Args:
        x (TensorOp): tensor.

    Returns:
        (TensorOp): The negative of x.

    """
    return NegativeOp(x)


class AbsoluteOp(UnaryElementWiseOp):
    """
    Absolute value of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sign(x) * delta)


def absolute(x):
    """
    Returns the absolute value of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The absolute value of x.

    """
    return AbsoluteOp(x)


class SinOp(UnaryElementWiseOp):
    """
    Sin of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * cos(x))


def sin(x):
    """
    Returns the sin of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: sin of x.

    """
    return SinOp(x)


class CosOp(UnaryElementWiseOp):
    """
    Cos of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta * sin(x))


def cos(x):
    """
    Returns the cos of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The cos of x.

    """
    return CosOp(x)


class TanhOp(UnaryElementWiseOp):
    """
    Tanh of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * (1.0 - self * self))


def tanh(x):
    """
    Returns the cos of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The tanh of x.

    """
    return TanhOp(x)


class ExpOp(UnaryElementWiseOp):
    """
    Exp of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * self)


def exp(x):
    """
    Returns the exp of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The exp of x.

    """
    return ExpOp(x)


class LogOp(UnaryElementWiseOp):
    """
    Log of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        def do_adjoints(delta, x):
            if isinstance(x, Divide):
                a, b = x.args
                do_adjoints(delta, a)
                do_adjoints(-delta, b)
            elif isinstance(x, ExpOp):
                x.args[0].generate_add_delta(adjoints, delta)
            else:
                x.generate_add_delta(adjoints, delta / x)

        do_adjoints(delta, x)


def log(x):
    """
    Returns the log of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The log of x.

    """
    return LogOp(x)


safelog_cutoff = 50.0


def safelog(x, limit=np.exp(-safelog_cutoff)):
    return log(maximum(x, limit))


class ReciprocalOp(UnaryElementWiseOp):
    """
    Reciprocal of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self * self * delta)


def reciprocal(x):
    """
    Returns the reciprocal of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The reciprocal of x.

    """
    return ReciprocalOp(x)


class SignOp(UnaryElementWiseOp):
    "Sign of a tensor."
    pass


def sign(x):
    """
    Returns the sign of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The sign of x.

    """
    return SignOp(x)


class SquareOp(UnaryElementWiseOp):
    """
    Square of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0 * delta * x)


def square(x):
    """
    Returns the square of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The square of x.

    """
    return SquareOp(x)


class SqrtOp(UnaryElementWiseOp):
    """
    Square root of a tensor.
    """

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 0.5 * delta / self)


def sqrt(x):
    """
    Returns the square root of x.

    Args:
        x (TensorOp): A tensor.

    Returns:
        TensorOp: The square root of x.

    """
    return SqrtOp(x)


class BinaryElementWiseOp(ElementWiseOp):

    def __init__(self, x, y, **kwargs):
        self.kwargs = kwargs
        x, y = as_ops((x, y))
        axes = x.axes | y.axes
        x = broadcast(x, axes)
        y = broadcast(y, axes)

        super(BinaryElementWiseOp, self).__init__(
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


def create_binary_elementwise(name,
                              func_name=None,
                              generate_adjoints=None):
    d = {}
    if generate_adjoints is not None:
        d['generate_adjoints'] = generate_adjoints
    BinClass = type(name, (BinaryElementWiseOp,), d)

    def func(*args, **kwargs):
        return BinClass(*args, **kwargs)
    func.__name__ = func_name

    return BinClass, func


def add_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta)
    y.generate_add_delta(adjoints, delta)


Add, add = create_binary_elementwise('AddOp', 'add', add_adjoints)


def subtract_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta)
    y.generate_add_delta(adjoints, -delta)


Subtract, subtract = create_binary_elementwise('Subtract', 'subtract', subtract_adjoints)


def multiply_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * y)
    y.generate_add_delta(adjoints, x * delta)


Multiply, multiply = create_binary_elementwise('Multiply', 'multiply', multiply_adjoints)


def divide_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * self / x)
    y.generate_add_delta(adjoints, -delta * self / y)


Divide, divide = create_binary_elementwise('Divide', 'divide', divide_adjoints)

Mod, mod = create_binary_elementwise('Mod', 'mod')


def maximum_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, greater(x, y) * delta)
    y.generate_add_delta(adjoints, greater(y, x) * delta)


Maximum, maximum = create_binary_elementwise('Maximum', 'maximum', maximum_adjoints)


def minimum_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, less(x, y) * delta)
    y.generate_add_delta(adjoints, less(y, x) * delta)


Minimum, minimum = create_binary_elementwise('Minimum', 'minimum', minimum_adjoints)


def power_adjoints(self, adjoints, delta, x, y):
    x.generate_add_delta(adjoints, delta * y * self / x)
    y.generate_add_delta(adjoints, delta * self * log(x))


Power, power = create_binary_elementwise('Power', 'power', power_adjoints)


Equal, equal = create_binary_elementwise('Equal', 'equal')


NotEqual, not_equal = create_binary_elementwise('NotEqual', 'not_equal')


Greater, greater = create_binary_elementwise('Greater', 'greater')


Less, less = create_binary_elementwise('Less', 'less')


GreaterEqual, greater_equal = create_binary_elementwise('GreaterEqual', 'greater_equal')


LessEqual, less_equal = create_binary_elementwise('LessEqual', 'less_equal')


class ContiguousOp(TensorOp):
    """
    Ensure that element layout is contiguous.

    Parameters:
        x (TensorOp): A possibly non-contiguous tensor.
    """

    def __init__(self, x, **kwargs):
        super(ContiguousOp, self).__init__(args=(x,), axes=x.axes, **kwargs)

    @property
    def old_axis_positions(self):
        return tuple(range(len(self.axes)))

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


class DotOp(TensorOp):

    def __init__(self, x, y, **kwargs):
        self.x_reduction_axes = x.axes & y.axes
        self.y_reduction_axes = self.x_reduction_axes
        assert self.x_reduction_axes == self.y_reduction_axes
        self.x_out_axes = x.axes - self.x_reduction_axes
        self.y_out_axes = y.axes - self.y_reduction_axes

        axes = self.x_out_axes | self.y_out_axes

        super(DotOp, self).__init__(
            args=(x, y), axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, x, y):
        """
        Generates the adjoint contributions for x and y.

        On input, x axes can be grouped as IJ and y axes as JK.

        Axes will be:
            Delta: IK.
            x adj: IJ
            y adj: JK

        Args:
            adjoints: The adjoints for the deriv being computed.
            delta (TensorOp): The backprop op.
            x (TensorOp): The x argument.
            y (TensorOp): The y argument.

        """
        x.generate_add_delta(
            adjoints,
            axes_with_order(dot(delta, y), x.axes)
        )
        y.generate_add_delta(
            adjoints,
            axes_with_order(dot(x, delta), y.axes)
        )


def dot(x, y):
    """
    The dot product of x and y.

    Reduction axes are the axes shared by x and y.

    Args:
        x (TensorOp): First argument.
        y (TensorOp): Second argumnent.
        name (String, optional): Name for the TensorOp.

    Returns:
        TensorOp: The dot product.

    """
    return DotOp(x, y)


def squared_L2(x, out_axes=None, reduction_axes=None):
    """
    Args:
        x (TensorOp): The first value, axes shifted down by 1.
        y (TensorOp): The second value.

    Returns:
        TensorOp: The result.

    """
    if reduction_axes is None:
        if out_axes is None:
            reduction_axes = x.axes.sample_axes()
        else:
            reduction_axes = x.axes - make_axes(out_axes)
    return sum(x * x, out_axes=out_axes, reduction_axes=reduction_axes)


class DotLowDimension(TensorOp):

    def __init__(self, x, y, axes, **kwargs):
        super(DotLowDimension, self).__init__(args=(x, y), axes=axes, **kwargs)


class SoftmaxOp(ValueOp):
    def __init__(self, x, normalization_axes=None, **kwargs):
        super(SoftmaxOp, self).__init__(**kwargs)

        if normalization_axes is None:
            normalization_axes = x.axes.sample_axes() - x.axes.recurrent_axis()
        self.x = x - max(x, reduction_axes=normalization_axes)
        self.exps = exp(self.x)
        self.Z = sum(self.exps, reduction_axes=normalization_axes)
        self.value_tensor = self.exps / self.Z
        self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
        """
        TODO.

        Arguments:
          adjoints: TODO
          delta: TODO
          op: TODO

        Returns:
          TODO
        """
        z = delta * self.value_tensor
        zs = sum(z)
        self.x.generate_add_delta(adjoints, (z - zs * self.value_tensor))


def softmax(x, normalization_axes=None, **kwargs):
    return SoftmaxOp(x, normalization_axes, **kwargs).value_tensor


class ReductionOp(TensorOp):

    def __init__(self, x, reduction_axes=None, out_axes=None, dtype=None, **kwargs):
        if reduction_axes is None and out_axes is None:
            reduction_axes = x.axes.sample_axes() - x.axes.recurrent_axis()
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
        assert (reduction_axes & out_axes) == make_axes(())

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


def create_reduction_op(name,
                        func_name=None,
                        generate_adjoints=None):
    d = {}
    if generate_adjoints is not None:
        d['generate_adjoints'] = generate_adjoints
    RedClass = type(name, (ReductionOp,), d)

    def func(*args, **kwargs):
        return RedClass(*args, **kwargs)
    func.__name__ = func_name
    return RedClass, func


def max_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, equal(x, self) * delta)


Max, max = create_reduction_op('Max', 'max', max_adjoints)


def min_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(adjoints, equal(x, self) * delta)


Min, min = create_reduction_op('Min', 'min', min_adjoints)


def sum_adjoints(self, adjoints, delta, x):
    x.generate_add_delta(
        adjoints,
        broadcast(delta, x.axes)
    )


Sum, sum = create_reduction_op('Sum', 'sum', sum_adjoints)


def prod_adjoints(self, adjoints, delta, x):
    # axes
    axes = x.axes
    reduction_axes = self.reduction_axes

    # x_equal_zero
    x_equal_zero = equal(x, 0)

    # count 0's occurrence by reduction axes
    x_zero_count = sum(x_equal_zero, reduction_axes=reduction_axes)

    # create mask for zero count 0 and 1
    mask_zero = broadcast(equal(x_zero_count, 0), axes=axes)
    mask_one = broadcast(equal(x_zero_count, 1), axes=axes)

    # replace all 0 to 1
    x_replaced = equal(x, 0.) * 1. + (1. - equal(x, 0.)) * x

    # do product of x_replace and gradient
    x_replaced_prod = prod(x_replaced, reduction_axes=reduction_axes)
    x_replaced_grad = x_replaced_prod / x_replaced

    # multiply mask with mask for the two cases
    x_grad = mask_zero * x_replaced_grad + mask_one * x_equal_zero * x_replaced_grad

    x.generate_add_delta(
        adjoints,
        broadcast(delta, x.axes) * x_grad
    )


Prod, prod = create_reduction_op('Prod', 'prod', prod_adjoints)


Argmax, _ = create_reduction_op('Argmax', 'argmax')


def argmax(x, dtype=None, **kwargs):
    return Argmax(x, dtype=default_int_dtype(dtype), **kwargs)


Argmin, _ = create_reduction_op('Argmin', 'argmin')


def argmin(x, dtype=None, **kwargs):
    return Argmin(x, dtype=default_int_dtype(dtype), **kwargs)


def variance(x, out_axes=None, reduction_axes=None):
    return mean(square(x - mean(x, out_axes=out_axes, reduction_axes=reduction_axes)),
                out_axes=out_axes, reduction_axes=reduction_axes)


class TensorSizeOp(TensorOp):
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
        super(TensorSizeOp, self).__init__(axes=())


def tensor_size(x, reduction_axes=None, out_axes=None):
    """
    A scalar returning the total size of a tensor in elements.

    Arguments:
        x: The tensor whose axes we are measuring.
        reduction_axes: if supplied, return the size
            of these axes instead.
    """
    return TensorSizeOp(x, reduction_axes=reduction_axes, out_axes=out_axes)


def batch_size(x):
    """

    Args:
        x: A Tensor

    Returns:
        The size of the batch axis in x.

    """
    return tensor_size(x, reduction_axes=x.axes.batch_axes())


def pad(x, paddings, axes=None):
    """
    Pads a tensor with zeroes along each of its dimensions.
    TODO: clean up slice / unslice used here

    Arguments:
      x: the tensor to be padded
      paddings: the length of the padding along each dimension.
        should be an array with the same length as x.axes.
        Each element of the array should be either an integer,
        in which case the padding will be symmetrical, or a tuple
        of the form (before, after)
      axes: the axes to be given to the padded tensor.
        If unsupplied, we create anonymous axes of the correct lengths.

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

    def to_slice(pad):
        s = (pad[0], -pad[1])
        s = tuple(None if p == 0 else p for p in s)
        return slice(s[0], s[1], 1)

    paddings = tuple(pad_to_tuple(pad) for pad in paddings)
    if axes is None:
        axes = make_axes(
            make_axis(length=axis.length + pad[0] + pad[1])
            if pad != (0, 0) else axis
            for axis, pad in zip(x.axes, paddings)
        )
    slices = tuple(to_slice(p) for p in paddings)

    return _unslice(x, slices, axes)


class OneHotOp(TensorOp):
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
        super(OneHotOp, self).__init__(
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
            out = OneHotTwoDimOp(x, self.axis)
            out = unflatten(
                out,
                [out.axes[0]] + list(out.axes[1].axes)
            )
            return out
        else:
            return OneHotTwoDimOp(x, self.axis)


def one_hot(x, axis):
    """

    Args:
        x: The one_hot tensor.
        axis: The hot axis.

    Returns:
        OneHotOp: The op.

    """
    return OneHotOp(x, axis)


class OneHotTwoDimOp(OneHotOp):
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
        super(OneHotTwoDimOp, self).__init__(x, axis, **kwargs)


class SigmoidOp(ValueOp):
    """
    Computes the sigmoid of x and handles autodiff for sigmoid.

    Arguments:
        x: The tensor argument.
        kwargs: Other construction arguments.

    Parameters:
        x: The tensor argument.
    """
    def __init__(self, x, **kwargs):
        super(SigmoidOp, self).__init__(**kwargs)
        self.x = x
        self.value_tensor = reciprocal(exp(-x) + 1)
        self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
        self.x.generate_add_delta(adjoints, delta * self.value_tensor * (1.0 - self.value_tensor))


def sigmoid(x):
    """
    Computes the sigmoid of x.

    Args:
        x:

    Returns:
        The sigmoid computation.
    """
    return SigmoidOp(x).value_tensor


def mean(x, reduction_axes=None, out_axes=None):
    """
    Computes the mean of x.

    Arguments:
        x (TensorOp): A tensor.
        reduction_axes (Axes, optional): If supplied, the mean is computed over these axes.
        out_axes (Axes, optional): If supplied, the result has these axes; the mean is computed
            over the remaining axes.

    Returns:
        TensorOp: The mean.
    """
    return sum(x, reduction_axes=reduction_axes, out_axes=out_axes) / \
        tensor_size(x, reduction_axes=reduction_axes, out_axes=out_axes)


class DerivOp(ValueOp):
    def __init__(self, dependent, independent, error):
        super(DerivOp, self).__init__()

        self.dependent = as_op(dependent)
        self.independent = as_op(independent)
        if error is None:
            # Get a singleton constant one for dependent. This ensures that all the
            # independents share the same backprop, which would not happen if we
            # made a constant 1 here, since we do not do common subexpression elimination,
            # while it also ensures that independent graphs do not share ops.
            error = self.dependent.one
        if not error.axes.is_equal_set(dependent.axes):
            raise ValueError("Dependent and error must have the same set of axes")

        self.error = as_op(error)
        adjoints = dependent.forwarded.adjoints(error)

        if independent.forwarded.tensor not in adjoints:
            self.value_tensor = constant(0, independent.axes)
        else:
            adjoint = adjoints[independent.forwarded.tensor]
            self.value_tensor = broadcast(adjoint.forwarded, axes=independent.axes)


def deriv(dependent, independent, error=None):
    """
    Computes the operation for [dDependent/dIndependent](error=1).

    The derivative is a multi-linear function.

    Args:
        dependent (TensorOp): Dependent op.
        independent(TensorOp): Independent op.
        error (TensorOp, optional): The tensor holding the error where the
            derivative will be computed at. Must have the same axes as dependent.

    Returns:
        TensorOp: Derivative applied to error. Has axes of independent.

    """
    return DerivOp(dependent, independent, error).value_tensor


class CrossEntropyMultiOp(ValueOp):
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

    def __init__(self, y, t, usebits=False, out_axes=None,
                 enable_softmax_opt=True,
                 enable_diff_opt=True, **kwargs):
        super(CrossEntropyMultiOp, self).__init__(**kwargs)
        if out_axes is None:
            # Compute along non-recurrent and non-batch axes
            index_axes = y.axes.sample_axes() - y.axes.recurrent_axis()
            out_axes = y.axes - index_axes
        if enable_softmax_opt and isinstance(y.deriv_handler, SoftmaxOp):
            # This depends on sum(t) being 1
            self.y = y
            self.x = y.deriv_handler.x
            self.s = -sum(self.x * t, out_axes=out_axes)
            self.value_tensor = self.s + safelog(y.deriv_handler.Z)
            if enable_diff_opt:
                self.value_tensor.deriv_handler = self
        else:
            self.value_tensor = -sum(safelog(y) * t, out_axes=out_axes)
        if usebits:
            self.value_tensor = self.value_tensor * np.float(1. / np.log(2.0))

    def generate_adjoints(self, adjoints, delta):
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

    return CrossEntropyMultiOp(y=y,
                               t=t,
                               usebits=usebits,
                               out_axes=out_axes,
                               enable_softmax_opt=enable_softmax_opt,
                               enable_diff_opt=enable_diff_opt).value_tensor


class CrossEntropyBinaryInnerOp(ValueOp):
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
    def __init__(self, y, t, enable_sig_opt=True, enable_diff_opt=True, **kwargs):
        super(CrossEntropyBinaryInnerOp, self).__init__(**kwargs)
        self.y = y
        self.t = t
        self.value_tensor = -(safelog(y) * t + safelog(1 - y) * (1 - t))
        if isinstance(y.deriv_handler, SigmoidOp):
            self.x = y.deriv_handler.x
            if enable_sig_opt:
                # Simpler equivalent
                self.value_tensor = (1 - t) * maximum(self.x, -safelog_cutoff) - safelog(y)
            if enable_diff_opt:
                self.value_tensor.deriv_handler = self

    def generate_adjoints(self, adjoints, delta):
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
    return CrossEntropyBinaryInnerOp(y=y, t=t,
                                     enable_sig_opt=enable_sig_opt,
                                     enable_diff_opt=enable_diff_opt).value_tensor


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
    result = sum(cross_entropy_binary_inner(y, t,
                                            enable_sig_opt=enable_sig_opt,
                                            enable_diff_opt=enable_diff_opt),
                 out_axes=out_axes
                 )

    if usebits:
        result = result * np.float(1. / np.log(2.0))
    return result
