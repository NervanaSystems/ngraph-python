.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Building graphs
***************
An operational graph is a device-independent program for performing computations. One can interact directly with the op-graph API, or use a variety of frontends (such as neon) to convert a model description into an op-graph. In order
to run the computations, use a transformer to compile the graph into a format that can be executed a desired backend (e.g. CPU, GPU, etc.)

To build op-graphs, we link together a sequence of operations that are instances of the class ``Op``. These operations are organized under several base classes:

* ``Op``: Base class for all ops.
* ``TensorOp (Op)``: Ops that produce a Tensor.
* ``ComputationOp (TensorOp)``: TensorOps with added backtrace functionality.
* ``ReductionOp (ComputationOp)``: Ops that reduce over some axes (e.g. sum).
* ``ElementWise (ComputationOp)``: Ops that perform element-wise calculations.
* ``ElementWiseBoolean (ElementWise)``: Boolean element-wise ops.

Supported ops are shown in the below figure:

.. image:: assets/op_heirarchy.png


Graph evaluation
================

During computation, the input and output values must be stored somehwere. To create a ``placeholder`` expression in the operational graph, we must import the operational backend symbols and then create the ``placeholder``::

    import ngraph as ng
    import ngraph.frontends.base.axis as ax

    x = ng.placeholder(axes=ng.Axes(ax.C, ax.W, ax.H, ax.N))

This will create an ``AllocationOp`` for a ``placeholder`` with the provided list of axes and assign the op to the python variable ``x``.  When the op is used in a graph, the op serves as a Python handle for the tensor stored in the device.

It is important to remember that ``x`` is a Python variable that holds an op.  Therefore, the following code::

    x = x + x

does not directly double the value of the tensor in the ``placeholder``. Instead, the ``__add__`` method is called with
both arguments pointing to the same ``placeholder`` object. This returns a new ``Op`` that is now stored as the python variable ``x``.
On the other hand, to directly modify the value of the ``placeholder``, use::

    ng.SetItem(x, x + x)

Constructing the graph mostly consists of manipulating expressions, so ``SetItem`` is rarely used, except for updating variables at the end of a minibatch. Consider::

    x1 = x + x
    y = x1 * x1 - x

The intermediate value ``x + x`` is only computed once, since the same op is used for both arguments of the multiplication in ``y``.
Furthermore, in this computation, all the computations will automatically be performed in place. If the computation is later modified such that the intermediate value ``x + x`` is needed, the op-graph will automatically adjust the computation's implementation to make the intermediate result ``x + x`` available.  This same flexibility exists with NumPy or PyCUDA, but those implementations always allocate tensors for the intermediate values, relying on Python's garbage collector clean them up; the peak memory usage will be higher and there will be more overhead.

Derivatives
===========

Because the ops describe computations, we have enough information to compute derivatives, using the ``deriv``
function::

    import ngraph as ng
    import ngraph.frontends.base.axis as ax

    x = ng.placeholder(axes=ng.Axes((ax.C, ax.W, ax.H, ax.N)))
    y0 = ng.placeholder(axes=ng.Axes((ax.Y, ax.N))
    w = ng.Variable(axes=(ng.Axes((ax.C, ax.W, ax.H, ax.Y))))
    b = ng.Variable(axes=(ng.Axes((ax.Y,)))
    y = ng.tanh(dot(w, x) + b)
    c = dot((y - y0), (y - y0))
    d = deriv(c, w)

The op `d` will be the op for the derivative of the value of `dc/dw`.

In this example, we knew which ops contain the variables to be trained (e.g. ``w``).  For a more general
optimizer, we could search through all the subexpressions looking for the dependant variables.  This is handled by the ``variables`` method, so ``c.variables()`` would be the list ``[w, b]``.

Graph execution
===============

A *computation* is a subset of ops whose values are desired and corresponds to a callable procedure on a backend.
Users define one or more computations by specifying sets of ops to be computed.  In addition, the transformer
will define four additional procedures:

`allocate`
    Allocate required storage required for all computations.  This includes all allocations for all ops
    marked as `in`.

`initialize`
    Run all initializations.  These are all the `initializers` for the ops needed for the computations.  These
    are analogous to C++ static initializers.

`save`
    Save all persistent state.  These are states with the `persistent` property set.

`restore`
    Restore saved state.


General properties of ops
=========================

All operational graph ops are instances of the class :py:class:`ngraph.op_graph.op_graph.Op`, which is a subclass of
the class :py:class:`ngraph.op_graph.nodes.Node`, which is itself a subclass of the classes
:py:class:`ngraph.op_graph.names.NameableValue` and :py:class:`ngraph.op_graph.nodes.DebugInfo`.

The constructor's required arguments are the subexpressions.  All ops also have key initializers for:

`axes`
    The axes of the result of the computation.  This only needs to be specified if the result is not correct.
    The `axes` are available as a gettable property.

`name`
    A string that can help identify the node during debugging, or when search for a node in a set of nodes.
    Some front ends may also make use of the `name`.  The `name` is a settable property.

`tags`
    A set of values that can be used to filter ops when manipulating them.  For example, tags may be used to
    indicate groups of trainable variables in conjunction with drop-out.

`initializers`
    A set of ops that must be executed during the `initialize` operation.

`follows`
    A set of ops, in addition to the `args`, that should be executed before the op using them is run.

Some useful properties of ops are:

`args`
    The subexpressions of the op.  These will be computed before the op is computed, since the operation needs their
    values to compute its value.

`users`
    The set of all nodes that use this node as an argument.

`filename`
    The file that created the op.

`lineno`
    The line number in the file where the op was created.

`file_info`
    The file and line number formatted for debuggers that support clicking on a file location to edit that location.
