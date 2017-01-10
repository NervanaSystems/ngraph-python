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
Frontends (or users who require the flexibility of constructing Nervana Graph ``Ops`` directly) utilize a set of graph construction functions to construct Nervana Graphs. We walk through the common patterns and arguments of these ``Ops`` here. We also discuss the underlying class structure of ``Op`` that is not typically a concern to users or frontends but gives a hierarchical structure that can be helpful.

Nervana Graph Structure
=======================

Data Dependencies
-----------------

An ``Op``'s primary role is to function as a node in a directed acyclic graph dependency computation graph. The ``Op`` class's attribute ``args`` is a list containing all upstream dependencies this ``Op`` operates upon. These operate as the directed edges of the graph.

For example,

.. code-block:: python

    >>> x = ng.constant(0)
    >>> y = ng.constant(1)
    >>> mysum = ng.add(x, y)
    >>> type(mysum)
    ngraph.op_graph.op_graph.AddOp

    >>> issubclass(mysum, ngraph.op_graph.op_graph.Op)
    True

    >>> mysum.args
    (<AssignableTensorOp(<Const(0)>):4500972432>,
     <AssignableTensorOp(<Const(1)>):4500974224>)

``mysum`` then refers to an instance of the class ``Add`` which is a subclass of ``Op``. ``mysum.args`` is a list containing the ``Ops`` pointed to by the python variables ``x`` and ``y``.

Initializers
------------
In addition to ``args``, there are two other types of edges in Nervana Graphs. Each op has an attribute ``initializers`` which contains a (possibly empty) set of ops needed to execute before any computations occur. To use our running example:

.. code-block:: python

    >>> mysum.initializers
    set()

    >>> x.initializers
    {<InitTensorOp(InitTensorOp_1):4500973392>}

We see here that ``mysum`` doesn't have any initializers because its value is only known at runtime. ``x`` on the other hand is a constant and can and must be initialized before any computations occur. Initializer subgraphs (the ops in ``initializers`` and all upstream ops) themselves contain ``SetItem``, ``Fill``, ``Flatten``, ``ConstantOp`` and other ops to manipulate a tensor to get it ready for computation.

Non-data Control Dependencies
-----------------------------
Finally, consider the following code:

.. code-block:: python

    >>> x = ng.placeholder((), initial_value=0)
    >>> a = ng.assign(x, 5)
    >>> z = x + 1

Here we create a scalar placeholder ``x``, then fill that placeholder with the value ``5``. Then we add one to ``x``. It is not clear initially if ``z`` when evaluated should equal ``1`` or ``6``. In Python at the last line of the previous code block, ``x`` still points to the initial ``placeholder`` with value ``0``. However, in Nervana Graph we believe most users intend the assign operation to occur before the final incrementing by one. Therefore the ordering semantics of ``ng.assign`` happen in accordance with the graph creation order.

In order to enforce these ordering semantics, Nervana Graph accounts for control dependencies between ``Ops`` when ``AssignOps`` are involved. To illustrate:

.. code-block:: python

    >>> z.other_deps
    {<AssignOp(AssignOp_1):4501621200>}

    >>> type(z.other_deps)
    ngraph.util.ordered.OrderedSet

    >>> z.other_deps.pop() is a
    True

All ``Ops`` have an ordered set in ``other_deps`` to contain the ops that must occur first in execution order before this op can be executed *even when those ops are not explicitly captured* as data dependencies of that ``Op``. The ``AddOp`` pointed to by the python variable ``z`` contains a ``other_deps`` control dependency on the ``AssignOp`` to ensure that it occurs first before z is computed.

Nervana graph also allows for contexts where the dependencies can be ignored, particularly when a variable has a self-assignment. For example, consider the following toy example:

.. code-block:: python

    import ngraph as ng
    import numpy as np
    from ngraph.transformers.nptransform import NumPyTransformer

    # set w
    w = ng.variable((), initial_value=0)

    # update op
    update_op = ng.assign(w, w + 1)

    # transformer
    transformer = NumPyTransformer()
    w_comp = transformer.computation(w)

    print(w_comp())
    print(w_comp())
    print(w_comp())

The above code will print ``1, 2, 3``. Even though the defined computation only retrieves the variable ``w``, the ``ng.assign`` dependencies get triggered such that the variable still updates with every call even though we simply want to retrieve the results.

We can guard the ``update_op`` with a context ``ng.Op.saved_user_deps`` to make sure that this dependency exists outside of the main stream.

.. code-block:: python

    with ng.Op.saved_user_deps():
        update_op = ng.assign(w, w + 1)

This modification will then allow the ``w_comp()`` to properly print ``0, 0, 0`` for each call. Ops that are defined inside the context are not included in the dependencies of the computation unless explicitly named. To recreate the ``1, 2, 3`` behavior now that the ``update_op`` is guarded, we would have to explicitly name the ``update_op`` in the computation:

.. code-block:: python

    w_comp = transformer.computation([w, update_op])

We see this context being used in the optimizer where velocities and parameters have a self-assignment with ``ng.assign``.

Note: The ``user_deps`` facility is likely to be replaced.

General properties of ops
=========================

All operational graph ops are instances of the class :py:class:`ngraph.op_graph.op_graph.Op`, which extends :py:class:`ngraph.op_graph.names.NameableValue` and :py:class:`ngraph.op_graph.nodes.DebugInfo`. The former provides ``Ops`` with automatically generated unique names and the latter provides debug info as to the line number and filename where this node was constructed.

In addition to the three graph properties explained above (``args``,
``initializers``, and ``other_deps``), all ops have the additional attributes:

`axes`
    The axes of the result of the computation. This only needs to be specified
    by the frontend or user during ``Op`` creation if the default result is not
    correct or not inferrable for a particular ``Op`` type. The `axes` are also
    available as a gettable property.

`name`
    A string that can help identify the node during debugging, or when search for a node in a set of nodes.
    Some front ends may also make use of the `name`.  The `name` is a settable property.

`metadata`
    A dictionary of key,value string pairs that can be used to select/filter
    ops when manipulating them. For example, ``stochastic=dropout`` may be used
    to indicate groups of trainable variables in conjunction with drop-out.

Some useful properties of ops are:

`filename`
    The file that created the op.

`lineno`
    The line number in the file where the op was created.

`file_info`
    The file and line number formatted for debuggers that support clicking on a file location to edit that location.

Op Hierarchy
============

Users and frontends do not typically need to worry about the implementation details of the various ``Op`` classes. This is why they are hidden behind graph construction functions.

.. All Nervana Graph nodes are instances of subclasses of the class ``Op`` which is captured in the full class hierarchy in the following figure.


.. .. image:: assets/op_hierarchy.*

Ops influencing evaluation
==========================

During computation (covered in more detail in :doc:`transformer_usage`), the input and output values must be stored somewhere. To create a ``placeholder`` expression in the operational graph, we must import the operational backend symbols and then create the ``placeholder``:

.. code-block:: python

    import ngraph as ng
    from ngraph.frontends.neon as ax

    x = ng.placeholder((ax.C, ax.W, ax.H, ax.N))

This ``placeholder`` will create an ``AssignableTensorOp`` to trigger the necessary storage to be allocated on the host device and trigger values to be transferred between the device and host. When the op is used in a graph computation, the op serves as a Python handle for the tensor stored on the device.

It is important to remember that ``x`` is a Python variable that holds an op.  Therefore, the following code:

.. code-block:: python

    x = x + x

does not directly double the value of the tensor in the ``placeholder``. Instead, the ``__add__`` method is called with
both arguments pointing to the same ``placeholder`` object. This returns a new ``Op`` that is now stored as the python variable ``x``.
On the other hand, to directly modify the value of the ``placeholder``, use:

.. code-block:: python

    ng.SetItem(x, x + x)

Constructing the graph consists mostly of manipulating expressions, so ``SetItem`` should rarely be used directly, except for updating variables at the end of a minibatch. Consider:

.. code-block:: python

    x1 = x + x
    y = x1 * x1 - x

The intermediate value ``x + x`` is only computed once, since the same op is used for both arguments of the multiplication in ``y``.
Furthermore, in this computation, all the computations will automatically be performed in place. If the computation is later modified such that the intermediate value ``x + x`` is needed, the op-graph will automatically adjust the computation's implementation to make the intermediate result ``x + x`` available.  This same flexibility exists with NumPy or PyCUDA, but those implementations always allocate tensors for the intermediate values, relying on Python's garbage collector clean them up; the peak memory usage will be higher and there will be more overhead.

Derivatives
===========

Because ``Ops`` describe computations, we have enough information to compute derivatives, using the ``deriv``
function:

.. code-block:: python

    import ngraph as ng
    from ngraph.frontends.neon import ax

    x = ng.placeholder((ax.C, ax.W, ax.H, ax.N))
    y0 = ng.placeholder((ax.Y, ax.N))
    w = ng.variable((ax.C, ax.W, ax.H, ax.Y)))
    b = ng.variable((ax.Y,))
    y = ng.tanh(dot(w, x) + b)
    c = ng.squared_L2(y - y0)
    d = ng.deriv(c, w)

The python variable ``d`` will hold an ``Op`` whose value is the derivative ``dc/dw``. In this example, we knew which ops contain the variables to be trained (e.g. ``w``).  For a more general optimizer, we could search through all the subexpressions looking for the dependant variables.  This is handled by the ``variables`` method, so ``c.variables()`` would return the list of ``Ops`` ``[w, b]``.

An important distinction to make here is that the ``deriv`` function does not perform symbolic or numeric differentiation. In fact it does not compute anything at all. Its sole job is to construct another computational graph using the existing upstream graph of ``c`` and return a handle to that new computational graph (``d``). No computation is therefore taking place at this point until a user evaluates a computation of ``d`` using a transformer.

.. Note::
  The following functionality is likely to be supplanted more composable abstractions involving op graph containers.

In some cases, it is convenient for an op graph construction function to associate additional information with an ``Op``. For example, the ``softmax`` function returns a ``DivideOp`` but when that output value is then used in a cross-entropy entropy calculation, the derivative computation would be numerically unstable if performed directly. To avoid this The ``softmax`` function can indicate that the ``DivideOp`` is part of a ``softmax`` computation and indicate the sub-graphs that are useful in cross-entropy and derivatives by adding a ``schema`` to the ``DivideOp``:

.. code-block:: python

    >>> x = ng.placeholder((ng.make_axis(20, 'C')))
    >>> s = ng.softmax(x)
    >>> s.schemas
    [<ngraph.op_graph.op_graph.Softmax at 0x10c5e2210>]

More details about the mechanics of automatic differiantion and how ``deriv`` works are covered in :doc:`autodiff`.

