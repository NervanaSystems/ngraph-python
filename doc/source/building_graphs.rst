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

Non-data Control Dependencies
-----------------------------
Finally, consider the following graph construction:

.. code-block:: python

    >>> x = ng.placeholder((), initial_value=0)
    >>> a = ng.assign(x, 5)
    >>> z = x + 1

Here we create a scalar placeholder ``x``, an assignment of 5 to the placeholder ``x``, and an addition of 1 to ``x``. It may not be clear if ``z`` when evaluated should equal ``1`` or ``6``. The sub-graph for ``z`` does not include the assignment, so the result would be ``1``. To include the assignment, we provide ``ng.sequential`` which causes ops to be executed in the order listed, with the last op serving as the value, subject to the constraint that ops in a computation are only executed once. To force the assignment, we would write:

.. code-block:: python

  >>> x = ng.placeholder((), initial_value=0)
  >>> z = ng.sequential([
            ng.assign(x, 5),
            x + 1
          ])

Now ``z`` performs the assignment and then returns the value of ``x + 1``.

General properties of ops
=========================

All operational graph ops are instances of the class :py:class:`ngraph.op_graph.op_graph.Op`, which extends :py:class:`ngraph.op_graph.names.ScopedNameableValue`. This provides ``Ops`` with automatically generated unique names.

In addition to the graph properties explained above (``args``) all ops have the additional attributes:

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
    ax_C = ng.make_axis(length=4, name='C')
    ax_W = ng.make_axis(length=2, name='W')
    ax_H = ng.make_axis(length=2, name='H')
    ax_N = ng.make_axis(length=128, name='N')

    x = ng.placeholder((ax_C, ax_W, ax_H, ax_N))

This ``placeholder`` will create an ``AssignableTensorOp`` to trigger the necessary storage to be allocated on the host device and trigger values to be transferred between the device and host. When the op is used in a graph computation, the op serves as a Python handle for the tensor stored on the device.

It is important to remember that ``x`` is a Python variable that holds an op.  Therefore, the following code:

.. code-block:: python

    x = x + x

does not directly double the value of the tensor in the ``placeholder``. Instead, the ``__add__`` method is called with
both arguments pointing to the same ``placeholder`` object. This returns a new ``Op`` that is now stored as the python variable ``x``.

Consider:

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

    ax_C = ng.make_axis(length=4, name='C')
    ax_Y = ng.make_axis(length=4, name='Y')
    ax_W = ng.make_axis(length=2, name='W')
    ax_H = ng.make_axis(length=2, name='H')
    ax_N = ng.make_axis(length=128, name='N')

    x = ng.placeholder((ax_C, ax_W, ax_H, ax_N))
    y0 = ng.placeholder((ax_Y, ax_N))
    w = ng.variable((ax_C, ax_W, ax_H, ax_Y))
    b = ng.variable((ax_Y,))
    y = ng.tanh(ng.dot(w, x) + b)
    c = ng.squared_L2(y - y0)
    d = ng.deriv(c, w)

The python variable ``d`` will hold an ``Op`` whose value is the derivative ``dc/dw``. In this example, we knew which ops contain the variables to be trained (e.g. ``w``).  For a more general optimizer, we could search through all the subexpressions looking for the dependant variables.  This is handled by the ``variables`` method, so ``c.variables()`` would return the list of ``Ops`` ``[w, b]``.

An important distinction to make here is that the ``deriv`` function does not perform symbolic or numeric differentiation. In fact it does not compute anything at all. Its sole job is to construct another computational graph using the existing upstream graph of ``c`` and return a handle to that new computational graph (``d``). No computation is therefore taking place at this point until a user evaluates a computation of ``d`` using a transformer.

.. Note::
  The following functionality is likely to be supplanted more composable abstractions involving op graph containers.

In some cases, it is convenient for an op graph construction function to associate additional information with an ``Op``. For example, the ``softmax`` function returns a ``DivideOp`` but when that output value is then used in a cross-entropy entropy calculation, the derivative computation would be numerically unstable if performed directly. To avoid this, the ``softmax`` function can indicate that the ``DivideOp`` is part of a ``softmax`` computation and indicate the sub-graphs that are useful in cross-entropy and derivatives by adding a ``deriv_handler`` to the ``DivideOp``:

More details about the mechanics of automatic differiantion and how ``deriv`` works are covered in :doc:`autodiff`.

