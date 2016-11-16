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

Axes
****

Introduction
------------

An Axis labels a dimension of a tensor. The op-graph uses
the identity of ``Axis`` objects to pair and specify dimensions in
symbolic expressions. This system has several advantages over
using the length and position of the axis as in other frameworks:

1. **Convenience.** The dimensions of tensors, which may be nested
deep in a computation graph, can be specified without having to
calculate their lengths.

2. **Safety.** Axis labels are analogous to types in general-purpose
programming languages, allowing objects to interact only when
they are permitted to do so in advance. In symbolic computation,
this prevents interference between axes that happen to have the
same lengths but are logically distinct, e.g. if the number of
training examples and the number of input features are both 50.


Core concepts
-------------

Axis and Axes
~~~~~~~~~~~~~
- ``Axis`` represents one dimension of a tensor. We can use ``ng.make_axis`` to
  create an ``Axis``.
  ::

    H = ng.make_axis(length=3, name='height')
    W = ng.make_axis(length=4, name='width')

- ``Axes`` represents multiple dimensions of a tensor. We can use ``ng.make_axes``
  to create ``Axes``.
  ::

    axes = ng.make_axes([H, W])

- After the ``Axes`` is created, we can apply it to a tensor. For example:
  ::

    image = ng.placholder(axes=axes)

- It's possible to delay the specification of axis length.
  ::

    H = ng.make_axis(length=3, name='height')
    W = ng.make_axis(length=4, name='width')
    image = ng.placholder(axes=ng.make_axes([H, W]))
    H.length = 3
    W.length = 4


AxisRole
~~~~~~~~
``AxisRole`` is the "type" for an ``Axis``.

- For example, in layer 1's feature
  map axes ``(C1, D1, H1, W1, N)`` and layer 2's feature map axes
  ``(C2, D2, H2, W2, N)``, ``C1`` and ``C2`` shares the same ``AxisRole`` as
  "channels", while ``D1`` and ``D2`` shares the same ``AxisRole`` as "depth".
- AxisRole is primarily for automatic axes inferencing. For example, a conv kernel
  can look at its input feature maps' ``AxisRole`` to determine whether a
  dimshuffle shall be applied prior to convolution.
- We can create ``AxisRole`` via ``ng.make_axis_role()``. For example:
  ::

    role_channel = ng.make_axis_role()
    axis_channel = ng.axis(length=3, roles=[role_channel])


DualAxis
~~~~~~~~
The nervana graph axes are agnostic to data layout on the compute device, so
the ordering of the axes does not matter. As a consequence, when two tensors
are provided to a ``ng.dot()`` operation, for example, one needs to indicate
which are the corresponding axes that should be matched together. We use
"dual offsets" of +/- 1 to mark which axes should be matched during a multi-axis
operation. For example:
::

  x_axes = ng.make_axes([ax.C, ax.H, ax.W, ax.N])
  x = ng.placeholder(axes=x_axes)
  w_axes = ng.make_axes([ax.M, ax.C - 1, ax.H - 1, ax.W - 1])
  w = ng.variable(initial_value=np.random.randn(*w_axes.lengths),
                  axes=w_axes)
  result = ng.dot(w, x)

- In the example, ``x`` is the right-hand side operand of ``ng.dot``, and we call
  ``x``'s axes the primary axes.
- Then to get the left-hand side matching dual axes of the primary axes, we use
  the ``-1`` operation to mark the matchin axes. That is, ``ax.C - 1``,
  ``ax.H - 1``, ``ax.W - 1`` match ``ax.C``, ``ax.H`` aond ``ax.W`` respectively.
  Similary, if we treat the left-hand side operand's axes to be the primary axes,
  we use the ``+1`` operation to mark its corresponding right-hand side
  operand's axes.
- When a dot operation is performed, the matching axes will be combined and
  cancelled out, leaving the unmatched axes in the result's axes. In the example
  above, the resulting axes of ``ng.dot(w, x)`` are [ax.M, ax.N].
- More examples on ``DualAxis`` in dot products
  ::

    # 2d dot
    (H, W - 1) • (W, N) -> (H, N)
    (H, W) • (W + 1, N) -> (H, N)

    # 4d dot
    (M, C - 1, H - 1, W - 1)  • (C, H, W, N) -> (M, N)
    (M, C, H, W)  • (C + 1, H + 1, W + 1, N) -> (M, N)

    # swapping the left and right operands is allowed
    (H, W - 1) • (W, N) -> (H, N)
    (W, N) • (H, W - 1) -> (H, N)

    # swapping the order of the axes is allowed
    (M, C - 1, H - 1, W - 1)  • (C, H, W, N) -> (M, N)
    (M, W - 1, H - 1, C - 1)  • (C, H, W, N) -> (M, N)


Properties
----------

1. The order of Axes does not matter. ::

  - Two tensors ``x`` and ``y`` are considered to be equal if

    - ``x`` and ``y`` have the same number of axes and same set of axes
    - After shuffling of ``y``'s axes to be the same order of ``x``'s, the
      underlying values are the same.

  - We can check element-wise tensor equality using ``ng.equal()``. In the
    following scripts, ``x`` and ``y`` are equal.  ::

      import numpy as np
      import ngraph as ng

      H = ng.make_axis(length=2)
      W = ng.make_axis(length=3)
      np_val = np.random.rand(2, 3)
      x = ng.constant(np_val, axes=ng.make_axes([H, W]))
      y = ng.constant(np_val.T, axes=ng.make_axes([W, H]))
      z = ng.equal(x, y)

      trans = ng.NumPyTransformer()
      comp = trans.computation([z])
      z_val = comp()[0]
      print(z_val)
      # [[ True  True  True]
      #  [ True  True  True]]

2. A tensor cannot have repetitive axes.

  For example: ::

      H = ng.make_axis(length=2)
      W = ng.make_axis(length=2)
      x = ng.constant(np.ones((2, 2)), axes=ng.make_axes([H, H]))  # throws exception
      x = ng.constant(np.ones((2, 2)), axes=ng.make_axes([H, W]))  # good

3. Axes have context

  A set of standard neon axes are defined for neon frontends.

  - Axes roles
  ::

    ar = ng.make_name_scope(name="ar")
    ar.Height = ng.make_axis_role()
    ar.Width = ng.make_axis_role()
    ar.Depth = ng.make_axis_role()
    ar.Channel = ng.make_axis_role()
    ar.Channelout = ng.make_axis_role()
    ar.Time = ng.make_axis_role()

  - Image / feature map
  ::

    ax = ng.make_name_scope(name="ax")
    ax.N = ng.make_axis(batch=True, docstring="minibatch size")
    ax.C = ng.make_axis(roles=[ar.Channel], docstring="number of input feature maps")
    ax.D = ng.make_axis(roles=[ar.Depth], docstring="input image depth")
    ax.H = ng.make_axis(roles=[ar.Height], docstring="input image height")
    ax.W = ng.make_axis(roles=[ar.Width], docstring="input image width")

  - Filter (convolution kernel)
  ::

    ax.R = ng.make_axis(roles=[ar.Height], docstring="filter height")
    ax.S = ng.make_axis(roles=[ar.Width], docstring="filter width")
    ax.T = ng.make_axis(roles=[ar.Depth], docstring="filter depth")
    ax.J = ng.make_axis(roles=[ar.Channel], docstring="filter channel size (for crossmap pooling)")
    ax.K = ng.make_axis(roles=[ar.Channelout], docstring="number of output feature maps")

  - Output
  ::

    ax.M = ng.make_axis(roles=[ar.Depth], docstring="output image depth")
    ax.P = ng.make_axis(roles=[ar.Height], docstring="output image height")
    ax.Q = ng.make_axis(roles=[ar.Width], docstring="output image width")

  - Recurrent
  ::

    ax.REC = ng.make_axis(roles=[ar.Time], recurrent=True, docstring="recurrent axis")

  - Target
  ::

    ax.Y = ng.make_axis(docstring="target")


Elementwise Binary Ops
----------------------

- When matches, output the same axis. ::

  (H,) + (H,) -> (H,)
  (H, W) + (H, W) -> (H, W)

- Automatic broadcasting / dim shuffle, the output axis order determined by input
  axis order of the left and right operands. ::

  (H, W) + (H,) -> (H, W)
  (H, W) + (W,) -> (H, W)
  (H, W) + (W, N) -> (H, W, N)
  (H, W) + (N, W) -> (H, W, N)
  (C, H) + (W, H, N) -> (C, H, W, N)

- Commutative property is as usual, though axis order of the equivalent tensors
  can be different. ::

  (H,) + (W,) -> (H, W)
  (W,) + (H,) -> (W, H)
  (C,) + (H, W) -> (C, H, W)
  (H, W) + (C,) -> (H, W, C)

  In the following example, `z` from left and right are equivalent, although
  the axis orders are different.

  ::

    x = ng.constant(np.ones((2, 3)),           | x = ng.constant(np.ones((2, 3)),
                    axes=ng.make_axes([H, W])) |                 axes=ng.make_axes([H, W]))
    y = ng.constant(np.ones((3, 2)),           | y = ng.constant(np.ones((3, 2)),
                    axes=ng.make_axes([W, H])) |                 axes=ng.make_axes([W, H]))
    z = x + y  # <==                           | z = y + x  # <==
                                               |
    trans = ng.NumPyTransformer()              | trans = ng.NumPyTransformer()
    comp = trans.computation([z])              | comp = trans.computation([z])
    z_val = comp()[0]                          | z_val = comp()[0]
    print(z_val)                               | print(z_val)
    print(z_val.shape)                         | print(z_val.shape)
    -------------------------------------------------------------------------------
    Output:                                    | Output:
    [[ 2.  2.  2.]                             | [[ 2.  2.]
      [ 2.  2.  2.]]                           |  [ 2.  2.]
    (2, 3)                                     |  [ 2.  2.]]
                                               | (3, 2)

- Associative property is as usual. ::

  ((H,) + (W,)) + (N,) -> (H, W) + (N,) -> (H, W, N)
  (H,) + ((W,) + (N,)) -> (H,) + (W, N) -> (H, W, N)

- Distributive property is as usual. ::

  (H,) * ((W,) + (N,)) = (H,) * (W, N) = (H, W, N)
  (H,) * (W,) + (H,) * (N,) = (H, W) * (H, N) = (H, W, N)


Axes Reduction
--------------

- We specify the reduction axes in ``reduction_axes``. Reduction operations can
  have arbitrary number of reduction axes. The order of the reduction axes
  can be arbitrary.
- When ``reduction_axes`` is empty, reduction is performed on NONE of the axes.

Examples: ::

    from ngraph.frontends.neon.axis import ax
    x = ng.placeholder([ax.C, ax.H, ax.W])
    ng.sum(x, reduction_axes=ng.make_axes([]))            -> [ax.C, ax.H, ax.W]
    ng.sum(x, reduction_axes=ng.make_axes([ax.C]))        -> [ax.H, ax.W]
    ng.sum(x, reduction_axes=ng.make_axes([ax.C, ax.W]))  -> [ax.H]
    ng.sum(x, reduction_axes=ng.make_axes([ax.W, ax.C]))  -> [ax.H]
    ng.sum(x, reduction_axes=x.axes)                      -> []


Axes Casting
------------

Use ``ng.cast_axes`` to cast at axes to targeting axes with the same dimensions.
For example, we might want to sum two layer's outputs, where they have the same
dimensions but different axes. ::

    # assume C1.length == C2.length == 100
    hidden_1 = ng.constant(np.ones((100, 128)), axes=ng.make_axes((C1, N)))
    hidden_2 = ng.constant(np.ones((100, 128)), axes=ng.make_axes((C2, N)))

    # if we add directly without casting
    sum_direct = hidden_1 + hidden_2  # sum_direct has axes: (C1, C2, N)

    # cast before sum
    hidden_2_cast = ng.cast_axes(hidden_2_cast, ng.make_axes((C1, N)))
    sum_cast = hidden_1 + hidden_2_cast  # sum_cast has axes: (C1, N)


Axes Broadcasting
-----------------

Use ``ng.broadcast`` to broadcast to new axes. The new axes must be a superset
of the original axes. The order of the new axes can be arbitrary.

Examples: ::

    from ngraph.frontends.neon.axis import ax
    x = ng.placeholder([ax.C, ax.H])
    ng.broadcast(x, axes=ng.make_axes([ax.C, ax.H, ax.W]))  -> [ax.C, ax.H, ax.W]
    ng.broadcast(x, axes=ng.make_axes([ax.W, ax.H, ax.C]))  -> [ax.W, ax.H, ax.C]


Axes dim-shuffle
----------------

Use ``ng.Dimshuffle`` to shuffle axes. The new axes must be the same set as the
original axes.

Examples: ::

    from ngraph.frontends.neon.axis import ax
    x = ng.placeholder([ax.C, ax.H, ax.W])
    ng.Dimshuffle(x, ng.make_axes([ax.H, ax.W, ax.C])).axes
