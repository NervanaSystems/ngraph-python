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

  For example, a set of standard neon axes are defined for neon frontends. ::

    N = Axis(name='N', batch=True)
    C = Axis(name='C')
    D = Axis(name='D')
    H = Axis(name='H')
    W = Axis(name='W')
    T = Axis(name='T', recurrent=True)
    R = Axis(name='R')
    S = Axis(name='S')
    K = Axis(name='K')
    M = Axis(name='M')
    P = Axis(name='P')
    Q = Axis(name='Q')
    Y = Axis(name='Y')


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

    x = ng.constant(np.ones((2, 3)),       | x = ng.constant(np.ones((2, 3)),
                    axes=ng.make_axes([H, W]))  |                 axes=ng.make_axes([H, W]))
    y = ng.constant(np.ones((3, 2)),       | y = ng.constant(np.ones((3, 2)),
                    axes=ng.make_axes([W, H]))  |                 axes=ng.make_axes([W, H]))
    z = x + y  # <==                       | z = y + x  # <==
                                           |
    trans = ng.NumPyTransformer()          | trans = ng.NumPyTransformer()
    comp = trans.computation([z])          | comp = trans.computation([z])
    z_val = comp()[0]                      | z_val = comp()[0]
    print(z_val)                           | print(z_val)
    print(z_val.shape)                     | print(z_val.shape)
    -------------------------------------------------------------------------------
    Output:                                | Output:
    [[ 2.  2.  2.]                         | [[ 2.  2.]
      [ 2.  2.  2.]]                       |  [ 2.  2.]
    (2, 3)                                 |  [ 2.  2.]]
                                           | (3, 2)

- Associative property is as usual. ::

  ((H,) + (W,)) + (N,) -> (H, W) + (N,) -> (H, W, N)
  (H,) + ((W,) + (N,)) -> (H,) + (W, N) -> (H, W, N)

- Distributive property is as usual. ::

  (H,) * ((W,) + (N,)) = (H,) * (W, N) = (H, W, N)
  (H,) * (W,) + (H,) * (N,) = (H, W) * (H, N) = (H, W, N)


Dot Products
------------

- 2D matrix dot with 2D matrix. ::

  (H, W) • (W, N) -> (H, N)

- Dot operation will be performed on overlapping axes of the left and right
  operands. That is, the overlapping axes will be eliminated in the output
  tensor. ::

  (C, H, W) • (H, W, N) -> (C, N)
  (H, W) • (H,) -> (W,)

- Left & right operands can be swapped, order of axis can be swapped, results
  are equivalent, though order can be different. ::

  (H, W) • (W, N) -> (H, N)
  (W, H) • (W, N) -> (H, N)
  (W, N) • (H, W) -> (N, H)


Axes Reduction
--------------

- We specify the reduction axes in ``reduction_axes``. Reduction operations can
  have arbitrary number of reduction axes. The order of the reduction axes
  can be arbitrary.
- When ``reduction_axes`` is empty, reduction is performed on NONE of the axes.

Examples: ::

    reduce((C, H, W), reduction_axes=())     -> (A, B, C)
    reduce((C, H, W), reduction_axes=(C,))   -> (B, C)
    reduce((C, H, W), reduction_axes=(C, W)) -> (H,)
    reduce((C, H, W), reduction_axes=(W, C)) -> (H,)

Axes casting
------------

Use ``AxesCastOp`` to cast at axes to targeting axes with the same dimensions.
For example, we might want to sum two layer's outputs, where they have the same
dimensions but different axes. ::

    # assume C1.length == C2.length == 100
    hidden_1 = ng.constant(np.ones((100, 128)), axes=ng.make_axes((C1, N)))
    hidden_2 = ng.constant(np.ones((100, 128)), axes=ng.make_axes((C2, N)))

    # if we add directly without casting
    sum_direct = hidden_1 + hidden_2  # sum_direct has axes: (C1, C2, N)

    # cast before sum
    hidden_2_cast = ng.make_axesCastOp(hidden_2_cast, ng.make_axes((C1, N)))
    sum_cast = hidden_1 + hidden_2_cast  # sum_cast has axes: (C1, N)

Axes broadcasting
-----------------

Use ``ng.Broadcast`` to broadcast to new axes. The new axes shall be a superset
of the original axes. The order of the new axes can be arbitrary.

Examples: ::

    broadcast((C, H), axes=(C, H, W)) -> (C, H, W)
    broadcast((C, H), axes=(W, H, C)) -> (W, H, C)
