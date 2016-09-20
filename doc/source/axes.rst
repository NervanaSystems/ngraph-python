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


Set Ups
-------

In this documentation, let's assume we predefine ``Axis`` objects as follows.
::

  import ngraph as ng
  import numpy as np

  A = ng.Axis(length=1)
  B = ng.Axis(length=2)
  C = ng.Axis(length=3)
  D = ng.Axis(length=4)

  A_ = ng.Axis(length=1)
  B_ = ng.Axis(length=2)
  C_ = ng.Axis(length=3)
  D_ = ng.Axis(length=4)

Sepecially, we let ``X`` and ``X_`` to be two axes of the same length but under
different labels.


Property of Axes
----------------

1. **A tensor cannot have repetitive axes.**

  For example, ::

      x = ng.Constant(np.ones((2, 2)), axes=ng.Axes([B, B]))  # throws exception
      x = ng.Constant(np.ones((2, 2)), axes=ng.Axes([B, B_]))  # correct


2. **The order of Axes does not matter.** ::

  - Two tensors ``x`` and ``y`` are considered "equivalent" if

    - ``x`` and ``y`` have the same number of axes and same set of axes
    - After shuffling of ``y``'s axes to be the same order of ``x``'s, the underlying value are the same.

  - For example, ``x`` and ``y`` are considered "equivalent" in the following scripts ::

        np_val = np.random.rand(2, 3)
        x = ng.Constant(np_val, axes=ng.Axes([B, C]))
        y = ng.Constant(np_val.T, axes=ng.Axes([C, B]))


Elementwise Binary Ops
----------------------

- When matches, output the same axis. ::

  (A,) + (A,) -> (A,)
  (A, B) + (A, B) -> (A, B)

- Automatic broadcasting / dim shuffle, the output axis order determined by input axis order of the left and right operands. ::

  (A, B) + (A,) -> (A, B)
  (A, B) + (B,) -> (A, B)
  (A, B) + (B, C) -> (A, B, C)
  (A, B) + (C, B) -> (A, B, C)
  (A, B) + (C, B, D) -> (A, B, C, D)

- Commutative property is as usual, though axis order of the equivalent tensors can be different. ::

  (A,) + (B,) -> (A, B)
  (B,) + (A,) -> (B, A)
  (A,) + (B, C) -> (A, B, C)
  (B, C) + (A,) -> (B, C, A)

- Associative property is as usual. ::

  ((A,) + (B,)) + (C,) -> (A, B) + (C,) -> (A, B, C)
  (A,) + ((B,) + (C,)) -> (A,) + (B, C) -> (A, B, C)

- Distributive property is as usual. ::

  (A,) * ((B,) + (C,)) = (A,) * (B, C) = (A, B, C)
  (A,) * (B,) + (A,) * (C,) = (A, B) * (A, C) = (A, B, C)


Dot Products
------------

- Standard 2d dot. ::

  (A, B) • (B, C) -> (A, C)

- Dot operation will take place in overlapping axes of the left and right operands. That is, the overlapping axes will be eliminated in the output tensor. ::

  (A, B, C) • (B, C, D) -> (A, D)
  (A, B) • (A,) -> (B,)

- Left & right operands can be swapped, order of axis can be swapped, results are equivalent, though order can be different. ::

  (A, B) • (B, C) -> (A, C)
  (B, A) • (B, C) -> (A, C)
  (B, C) • (A, B) -> (C, A)


Axes Reduction
--------------

- Reduction operations can have arbitary number of reduction axis, which, alternatively we specify them as ``out_axis``.
- The ``out_axis`` order can be arbitary, as long as all axes in ``out_axis`` is in the original tensor's axes.
- When ``out_axis`` is empty list or tuple, reduction is performed on all axes.

Examples: ::

    reduce((A, B, C), out_axis=()) -> ()
    reduce((A, B, C), out_axis=(A,)) -> (A,)
    reduce((A, B, C), out_axis=(A, B)) -> (A, B)
    reduce((A, B, C), out_axis=(C, B)) -> (C, B)



Casting Axis
------------

Use ``AxesCastOp`` to cast at tensor to known axes. The user must user that the
targeting axes has the same length as the origin tensor's axes at all
coordinates.

- Example 1: adding two tensors of shape ``(2, 3)`` but with differently named axis ::

    x = ng.Constant(np.ones((2, 3)), axes=ng.Axes([B, C]))
    y = ng.Constant(np.ones((2, 3)), axes=ng.Axes([B_, C_]))
    # z1 have axes: (B, C, B_, C_)
    z1 = x + y
    # z2 have axes: (B, C), which is what we expect
    z2 = x + ng.AxesCastOp(y, x.axes)

- Example 2: invalid casting::

    y = ng.Constant(np.ones((2, 3)), axes=ng.Axes([B_, C_]))
    z1 = ng.AxesCastOp(y, axes=ng.Axes([B, C]))  # valid
    z2 = ng.AxesCastOp(y, axes=ng.Axes([C, B]))  # exception when evaluated
    z3 = ng.AxesCastOp(y, axes=ng.Axes([B, D]))  # exception when evaluated
