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

- Left & right operands can be swapped, order of axis can be swapped, results are equivalent. ::

  (A, B) • (B, C) -> (A, C)
  (B, A) • (B, C) -> (A, C)
  (B, C) • (A, B) -> (C, A)


Casting Axis
------------


Default Axis for neon
---------------------
