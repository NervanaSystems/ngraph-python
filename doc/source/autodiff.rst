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

Autodiff
********

Note: There is a lot of material to be moved here from GoogleDocs.

Computing Derivatives
=====================

Derivatives are computed with backprop computations that are generated in two steps.  First, users of the ``Op`` accumulate their adjoints (backprops) into the adjoint for the ``Op``, and then the ``Op`` propagates its adjoint into the adjoints of its args.  The adjoint of an ``Op`` is a tensor with the same tensor description as the ``Op``, but which may have a different storage representation for ``Ops`` with axes information.

The ``generate_adjoints`` method will be passed ``adjoints``, ``delta``, and the ``*args`` of the op.
The ``adjoints`` contains
partially computed backprop ``Ops``, while ``delta`` is the complete adjoint of the ``Op``.

To implement ``generate_adjoints`` for an ``Op`` for the function

.. math:: f(x_1, x_2, \ldots, x_n)

write out

.. math:: df = a_1 dx_1 + a_2 dx_2 + \ldots + a_n dx_n

Then::

    def generate_adjoints(adjoints, delta, x1, x2, ..., xn):
        x1.generate_add_delta(adjoints, a1 * delta)
        x2.generate_add_delta(adjoints, a2 * delta)
        ...
        xn.generate_add_delta(adjoints, an * delta)


For example,

.. math:: f(x,y) = xy

    df = y dx + x dy

So::

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, y * delta)
        y.generate_add_delta(adjoints, x * delta)

