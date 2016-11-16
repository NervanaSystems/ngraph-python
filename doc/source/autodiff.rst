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

We use the autodiff algorithm to generate the *backprop* computations for derivatives. Autodiff is based on symbolic differentiation via the chain rule.

Computing Derivatives in Op-Graph
=================================

Each ``Op`` node in our graph implements the ``generate_adjoints`` method, which defines the local gradient for that operation and propagates the deltas to its arguments. To compute the derivatives, we first perform a topological sort on the graph, then traverse the graph in order, calling each node's ``generate_adjoints`` method to add the required backprop computations to the graph.

The ``generate_adjoints`` method accepts ``adjoints``, ``delta``, and the arguments of the op.
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


Technical Details
=================

Although we write a computation in a program as a series of expressions, they are converted into a series of steps, each a function producing a value from previously computed values.  We will use the notation :math:`t_{mj}` for a value which is computed from :math:`\{t_{ij} | i<m\}`.  We start with the independent variables,

.. math:: t_{00}, t_{01}, \ldots

Then we apply functions to these variables to obtain

.. math:: t_{1i} = f_{1i}(t_{00}, t_{01}, \ldots).

It is not necessary for each function to use all of the values, but it must only use previous values.

From the original values and the newly computed values we compute new values,

.. math:: t_{2i} = f_{2i}(t_{00}, t_{01}, \ldots, t_{10}, t_{11}, \ldots).

We proceed until we finally have :math:`y=t_{n0}`.

For each computation, we have

.. math:: dt_{mi} = \sum_{jk} D_{jk}f_{mi}(\ldots)d_{jk}

where :math:`D_{jk}f_{mi}` is the derivative of :math:`f_{mi}` with respect to argument :math:`jk`.

If we continue expanding the :math:`dt_{mi}` expressions we will eventually have an expression that can be written as

.. math:: dy = \sum a_{00k}dt_{0k}.

Since the layer 0 values are independent,

.. math:: \frac{dt_{0i}}{dt_{0j}} = \delta_{ij}

so

.. math:: \frac{dy}{dt_{0j}} = a_{00j}.

If we expand the computation level by level we find that at level :math:`m` we have

.. math:: dy = \left(\sum_{mj} a_{mj}dt_{mj}\right) + \left(\sum_{ij, i<m} b_{ij}dt_{ij}\right),

where :math:`a_{mj}` is called the *adjoint* of :math:`dt_{mj}`.

If we expand the :math:`dt_{mj}` terms, we will be left with only :math:`dt_{ij}` terms for :math:`i<m`.  During the expansion, we can push the :math:`a_{mij}` adjoints down to the next level.

Example
-------

To compute :math:`y = ax^2+bx+c` we have:

.. math::
    t_{00} &= a, &t_{01} = b, &t_{02} = c, &t_{03} = x \\
    t_{10} &= t_{03}^2, &t_{11} = t_{01} t_{03}\\
    t_{20} &= t_{00}t_{10}, &t_{21} = t_{11} + t_{02} \\
    t_{30} &= t_{20} + t_{21} \\
    y &= t_{30}.

The derivatives of these computations are:

.. math::
    dt_{10} &= 2t_{03}dt_{03}, &dt_{11}=t_{01}dt_{03} + t_{03}dt_{01} \\
    dt_{20} &= t_{00}dt_{10} + t_{10}dt_{00}, &dt_{21} = dt_{11} + dt_{02} \\
    dt_{30} &= dt_{20}+dt_{21}\\
    dy &= dt_{30}

Now we start expanding:

.. math::
    dy &= 1 dt_{30}\\
    &= 1(dt_{20}+dt_{21})\\
    &= 1 dt_{20} + 1 dt_{21}

In the expansion, we pushed the adjoint of 1 on :math:`dt_{30}` down to the terms in the expansion.

We then expand the :math:`dt_{21}` terms to get:

.. math::
    dy &= 1(t_{00}dt_{10} + t_{10}dt_{00}) + 1(dt_{11} + dt_{02})\\
    &= t_{00}dt_{10} + t_{10}dt_{00} + 1dt_{11} + 1dt_{02}

Finally, we expand the first level terms to get

.. math::
    dy &= t_{00}(2(t_{03}dt_{03})+t_{10}dt_{00}+1(t_{01}dt_{03}+t_{03}dt_{01})+1dt_{02}\\
    &= t_{10}dt_{00}+t_{03}dt_{01}+1dt_{02}+(2t_{00}t_{01}+t_{01})dt_{03}

The Algorithm
-------------

Every intermediate value in the computation supports three adjoint methods, initialize, increment, and finalize.  The initialize step is performed when the intermediate value is computed, the increment is called when a node which uses the value sends a contribution to the adjoint, and finalize is called when there will be no more contributions to the adjoint; processing at its level is complete.

There are two ways to implement the three methods.
    1. The initialize and finalize methods do nothing, while the increment method propagates to increment methods at lower levels.
    2. We associate an adjoint array of the same kind as the value.  Initialize initializes the adjoint to 0 (possibly also allocating it), increment increments the adjoint, and finalize propagates the appropriate values to increment methods for lower level adjoints, and possibly frees the adjoint storage.

    For values at level 0 that we want derivatives for we use the second approach, and the remaining values at level 0 use the first approach, which ignores the updates.  At higher levels, the approach depends on the computation and how many computations use the value.  If the update is simple, or if the value is only used once, the first approach should be used, while if it is cheaper to accumulate the adjoint and process it all at once, the second approach is used.

For example, if we have a computation :math:`t_m = t_a t_b` then, since :math:`dt_m = t_b dt_a+t_a dt_b`, we perform

.. math::
    \overline{t_a} += \overline{t_m} t_b\\
    \overline{t_b} += \overline{t_m} t_a

where we use :math:`\overline{t}` to denote the adjoint we are accumulating for :math:`t`.

We use method 2 so that we only need to perform the multiplication once.  Compare this with :math:`t_m=t_a+t_b` with derivative :math:`dt_a+dt_b`.  If there are two uses of the value, using approach 2 requires allocating and initializing an array for the adjoint (we could have the first update perform the initialization), followed by one addition to the adjoint, and then two additions as the adjoint is passed to the next level, while approach 1 requires four additions to the adjoints at the next level, but no additional storage.
