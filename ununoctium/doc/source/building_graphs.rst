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
Although the name "operational graph API" contains the word "graph," the API is for defining, analyzing
and manipulating machine learning computations.  Although the API is supported by a few graphs behind the scenes,
the important thing for the user is the definition of models and frameworks for defining models.

Expression basics
=================
Manipulating expressions in a programming language that already has expressions can get a little
confusing, so we will start with a mathematical expression that is not from a program:

.. math:: y = \tanh(w*x+b)

where :math:`w` and :math:`b` are parameters, :math:`x` will be provided as input, and :math:`y`
will be returned as the result.  When you type the expression, :math:`y` will contain the result,
but behind the scenes many more actions take place.  When we work with computations, we need to think
about both what the computation is doing and how the computation is performed.

When we work evaluate an expression, we start with the things that have no dependencies, like :math:`w, x` and
:math:`b,` and compute something that only depends on things we already have values for, until every operation
has been performed.

When we work with expressions, we go in the other direction; we start with the last
thing we would evaluate, in this case the assignment to :math:`y.`  The assignment has two subexpressions,
the :math:`y` and the  :math:`\tanh(w*x+b).`  The :math:`=` is in the expression, but its role is to
specify what is being done with the two subexpressions, so it is not a subexpression, but, instead, identifies
the expresssion as an assignment.

The :math:`\tanh(w*x+b)` has only one subexpression, :math:`w*x+b` and is a :math:`\tanh` expression.  The
:math:`w*x+b` is a :math:`+` expression with two subexpressions, :math:`w*x` and :math:`b.`  The :math:`b`
is a variable reference with no subexpressions.

Although the :math:`b` expression has no subexpressions, it is different
from the variable expressins :math:`w` and :math:`x.`  We call ``b, w`` and ``x`` *parameters* of the variable;
:math:`b, w` and :math:`x` are all variables with no subexpressions, but they do differ in their parameters.
This is similar to the difference between :math:`+` and :math:`*`, but the difference between :math:`+` and
:math:`*` is more significant so for our convenience we will call them different kinds of expressions rather
than the same kind of expression with different parameters.

Expressions with Python
=======================
If we included the following in a Python program::

    tanh(w*x+b)

we would get an error that ``w`` is not defined.  If gave values ``w, x,`` and ``b`` we would get a result,
such as ``0.76159415595576485``, not the expression.  There are two ways to get the expression, write a
parser and pass the expression to it as a string to be parsed, or trick Python into returning an expression.

It is easy to turn ``tanh`` into an expression object.  All we need to do is define a class called ``tanh``
that is an expression object.  We can have the ``__init__`` try to coerce its argument to an expression if
it is not already one, and this becomes the subexpression of ``tanh``.  Normally a class would not be lowercase
as in ``tanh`` but since we want to think of it as a function we type it as a function.

Python supportds limited overloading with something called "magic methods."  Certain functions and operators
can be extended to new kinds of objects by defining their magic methods.  For example, if you define a class
with a ``__add__`` method, and ``x`` is an instance of your class, ``x+y`` will call ``x``'s ``__add__`` with the value
of ``y`` as an argument.  Likewise, if you define ``__radd__`` and ``y`` is an instance of your class, but
``x`` is neither a number nor an instance of a class with an ``__add__`` method, ``x+y`` would cause ``y``'s
``__radd__`` to be called with ``x`` as an argument.

In ``x+y``, if ``x`` is some kind of expression object with an ``__add__`` method, the ``__add__`` method can
coerce the ``y`` to be an expression object if it is not already one, and return an expression object for
the sum.  This expression object would have two subexpressions, the expression object that was in ``x`` and
the expression object that ``y`` was coerced into.

When Python evaluates a Python expression, each subexpression must be evaluated before the expression can be
evaluated, so we just need to ensure that Python expressions with no subexpressions are objects of our
expression class, or can be coerced into objects of our expression class, and that functions without magic
method support are expression-aware functions of our own, something we can arrange via imports.  Then the result
of Python evaluating one of these expressions will be an expression object.

Operational Graph Expressions
=============================
Almost all operational graph expressions describe tensor computations.  Associated with every tensor is a dtype and a
sequence of zero or more axes.  A tensor with zero axes is also called a scalar, a tensor with one axis a vector,
and two axes a matrix.  Axes are described elsewhere.
Unlike some graph/tensor languages, a tensor does not need to be associated with storage.

In the operational graph, all the expressions are operations of some kind, so we call them ops, and they are all
instances of the class ``Op``.  Most expressions are tensors, and are instance of the ``Tensor`` class.  However,
some ops are used for side-effects and produce no value; these are instance of ``VoidOp``.  Ops that actually
perform a computation on their arguments are instances of ``ComputationOp``.

During a computation, the values must be stored somewhere, but only those tensors whose values are explicitly
marked as being needed at the end of a computation are available when the computation completes.
The general computation model is that computation may occur on a device with its own memory, so we need a way
to copy data to/from the device.  A tensor marked as ``in`` can be written to before a computation, and as
``out`` can be read after a computation.  A tensor can be both ``in`` and ``out``.

A ``placeholder`` is a tensor marked as ``in``, so it can be written to before a computation.
To create a ``placeholder`` expression in the operational graph, we must import the operational backend symbols
and then make the ``placeholder``::code

    import geon as be
    import geon.frontends.base.axis as ax

    x = be.placeholder(axes=be.Axes(ax.C, ax.W, ax.H, ax.N))


This will create an op for a ``placeholder`` with the indicated list of axes and assign the Python
variable `x` to the op.  When the op is used in a graph, the op serves as a Python handle
for the tensor stored in the device.

It is important to remember that ``x`` is a Python variable that holds an op.  There are no magic methods for
Python variable assignment or use, so assigning a new value to ``x`` has no effect on the the tensor
previously represented by ``x``.  In other words::code

    x = x + x

does not double the value of the tensor in the ``placeholder``.  Instead, the Python variable ``x`` is now an
op that is the sum of the ``placeholder`` and itself.  In order to change the value of the ``placeholder``
you would need to say::code

    be.SetItem(x, x + x)

Perhaps surprisingly, because we are manipulating expressions, you rarely need to use ``SetItem``, other than
when updating variables after training.  Consider::code

    x1 = x + x
    y = x1 * x1 - x

The Python variable ``y`` holds an op for a computation that adds the ``placeholder`` to itself, then multiplies
that value by itself, and then subtracts the original value of the ``placeholder``.  The intermediate
value ::code``x + x`` is only computed once, since the same op is used for both arguments of the multiplication.
Furthermore, in this computation, all the computations will automatically be performed in place.  In NumPy
it would be like::code

    y = x + x
    np.multiply(y, y, out=y)
    bp.subtract(y, x, out=y)

However, if you later modified the computation so that you needed ``x + x`` in some other operation, we would
automatically adjust the computation's implementation so that the intermediate result ``x + x`` was available
wherever it was needed.  You can get this flexibility with NumPy or PyCUDA with the original expression, but they
will be allocating tensors for the intermediate values and letting Python's garbage collector clean them up; the
peak memory usage will be higher and there will be more overhead.



