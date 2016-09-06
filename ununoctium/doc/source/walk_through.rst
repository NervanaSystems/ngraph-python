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

.. include:: <isonum.txt>

Walk-through
************

Let's begin with a very simple example, computing :math:`x+1` for several values of :math:`x` using the front end
API.  We should think of the computation as being invoked from our program, *the CPU*, but possibly taking place
somewhere else, which we will refer to as *the device.*
Our program will provide values for :math:`x` and receive :math:`x+1` for each :math:`x` provided.

The x + 1 program
=================

The complete program source can be found in :download:`../../examples/walk_through/x_plus_one.py`.

The complete program is::

    import geon

    x = geon.placeholder(axes=geon.Axes())
    x_plus_one = x + 1

    transformer = geon.NumPyTransformer()

    plus_one = transformer.computation(x_plus_one, x)

    for i in range(5):
        print(plus_one(i))


We begin by importing ``geon``, the Python module for the front end API.

Next we create the operational graph (opgraph) for the computation.  Following |TF| terminology, we call the
parameter that receives the value of :math:`x` a ``placeholder``.  A placeholder has a tensor value, so we need
to indicate what kind of tensor by specifying its axes.  In this simple example, :math:`x` is a scalar,
so the axes are empty.  We follow this with the computation that adds 1 to the ``placeholder.``  Even though
this looks like we are adding 1, the opgraph objects overload arithmetic, so ``x_plus_one`` is really
an opgraph object.

Once the graph is set up, we can compile it with a *transformer*.  Here we use the transformer that uses NumPy, but
the procedure would be the same for any other transformer.  We tell the transformer that we will want a function
that computes ``x_plus_one`` and that it has one parameter, ``x``.  We could specify additional computations,
as well as have computations return more than one value.

Once all our computations have been specified, we can run them.  The first time some computation for a transformer
is run, all the computations for the transformer are analyzed and compiled, and storage is allocated and initialized
on the device.  These steps can be performed manually, for example if some device state is to be restored from
previously saved state.  Once compiled, the computations are callable Python objects.

On each call to ``plus_one`` the value of ``x`` is copied to the device, 1 is added, and then the result is copied
back from the device.

The Compiled x + 1 Program
--------------------------

We can examine the compiled code to get an idea of the runtime device model::

    class Model(object):
        def __init__(self):
            self.a_t7 = None
            self.v_t7_ = None
            self.a_t8 = None
            self.v_t8_ = None
            self.a_t6 = None
            self.v_t6_ = None

        def alloc_a_t7(self):
            self.update_a_t7(np.empty(1, dtype=np.dtype('float32')))

        def update_a_t7(self, buffer):
            self.a_t7 = buffer
            self.v_t7_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def alloc_a_t8(self):
            self.update_a_t8(np.empty(1, dtype=np.dtype('float32')))

        def update_a_t8(self, buffer):
            self.a_t8 = buffer
            self.v_t8_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def alloc_a_t6(self):
            self.update_a_t6(np.empty(1, dtype=np.dtype('float32')))

        def update_a_t6(self, buffer):
            self.a_t6 = buffer
            self.v_t6_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def allocate(self):
            self.alloc_a_t7()
            self.alloc_a_t8()
            self.alloc_a_t6()


        def c_0(self):
            np.add(self.v_t6_, self.v_t7_, out=self.v_t8_)

        def c_1(self):
            pass


Tensors have two components, storage for their elements, and views of that storage.  The ``alloc_`` methods allocate
storage and then create the views of the storage that will be needed.  The view creation is separated from the
allocation because we also allow storage to be allocated by other mechanisms.  The ``allocate`` method calls each
allocator, and each allocator creates the needed views.  The NumPy transformer's ``allocate`` method calls the
``allocate`` method.  A GPU transformer might handle allocation off device.

Initializations can be associated with allocated storage.  For example, trainable variables could be intialized
uniformly.  In this case, there are no initializations, so the method `c_1` which performs one-time device
initialization is empty.  Constants, such as 1, are copied to the device as part of the allocation process.

The method ``c_1`` handles the ``plus_one`` computation.  Clearly this is not the optimal way to add 1 to a scalar,
so let's look at a more complex example next.

Logistic Regression
===================

The complete program source can be found in :download:`../../examples/walk_through/logres.py`.

The next example is logistic regression.  We want to classify an observation :math:`x` as having some property,
where we will say :math:`y=1` if it has the property, and :math:`y=0` if it does not.  We want to find the best values
for :math:`W` for the model :math:`\hat{y}=\sigma(Wx)`, using binary cross-entropy of the samples as the
error function and gradient descent with a learning rate of :math:`\alpha`.

We start with basic setup and some training data::

    import numpy as np
    import geon

    xs = np.array([[0.52, 1.12, 0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]]).T

    ys = np.array([1, 1, 0, 1])

    C, N = xs.shape

Our model will have three placeholders, ``X``, ``Y``, and ``alpha``.  ``alpha`` is a scalar, so we already know how
to specify its axes.  ``X`` and ``Y`` do have shape, and we will need to provide the shape to the placeholders.
|Geon| has a unique Axes facility for making it easier to describe tensor computations, but we will begin with
conventional shapes.  Our convention is to use the last axis for samples.  The placeholders can be specified as::

    X = geon.placeholder(axes=geon.Axes([C, N]))
    Y = geon.placeholder(axes=geon.Axes([N]))
    alpha = geon.placeholder(axes=geon.Axes())

We also need our training weights, ``W``.  Unlike the placeholders, we want ``W`` to retain its value from computation
to computation.  Following |TF|, we call this a *Variable*.  Again, we need to specify the axes.  In addition, we
want to specify an initial value for the tensor::

    W = geon.Variable(axes=geon.Axes([C]), initial_value=0)

Other than the axes, the syntax is the same as |TF|. The transformer's initialization function will initialize `W`
to 0 after allocating storage.

Now we can estimate :math:`y` and compute the loss::

    Y_hat = geon.sigmoid(geon.dot(W, X))
    L = geon.cross_entropy_binary(Y_hat, Y)

To do gradient descent we will need the gradient, i.e. :math:`dL/dW`::

    grad = geon.deriv(L, W)

The ``geon.deriv`` function computes the backprop computation that computes the derivative using autodiff.

We are almost done.  The update is the gradient descent operation::

    update = geon.assign(W, W - alpha * grad / geon.tensor_size(Y_hat))

Now we can make a transformer and define a computation::

    transformer = geon.NumPyTransformer()
    update_fun = transformer.computation([L, W, update], alpha, X, Y)

Here, the computation returns three values, although the update's value is ``None``.  We just need to make sure that
it happens.  We pass in values for the training rate and samples.

Finally, we run it::

    for i in range(10):
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, loss %s" % (w_val, loss_val))

After each update, we return the loss and the new weights.

Logistic Regression with Axes
=============================

The complete program source can be found in :download:`../../examples/walk_through/logres_axes.py`.

When implementing front ends, the length of tensor axes, or even their dimensions, may not be known until later.
|Geon| provides a facility called axes for making it easier to work with tensors at a more abstract level.  We begin
by converting the logistic regression example to using axes rather than specific lengths::

    import numpy as np
    import geon

    C = geon.Axis("C")
    N = geon.Axis("N")

    X = geon.placeholder(axes=geon.Axes([C, N]))
    Y = geon.placeholder(axes=geon.Axes([N]))
    alpha = geon.placeholder(axes=geon.Axes())

    W = geon.Variable(axes=geon.Axes([C]), initial_value=0)

    Y_hat = geon.sigmoid(geon.dot(W, X))
    L = geon.cross_entropy_binary(Y_hat, Y, out_axes=geon.Axes())

    grad = geon.deriv(L, W)

    update = geon.assign(W, W - alpha * grad / geon.tensor_size(Y_hat))

Rather than ``C`` and ``N`` being lengths, they are now ``Axis`` objects of unspecified length.  Here, an ``Axis``
is something like a variable for an axis length, but we will later see that an ``Axis`` is more like a type in
the opgraph.

When we are ready to use our model, we do need to specify lengths for the axes we are using::

    xs = np.array([[0.52, 1.12, 0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]]).T

    ys = np.array([1, 1, 0, 1])

    C.length, N.length = xs.shape
    transformer = geon.NumPyTransformer()
    update_fun = transformer.computation([L, W, update], alpha, X, Y)

    for i in range(20):
        loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, loss %s" % (w_val, loss_val))

Rather than setting ``C`` and ``N`` to the components of the shape of ``xs``, we the axis lengths.

Adding a Bias
=============

The complete program source can be found in :download:`../../examples/walk_through/logres_bias.py`.

We can add a bias to the model: :math:`\hat{y}=\sigma(Wx+b)`.  This changes the model to::

    W = geon.Variable(axes=geon.Axes([C]), initial_value=0)
    b = geon.Variable(axes=geon.Axes(), initial_value=0)

    Y_hat = geon.sigmoid(geon.dot(W, X) + b)

Now we have two variables to update, ``W`` and ``b``.  However, all the updates are essentially the same, and
we know that everything to be udated is a variable.  We can use the ``variables`` method to find all the
trainable variables used in an ``Op``'s computation::

    updates = [geon.assign(v, v - alpha * geon.deriv(L, v) / geon.tensor_size(Y_hat))
               for v in L.variables()]

    all_updates = geon.doall(updates)

The function ``geon.doall`` is a short-hand for ensuring that all the updates get run.  We can change the computation
and printing of results to::

    update_fun = transformer.computation([L, W, b, all_updates], alpha, X, Y)

    for i in range(20):
        loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("W: %s, b: %s, loss %s" % (w_val, b_val, loss_val))

Multi-dimensional Logistic Regression
=====================================