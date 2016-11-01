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

Let's begin with a very simple example: computing :math:`x+1` for several values of :math:`x` using the front end
API.  We should think of the computation as being invoked from our program, *the CPU*, but possibly taking place
somewhere else, which we will refer to as *the device.*

Our program will provide values for :math:`x` and receive :math:`x+1` for each :math:`x` provided.

The x + 1 program
=================

The source code can be found in :download:`../../examples/walk_through/x_plus_one.py`.

The complete program is

.. code-block:: python

    import ngraph as ng

    x = ng.placeholder(axes=ng.Axes())
    x_plus_one = x + 1

    transformer = ng.NumPyTransformer()
    plus_one = transformer.computation(x_plus_one, x)

    for i in range(5):
        print(plus_one(i))


We begin by importing ``ngraph``, the Python module for the front end API.

Next we create an operational graph (op-graph) for the computation.  Following |TF| terminology, we call the
parameter that receives the value of :math:`x` a ``placeholder``.  A placeholder has a tensor value, so we need
to indicate the tensor shape by specifying its axes.  In this simple example, :math:`x` is a scalar,
so the axes are empty.  We follow this with the computation that adds 1 to the ``placeholder.``  Even though
this looks like we are adding 1, the op-graph objects overload the arithmetic method, so ``x_plus_one`` is really
an op-graph object.

Once the op-graph is set up, we can compile it with a *transformer*.  Here the transformer uses NumPy and runs on the CPU, but
the procedure would be the same for any other transformer.  We tell the transformer the function to compute (``x_plus_one``) and the associated parameter (``x``).

The first time the transformer executes a computation, the op-graph is analyzed and compiled, and the storage is allocated and initialized on the device.  These steps can be performed manually, for example if some device state is to be restored from
previously saved state.  Once compiled, the computations are callable Python objects.

On each call to ``plus_one`` the value of ``x`` is copied to the device, 1 is added, and then the result is copied
back from the device.

The Compiled x + 1 Program
--------------------------

We can examine the compiled code (currently located in ``/tmp`` folder) to view the runtime device model. Here we show the code with some clarifying comments.

.. code-block:: python

    class Model(object):
        def __init__(self):
            self.a_t7 = None  # allocated storage for tensor t7
            self.v_t7_ = None  # a view of tensor t7
            self.a_t8 = None
            self.v_t8_ = None
            self.a_t6 = None
            self.v_t6_ = None

        # allocate and create the views on tensor t7
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

        # allocate and create the views on tensor t8
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

        # allocate and create the views on tensor t6
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

        # allocate all tensors
        def allocate(self):
            self.alloc_a_t7()
            self.alloc_a_t8()
            self.alloc_a_t6()

        # perform the addition computation
        def c_0(self):
            np.add(self.v_t6_, self.v_t7_, out=self.v_t8_)

        # tensor initialization (not used in this example)
        def c_1(self):
            pass


Tensors have two components, storage for their elements (using the convention ``a_t7`` for the allocated storage of the tensor ``t7``) and views of that storage (denoted as ``v_t7_``).  The ``alloc_`` methods allocate
storage and then create the views of the storage that will be needed.  The view creation is separated from the
allocation because storage may be allocated in multiple ways.  The ``allocate`` method calls each
allocator, and each allocator creates the needed views.  The NumPy transformer's ``allocate`` method calls the
``allocate`` method.

Each allocated storage can also be initialized to for example, random gaussian variables. In this example, there are no initializations, so the method ``c_1`` which performs  the one-time device
initialization is empty.  Constants, such as 1, are copied to the device as part of the allocation process.

The method ``c_0`` handles the ``plus_one`` computation.  Clearly this is not the optimal way to add 1 to a scalar,
so let's look at a more complex example next.

Logistic Regression
===================

This example performs logistic regression. We want to classify an obervation :math:`x` into one of two classes, denoted by :math:`y=0` and :math:`y=1`. With a simple linear model :math:`\hat{y}=\sigma(Wx)`, we want to find the optimal values for :math:`W`. Here, we use gradient descent with a learning rate of :math:`\alpha` and the cross-entropy as the error function.

The complete program source can be found in :download:`../../examples/walk_through/logres.py`.

The data is synthetically generated as a mixture of two Gaussian distributions in 4-d space.  Our dataset consists of 10
mini-batches of 128 samples each::

    import gendata
    N = 128
    C = 4
    g = gendata.MixtureGenerator([.5, .5], C)
    XS, YS = g.gen_data(N, 10)

Our model has three placeholders, ``X``, ``Y``, and ``alpha``. Each placeholder needs to have axis specified.

``alpha`` is a scalar, so we pass in empty axes::

    alpha = ng.placeholder(axes=ng.Axes())

``X`` and ``Y`` have shape, which we provide to the placeholders. Our convention is to use the last axis for samples.  The placeholders can be specified as::

    X = ng.placeholder(axes=ng.Axes([C, N]))  # input data has 4 features for each datapoint
    Y = ng.placeholder(axes=ng.Axes([N]))

We also need to specify the training weights, ``W``.  Unlike placeholders, ``W`` should retain its value from computation
to computation (for example, across mini-batches of training).  Following |TF|, we call this a *Variable*.  We specify the variable with both an axes and also an initial value::

    W = ng.Variable(axes=ng.Axes([C]), initial_value=0)

Other than the axes, the syntax is the same as |TF|. The transformer's initialization function will initialize `W`
to 0 after allocating storage.

Now we can estimate :math:`y` and compute the average loss::

    Y_hat = ng.sigmoid(ng.dot(W, X))
    L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

Gradient descent requires computing the gradient, :math:`dL/dW`::

    grad = ng.deriv(L, W)

The ``ng.deriv`` function computes the backprop computation using autodiff.

We are almost done.  The update is the gradient descent operation::

    update = ng.assign(W, W - alpha * grad / ng.tensor_size(Y_hat))

Now we create a transformer and define a computation. We pass the ops from which we want to retrieve the results for, followed by the placeholders::

    transformer = ng.NumPyTransformer()
    update_fun = transformer.computation([L, W, update], alpha, X, Y)

Here, the computation will return three values for the ``L``, ``W``, and ``update``, given inputs to fill the placeholders.

Finally, we train the model across 10 epochs::

    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
            print("W: %s, loss %s" % (w_val, loss_val))

After each update, we return the loss and the new weights.

Adding a second computation for Evaluation
==========================================

The complete program source can be found in :download:`../../examples/walk_through/logres_eval.py`.

If we want to evaluate our model, we can also generate some evaluation data::

    EVAL_XS, EVAL_YS = g.gen_data(N, 4)

We need to add a second computation, which just computes the average batch loss, with no update::

    eval_fun = transformer.computation(L, X, Y)

Finally, we use this computation to evaluate the model's performance on the test set during the course of training::

    def avg_loss():
        total_loss = 0
        for xs, ys in zip(EVAL_XS, EVAL_YS):
            loss_val = eval_fun(xs, ys)
            total_loss += loss_val
        return total_loss/len(xs)

    print("Starting avg loss: {}".format(avg_loss()))
    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("After epoch %d: W: %s, avg loss %s" % (i, w_val, avg_loss()))

Which demonstrates reasonable learning behavior::

    Starting avg loss: 0.693147301674
    After epoch 0: W: [ 1.31084263  3.54553676  0.83918822  0.47578019], avg loss 0.210895072669
    After epoch 1: W: [ 1.61401999  4.14274025  0.80241382  0.70635045], avg loss 0.188071470708
    After epoch 2: W: [ 1.78632712  4.44820547  0.75676179  0.8398425 ], avg loss 0.17810350284
    After epoch 3: W: [ 1.90496778  4.6451354   0.71686864  0.93109995], avg loss 0.17216200754
    After epoch 4: W: [ 1.9946698   4.78712606  0.68277711  0.99926096], avg loss 0.168086335063
    After epoch 5: W: [ 2.06639457  4.89654636  0.65336621  1.05305564], avg loss 0.165055405349
    After epoch 6: W: [ 2.12592459  4.98467636  0.62764722  1.09714031], avg loss 0.162679858506
    After epoch 7: W: [ 2.17666841  5.05792665  0.60486782  1.1342684 ], avg loss 0.160748042166
    After epoch 8: W: [ 2.2207973   5.12026215  0.58446652  1.1661936 ], avg loss 0.15913354978
    After epoch 9: W: [ 2.25977612  5.17428732  0.56602061  1.19409537], avg loss 0.157755594701

Logistic Regression with Axes
=============================

The complete program source can be found in :download:`../../examples/walk_through/logres_axes.py`.

When implementing front ends, the length of tensor axes, or even their dimensions, may not be known until later.
|Geon| provides a facility called axes for making it easier to work with tensors at a more abstract level.  We begin
by converting the logistic regression example to using axes rather than specific lengths::

    import numpy as np
    import ngraph as ng
    import gendata

    C = ng.Axis("C")
    N = ng.Axis("N")

    X = ng.placeholder(axes=ng.Axes([C, N]))
    Y = ng.placeholder(axes=ng.Axes([N]))
    alpha = ng.placeholder(axes=ng.Axes())

    W = ng.Variable(axes=ng.Axes([C]), initial_value=0)

    Y_hat = ng.sigmoid(ng.dot(W, X))
    L = ng.cross_entropy_binary(Y_hat, Y, out_axes=ng.Axes())

    grad = ng.deriv(L, W)

    update = ng.assign(W, W - alpha * grad)

Rather than ``C`` and ``N`` holding integers, they are now ``Axis`` objects of unspecified length.  Here, an ``Axis``
is something like a variable for an axis length, but we will later see that an ``Axis`` is more like a type in
the op-graph.

When we are ready to use our model, we specify the lengths for the axes we are using::

    C.length = 4
    N.length = 128

    g = gendata.MixtureGenerator([.5, .5], C.length)
    XS, YS = g.gen_data(N.length, 10)
    EVAL_XS, EVAL_YS = g.gen_data(N.length, 4)

    transformer = ng.NumPyTransformer()
    update_fun = transformer.computation([L, W, update], alpha, X, Y)
    eval_fun = transformer.computation(L, X, Y)

    def avg_loss():
        total_loss = 0
        for xs, ys in zip(EVAL_XS, EVAL_YS):
            loss_val = eval_fun(xs, ys)
            total_loss += loss_val
        return total_loss/len(xs)

    print("Starting avg loss: {}".format(avg_loss()))
    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("After epoch %d: W: %s, avg loss %s" % (i, w_val, avg_loss()))

Rather than setting ``C`` and ``N`` to the components of the shape of ``xs``, we use the axis lengths.

Adding a Bias
=============

The complete program source can be found in :download:`../../examples/walk_through/logres_bias.py`.

We can add a bias :math:`b` to the model: :math:`\hat{y}=\sigma(Wx+b)`.  This changes the model to::

    W = ng.Variable(axes=ng.Axes([C]), initial_value=0)
    b = ng.Variable(axes=ng.Axes(), initial_value=0)

    Y_hat = ng.sigmoid(ng.dot(W, X) + b)

Now we have two variables to update, ``W`` and ``b``.  However, all the updates are essentially the same, and
we know that everything to be updated is a variable.  We can use the ``variables`` method to find all the
trainable variables used in an ``Op``'s computation::

    updates = [ng.assign(v, v - alpha * ng.deriv(L, v))
               for v in L.variables()]

    all_updates = ng.doall(updates)

The function ``ng.doall`` is a short-hand for ensuring that all the updates get run.  We can change the computation
and printing of results to::

    update_fun = transformer.computation([L, W, b, all_updates], alpha, X, Y)
    eval_fun = transformer.computation(L, X, Y)

    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
            print("W: %s, b: %s, loss %s" % (w_val, b_val, loss_val))

    def avg_loss():
        total_loss = 0
        for xs, ys in zip(EVAL_XS, EVAL_YS):
            loss_val = eval_fun(xs, ys)
            total_loss += loss_val
        return total_loss/len(xs)

    print("Starting avg loss: {}".format(avg_loss()))
    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, b_val, _ = update_fun(5.0 / (1 + i), xs, ys)
        print("After epoch %d: W: %s, b: %s, avg loss %s" % (i, w_val, b_val, avg_loss()))

Multi-dimensional Logistic Regression
=====================================

The complete program source can be found in :download:`../../examples/walk_through/logres_multi.py`.

In this example, we begin by introducing a class, ``NameScope``, than can be useful for naming values::

    ax = ng.NameScope(name="ax")

    ax.W = ng.Axis()
    ax.H = ng.Axis()
    ax.N = ng.Axis()

Many |ngraph| objects are ``NameableValue``s, which means they have a ``name`` attribute.  When a ``NameableValue``
is assigned to a ``NameScope``'s attribute, the name of the ``NameableValue`` will be set.  Here, we give
``ax`` the name ``ax``.  Then the axis ``ax.W`` will have the name ``ax.W``.  Referring to the axes with
``ax.`` prefixes makes it easier to identify axes in programs, and keeps them from using up the desirable
short variable names.

Also notice the ``batch`` parameter to ``ax.N``.  This tells |ngraph| that ``ax.N`` is used as the axis for
samples within a batch.

We are switching from a flat :math:`C`-dimensional featurespace to an :math:`W\times H` feature space.  The
weights are now also a :math:`W\times H` tensor::

    X = ng.placeholder(axes=ng.Axes([ax.W, ax.H, ax.N]))
    Y = ng.placeholder(axes=ng.Axes([ax.N]))
    alpha = ng.placeholder(axes=ng.Axes())

    W = ng.Variable(axes=ng.Axes([ax.W, ax.H]), initial_value=0)
    b = ng.Variable(axes=ng.Axes(), initial_value=0)

The calculation remains::

    Y_hat = ng.sigmoid(ng.dot(W, X) + b)

What does it mean to ``dot`` tensors with axes?  The tensor ``dot`` operation has *reduction axes*, which
defaults to the intersection of the axes.  Both arguments
have their axes extended (broadcast) by axes they are missing in the reduction axes.  Then for each set of all indices
not in the reduction axes, the elements matching in the reduction axes are multiplied and all of these are summed
to form the result.  In this case, ``W`` has axes ``(ax.W, ax.H)`` and ``X`` has axes ``(ax.W, ax.H, ax.N)``.
The reduction axes are ``(ax.W, ax.H)``, so for each index in ``ax.N`` the matching pairs elements are multiplied
and summed, resulting in a tensor with one axes, ``ax.N``.

In general, when axes are missing in a computation, they will be automatically broadcast; axis identity indicates
which axes are missing.

The data generator is able to generate multi-dimensional data; it just reshapes::

    ax.W.length = 2
    ax.H.length = 2
    ax.N.length = 128

    g = gendata.MixtureGenerator([.5, .5], (ax.W.length, ax.H.length))
    XS, YS = g.gen_data(ax.N.length, 10)
    EVAL_XS, EVAL_YS = g.gen_data(ax.N.length, 4)

Note: Some bugs in ngraph.dot and its derivative were discovered while making this example.  They are not fixed yet.
