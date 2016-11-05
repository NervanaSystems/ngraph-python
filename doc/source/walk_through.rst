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

Let's begin with a very simple example: computing `x+1` for several values of `x` using the `ngraph`
API.  We should think of the computation as being invoked from the *host*, but possibly taking place
somewhere else, which we will refer to as *the device.*

The nervana graph currently uses a compilation model. You define computations you want to run, compile them, and
then run them. In the future, we plan to make this even more compiler-like, where you produce something like an
executable that can later be run on various platforms, as well as provide a more interactive version.

Our first program will provide values for :math:`x` and receive :math:`x+1` for each :math:`x` provided.

The x + 1 program
=================

The source code can be found in :download:`../../examples/walk_through/wt_1_x_plus_one.py`.

The complete program is

.. code-block:: python

    from __future__ import print_function
    import ngraph as ng
    import ngraph.transformers as ngt

    # Build the graph
    x = ng.placeholder(axes=ng.make_axes())
    x_plus_one = x + 1

    # Select a transformer
    transformer = ngt.make_transformer()

    # Define a computation
    plus_one = transformer.computation(x_plus_one, x)

    # Run the computation
    for i in range(5):
        print(plus_one(i))


We begin by importing ``ngraph``, the Python module for frontend graph construction, and ``ngraph.transformers``, the module for frontend transformer operations.

Next we create an operational graph (op-graph) for the computation.  Following |TF| terminology, we use ``placeholder`` to define a port for transferring tensors between the host and the device. We use `Axes` to tell nervana graph about these tensors. Axes are like tensor shapes, with some semantics added. In this example, `x` is a scalarm so the axes are empty.

The `ngraph` graph construction API uses functions to build a graph of ``Op`` objects. Each function may add one or more operations to the graph, and will return an object that represents the computation. Once the computation has been instantiated, the object can also be used as a handle to the tensor when that part of the computation is associated with persistent storage. In this case, an ``AssignableTensorOp`` is returned, which represents a tensor associated with storage.

An ``AssignableTensorOp`` is a kind of ``TensorOp``, which is a kind of ``Op``. The ``TensorOp`` defines the Python "magic methods" for arithmetic, so we can use a ``TensorOp`` in an arithmetic expression, the result is an ``Op`` for the result of that operation. We could less concisely have written
.. code-block:: python

    x_plus_one = ng.add(x, 1)

which may be more convenient when implementing a frontend.

Another bit of behind the scenes magic occurs with the Python ``1``, which is not an ``Op``. When arguments to graph constructors are not ``Op``s, nervana graph will attempt to convert them to an ``Op`` with ``ng.constant``, the graph function for making a constant. Thus, what is really happening is
.. code-block:: python

    x_plus_one = ng.add(x, ng.constant(1))

Once the op-graph is set up, we can compile it with a *transformer*.  Here we use ``make_transformer`` to make a defuault transformer.  We tell the transformer the function to compute (``x_plus_one``) and the associated parameter (``x``).

The first time the transformer executes a computation, the graph is analyzed and compiled, and the storage is allocated and initialized on the device.  These steps can be performed manually, for example if some device state is to be restored from
previously saved state.  Once compiled, the computations are callable Python objects.

On each call to ``plus_one`` the value of ``x`` is copied to the device, 1 is added, and then the result is copied
back from the device.

The Compiled x + 1 Program
--------------------------

We can examine the compiled code (currently located in ``/tmp`` folder) to view the runtime device model. Here we show the code with some clarifying comments.

.. code-block:: python

    class Model(object):
        def __init__(self):
            self.a_AssignableTensorOp_0_0 = None
            self.a_AssignableTensorOp_0_0_v_AssignableTensorOp_0_0_ = None
            self.a_AssignableTensorOp_1_0 = None
            self.a_AssignableTensorOp_1_0_v_AssignableTensorOp_1_0_ = None
            self.a_AddZeroDim_0_0 = None
            self.a_AddZeroDim_0_0_v_AddZeroDim_0_0_ = None
            self.be = NervanaObject.be

        def alloc_a_AssignableTensorOp_0_0(self):
            self.update_a_AssignableTensorOp_0_0(np.empty(1, dtype=np.dtype('float32')))

        def update_a_AssignableTensorOp_0_0(self, buffer):
            self.a_AssignableTensorOp_0_0 = buffer
            self.a_AssignableTensorOp_0_0_v_AssignableTensorOp_0_0_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def alloc_a_AssignableTensorOp_1_0(self):
            self.update_a_AssignableTensorOp_1_0(np.empty(1, dtype=np.dtype('float32')))

        def update_a_AssignableTensorOp_1_0(self, buffer):
            self.a_AssignableTensorOp_1_0 = buffer
            self.a_AssignableTensorOp_1_0_v_AssignableTensorOp_1_0_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def alloc_a_AddZeroDim_0_0(self):
            self.update_a_AddZeroDim_0_0(np.empty(1, dtype=np.dtype('float32')))

        def update_a_AddZeroDim_0_0(self, buffer):
            self.a_AddZeroDim_0_0 = buffer
            self.a_AddZeroDim_0_0_v_AddZeroDim_0_0_ = np.ndarray(
                shape=(),
                dtype=np.float32,
                buffer=buffer,
                offset=0,
                strides=())

        def allocate(self):
            self.alloc_a_AssignableTensorOp_0_0()
            self.alloc_a_AssignableTensorOp_1_0()
            self.alloc_a_AddZeroDim_0_0()


        def Computation_0(self):
            np.add(self.a_AssignableTensorOp_0_0_v_AssignableTensorOp_0_0_, self.a_AssignableTensorOp_1_0_v_AssignableTensorOp_1_0_, out=self.a_AddZeroDim_0_0_v_AddZeroDim_0_0_)

        def init(self):
            pass


Tensors have two components, storage for their elements (using the convention ``a_`` for the allocated storage of a tensor) and views of that storage (denoted as ``a_...v_``).  The ``alloc_`` methods allocate
storage and then create the views of the storage that will be needed.  The view creation is separated from the
allocation because storage may be allocated in multiple ways.  The ``allocate`` method calls each
allocator, and each allocator creates the needed views.  The NumPy transformer's ``allocate`` method calls the
``allocate`` method.

Each allocated storage can also be initialized to for example, random gaussian variables. In this example, there are no initializations, so the method ``init`` which performs  the one-time device
initialization is empty.  Constants, such as 1, are copied to the device as part of the allocation process.

The method ``Computation_0`` handles the ``plus_one`` computation.  Clearly this is not the optimal way to add 1 to a scalar,
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
