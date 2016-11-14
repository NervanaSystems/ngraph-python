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

Let's begin with a very simple example: computing ``x+1`` for several values of ``x`` using the ``ngraph``
API.  We should think of the computation as being invoked from the *host*, but possibly taking place
somewhere else, which we will refer to as *the device.*

The nervana graph currently uses a compilation model. Users first define the computations, then they are compiled and run. In the future, we plan an even more compiler-like approach, where an executable is produced can later be run on various platforms, in addition to an interactive version.

Our first program will provide values for ``x`` and receive ``x+1`` for each ``x`` provided.

The x + 1 program
=================

The source code can be found in :download:`../../examples/walk_through/wt_1_x_plus_one.py`.

The complete program is

.. code-block:: python

    from __future__ import print_function
    import ngraph as ng
    import ngraph.transformers as ngt

    # Build the graph
    x = ng.placeholder(())
    x_plus_one = x + 1

    # Select a transformer
    transformer = ngt.make_transformer()

    # Define a computation
    plus_one = transformer.computation(x_plus_one, x)

    # Run the computation
    for i in range(5):
        print(plus_one(i))


We begin by importing ``ngraph``, the Python module for graph construction, and ``ngraph.transformers``, the module for transformer operations.

.. code-block:: python

    import ngraph as ng
    import ngraph.transformers as ngt

Next, we create an operational graph (op-graph) for the computation.  Following |TF| terminology, we use ``placeholder`` to define a port for transferring tensors between the host and the device. ``Axes`` are used to tell the graph the tensor shape. In this example, ``x`` is a scalar so the axes are empty.

The ``ngraph`` graph construction API uses functions to build a graph of ``Op`` objects. Each function may add operations to the graph, and will return an ``Op`` that represents the computation. Here, the ``Op`` returned is a ``TensorOp``, which defines the Python "magic methods" for arithmetic (for example, ``__add__()``). We could less concisely have written

.. code-block:: python

    x_plus_one = ng.add(x, 1)

Another bit of behind the scenes magic occurs with the Python ``1``, which is not an ``Op``. When an argument to a graph constructor is not an ``Op``, nervana graph will attempt to convert it to an ``Op`` using ``ng.constant``, the graph function for creating a constant. Thus, what is really happening is:

.. code-block:: python

    x_plus_one = ng.add(x, ng.constant(1))

Once the op-graph is defined, we can compile it with a *transformer*.  Here we use ``make_transformer`` to make a default transformer.  We tell the transformer the function to compute, ``x_plus_one``, and the associated parameter ``x``. The current default transformer uses NumPy for execution.

.. code-block:: python

    # Select a transformer
    transformer = ngt.make_transformer()

    # Define a computation
    plus_one = transformer.computation(x_plus_one, x)

The first time the transformer executes a computation, the graph is analyzed and compiled, and storage is allocated and initialized on the device. Once compiled, the computations are callable Python objects.

On each call to ``x_plus_one`` the value of ``x`` is copied to the device, 1 is added, and then the result is copied
back from the device.

The Compiled x + 1 Program
--------------------------

The compiled code can be examined (currently located in ``/tmp`` folder) to view the runtime device model. Here we show the code with some clarifying comments.

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


Tensors have two components: storage for their elements (using the convention ``a_`` for the allocated storage of a tensor) and views of that storage (denoted as ``a_...v_``).  The ``alloc_`` methods allocate
storage and then create the views of the storage that will be needed.  The view creation is separated from the
allocation because storage may be allocated in multiple ways.  The ``allocate`` method calls each
allocator, and each allocator creates the needed views.  The NumPy transformer's ``allocate`` method calls the
``allocate`` method.

Each allocated storage can also be initialized to, for example, random Gaussian variables. In this example, there are no initializations, so the method ``init``, which performs the one-time device
initialization, is empty.  Constants, such as 1, are copied to the device as part of the allocation process.

The method ``Computation_0`` handles the ``plus_one`` computation.  Clearly this is not the optimal way to add 1 to a scalar,
so let's look at a more complex example next.

Logistic Regression
===================

This example performs logistic regression. We want to classify an observation :math:`x` into one of two classes, denoted by :math:`y=0` and :math:`y=1`. Using a simple linear model :math:`\hat{y}=\sigma(Wx)`, we want to find the optimal values for :math:`W`. Here, we use gradient descent with a learning rate of :math:`\alpha` and the cross-entropy as the error function.

The complete program source can be found in :download:`../../examples/walk_through/wt_2_logres.py`.

We first define the axes for our tensors. In the nervana graph, `Axes` are similar to tensor shapes, except with additional semantics added. The function ``ng.make_axis`` will create an ``Axis`` object with an optionally supplied `name` argument. For example

.. code-block:: python

   my_axis = ng.make_axis(length=256, name='my_axis')

Here, we use a ``NameScope`` to set the names of the various axes. A ``NameScope`` is an object that sets the name of an object to that of its assigned attribute. So when we set ``ax.N`` to an ``Axis`` object, the ``name`` of the object is automatically set to ``ax.N``.

.. code-block:: python

    ax = ng.make_name_scope("ax")
    ax.N = ng.make_axis(length=128)
    ax.C = ng.make_axis(length=4)

The input data is synthetically generated as a mixture of two Gaussian distributions in 4-d space.  Our dataset consists of 10
mini-batches of 128 samples each:

.. code-block:: python

    g = gendata.MixtureGenerator([.5, .5], (ax.C.length,))
    XS, YS = g.gen_data(ax.N.length, 10)


Our model has three placeholders, ``X``, ``Y``, and ``alpha``, each of which need to have axes defined. ``alpha`` is a scalar, so we pass in empty axes:

.. code-block:: python

    alpha = ng.placeholder(())

``X`` and ``Y`` are tensors for the input and output data, respectively. Our convention is to use the last axis for samples.  The placeholders can be specified as:

.. code-block:: python

    X = ng.placeholder([ax.C, ax.N])
    Y = ng.placeholder([ax.N])

We also need to specify the training weights, ``W``.  Unlike a placeholder, ``W`` should retain its value from computation
to computation (for example, across mini-batches of training).  Following |TF|, we call this a *variable*.  We specify the variable with both ``Axes`` and also an initial value:

.. code-block:: python

    W = ng.variable([ax.C - 1], initial_value=0)

Now we can estimate :math:`y` and compute the average loss:

.. code-block:: python

    Y_hat = ng.sigmoid(ng.dot(W, X))
    L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

Here we use several ngraph functions, including ``ng.dot`` and ``ng.sigmoid``. Since a tensor can have multiple axes, we need a way to mark which axes in the first argument of ``ng.dot`` are to act on which axes in the second argument.

Every axis is a member of a family of axes we call duals of the axis, and each axis in the family has a position. When you create an axis, its dual position is 0. ``dot`` pairs axes in the first and second arguments that are of the same dual family and have consecutive positions.

We want the variable `W` to act on the `ax.C` axis, so we want the axis for `W` to be in the position before `ax.C`, which we can obtain with `ax.C - 1`. We initialize ``W`` to ``0``.

NOTE: The ``dot`` operation previously matched axes by identity, which was problematic for RNNs.

Gradient descent requires computing the gradient, :math:`dL/dW`:

.. code-block:: python

    grad = ng.deriv(L, W)

The ``ng.deriv`` function computes the backprop computation using autodiff.

We are almost done.  The update is the gradient descent operation:

.. code-block:: python

    update = ng.assign(W, W - alpha * grad / ng.tensor_size(Y_hat))

Now we create a transformer and define a computation. We pass the ops from which we want to retrieve the results for, followed by the placeholders:

.. code-block:: python

    ngt.make_transformer()

    transformer = ngt.make_transformer()
    update_fun = transformer.computation([L, W, update], alpha, X, Y)

Here, the computation will return three values for the ``L``, ``W``, and ``update``, given inputs to fill the placeholders.

Finally, we train the model across 10 epochs:

.. code-block:: python

    for i in range(10):
        for xs, ys in zip(XS, YS):
            loss_val, w_val, _ = update_fun(5.0 / (1 + i), xs, ys)
            print("W: %s, loss %s" % (w_val, loss_val))

After each update, we return the loss and the new weights.

Adding a second computation for Evaluation
==========================================

The complete program source can be found in :download:`../../examples/walk_through/wt_3_logres_eval.py`.

If we want to evaluate our model, we can also generate some evaluation data:

.. code-block:: python

    EVAL_XS, EVAL_YS = g.gen_data(ax.N.length, 4)

We need to add a second computation, which just computes the average batch loss, with no update:

.. code-block:: python

    eval_fun = transformer.computation(L, X, Y)

Finally, we use this computation to evaluate the model's performance on the test set during the course of training:

.. code-block:: python

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

Which demonstrates reasonable learning behavior:

.. code-block:: python

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

Adding a Bias
=============

The complete program source can be found in :download:`../../examples/walk_through/wt_4_logres_bias.py`.

We can add a bias :math:`b` to the model: :math:`\hat{y}=\sigma(Wx+b)`.  This changes the model to:

.. code-block:: python

    W = ng.variable([ax.C - 1], initial_value=0)
    b = ng.variable((), initial_value=0)

    Y_hat = ng.sigmoid(ng.dot(W, X) + b)

Now we have two variables to update, ``W`` and ``b``.  However, all the updates are essentially the same, and
we know that everything to be updated is a variable.  We can use the ``variables()`` method to find all the
trainable variables used in an ``Op``'s computation:

.. code-block:: python

    updates = [ng.assign(v, v - alpha * ng.deriv(L, v))
               for v in L.variables()]

    all_updates = ng.doall(updates)

The function ``ng.doall`` is a short-hand for ensuring that all the updates get run.  We can change the computation
and printing of results to:

.. code-block:: python

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

The complete program source can be found in :download:`../../examples/walk_through/wt_5_logres_multi.py`.

We are switching from a flat :math:`C`-dimensional feature space to an :math:`W\times H` feature space.  The
weights are now also a :math:`W\times H` tensor:

.. code-block:: python

    alpha = ng.placeholder(())
    X = ng.placeholder([ax.W, ax.H, ax.N])
    Y = ng.placeholder([ax.N])

    W = ng.variable([ax.W - 1, ax.H - 1], initial_value=0)
    b = ng.variable((), initial_value=0)

The calculation remains:

.. code-block:: python

    Y_hat = ng.sigmoid(ng.dot(W, X) + b)

The two dual axes of ``W`` will match the corresponding axes of ``X`` in the dot product.

Note: Some bugs in ngraph.dot and its derivative were discovered while making this example.  They are not fixed yet.
