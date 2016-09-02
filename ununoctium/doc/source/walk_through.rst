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

Walk Through
************

Let's begin with a very simple example, computing :math:`x+1` for several values of :math:`x` using the front end
API.  We should think of the computation as being invoked from our program, *the CPU*, but possibly taking place
somewhere else, which we will refer to as *the device.*
Our program will provide values for :math:`x` and receive :math:`x+1` for each :math:`x` provided.

The x + 1 program
=================
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
