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

TensorFlow
==========

In ngraph, we aim to provide utilities that enables frontend interoperability
with other frameworks such as TensorFlow. The TensorFlow importer allows user
to define models in TensorFlow and then execute computations in ngrah.

Minimal Example
---------------
Here's a minimal example for the TensorFlow importer.

::

    from __future__ import print_function
    from ngraph.frontends.TensorFlow.tf_importer.importer import TFImporter
    import ngraph.transformers as ngt
    import TensorFlow as tf
    import ngraph as ng

    # TensorFlow ops
    x = tf.constant(1.)
    y = tf.constant(2.)
    f = x + y

    # import
    importer = TFImporter()
    importer.parse_graph_def(tf.Session().graph_def)

    # get handle
    f_ng = importer.get_op_handle(f)

    # execute
    transformer = ngt.make_transformer()
    f_result = transformer.computation(f_ng)()
    print(f_result)


Walk-through of Mnist MLP Example
---------------------------------
Here's a walk-through of the Mnist MLP example. For full source code of the
example, please head on to the ``examples`` directory under importer.

1. Define Mnist MLP model in TensorFlow
::

    x = tf.placeholder(tf.float32, [args.batch_size, 784])
    t = tf.placeholder(tf.float32, [args.batch_size, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b
    cost = tf.reduce_mean(-tf.reduce_sum(
        t * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    init = tf.initialize_all_variables()

In the example, we need to explicitly set ``init`` to
``tf.initialize_all_variables()`` since we need to use the handle of the
``init`` op for ngrpah to execute the correct initialization.

2. Import TensorFlow ``GraphDef``
::

    importer = TFImporter()
    importer.parse_graph_def(tf.Session().graph_def)

- We use the ``TFImporter.parse_graph_def()`` function to import from
  TensorFlow sessions's ``graph_def``.
- The importer also support importeing from a ``graph_def`` protobuf file
  using ``TFImporter.parse_protobuf()``. For example, a ``graph_def`` file can
  be dump by ``tf.train.SummaryWriter()``.

3. Get handles of corresponding ngraph ops
::

    x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle([x, t, cost, init])

TensorFlow nodes are converted to an ngraph ops. In order to evaluate a
TensorFlow node, we need to get its corresponding ngraph node using
``TFImporter.get_op_handle()``.

4. Perform autodiff and define computations
::

    updates = SGDOptimizer(args.lrate).minimize(cost_ng)
    transformer = ngt.make_transformer()
    train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
    init_comp = transformer.computation(init_op_ng)
    transformer.initialize()

As we only import the forward graph from TensorFlow, we should use ngraph's
autodiff to compute gradients and get optimizers.

5. Traininig using ngraph
::

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    init_comp()
    for idx in range(args.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
        cost_val, _ = train_comp(batch_xs, batch_ys)
        print("[Iter %s] Cost = %s" % (idx, cost_val))

Now we can train the model in ngraph as if it were a native ngraph model. All
ngrpah functionalities and syntax can be applied after the graph is imported.

6. Training using TensorFlow as comparison
::

    with tf.Session() as sess:
        # train in tensorflow
        train_step = tf.train.GradientDescentOptimizer(args.lrate).minimize(cost)
        sess.run(init)

        mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        for idx in range(args.max_iter):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            cost_val, _ = sess.run([cost, train_step],
                                   feed_dict={x: batch_xs, t: batch_ys})
            print("[Iter %s] Cost = %s" % (idx, cost_val))

Finally, we train the model using standard TensorFlow. The ngraph results above
shall all match TensorFlow's result.


Current Limitations
-------------------

1. Only a subset of operations are supported.

  - Currently we only support a subset of operations from TensorFlow that are
    related to neural networks. We are working on getting more ops supported in
    the importer.
  - A util function ``TFImporter._get_unimplemented_ops()`` is provided for
    getting a list of unimplemented ops from a particular model.

2. The importer shall be used to imports forward graph.

  - User shall use the importer to import forward pass of the TensorFlow graph,
    and then perform autodiff and training updates in ngraph.
  - TensorFlow ops related to gradient computation is not supported.
  - In the future, two-way weights exchange between TensorFlow and ngraph will
    also be supported.

3. Static-ness

  - In ngraph, transformer may alter the computation graph at during
    transformation phase, thus we need to declare all computations before
    executing any one of them. Alternating the imported graph after transformer
    initialization is not supported.
  - TensorFlow allows dynamic parameters to its ops. For example, the kernel
    size of a ``Conv2d`` of can be results from another computation. Since
    ngraph needs to know dimension information prior to execution to allocate
    memory, dynamic parameters is not supported in importer.
