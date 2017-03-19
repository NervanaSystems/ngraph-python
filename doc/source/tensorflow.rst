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

In ngraph, we aim to provide utilities that enable frontend interoperability
with other frameworks such as `TensorFlow <https://www.tensorflow.org/>`__.
The TensorFlow importer allows users to define a limited set models in
TensorFlow and then execute computations using ngraph transformers.


Minimal Example
---------------
Here's a minimal example for the TensorFlow importer.

::

    from __future__ import print_function
    from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
    import ngraph.transformers as ngt
    import tensorflow as tf
    import ngraph as ng

    # TensorFlow ops
    x = tf.constant(1.)
    y = tf.constant(2.)
    f = x + y

    # import
    importer = TFImporter()
    importer.import_graph_def(tf.Session().graph_def)

    # get handle
    f_ng = importer.get_op_handle(f)

    # execute
    transformer = ngt.make_transformer()
    f_result = transformer.computation(f_ng)()
    print(f_result)


Walk-through of MNIST MLP Example
---------------------------------
Here's a walk-through of the MNIST MLP example. For full source code of the
example, please see the
`examples <https://github.com/NervanaSystems/ngraph/tree/master/ngraph/frontends/tensorflow/examples/>`__
directory.

1. Define MNIST MLP model in TensorFlow
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
``init`` op for ngraph to execute the correct initialization.

2. Import TensorFlow ``GraphDef``
::

    importer = TFImporter()
    importer.import_graph_def(tf.Session().graph_def)

- We use the ``TFImporter.import_graph_def()`` function to import from
  TensorFlow sessions's ``graph_def``.
- The importer also support importing from a ``graph_def`` protobuf file
  using ``TFImporter.import_protobuf()``. For example, a ``graph_def`` file can
  be dumped by ``tf.train.SummaryWriter()``.

3. Get handles of corresponding ngraph ops
::

    x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle([x, t, cost, init])

TensorFlow nodes are converted to ngraph ops. In order to evaluate a
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

5. Training using ngraph
::

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    init_comp()
    for idx in range(args.max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
        cost_val, _ = train_comp(batch_xs, batch_ys)
        print("[Iter %s] Cost = %s" % (idx, cost_val))

Now we can train the model in ngraph as if it were a native ngraph model. All
ngraph functionalities and syntax can be applied after the graph is imported.

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
match TensorFlow's results.


Current Limitations
-------------------

1. Only a subset of operations are supported.

  - Currently we only support a subset of operations from TensorFlow that are
    related to neural networks. We are working on getting more ops supported in
    the importer.
  - A util function ``TFImporter._get_unimplemented_ops()`` is provided for
    getting a list of unimplemented ops from a particular model.

2. The importer should be used to import the forward graph.

  - User should use the importer to import the forward pass of the TensorFlow graph,
    and then perform autodiff and training updates in ngraph.
  - TensorFlow ops related to gradient computation are not supported.
  - In the future, bidirectional weight exchange between TensorFlow and ngraph will
    also be supported.

3. Static-ness

  - In ngraph, the transformer may alter the computation graph during the
    transformation phase, thus we need to declare all computations before
    executing any of them. Altering the imported graph after transformer
    initialization is not supported.
  - TensorFlow allows dynamic parameters to its ops. For example, the kernel
    size of a ``Conv2d`` can be the result of another computation. Since
    ngraph needs to know dimension information prior to execution to allocate
    memory, dynamic parameters are not supported in importer.
