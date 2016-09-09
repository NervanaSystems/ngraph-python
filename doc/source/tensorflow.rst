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

TensorFlow Importer
===================

Example Usage
-------------

1. Preparation.
::

    $ ./mnist_prepare.sh

This will 1) Fetch TensorFlow fetch data and train for 2 epochs. Dump the
training graph and model checkpoints. 2) Freeze the model checkpoints to
protobuf using `this
tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py>`__.
The ``--output_node_names`` option is the name of the last operation for
inference, which is currently manually identified on TensorBoard.

2. Inference. Now we can import the TensorFlow-trained weights and perform
inference using ``ngraph``. Notes that we need to manually identify the last op
used in inference, which is ``softmax_linear/add`` for this example.
::

    $ python mnist_mlp_inference.py

3. Train. Or, we can use ``ngraph`` to train and eval from ``GraphDef`` directly.
::

    $ python mnist_mlp_train.py

4. A tool to clean up the mnist data is also provided.
::

    $ python mnist_clean.py

Notes on TensorFlow
------------------------

-  TensorFlow is now automatically installed when ``ngraph`` installs.
   We may remove this dependency in the future.
-  Save graph definition to protobuf
::

    tf.train.write_graph(sess.graph_def, "./", "graph.pb.txt", True)  # text proto
    tf.train.write_graph(sess.graph_def, "./", "graph.pb", False)  # binary proto

-  Save trainable parameters to checkpoints.
::

    saver = tf.train.Saver()
    saver.save(sess, "model.ckpt")  # done periodically or at the end of training

- We made a copy of ``3rd_party/mnist/fully_connected_feed.py`` and
modified it by adding code for saving ``GraphDef`` and trained
parameters.

- Visualize using TensorBoard (default on ``http://0.0.0.0:6006``).
::

    $ tensorboard --logdir=.

-  The GraphDef contains several sub-graphs for different purpose:

   -  Variable initialization: executed only once before training
   -  Fprop and bprop: executed for each mini-batch optimization
   -  Serialization & statistics report

-  MetaGraph

   -  The ``save()`` API also automatically exports a
      `MetaGraphDef <https://www.tensorflow.org/versions/r0.9/how_tos/meta_graph/index.html/>`__
      (.meta) file, which contains MetaInfoDef, GraphDef, SaverDef and
      CollectionDef.
   -  As explained
      `here <http://stackoverflow.com/questions/36195454/what-is-the-tensorflow-checkpoint-meta-file#>`__,
      the ``MetaGraphDef`` includes all information required to restore
      a training or inference process, including the GraphDef that
      describes the dataflow, and additional annotations that describe
      the variables, input pipelines, and other relevant information.
