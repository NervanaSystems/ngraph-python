# TensorFlow importer for ngraph

## Minimal example

```python
from __future__ import print_function
from tf_importer.tf_importer.importer import TFImporter
import tensorflow as tf
import ngraph as ng

# tensorflow ops
x = tf.constant(1.)
y = tf.constant(2.)
f = x + y

# import
importer = TFImporter()
importer.parse_graph_def(tf.Session().graph_def)

# get handle
f_ng = importer.get_op_handle(f)

# execute
f_result = ng.NumPyTransformer().computation(f_ng)()
print(f_result)
```

## Supported models

- MNIST MLP

## Some notes on TensorFlow
- TensorFlow is now automatically installed when `ngraph` installs. We may
  remove this dependency in the future.
- Save graph definition to protobuf

    ```python
    tf.train.write_graph(sess.graph_def, "./", "graph.pb.txt", True)  # text proto
    tf.train.write_graph(sess.graph_def, "./", "graph.pb", False)  # binary proto
    ```

- Save trainable parameters to checkpoints.

    ```python
    saver = tf.train.Saver()
    saver.save(sess, "model.ckpt")    # done periodically or at the end of training
    ```

  We made a copy of `3rd_party/mnist/fully_connected_feed.py` and modified it by
  adding code for saving `GraphDef` and trained parameters.
- Visualize using TensorBoard (default on <http://0.0.0.0:6006>).

    ```sh
    $ tensorboard --logdir=.
    ```

- The GraphDef contains several sub-graphs for different purpose:
    - Variable initialization: executed only once before training
    - Fprop and bprop: executed for each mini-batch optimization
    - Serialization & statistics report
- MetaGraph
    - The `save()` API also automatically exports a [MetaGraphDef] (.meta) file,
        which contains MetaInfoDef, GraphDef, SaverDef and CollectionDef.
    - As explained [here], the `MetaGraphDef` includes all information required to
        restore a training or inference process, including the GraphDef that
        describes the dataflow, and additional annotations that
        describe the variables, input pipelines, and other relevant information.

[MetaGraphDef]: https://www.tensorflow.org/versions/r0.9/how_tos/meta_graph/index.html/
[here]: http://stackoverflow.com/questions/36195454/what-is-the-tensorflow-checkpoint-meta-file#

