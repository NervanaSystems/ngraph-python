# Run TensorFlow Graph with Neon

## Supported ops after refactor
```python
Add
Assign
Const
Div
DummyOp
Identity
MatMul
Maximum
Mean
Mul
NoOp
Placeholder
Range
Rank
Relu
Sigmoid
Sum
Tanh
Variable
```

## Minimal Example

```python
import tensorflow as tf
import ngraph as ng
from tensorflow_import.importer import TensorFlowImporter

# build TensorFlow graph
a = tf.constant(10)
b = tf.constant(20)
c = a + b
d = c * a

# write to protobuf
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, "./", "my_graph.pb.txt", True)

# import from protobuf
importer = TensorFlowImporter("my_graph.pb.txt")

# run imported graph
transformer = ng.NumPyTransformer()
result_comp = transformer.computation([importer.last_op])
result_val = result_comp()[0]
print(result_val)  # prints 300
```

## Example Usage
1. Preparation.

    ```sh
    $ ./mnist_prepare.sh
    ```

  This will
    - Fetch TensorFlow fetch data and train for 2 epochs. Dump the training
      graph and model checkpoints.
    - Freeze the model checkpoints to protobuf using [this tool]. The
      `--output_node_names` option is the name of the last operation for
      inference, which is currently manually identified on TensorBoard.
2. Inference. Now we can import the TensorFlow-trained weights and perform
   inference using `ngraph`.  Notes that we need to manually identify the last op
   used in inference, which is `softmax_linear/add` for this example.

    ```sh
    $ python mnist_mlp_inference.py
    ```

3. Or, we can use `ngraph` to train and eval from `GraphDef` directly.

    ```sh
    $ python mnist_mlp_train.py
    ```

4. A tool to clean up the mnist data is also provided.

    ```sh
    $ ./mnist_clean.sh
    ```

[this tool]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

## Some Notes on TensorFlow
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

