# Use TF's Graph with Neon

## Setup

Tensorflow is currently installed as part of ununoctium:

```
cd ununoctium
make install
```

## Run TF's Inference Graph

### 1. Save TF's graph and weights

TensorFlow provides an [example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist) of training a simple MLP on the MNIST dataset. We made a new copy and modified the training script, adding code for graph/variable exportation.


#### 1.1 Save the computation graph

To save the GradDef as protobuf, add the following code at the end of the training.

```  
graph_pbtxt = "mnist_mlp_graph.pb.txt"  
graph_pb = "mnist_mlp_graph.pb"  

tf.train.write_graph(sess.graph_def, "./", graph_pbtxt, True) # text protobuf  
tf.train.write_graph(sess.graph_def, "./", graph_pb, False) # binary protobuf
```

The compuation graph is saved in .pb.txt file and its binary version is saved in the .pb file.

#### 1.2 Save the variables  

To save the variable values, add following code at the beggining of the training

```
saver = tf.train.Saver() 
```

and the following code at the end of training.

```  
checkpoint = "mnist_mlp_model.ckpt"  
saver.save(sess, checkpoint)
```

The variable values are saved in the **checkpoint** (.ckpt) file.

Note that TensorFlow also saves a [MetaGraph](https://www.tensorflow.org/versions/r0.9/how_tos/meta_graph/index.html) (.meta) file, which contans MetaInfoDef, GraphDef, SaverDef and CollectionDef.

#### 1.3 Starts training

To run the training script, first activate the virtual environment, for example:

```
$ source ~/code/private-neon/.venv/bin/activate
```

Then execute the stript `tf_benchmark/mnist/fully_connected_feed.py`

```
(tensorflow)$ cd tf_benchmark/mnist/
(tensorflow)$ python fully_connected_feed.py
```


### 2. Combine the graph and checkpoint

For inference only application, we need to load two files: the model's computation graph and its weights (saved in checkpionts). 
However, for some applications, loading one file is more convenient.
In addition, the model parameters (variables) would not change any more and can be converted to constant values. 

TensorFlow provides a convenient tool [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) to combine the computation graph and the checkpoint file into a single file, in which the variables are converted to constant values. An example of freezing a trained mnist graph is as follows:

```
python freeze_graph.py \
  --input_graph=mnist_mlp_graph.pb.txt \
  --input_checkpoint=mnist_mlp_model.ckpt \
  --output_graph=mnist_mlp_graph_froze.pb \
  --output_node_names=softmax_linear/add
```
Note that the `--output_node_names` option is the name of the last operation for inference, which is currently manually identified in the `mnist_mlp_graph.pb.txt` file.

We provide bash script inside each example folder for convenience. So simply `./freeze_graph.sh` would work.

### 3. Execute the frozen graph with Neon's graph backend

Activate the virtual environment, for example:

```
$ source ~/code/private-neon/.venv/bin/activate
```

Execute the following script will convert the frozen graph into Neon's graph and execute it.

```
(.venv2)$ cd ununoctium/tests/
(.venv2)$ python test_mnist_mlp.py
``` 

## Run TF's Training Graph
