'''
Create a TensorFlow graph with variables.
Test exporting and importing meta graph and checkpoints.

To run this script, you need to install TensorFlow and activate it if installed virtually.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training.saver import read_meta_graph_file

def create_variable_graph(graph_pbtxt, graph_pb, checkpoint):
  '''
    Create a sample graph with two variables.
    Save the graph in metagraph, checkpoint and
  '''

  x = tf.placeholder(tf.float32, shape=(1, 3))

  weight = tf.Variable(tf.random_normal([3, 5], stddev=0.35), name="weights")
  biases = tf.Variable(tf.ones([1]), name='biases')

  result = tf.matmul(x, weight) + biases

  init_op = tf.initialize_all_variables()

  saver = tf.train.Saver([biases, weight])

  with tf.Session() as sess:
    sess.run(init_op)
    sess.run(result, feed_dict={x: [[1.0, 2.0, 3.0]]})

    # Saver saves variables into a checkpoint file.
    # In addition, the save function implicitly calls tf.export_meta_graph(), which generates ckpt.meta file.
    save_path = saver.save(sess, checkpoint)
    print("Variables saved in file: %s" % checkpoint)

    # Save the computation graph only
    tf.train.write_graph(sess.graph_def, "./", graph_pbtxt, True)  # The graph is written as a text proto
    tf.train.write_graph(sess.graph_def, "./", graph_pb, False)  # The graph is written as a binary proto
    print("GraphDef saved in file: %s" % graph_pb)

def restore_graph_pb(graph_pb):
  '''
    Restore from the graph protobuf file and the checkpoint file.
    This needs the original graph construction steps.
  '''

  biases = tf.Variable(tf.zeros([200]), name='biases')
  weight = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
  saver = tf.train.Saver([biases, weight])

  with tf.Session() as sess:
    # Restore the computation graph
    # the computation graph can also be restored from ckpt.meta file
    print("loading graph")
    graph_def = tf.GraphDef()
    with open(graph_pb, 'rb') as f:
      graph_def.ParseFromString(f.read())  # read serialized binary file only
      tf.import_graph_def(graph_def, name='')

    # Restore variable value
    ckpt = tf.train.get_checkpoint_state("./")
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("variable restored.")

def restore_meta_graph(meta_graph):
  '''
    Restore from the metagraph (.meta) file and the checkpoint file.
    No need for building graph from scratch.
  '''

  with tf.Session() as sess:
    meta_graph_def = read_meta_graph_file(meta_graph)
    saver = tf.train.import_meta_graph(meta_graph_def)
    print(meta_graph_def.graph_def)

    ckpt = tf.train.get_checkpoint_state("./")
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

      for v in tf.all_variables():
        print(v.name)
        print(v.op)
        shape = v.get_shape()
        print(len(v.get_shape()))
        for s in shape: print(s)
        print(v.value)
        tensor_value = v.eval()
        print(tensor_value)

def main(_):
  graph_pbtxt = "variable_graph.pb.txt"
  graph_pb = "variable_graph.pb"
  checkpoint = "model.ckpt"
  meta_graph = "model.ckpt.meta"

  create_variable_graph(graph_pbtxt, graph_pb, checkpoint)
  restore_meta_graph(meta_graph)

if __name__ == '__main__':
  tf.app.run()
