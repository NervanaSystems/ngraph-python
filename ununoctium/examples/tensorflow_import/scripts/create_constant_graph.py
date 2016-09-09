# ----------------------------------------------------------------------------
# To run this script, you need to first install TensorFlow.
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_tf_graph():
    a = tf.constant(10)
    b = tf.constant(32)
    c = a + b
    d = c * a

    sess = tf.Session()
    print(sess.run(d))

    graph_name = "constant_graph"

    # write text proto and binary proto
    tf.train.write_graph(sess.graph_def, "./", graph_name + ".pb.txt", True)
    tf.train.write_graph(sess.graph_def, "./", graph_name + ".pb", False)
    

if __name__ == '__main__':
    create_tf_graph()
