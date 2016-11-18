# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import print_function

from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
from ngraph.frontends.tensorflow.tf_importer.utils import SGDOptimizer
import numpy as np
import tensorflow as tf
import ngraph.transformers as ngt

# setups -> xs: (N, C), y: (N, 1)
xs_np = np.array([[0.52, 1.12, 0.77], [0.88, -1.08, 0.15],
                  [0.52, 0.06, -1.30], [0.74, -2.49, 1.39]])
ys_np = np.array([[1], [1], [0], [1]])
max_iter = 10
lrate = 0.1

# placeholders
x = tf.placeholder(tf.float32, shape=(4, 3))
t = tf.placeholder(tf.float32, shape=(4, 1))
w = tf.Variable(tf.zeros([3, 1]))
y = tf.nn.sigmoid(tf.matmul(x, w))
log_likelihoods = tf.log(y) * t + tf.log(1 - y) * (1 - t)
cost = -tf.reduce_sum(log_likelihoods)
init_op = tf.initialize_all_variables()

# import graph_def
importer = TFImporter()
importer.parse_graph_def(tf.get_default_graph().as_graph_def())

# get handle of ngraph ops
x_ng, t_ng, cost_ng, init_op_ng = importer.get_op_handle([x, t, cost, init_op])

# transformer and computations
transformer = ngt.make_transformer()
updates = SGDOptimizer(lrate).minimize(cost_ng)
train_comp = transformer.computation([cost_ng, updates], x_ng, t_ng)
init_comp = transformer.computation(init_op_ng)
transformer.initialize()

# train
init_comp()
for idx in range(max_iter):
    loss_val, _ = train_comp(xs_np, ys_np)
    print("[Iter %s] Cost = %s" % (idx, loss_val))

with tf.Session() as sess:
    train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost)
    sess.run(init_op)

    for idx in range(max_iter):
        cost_val, _ = sess.run([cost, train_step],
                               feed_dict={x: xs_np,
                                          t: ys_np})
        print("[Iter %s] Cost = %s" % (idx, cost_val))
