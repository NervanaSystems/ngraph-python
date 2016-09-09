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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import geon as be
from util.importer import TensorFlowImporter


def test_import_constant_graph():
    # build constant graph
    a = tf.constant(10)
    b = tf.constant(32)
    c = a + b
    d = c * a

    # get tensorflow result
    with tf.Session() as sess:
        tf_result = sess.run(d)

    # write to protobuf
    pb_txt_path = "constant_graph.txt"

    tf.train.write_graph(sess.graph_def, "./", pb_txt_path, True)

    # init importer, transformer
    importer = TensorFlowImporter(pb_txt_path)
    transformer = be.NumPyTransformer()

    # now, assumes last op is the result we want to get
    ng_result_comp = transformer.computation([importer.last_op])
    ng_result = ng_result_comp()[0]

    print(tf_result, ng_result)
    assert tf_result == ng_result


if __name__ == '__main__':
    test_import_constant_graph()
