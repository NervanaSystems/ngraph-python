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
import tensorflow as tf
import ngraph as ng
import ngraph.transformers as ngt

# tensorflow ops
x = tf.constant(1.)
y = tf.constant(2.)
f = x + y

# import
importer = ng.make_tf_importer()
importer.import_graph_def(tf.get_default_graph().as_graph_def())

# get handle
f_ng = importer.get_op_handle(f)

# execute
f_result = ngt.make_transformer().computation(f_ng)()
print(f_result)
