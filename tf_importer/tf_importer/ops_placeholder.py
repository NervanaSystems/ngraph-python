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

from tf_importer.tf_importer.utils import tf_to_shape_axes
from tf_importer.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsPlaceholder(OpsBase):
    """
    Mix-in class placeholder op
    """

    def Placeholder(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Inserts a placeholder for a tensor that will be always fed.

        **Important**: This tensor will produce an error if evaluated. Its value must
        be fed using the `feed_dict` optional argument to `Session.run()`,
        `Tensor.eval()`, or `Operation.run()`.

        For example:

        ```python
        x = tf.placeholder(tf.float32, shape=(1024, 1024))
        y = tf.matmul(x, x)

        with tf.Session() as sess:
            print(sess.run(y))  # ERROR: will fail because x was not fed.

            rand_array = np.random.rand(1024, 1024)
            print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
        ```

        Args:
            dtype: The type of elements in the tensor to be fed.
            shape: The shape of the tensor to be fed (optional). If the shape is not
                   specified, you can feed a tensor of any shape.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` that may be used as a handle for feeding a value, but not
            evaluated directly.
        """
        axes = tf_to_shape_axes(tf_node)
        ng_op = ng.placeholder(axes, name=tf_node.name)
        return ng_op
