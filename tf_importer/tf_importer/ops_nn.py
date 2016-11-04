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

from tf_importer.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsNN(OpsBase):
    """
    Mix-in class for tf.nn related ops
    """

    def SparseSoftmaxCrossEntropyWithLogits(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes softmax cross entropy between `logits` and `labels`.

        Measures the probability error in discrete classification tasks in which the
        classes are mutually exclusive (each entry is in exactly one class).  For
        example, each CIFAR-10 image is labeled with one and only one label: an image
        can be a dog or a truck, but not both.

        **NOTE:**  While the classes are mutually exclusive, their probabilities
        need not be.  All that is required is that each row of `labels` is
        a valid probability distribution.  If they are not, the computation of the
        gradient will be incorrect.

        If using exclusive `labels` (wherein one and only
        one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

        **WARNING:** This op expects unscaled logits, since it performs a `softmax`
        on `logits` internally for efficiency.  Do not call this op with the
        output of `softmax`, as it will produce incorrect results.

        `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        and the same dtype (either `float32` or `float64`).

        Args:
            logits: Unscaled log probabilities.
            labels: Each row `labels[i]` must be a valid probability distribution.
            name: A name for the operation (optional).

        Returns:
            A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
            softmax cross entropy loss.

        # C++ ref: https://goo.gl/z5T2my
        """

        # logits: (N1, Y1), labels: (N2,)
        logits, labels = inputs

        # check input dimension
        try:
            check_0 = len(logits.axes) == 2
            check_1 = len(labels.axes) == 1
            check_2 = logits.axes[0].length == labels.axes[0].length
            assert check_0 and check_1 and check_2
        except:
            raise NotImplementedError("logits' shape must be (Y, N), "
                                      "labels' shape must be (N,), "
                                      "other shapes not supported yet.")
        # get axis
        axis_y = logits.axes[1]

        # labels_one_hot: (Y2, N2)
        labels_one_hot = ng.onehot(labels, axis=axis_y)

        # predicts: (N1, Y1)
        predicts = ng.softmax(logits, normalization_axes=axis_y)

        # dim-shuffle / cast to (Y1, N1)
        predicts_axes = ng.make_axes([axis for axis in reversed(predicts.axes)])
        predicts = ng.Dimshuffle(predicts, axes=predicts_axes)
        labels_one_hot = ng.cast_axes(labels_one_hot, axes=predicts_axes)

        # cross_entropy: (N1,)
        cross_entropy = ng.cross_entropy_multi(predicts, labels_one_hot,
                                               out_axes=(logits.axes[0],))

        return cross_entropy

    def Softmax(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Computes softmax activations.

        For each batch `i` and class `j` we have

                softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

        Args:
            logits: A `Tensor`. Must be one of the following types: `half`,
                    `float32`, `float64`. 2-D with shape `[batch_size, num_classes]`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
        """
        # TODO: only support tf.nn.softmax(logits, dim=-1) now
        logits = inputs[0]
        return ng.softmax(logits, normalization_axes=logits.axes[1])
