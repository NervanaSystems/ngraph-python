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

from ngraph.frontends.tensorflow.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsNN(OpsBase):
    """
    Mix-in class for tf.nn related ops
    """

    def SparseSoftmaxCrossEntropyWithLogits(self, tf_node, inputs):
        """
        Computes softmax cross entropy. The inputs `logits` are unscaled log
        probabilities, and each row of `labels[i]` must be a valid distribution.
        Reference: https://goo.gl/z5T2my

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            logits, labels, name
        """

        # logits: (N1, Y1), labels: (N2,)
        logits, labels = inputs

        # check input dimension
        try:
            assert len(logits.axes) == 2
            assert len(labels.axes) == 1
            assert logits.axes[0].length == labels.axes[0].length
        except:
            raise NotImplementedError("logits' shape must be (Y, N), "
                                      "labels' shape must be (N,), "
                                      "other shapes not supported yet.")
        # get axis
        axis_y = logits.axes[1]

        # labels_one_hot: (Y2, N2)
        labels_one_hot = ng.one_hot(labels, axis=axis_y)

        # predicts: (N1, Y1)
        predicts = ng.softmax(logits, normalization_axes=axis_y)

        # dim-shuffle / cast to (Y1, N1)
        predicts_axes = ng.make_axes(
            [axis for axis in reversed(predicts.axes)])
        predicts = ng.Dimshuffle(predicts, axes=predicts_axes)
        labels_one_hot = ng.cast_axes(labels_one_hot, predicts_axes)

        # cross_entropy: (N1,)
        cross_entropy = ng.cross_entropy_multi(
            predicts, labels_one_hot, out_axes=(logits.axes[0], ))

        return cross_entropy

    def Softmax(self, tf_node, inputs):
        """
        Computes softmax activations.

        Arguments:
            tf_node: NodeDef object, the tensorflow node to convert.
            inputs: List of ngraph Ops as inputs to this node.

        Returns:
            A ngraph Op corresponding to the tensorflow node.

        Inputs to tf_node:
            logits, name
        """
        # TODO: only support tf.nn.softmax(logits, dim=-1) now, should add more
        logits = inputs[0]
        return ng.softmax(logits, normalization_axes=logits.axes[1])
