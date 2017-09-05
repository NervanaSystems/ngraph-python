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

from ngraph.op_graph.op_graph import TensorOp


class BatchnormOp(TensorOp):

    def __init__(self, inputs, gamma, beta, epsilon, mean, variance, **kwargs):
        super(
            BatchnormOp,
            self).__init__(
            args=(
                inputs,
                gamma,
                beta,
                epsilon,
                mean,
                variance),
            axes=inputs.axes,
            **kwargs)
        self.eps = epsilon

    def copy_with_new_args(self, args):
        return type(self)(args[0], args[1], args[2], args[3], args[4], args[5])

    def generate_adjoints(self, adjoints, delta, inputs):
        bprop_batchnorm_op = BpropBatchnormOp(delta, inputs, self)
        bprop_batchnorm_op.add_control_dep(self)
        inputs.generate_add_delta(adjoints, bprop_batchnorm_op)


class BpropBatchnormOp(TensorOp):
    """
    Arguments:
    fprop: corrosponding batchnormOp.
    delta: global gradients from the previous layer
    inputs: fprop src input to the batchnormOp
    """

    def __init__(self, delta, inputs, fprop, **kwargs):
        gamma = fprop.args[1]
        beta = fprop.args[2]
        mean = fprop.args[4]
        variance = fprop.args[5]
        super(
            BpropBatchnormOp,
            self).__init__(
            args=(
                delta,
                inputs,
                gamma,
                beta,
                mean,
                variance),
            axes=inputs.axes,
            **kwargs)
        self.fprop = fprop

    def copy_with_new_args(self, args):
        return type(self)(args[0], args[1], self.fprop)
