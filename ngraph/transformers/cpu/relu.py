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
# ----------------------------------------------------------------------------
from ngraph.op_graph.op_graph import UnaryElementWiseOp


class ReluOp(UnaryElementWiseOp):
    _index = 0

    def __init__(self, inputs, slope, *args, **kwargs):
        super(ReluOp, self).__init__(inputs, *args, **kwargs)
        self.slope = slope
        self.index = ReluOp._index
        ReluOp._index += 1

    def generate_adjoints(self, adjoints, delta, inputs):
        bprop_relu_op = BpropReluOp(delta, inputs, self)
        bprop_relu_op.add_control_dep(self)
        inputs.generate_add_delta(adjoints, bprop_relu_op)


class ReluDerivOp(UnaryElementWiseOp):
    """
    Maintains index and conv_params through forwarding of the original relu.

    Arguments:
        fprop: The original relu.
    """
    def __init__(self, delta, fprop):
        super(ReluDerivOp, self).__init__(delta)
        self.fprop = fprop

    @property
    def index(self):
        """

        Returns:
            The slice index of the relu.
        """
        return self.fprop.forwarded.index


class BpropReluOp(ReluDerivOp):

    def __init__(self, delta, inputs, fprop, **kwargs):
        super(BpropReluOp, self).__init__(delta, fprop)
        self.inputs = inputs
