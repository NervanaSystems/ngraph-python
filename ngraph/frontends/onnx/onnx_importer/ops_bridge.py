# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from __future__ import division

import ngraph as ng


class OpsBridge:
    """
    Bridging op between ONNX and ngraph.
    """

    def get_ng_node(self, onnx_node):
        op_type = onnx_node.op_type
        ng_node_factory = getattr(self, op_type, None)
        ng_inputs = onnx_node.get_ng_inputs()

        if not ng_node_factory:
            raise NotImplementedError("Unknown operation: %s", op_type)

        return ng_node_factory(onnx_node, ng_inputs)

    def Abs(self, onnx_node, ng_inputs):
        return ng.absolute(ng_inputs[0])

    def Add(self, onnx_node, ng_inputs):
        return ng.add(ng_inputs[0], ng_inputs[1])

    def Relu(self, onnx_node, ng_inputs):
        return ng.maximum(ng_inputs[0], 0.)
