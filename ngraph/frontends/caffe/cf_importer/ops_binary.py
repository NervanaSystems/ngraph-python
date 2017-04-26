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

import ngraph as ng
import caffe_pb2
from ngraph.frontends.caffe.cf_importer.ops_bridge import register_func_with_ops_bridge


class OpsBinary():
    """
    This class supports all the binary operations required for caffe.
    It provides ngraph ops corresponding to caffe operations

    Note : register all the Op functions in this class with OpsBridge
    """
    def Eltwise(self, layer, inputs):
        """
        To support the Eltwise layer of caffe.

        Arguments:
            layer: Layer which needs to be be mapped to ngrpah op
            inputs: input ops on which current op depends on
        return:
            ngraph output operation corresponding to the given layer
        """
        operation = layer.eltwise_param.operation

        if operation == caffe_pb2.EltwiseParameter.SUM:
            ax = inputs[0].axes
            out = ng.add(inputs[0], ng.cast_axes(inputs[1], ax))
            for inp in inputs[2:]:
                out = ng.add(out, ng.cast_axes(inp, ax))
            out.named = layer.name
            return out

    def __call__(self, func, layer, inputs):
        return getattr(self, func)(layer, inputs)

# register all functions in this class with opbridge
ops_binary = OpsBinary()
register_func_with_ops_bridge("Eltwise", ops_binary)
# EOF
