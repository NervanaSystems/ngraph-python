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

from __future__ import print_function, division

import onnx
import pytest
from onnx.helper import make_node, make_graph, make_tensor_value_info

from ngraph.frontends.onnx.onnx_importer.model_wrappers import NodeWrapper, GraphWrapper

from ngraph.frontends.onnx.onnx_importer.utils.misc import split_into_pairs, \
    verify_symmetric_padding


def test_split_into_pairs():
    assert split_into_pairs([1, 1, 2, 2, 3, 3]) == [(1, 1), (2, 2), (3, 3)]
    assert split_into_pairs([1, 1, 2, 2, 3]) == [(1, 1), (2, 2)]


def test_verify_symmetric_padding():
    def wrap_node(node):
        graph = make_graph([node], "test_graph",
                           [make_tensor_value_info("X", onnx.TensorProto.FLOAT, (1, 1, 1, 1)),
                            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, (1, 1, 1, 1))],
                           [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, ())])
        return NodeWrapper(node, GraphWrapper(graph))

    node = wrap_node(make_node("Conv", ["X", "Y"], ["Z"], pads=(1, 1, 2, 2, 3, 3)))
    assert verify_symmetric_padding(node)

    with pytest.raises(NotImplementedError):
        node = wrap_node(make_node("Conv", ["X", "Y"], ["Z"], pads=(1, 1, 2, 4)))
        assert verify_symmetric_padding(node)
