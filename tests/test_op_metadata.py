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
from ngraph.util.names import NameableValue


class Dummy(NameableValue):
    metadata = {"layer_type": "convolution"}

    @ng.with_op_metadata
    def configure(self, input_op):
        return ng.exp(input_op)


def test_metadata():
    n = ng.Op(metadata=dict(something=3))
    m = ng.Op()
    assert len(m.metadata) == 0
    assert n.metadata['something'] == 3


def test_metadata_capture():
    layer = Dummy()
    x = ng.constant(2)
    ret = layer.configure(x)
    assert ret.metadata['layer_type'] == 'convolution'
