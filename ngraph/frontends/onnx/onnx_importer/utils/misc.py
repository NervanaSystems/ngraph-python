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


def split_into_pairs(items):  # type: (Iterable) -> List[Tuple]
    """
    Split a list or tuple of items into a list of pairs (tuples).

    e.g. [1, 2, 3, 4, 5, 6] -> [(1, 2), (3, 4), (5, 6)]

    :param items: an iterable with items
    :return: list of tuples with pairs of items
    """
    return list(zip(*[iter(items)] * 2))


def verify_symmetric_padding(onnx_node):  # type: (NodeWrapper) -> bool
    """
    Checks if the "pads" attribute of an ONNX node contains only symmetric padding pairs.
    """
    pads = onnx_node.get_attribute_value('pads', ())

    for pad_left, pad_right in split_into_pairs(pads):
        if pad_left != pad_right:
            raise NotImplementedError('%s node (%s): asymmetric padding is not supported '
                                      'by ngraph.', onnx_node.op_type, onnx_node.name)

    return True
