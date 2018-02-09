# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from __future__ import division


def split_pads_into_pairs(pads):  # type: (Sequence[int]) -> List[Tuple[int]]
    """
    Convert ONNX padding format to ngraph padding format.

    :param pads: ONNX `pads` format: [x1_begin, x2_begin..., x1_end, x2_end,...]
    :return: ngraph format: [(x1_begin, x1_end), (x2_begin, x2_end), ...]
    """
    if not pads:
        return []

    first_end_pad_index = int(len(pads) / 2)
    begin_pads = pads[:first_end_pad_index]
    end_pads = pads[first_end_pad_index:]
    return list(zip(begin_pads, end_pads))


def verify_symmetric_padding(onnx_node, pads):
    # type: (NodeWrapper, Sequence[int]) -> bool
    """
    Check if the `pads` value of an ONNX node contains only symmetric padding pairs.

    :param onnx_node: an ONNX node
    :param pads: the value for `pads` already extracted or calculated base on `auto_pad`
    :return: True if padding is symmetric, otherwise raises a NotImplementedError
    """
    for pad_left, pad_right in split_pads_into_pairs(pads):
        if pad_left != pad_right:
            raise NotImplementedError('%s node (%s): asymmetric padding is not supported '
                                      'by ngraph.', onnx_node.op_type, onnx_node.name)

    return True
