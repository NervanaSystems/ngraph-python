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

class IncompatibleNodeTypes(Exception):
    def __init__(self, value="Node types are incompatible"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def sum_broadcast_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        raise IncompatibleNodeTypes()
    def broadcast_size(s1, s2):
        if s1 == 1:
            return s2
        if s2 == 1:
            return s1
        if s1 == s2:
            return s1
        raise IncompatibleNodeTypes()

    return tuple(broadcast_size(s1, s2) for (s1, s2) in zip(shape1, shape2))


class NodeType(object):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return False


class TensorType(NodeType):
    def __init__(self, shape, dtype=None):
        super(TensorType, self).__init__('TensorType')
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other):
        """

        :type other: object
        """
        if not isinstance(other, TensorType):
            return False

        return self.shape == other.shape and self.dtype == other.dtype

    def clone(self):
        return TensorType(shape=self.shape, dtype=self.dtype)
