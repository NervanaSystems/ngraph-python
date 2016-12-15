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


class OpsBase(object):
    """
    Abstract base op for implementing TensorFlow ops using ngraph. The class
    diagram is as follows:

    OpsBase
        ^
        |_____________________________________________________ ...
        |                 |                 |
    OpsBinary         OpsUnary           OpsReduction          ...
        ^                 ^                 ^
        |def Add()        |def Tanh()       |
        |def Mul()        |def Sigmoid()    |
        |...              |...              |
        |_________________|_________________|_________________ ...
        |
        |
    OpsBridge (contains mix-ins from OpsBinary, OpsUnary, ...)
    """

    def DummyOp(self, input_ops):
        """
        An example of how actual op function shall be added

        Arguments:
            input_ops (List): list of ngraph op

        Returns:
            The resulting ngraph op
        """
        raise NotImplementedError
