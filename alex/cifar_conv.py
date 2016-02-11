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

import neon.alex.nodes as g


class CifarSimpleModel(object):
    def __init__(self):
        # This is a container where we will put new data in on every iteration
        self.inputs = g.data_tensor((32, 32, 3), name='inputs')
        self.targets = g.data_tensor((10), name='targets')

        w1 = g.variable_tensor((5,5,16))
        c1 = g.max_pool(g.relu(g.conv(self.inputs, w1, pad=2, stride=2)), (2,2), pad=0, stride=2)
        w2 = g.variable_tensor((5,5,32))
        c2 = g.max_pool(g.relu(g.conv(c1, w2, pad=2, stride=2)), (2.2), pad=0, stride=2)
        #y3 = g.relu(g.affine(c2, (500,)))
        #y4 = g.softmax(g.affine(y3, (10,)))
        print(c2)

model = CifarSimpleModel()
