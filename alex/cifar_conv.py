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
        self.inputs = g.data_tensor((3, 32, 32), name='inputs')
        self.targets = g.data_tensor((10), name='targets')

        c1 = g.conv(self.inputs, filter=(3,5,5), padding=(None, 0, 0), count=16, stride=(1, 2, 2))
        p1 = g.max_pool(g.relu(c1), filter=(1,2,2), padding=(None, 0, 0), stride=(1, 1, 1))
        c2 = g.conv(p1, filter=(3,5,5), padding=(None, 0, 0), count=16, stride=(1,2,2))
        p2 = g.max_pool(g.relu(c1), filter=(1,2,2), padding=(None, 0, 0), stride=(1,1,1))
        #y3 = g.relu(g.affine(c2, (500,)))
        #y4 = g.softmax(g.affine(y3, (10,)))
        print(p2)

model = CifarSimpleModel()
