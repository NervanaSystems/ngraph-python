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

        # These are the persistent tensors that will be initialized and evolve
        w1 = g.variable_tensor((64, 3, 3), name='w1')
        w2 = g.variable_tensor((10, 64), name='w2')

        y1 = g.relu(g.conv(self.inputs, stride=2, pad=0, weights=w1))
        print(y1)

        # y2 = g.pad(self.inputs) + y1
        # y3 = g.affine(y2, weights=w2)


        # cost = g.cross_entropy(g.softmax(y3), self.targets)

        # params = [w1, w2]

        # grads will return [dw1, dw2]
        # grads = g.calc_gradients(cost, params)

        # update rule
        # for w, d in zip(params, grads):
        #    w = SGD(w, d)

model = CifarSimpleModel()

if __name__ == '__main__':
    pass

