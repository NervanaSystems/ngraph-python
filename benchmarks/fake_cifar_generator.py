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
import numpy as np


class FakeDataset(object):
    def __init__(self):
        self.rand_state = np.random.RandomState()

    def next_batch(self, batch_size):
        batch_xs = self.rand_state.rand(batch_size, 3, 32, 32).astype(np.float32)
        labels = self.rand_state.randint(low=0, high=9, size=batch_size)
        batch_ys = np.eye(10)[labels, :]
        return (batch_xs, batch_ys)


class FakeCIFAR(object):
    def __init__(self, random_seed=None):
        self.train = FakeDataset()
        if random_seed is not None:
            self.reset(random_seed)

    def reset(self, random_seed):
        self.train.rand_state.seed(random_seed)
