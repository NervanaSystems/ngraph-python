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


def generate_data(dataset, batch_size):
    rand_state = np.random.RandomState()
    # rand_state.seed(0)
    if dataset == 'cifar10':
        batch_xs = rand_state.rand(batch_size, 3, 32, 32).astype(np.float32)
        labels = rand_state.randint(low=0, high=9, size=batch_size)
        batch_ys = np.eye(10)[labels, :]
        x_train = np.vstack(batch_xs).reshape(-1, 3, 32, 32)
        y_train = np.vstack(batch_ys).ravel()
        return (x_train, y_train)

    elif dataset == 'i1k':
        batch_xs = rand_state.rand(batch_size, 3, 224, 224).astype(np.float32)
        labels = rand_state.randint(low=0, high=999, size=batch_size)
        batch_ys = np.eye(1000)[labels, :]
        x_train = np.vstack(batch_xs).reshape(-1, 3, 224, 224)
        y_train = np.vstack(batch_ys).ravel()
        return (x_train, y_train)

    else:
        raise ValueError("Incorrect dataset provided")
