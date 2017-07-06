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
from __future__ import division
from ngraph.frontends.neon import ax
import ngraph as ng
import numpy as np


def generate_data(dataset, batch_size):
    rand_state = np.random.RandomState()
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


def generate_ds2_data(max_length, str_w, nbands, batch_size, num_iter):
    frame_stride = 0.01  # seconds, hard-coded value in make_aeon_dataloaders
    max_utt_len = ((int(max_length / frame_stride) - 1) // str_w) + 1

    train_set, eval_set = make_fake_dataloader(nbands, max_utt_len, batch_size, num_iter)

    inputs = train_set.make_placeholders()
    return inputs, train_set, eval_set


def make_fake_dataloader(nbands, max_utt_len, batch_size, num_iter):

    # mimic aeon data format
    dataset_info = {'audio': {"axes": [('C', 1),
                                       ('frequency', nbands),
                                       ('time', int(max_utt_len))]}}

    train_set = FakeDataIterator(dataset_info, batch_size, num_iter, same=False)
    eval_set = FakeDataIterator(dataset_info, batch_size, num_iter, same=False)

    return train_set, eval_set


class FakeDataIterator(object):

    def __init__(self, dataset_info, batch_size, total_iterations=np.inf, same=False):
        self.batch_size = batch_size
        self.total_iterations = total_iterations
        self.same = same

        self.axis_names = dict()
        self.randomizers = dict()
        self.dtypes = dict()
        ax.N.length = batch_size
        batch_axes = ng.make_axes([ax.N])
        for name, info in dataset_info.items():
            self.axis_names[name] = batch_axes + ng.make_axes([ng.make_axis(name=k, length=v)
                                                               for k, v in info["axes"]])
            self.randomizers[name] = info.get("random", lambda s: np.random.uniform(-1, 1, s))
            self.dtypes[name] = info.get("dtype", np.float32)

        self.keys = list(self.axis_names.keys())
        self.ndata = total_iterations * batch_size
        self._sample = dict()
        self.start = 0
        self.index = 0

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return self.total_iterations

    def make_placeholders(self):
        placeholders = {}
        ax.N.length = self.batch_size
        for k, axes in self.axis_names.items():
            placeholders[k] = ng.placeholder(axes, dtype=np.dtype(self.dtypes[k])).named(k)

        return placeholders

    def reset(self):
        """
        Reset the counter
        """
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.total_iterations:
            raise StopIteration

        self.index += 1
        feed_dict = dict()
        if self.same:
            for name, axes in self.axis_names.items():
                if name in self._sample:
                    feed_dict[name] = self._sample[name]

        for name, axes in self.axis_names.items():
            if name not in feed_dict:
                feed_dict[name] = self.randomizers[name](axes.lengths).astype(self.dtypes[name])
                self._sample[name] = feed_dict[name]

        return feed_dict

    next = __next__
