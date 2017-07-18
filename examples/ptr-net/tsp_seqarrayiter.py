# ----------------------------------------------------------------------------
<<<<<<< HEAD
# Copyright 2017 Nervana Systems Inc.
=======
# Copyright 2016 Nervana Systems Inc.
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
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
import ngraph as ng
from future.utils import viewitems
import six
from ngraph.frontends.neon import ax
import collections

class TSPSequentialArrayIterator(object):
    """
<<<<<<< HEAD
    modification of SequentialArrayIterator class.
=======
    modification of SequentialArrayIterator class in arrayiterator.py. 
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
    Add number of features argument to handle variable feature sizes.

    Args:
        data_arrays (ndarray): Input features of the dataset.
        time_steps (int): number of time steps.
        batch_size (int): number of examples in each minibatch.
        n_features (int): nubmer of input features (dimensions) of each cities.
        total_iterations (int): number of minibatches to cycle through on this iterator.
                                If not provided, it will cycle through all of the data once.
<<<<<<< HEAD
    """
    def __init__(self, data_arrays, time_steps, batch_size, nfeatures,
                 total_iterations=None):
        self.nfeatures = nfeatures
=======
        get_prev_target (bool): default to be True to use Teach Forcing (Williams and Zipser, 1989)
                                during training.
    """
    def __init__(self, data_arrays, time_steps, batch_size, nfeatures,
                 total_iterations=None, get_prev_target=True):
        self.nfeatures = nfeatures
        self.get_prev_target = get_prev_target

>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.index = 0

        # make sure input is in dict format
        if isinstance(data_arrays, dict):
            self.data_arrays = {k: v for k, v in viewitems(data_arrays)}
        else:
            raise ValueError("Must provide dict as input")

        # Number of examples
        self.ndata = len(six.next(six.itervalues(self.data_arrays)))

        # Number of examples (with integer multiplication of batch sizes)
        self.ndata = self.ndata // (self.batch_size) * self.batch_size

        self.nbatches = self.ndata // self.batch_size

        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.total_iterations = self.nbatches if total_iterations is None else total_iterations

<<<<<<< HEAD
        # reshape array for batch and batch size dimensions
=======
        # reshape array for placeholders
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
        self.data_arrays['inp_txt'] = self.data_arrays['inp_txt'][:self.ndata][:][:].reshape(self.batch_size, self.nbatches, self.time_steps, self.nfeatures)

        self.data_arrays['tgt_txt'] = self.data_arrays['tgt_txt'][:self.ndata][:].reshape(self.batch_size, self.nbatches, self.time_steps)

<<<<<<< HEAD
        self.data_arrays['teacher_tgt'] = self.data_arrays['teacher_tgt'][:self.ndata][:][:].reshape(self.batch_size, self.nbatches, self.time_steps, self.nfeatures)

        # Teacher Forcing
        self.data_arrays['teacher_tgt'] = np.roll(self.data_arrays['teacher_tgt'], shift=1, axis=2)
        # put a start token (0, 0) as the first decoder input
        for i in range(self.batch_size):
            for j in range(self.nbatches):
                for k in range(self.nfeatures):
                    np.put(self.data_arrays['teacher_tgt'][i][j][0], [k], [0])
=======
        # Teacher Forcing
        if self.get_prev_target:
            self.data_arrays['prev_tgt'] = np.roll(self.data_arrays['tgt_txt'], shift=1, axis=2)
            # put a start token (0 here) as the first input to the decoder
            for i in range(self.batch_size):
                for j in range(self.nbatches):
                    np.put(self.data_arrays['prev_tgt'][i][j], [0], [0])
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4

    def make_placeholders(self):
        batch_axis = ng.make_axis(length=self.batch_size, name="N")
        time_axis = ng.make_axis(length=self.time_steps, name="REC")
        feature_axis = ng.make_axis(length=self.nfeatures, name="feature_axis")

        dict = {}
        for k in self.data_arrays.keys():
<<<<<<< HEAD
            if k == 'inp_txt' or k == 'teacher_tgt':
=======
            if k == 'inp_txt':
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
                p_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
            else:
                p_axes = ng.make_axes([batch_axis, time_axis])
            dict[k] = ng.placeholder(p_axes)

        return dict

    def reset(self):
        self.index = 0

    def __iter__(self):
        while self.index < self.total_iterations:
            idx = self.index % self.nbatches
            self.index += 1
<<<<<<< HEAD
            dict = {}
            for k, x in viewitems(self.data_arrays):
                if k == 'inp_txt' or k == 'teacher_tgt':
=======

            dict = {}
            for k, x in viewitems(self.data_arrays):
                if k == 'inp_txt':
>>>>>>> 4b7eb0159d671ed0226fe820069655d9a2aeb6d4
                    dict[k] = np.squeeze(x[:, idx:(idx + 1), :, :])
                else:
                    dict[k] = np.squeeze(x[:, idx:(idx + 1), :])

            yield dict
