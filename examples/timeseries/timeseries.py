#!/usr/bin/env python
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
import numpy as np


class TimeSeries(object):
    """
    Class that generates a time-series of Lissajous function
    Attributes:
        batch_size (int)
        npoints (int,  optional): points per cycle (resolution of the series)
        ncycles (int, optional): how many cycles to get for the function
        train_ratio (float, optional): percentage of the function to be used for training
        curvetype (str, optional): Type of Lissajous Curve, Lissajous1 or Lissajous2
        seq_len (int, optional): length of the sequence for each sample
        predict_seq (boolean, optional):
            False : Inputs - X[no_samples, seq_len, no_input_features]
                    Labels - y[no_samples, no_output_features]
            True : Inputs - X[no_samples, seq_len, no_input_features]
                   Labels - y[no_samples, seq_len, no_output_features]
        look_ahead (int, optional): How far ahead the predicted sequence starts from the input seq
                    Set to 1 to start predicting from next time point onwards
                    Only used when predict_seq is False
    """
    def __init__(self, batch_size, npoints=100, ncycles=10, train_ratio=0.8,
                 curvetype='Lissajous1', seq_len=30,
                 predict_seq=True, look_ahead=1):
        """
        curvetype (str, optional): 'Lissajous1' or 'Lissajous2'
        """
        self.nsamples = npoints * ncycles
        self.x = np.linspace(0, ncycles * 2 * np.pi, self.nsamples)

        if curvetype not in ('Lissajous1', 'Lissajous2'):
            raise NotImplementedError()

        sin_scale = 2 if curvetype is 'Lissajous1' else 6

        def y_x(x):
            return 4.0 / 5 * np.sin(x / sin_scale)

        def y_y(x):
            return 4.0 / 5 * np.cos(x / 2)

        self.data = np.zeros((self.nsamples, 2))
        self.data[:, 0] = np.asarray([y_x(xs)
                                      for xs in self.x]).astype(np.float32)
        self.data[:, 1] = np.asarray([y_y(xs)
                                      for xs in self.x]).astype(np.float32)

        if (predict_seq is False):
            # X will be (no_samples, time_steps, feature_dim)
            X = self.rolling_window(a=self.data, seq_len=seq_len)

            # Get test samples; number of test samples will be an integer multiple of batch_size
            test_samples = (int((1 - train_ratio) * X.shape[0]) // batch_size) * batch_size
            train_samples = X.shape[0] - test_samples - 1

            self.train = {'X': {'data': X[:train_samples, ...], 'axes': ('N', 'REC', 'F')},
                          'y': {'data': self.data[seq_len:train_samples + seq_len, ...],
                                'axes': ('N', 'Fo')}}

            self.test = {'X': {'data': X[train_samples:-1, ...], 'axes': ('N', 'REC', 'F')},
                         'y': {'data': self.data[train_samples + seq_len:, ...],
                               'axes': ('N', 'Fo')}}
        else:
            # Make number of samples an integer multiple of seq_len * batch_size
            no_samples = (self.nsamples // (batch_size * seq_len)) * (batch_size * seq_len)
            X = self.data[:no_samples, :]
            y = np.concatenate((X[look_ahead:], X[:look_ahead]))

            # Reshape X and y
            X = X.reshape((no_samples // seq_len, seq_len, 2))
            y = y.reshape((no_samples // seq_len, seq_len, 2))

            # Get test samples; number of test samples will be an integer multiple of batch_size
            test_samples = (int((1 - train_ratio) * X.shape[0]) // batch_size) * batch_size
            train_samples = X.shape[0] - test_samples

            self.train = {'X': {'data': X[:train_samples], 'axes': ('N', 'REC', 'F')},
                          'y': {'data': y[:train_samples],
                                'axes': ('N', 'REC', 'Fo')}}

            self.test = {'X': {'data': X[train_samples:], 'axes': ('N', 'REC', 'F')},
                         'y': {'data': y[train_samples:], 'axes': ('N', 'REC', 'Fo')}}

    def rolling_window(self, a=None, seq_len=None):
        """
        Convert sequence a into time-lagged vectors
        a           : (time_steps, feature_dim)
        seq_len     : length of sequence used for prediction
        returns  (time_steps - seq_len + 1, seq_len, feature_dim)  array
        """
        assert a.shape[0] > seq_len

        shape = [a.shape[0] - seq_len + 1, seq_len, a.shape[-1]]
        strides = [a.strides[0], a.strides[0], a.strides[-1]]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
