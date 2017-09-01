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
"""
Helper classes and functions for gan_toy.py
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


class UniformNoise(object):

    def __init__(self, shape, seed):
        self.shape = shape
        self.seed = seed
        np.random.seed(seed)

    def __next__(self):
        return np.random.uniform(size=self.shape)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class NormalNoise(object):

    def __init__(self, shape, seed):
        self.shape = shape
        self.seed = seed
        np.random.seed(seed)

    def __next__(self):
        return np.random.normal(size=self.shape)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class DataGenerator(object):

    def __init__(self, shape, seed, data_type):
        self.shape = shape
        self.seed = seed
        self.data_type = data_type
        np.random.seed(seed)

    def __next__(self):
        if self.data_type == 'Rectangular':
            return self._25_gaussians(self.shape)
        elif self.data_type == 'Circular':
            return self._n_gaussians(self.shape, n=8)
        elif self.data_type == 'Roll':
            return self._swissroll(self.shape)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def _25_gaussians(self, shape):
        '''
        25 gaussians with centers on a euclidean grid
        '''

        mean = 2 * np.random.randint(low=-2, high=3, size=shape)
        rndn_pos = 0.05 * np.random.randn(shape[0], shape[1], shape[2], shape[3], shape[4])

        result = (mean + rndn_pos) / 2.828
        return result

    def _n_gaussians(self, shape, n=8):
        '''
        n gaussians with centers on a polar grid
        '''
        center_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        center_angle = np.random.choice(center_angles, shape[4])
        rndn_pos = 0.02 * np.random.randn(shape[0], shape[1], shape[2], shape[3], shape[4])
        mean = (2 * np.array([np.sin(center_angle), np.cos(center_angle)])).reshape(shape)

        result = (mean + rndn_pos) / 1.414
        return result

    def _swissroll(self, shape):
        '''
        swissroll dataset
        '''

        roll = sklearn.datasets.make_swiss_roll(n_samples=shape[-1], noise=0.25)[0]
        result = roll.astype('float32')[:, [0, 2]].T
        result /= 7.5

        return result


def generate_plot(plot_dir, iteration, data_in, output_g, output_d, train_data, args):
    data_in = data_in.squeeze()
    generated = output_g['generated']
    plt.plot(data_in[0], data_in[1], 'gx')
    plt.plot(generated[0], generated[1], 'r.')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    title = 'Iteration {} \n Gen. Cost {:.2E}  Disc. Cost {:.2E}'.format(
        iteration, float(output_g['batch_cost']), float(output_d['batch_cost']))
    plt.title(title)
    plt.savefig(plot_dir + '/' + str(iteration) + 'Generated.png')
    plt.clf()

    # plot and save loss and gradients
    for key in train_data.keys():
        data = np.array(train_data[key]).T
        plt.plot(data[0], data[1])
        plt.title(key + ' for ' + args.loss)
        plt.xlabel('Iterations')
        plt.ylabel(key)
        plt.savefig(plot_dir + '/' + key + '.png')
        plt.clf()
