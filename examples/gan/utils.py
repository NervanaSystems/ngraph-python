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
Utilities for mnist_gan example
"""
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed')


def save_plots(data, filename, train_data, args, title=None):
    # format incoming data
    data = (data.astype(np.float) + 1.0) / 2.0
    data = data.squeeze().transpose(2, 0, 1)

    # auto size plots
    y_dim = 2 ** (int(np.log2(args.batch_size) / 2))
    x_dim = args.batch_size / y_dim

    data = data.reshape(y_dim, x_dim, 28, 28).transpose(0, 2, 1, 3).reshape(y_dim * 28, x_dim * 28)

    # plot and save file
    plt.gray()
    plt.imshow(data)
    plt.title(title)
    plt.axis('off')
    plt.savefig(args.plot_dir + '/' + filename)
    plt.clf
    plt.cla
    plt.close()

    # plot and save loss and gradients
    for key in train_data.keys():
        data = np.array(train_data[key]).T
        plt.plot(data[0], data[1])
        plt.title(key + ' for ' + args.loss_type)
        plt.xlabel('Iterations')
        plt.ylabel(key)
        plt.savefig(args.plot_dir + '/' + key + '.png')
        plt.clf
        plt.cla
        plt.close()


def train_schedule(idx, args):
    '''
    Generate number of critic iterations
    as a function of minibatch index
    '''

    if args.loss_type == 'DCGAN':
        return 1

    if args.loss_type == 'WGAN' or args.loss_type == 'WGAN-GP':
        if idx < 25 or (idx + 1) % 500 == 0:
            return 100
        else:
            return 5


def get_image(datadict):
    '''
    Place the input data into proper format
    '''

    image_samp = 2. * (datadict['image'].astype(np.float) / 255.0) - 1.0
    # reshape from NHW to DHWN
    image_samp = np.expand_dims(image_samp.transpose([1, 2, 0]), axis=0)
    # reshape from DHWN to CDHWN
    image_samp = np.expand_dims(image_samp, axis=0)

    return image_samp


class Noise(object):

    def __init__(self, shape, mean=0, std=1, seed=0):
        self.shape = shape
        self.mean = mean
        self.std = std
        self.seed = seed
        np.random.seed(seed)

    def __next__(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=self.shape)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
