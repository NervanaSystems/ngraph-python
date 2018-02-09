# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Utilities for wgan example
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
    # hard limit values outside of range
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0

    if len(data.shape) == 5:  # MNIST example generated: C,D,H,W,N
        data = data.squeeze().transpose(2, 0, 1)
    elif len(data.shape) == 4:  # LSUN color image genrated: N,C,H,W
        data = data.squeeze().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    # auto size plots
    y_dim = int(2 ** (int(np.log2(args.batch_size) / 2)))
    x_dim = int(args.batch_size / y_dim)

    if len(data.shape) == 3:  # mnist image : N,H,W
        data = data.reshape(y_dim, x_dim, args.im_size, args.im_size).transpose(0, 2, 1, 3).\
            reshape(y_dim * args.im_size, x_dim * args.im_size)
        plt.gray()
    elif len(data.shape) == 4:  # color image : N,H,W,C
        data = data.reshape(y_dim, x_dim,
                            args.im_size, args.im_size, 3).transpose(0, 2, 1, 3, 4).\
            reshape(y_dim * args.im_size, x_dim * args.im_size, 3)

    # plot and save file
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

    if len(image_samp.shape) == 3:  # grey image
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
