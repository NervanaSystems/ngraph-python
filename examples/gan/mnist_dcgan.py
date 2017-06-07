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
MNIST DCGAN
"""
import numpy as np
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import Sequential, Layer, Deconvolution, Convolution, \
    Rectlin, Logistic, Tanh, GaussianInit, Adam, ArrayIterator, make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import MNIST
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    create_plots = True
except ImportError:
    create_plots = False
    print('matplotlib not installed')


def make_optimizer(name=None):
    optimizer = Adam(learning_rate=5e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    return optimizer


def save_plots(data, filename, title=None):
    # convert to matplotlib format: [0.0, 1.0]
    data = (data.astype(np.float) + 1.0) / 2.0
    # hard limit values outside of range
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    # select some images
    for i in range(8):
        gen_image = data[0, 0, :, :, i]  # CDHWN --> HW
        plt.subplot(2, 4, i + 1)
        plt.imshow(gen_image)
    if title is not None:
        plt.suptitle(title)
    plt.savefig(filename)


class Noise(object):
    def __init__(self, samples, shape, mean=0, std=1, seed=0):
        self.samples = samples
        self.shape = shape
        self.mean = mean
        self.std = std
        self.seed = seed
        np.random.seed(seed)
        self.cnt = 0

    def __next__(self):
        if self.cnt < self.samples:
            self.cnt += 1
            return np.random.normal(loc=self.mean, scale=self.std, size=self.shape)
        else:
            raise StopIteration

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


# parse command line arguments
parser = NgraphArgparser()
parser.add_argument('--plot_interval', type=int, default=200,
                    help='save generated images with a period of this many iterations')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
args = parser.parse_args()
np.random.seed(args.rng_seed)

args.batch_size = 32

# Create the dataloader
train_data, valid_data = MNIST(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size)

# noise source
noise_dim = (2, 1, 3, 3)
noise_generator = Noise(train_set.ndata, shape=noise_dim + (args.batch_size,), seed=args.seed)

# generator network
g_scope = 'generator'
filter_init = GaussianInit(var=0.05)
relu = Rectlin(slope=0)
with Layer.variable_scope(g_scope) as scope:
    deconv_layers = [Deconvolution((1, 1, 16), filter_init, strides=1, padding=0,
                                    activation=relu, batch_norm=True),
                     Deconvolution((3, 3, 192), filter_init, strides=1, padding=0,
                                    activation=relu, batch_norm=True, deconv_out_shape=(1, 5, 5)),
                     Deconvolution((3, 3, 192), filter_init, strides=2, padding=0,
                                    activation=relu, batch_norm=True, deconv_out_shape=(1, 11, 11)),
                     Deconvolution((3, 3, 192), filter_init, strides=1, padding=0,
                                    activation=relu, batch_norm=True, deconv_out_shape=(1, 13, 13)),
                     Deconvolution((3, 3, 96), filter_init, strides=2, padding=0,
                                    activation=relu, batch_norm=True, deconv_out_shape=(1, 27, 27)),
                     Deconvolution((3, 3, 96), filter_init, strides=1, padding=0,
                                    activation=relu, batch_norm=True, deconv_out_shape=(1, 28, 28)),
                     Deconvolution((3, 3, 1), filter_init, strides=1, padding=1,
                                    activation=Tanh(), batch_norm=False, deconv_out_shape=(1, 28,28))]
    generator = Sequential(deconv_layers)

# discriminator network
d_scope = 'discriminator'
lrelu = Rectlin(slope=0.1)
with Layer.variable_scope(d_scope) as scope:
    conv_layers = [Convolution((3, 3, 96), filter_init, strides=1, padding=1,
                                activation=lrelu, batch_norm=True),
                   Convolution((3, 3, 96), filter_init, strides=2, padding=1,
                                activation=lrelu, batch_norm=True),
                   Convolution((3, 3, 192), filter_init, strides=1, padding=1,
                                activation=lrelu, batch_norm=True),
                   Convolution((3, 3, 192), filter_init, strides=2, padding=1,
                                activation=lrelu, batch_norm=True),
                   Convolution((3, 3, 192), filter_init, strides=1, padding=1,
                                activation=lrelu, batch_norm=True),
                   Convolution((1, 1, 16), filter_init, strides=1, padding=0,
                                activation=lrelu, batch_norm=True),
                   Convolution((7, 7, 1), filter_init, strides=1, padding=0,
                                activation=Logistic(), batch_norm=False)]
    discriminator = Sequential(conv_layers)

# noise placeholder
N = ng.make_axis(name='N', length=args.batch_size)
noise_ax_names = 'CDHW'
noise_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                           for nm, l in zip(noise_ax_names, noise_dim)])
z_ax = noise_axes + N
z = ng.placeholder(axes=z_ax)

# image placeholder
C = ng.make_axis(name='C', length=1)
D = ng.make_axis(name='D', length=1)
H = ng.make_axis(name='H', length=28)
W = ng.make_axis(name='W', length=28)
image_axes = ng.make_axes([C, D, H, W, N])
image = ng.placeholder(axes=image_axes)

# build network graph
generated = generator(z)
D1 = discriminator(image)
D2 = discriminator(generated)

loss_d = -ng.log(D1) - ng.log(1 - D2)
mean_cost_d = ng.mean(loss_d, out_axes=[])
loss_g = -ng.log(D2)
mean_cost_g = ng.mean(loss_g, out_axes=[])

optimizer_d = make_optimizer(name='discriminator_optimizer')
optimizer_g = make_optimizer(name='generator_optimizer')
updates_d = optimizer_d(loss_d, variable_scope=d_scope)
updates_g = optimizer_g(loss_g, variable_scope=g_scope)

# compile computations
generator_train_inputs = {'noise': z}
discriminator_train_inputs = {'image': image, 'noise': z}

generator_train_outputs = {'batch_cost': mean_cost_g, 'updates': updates_g,
                           'generated': generated}  # for plots
discriminator_train_outputs = {'batch_cost': mean_cost_d, 'updates': updates_d}

with closing(ngt.make_transformer()) as transformer:

    train_computation_g = make_bound_computation(transformer,
                                                 generator_train_outputs,
                                                 generator_train_inputs)
    train_computation_d = make_bound_computation(transformer,
                                                 discriminator_train_outputs,
                                                 discriminator_train_inputs)

    # train loop
    k = 1  # variable rate training of discriminator if k > 1
    print('start train loop')
    for mb_idx, (z_samp, datadict) in enumerate(zip(noise_generator, train_set)):

        image_samp = 2. * (datadict['image'].astype(np.float) / 255.0) - 1.0
        # reshape from NHW to DHWN
        image_samp = np.expand_dims(image_samp.transpose([1, 2, 0]), axis=0)
        # reshape from DHWN to CDHWN
        image_samp = np.expand_dims(image_samp, axis=0)

        # update generator
        for iter_g in range(1):
            output_g = train_computation_g({'noise': z_samp})

        # update discriminator
        for iter_d in range(k):
            output_d = train_computation_d({'noise': z_samp, 'image': image_samp})

        # print losses
        if mb_idx % 100 == 0:
            msg = "Iteration {} complete. Discriminator avg loss: {} Generator avg loss: {}"
            print(msg.format(mb_idx,
                             float(output_d['batch_cost']),
                             float(output_g['batch_cost'])))

        # plots
        if create_plots:
            # plot some examples from dataset
            if mb_idx == 0:
                save_plots(image_samp,
                           'mnist-examples.png',
                           title='mnist data examples')

            # display some generated images
            if mb_idx % args.plot_interval == 0:
                generated = output_g['generated']  # C, D, H, W, N
                pltname = 'dcgan-mnist-iter-{}.png'.format(mb_idx)
                save_plots(generated, pltname, title='iteration {}'.format(mb_idx))
