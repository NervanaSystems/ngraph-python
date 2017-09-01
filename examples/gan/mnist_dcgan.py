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
MNIST DCGAN, WGAN, WGAN with Gradient Penalty
"""
import numpy as np
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import (Sequential, Deconvolution, Convolution,
                                   Rectlin, Logistic, Tanh, GaussianInit,
                                   Adam, RMSProp, ArrayIterator, Affine,
                                   KaimingInit, make_bound_computation)
from ngraph.frontends.neon import regularize, ConstantInit
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import MNIST
import copy
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    create_plots = True
except ImportError:
    create_plots = False
    print('matplotlib not installed')


# helper functions
def make_optimizer(name=None, weight_clip_value=None):
    if args.loss_type == "DCGAN":
        optimizer = Adam(learning_rate=3e-4, beta_1=0.5, beta_2=0.999,
                         epsilon=1e-8, weight_clip_value=weight_clip_value)

    if args.loss_type == "WGAN":
        optimizer = RMSProp(learning_rate=5e-5, decay_rate=0.99,
                            weight_clip_value=weight_clip_value)

    if args.loss_type == "WGAN-GP":
        optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
                         epsilon=1e-8, weight_clip_value=weight_clip_value)

    return optimizer


def save_plots(data, filename, train_data, args, title=None):
    # format incoming data
    data = (data.astype(np.float) + 1.0) / 2.0
    data = data.squeeze().transpose(2, 0, 1)
    data = data.reshape(4, 8, 28, 28).transpose(0, 2, 1, 3).reshape(4 * 28, 8 * 28)

    # plot and save file
    plt.gray()
    plt.imshow(data)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
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
        plt.savefig(key + '.png')
        plt.clf
        plt.cla
        plt.close()


def train_schedule(idx):
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


# parse command line arguments
parser = NgraphArgparser()
parser.add_argument('--plot_interval', type=int, default=200,
                    help='save generated images with a period of this many iterations')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--loss_type', default="WGAN", help='Loss Function')

args = parser.parse_args()
args.rng_seed = 0
np.random.seed(args.rng_seed)

args.batch_size = 32

# define noise dimension
noise_dim = (2, 1, 3, 3)

# common layer parameters
filter_init = KaimingInit()
relu = Rectlin(slope=0)
lrelu = Rectlin(slope=0.2)

# TODO implement it using linear at the begining once fixed
# generator network


def make_generator(bn=True):
    deconv_layers = [Deconvolution((1, 1, 16), filter_init, strides=1, padding=0,
                                   activation=relu, batch_norm=bn),
                     Deconvolution((3, 3, 192), filter_init, strides=1, padding=0,
                                   activation=relu, batch_norm=bn, deconv_out_shape=(1, 5, 5)),
                     Deconvolution((3, 3, 192), filter_init, strides=2, padding=0,
                                   activation=relu, batch_norm=bn, deconv_out_shape=(1, 11, 11)),
                     Deconvolution((3, 3, 192), filter_init, strides=1, padding=0,
                                   activation=relu, batch_norm=bn, deconv_out_shape=(1, 13, 13)),
                     Deconvolution((3, 3, 96), filter_init, strides=2, padding=0,
                                   activation=relu, batch_norm=bn, deconv_out_shape=(1, 27, 27)),
                     Deconvolution((3, 3, 96), filter_init, strides=1, padding=0,
                                   activation=relu, batch_norm=bn, deconv_out_shape=(1, 28, 28)),
                     Deconvolution((3, 3, 1), filter_init, strides=1, padding=1,
                                   activation=Tanh(), batch_norm=False,
                                   deconv_out_shape=(1, 28, 28))]
    return Sequential(deconv_layers, name="Generator")

# discriminator network


def make_discriminator(bn=True, disc_activation=None):
    conv_layers = [Convolution((3, 3, 96), filter_init, strides=1, padding=1,
                               activation=lrelu, batch_norm=bn),
                   Convolution((3, 3, 96), filter_init, strides=2, padding=1,
                               activation=lrelu, batch_norm=bn),
                   Convolution((3, 3, 192), filter_init, strides=1, padding=1,
                               activation=lrelu, batch_norm=bn),
                   Convolution((3, 3, 192), filter_init, strides=2, padding=1,
                               activation=lrelu, batch_norm=bn),
                   Convolution((3, 3, 192), filter_init, strides=1, padding=1,
                               activation=lrelu, batch_norm=bn),
                   Convolution((1, 1, 16), filter_init, strides=1, padding=0,
                               activation=lrelu, batch_norm=bn),
                   Convolution((7, 7, 1), filter_init, strides=1, padding=0,
                               activation=disc_activation, batch_norm=False)]
    return Sequential(conv_layers, name="Discriminator")


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

# Vanilla GAN
if args.loss_type == "DCGAN":

    generator = make_generator(bn=True)
    discriminator = make_discriminator(bn=True, disc_activation=Logistic())

    # build network graph
    generated = generator(z)
    D1 = discriminator(image)
    D2 = discriminator(generated)

    weight_clip_value = None  # no weight clipping
    gp_scale = None  # no gradient penalty

    loss_d = -ng.log(D1) - ng.log(1 - D2)
    loss_g = -ng.log(D2)

    args.num_iterations = 10000
    args.plot_interval = 200

# Wasserstein GAN
if args.loss_type == "WGAN":

    generator = make_generator(bn=True)
    discriminator = make_discriminator(bn=True, disc_activation=None)

    # build network graph
    generated = generator(z)
    D1 = discriminator(image)
    D2 = discriminator(generated)

    weight_clip_value = 0.1  # apply weight clipping
    gp_scale = None  # no gradient penalty

    loss_d = D1 - D2
    loss_g = D2

    args.num_iterations = 200000
    args.plot_interval = 200

# WGAN with Gradient Penalty
if args.loss_type == "WGAN-GP":

    generator = make_generator(bn=False)
    discriminator = make_discriminator(bn=False, disc_activation=None)

    # build network graph
    generated = generator(z)
    D1 = discriminator(image)
    D2 = discriminator(generated)

    weight_clip_value = None  # no weight clipping
    gp_scale = 10  # gradient penalty coefficient

    loss_d = D1 - D2
    loss_g = D2
    args.num_iterations = 35000
    args.plot_interval = 200

# calculate gradient for all losses
# TODO: change constant 0.5 to uniform random
# once ng.uniform is fixed for GPU

#x = ng.variable(initial_value=0.5, axes=[])
#epsilon = ng.uniform(x)
epsilon = ng.constant(0.5)  # delete after uniform is fixed
interpolated = epsilon * image + (1 - epsilon) * generated  # delete after uniform is fixed
D3 = discriminator(interpolated)
gradient = ng.deriv(ng.sum(D3, out_axes=[]), interpolated)
grad_norm = ng.L2_norm(gradient)
gradient_penalty = ng.square(grad_norm - 1)

# add gradient penalty
if gp_scale:
    # TODO: Simply having 0*gradient_penalty does not work
    # This may be a problem in graph optimization
    loss_d = loss_d + gp_scale * gradient_penalty

mean_cost_d = ng.mean(loss_d, out_axes=[])
mean_cost_g = ng.mean(loss_g, out_axes=[])
mean_grad_norm = ng.mean(grad_norm, out_axes=[])

optimizer_d = make_optimizer(name='discriminator_optimizer', weight_clip_value=weight_clip_value)
optimizer_g = make_optimizer(name='generator_optimizer')
updates_d = optimizer_d(loss_d, subgraph=discriminator)
updates_g = optimizer_g(loss_g, subgraph=generator)

# compile computations
generator_train_inputs = {'noise': z}
discriminator_train_inputs = {'image': image, 'noise': z}

generator_train_outputs = {'batch_cost': mean_cost_g, 'updates': updates_g,
                           'generated': generated}  # for plots
discriminator_train_outputs = {'batch_cost': mean_cost_d, 'updates': updates_d,
                               'grad_norm': mean_grad_norm}

# create the dataloader
train_data, valid_data = MNIST(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size, args.num_iterations)

# noise source
noise_generator = Noise(shape=noise_dim + (args.batch_size,), seed=args.seed)

with closing(ngt.make_transformer()) as transformer:

    train_computation_g = make_bound_computation(transformer,
                                                 generator_train_outputs,
                                                 generator_train_inputs)
    train_computation_d = make_bound_computation(transformer,
                                                 discriminator_train_outputs,
                                                 discriminator_train_inputs)

    # train loop
    train_data = {'Discriminator Cost': [],
                  'Generator Cost': [],
                  'Log Gradient Norm': []}

    print('start train loop')
    for mb_idx in range(args.num_iterations):

        k = train_schedule(mb_idx)

        # update generator
        for iter_g in range(1):
            z_samp = noise_generator.next()
            output_g = train_computation_g({'noise': z_samp})

        # update discriminator
        for iter_d, datadict in enumerate(train_set):
            if iter_d == k:
                break
            z_samp = noise_generator.next()
            image_samp = get_image(datadict)
            output_d = train_computation_d({'noise': z_samp, 'image': image_samp})

        # save loss and gradient data to plot
        if mb_idx % 10 == 0:
            train_data['Discriminator Cost'].append([mb_idx, output_d['batch_cost']])
            train_data['Generator Cost'].append([mb_idx, output_g['batch_cost']])
            train_data['Log Gradient Norm'].append([mb_idx, np.log10(output_d['grad_norm'])])

        # print losses
        if mb_idx % 100 == 0:
            msg = ("Iteration {} complete.\nDiscriminator avg loss: {} "
                   "Generator avg loss: {} Gradient Norm: {}")
            print(msg.format(mb_idx,
                             float(output_d['batch_cost']),
                             float(output_g['batch_cost']),
                             float(output_d['grad_norm'])))

        # generate plots
        if create_plots:
            # save some examples from dataset
            if mb_idx == 0:
                save_plots(image_samp, 'mnist-examples.png', train_data, args,
                           title='mnist data examples')

            # save generated images
            if mb_idx % args.plot_interval == 0:
                output_g = train_computation_g({'noise': z_samp})
                generated = output_g['generated']
                pltname = 'dcgan-mnist-iter-{}.png'.format(mb_idx)
                title = 'iteration {} disc_cost{:.2E} gen_cost{:.2E}'.format(
                    mb_idx, float(output_d['batch_cost']), float(output_g['batch_cost']))
                save_plots(generated, pltname, train_data, args, title=title)
