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
MNIST Example with different losses: DCGAN, WGAN, WGAN with Gradient Penalty
usage(DCGAN): python mnist_gan.py -b gpu -z 64 -t 15000 --loss_type 'DCGAN'
usage(WGAN): python mnist_gan.py -b gpu -z 64 -t 30000 --loss_type 'WGAN'
usage(WGAN-GP): python mnist_gan.py -b gpu -z 64 -t 15000 --loss_type 'WGAN-GP'
"""

import numpy as np
from contextlib import closing
import ngraph as ng
import os
import ngraph.transformers as ngt
from ngraph.frontends.neon import (Sequential, Deconvolution, Convolution,
                                   Rectlin, Logistic, Tanh,
                                   Adam, ArrayIterator,
                                   KaimingInit, make_bound_computation)
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import MNIST
from ngraph.frontends.neon.logging import ProgressBar
from ngraph.util.names import name_scope
from utils import save_plots, get_image, train_schedule, Noise

# parse command line arguments
parser = NgraphArgparser()
parser.add_argument('--plot_interval', type=int, default=200,
                    help='Save generated images with a period of this many iterations')
parser.add_argument('--loss_type', default='WGAN-GP',
                    help='Loss Function', choices=['DCGAN', 'WGAN', 'WGAN-GP'])
parser.add_argument('--gp_scale', type=int, default=10,
                    help='Scale of the gradient penalty')
parser.add_argument('--w_clip', type=int, default=0.01,
                    help='Weight clipping value for WGAN')
parser.add_argument('--plot_dir', type=str, default='MNIST_Plots',
                    help='Directory name to save the results')
args = parser.parse_args()

if not os.path.isdir(args.plot_dir):
    os.makedirs(args.plot_dir)

# define noise dimension
noise_dim = (2, 1, 3, 3)

# common layer parameters
filter_init = KaimingInit()
relu = Rectlin(slope=0)
lrelu = Rectlin(slope=0.2)


# helper function
def make_optimizer(name=None, weight_clip_value=None):

    optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
                     epsilon=1e-8, weight_clip_value=weight_clip_value)

    return optimizer


# generator network
def make_generator(bn=True):
    # TODO
    # add affine before conv once that is corrected
    # https://github.com/NervanaSystems/private-ngraph/issues/2054
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

# DCGAN
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

# Wasserstein GAN
elif args.loss_type == "WGAN":

    generator = make_generator(bn=True)
    discriminator = make_discriminator(bn=True, disc_activation=None)

    # build network graph
    generated = generator(z)
    D1 = discriminator(image)
    D2 = discriminator(generated)

    weight_clip_value = args.w_clip  # apply weight clipping
    gp_scale = None  # no gradient penalty

    loss_d = D1 - D2
    loss_g = D2

# WGAN with Gradient Penalty
elif args.loss_type == "WGAN-GP":

    generator = make_generator(bn=False)
    discriminator = make_discriminator(bn=False, disc_activation=None)

    # build network graph
    generated = generator(z)
    D1 = discriminator(image)
    D2 = discriminator(generated)

    weight_clip_value = None  # no weight clipping
    gp_scale = args.gp_scale  # gradient penalty coefficient

    loss_d = D1 - D2
    loss_g = D2

# calculate gradient for all losses

# TODO
# change constant 0.5 to uniform random
# once ng.uniform is fixed for GPU
# https://github.com/NervanaSystems/private-ngraph/issues/2011
# x = ng.variable(initial_value=0.5, axes=[])
# epsilon = ng.uniform(x)
epsilon = ng.constant(0.5)  # delete after uniform is fixed
interpolated = epsilon * image + (1 - epsilon) * generated
D3 = discriminator(interpolated)

with name_scope(name="GradientPenalty"):
    gradient = ng.deriv(ng.sum(D3, out_axes=[]), interpolated)
    grad_norm = ng.L2_norm(gradient)
    gradient_penalty = ng.square(grad_norm - 1)

# add gradient penalty
# TODO
# when gp_scale is set to 0 the behavior is not as expected
# loss_d = loss_d + 0 * gp is not loss_d = loss_d + 0
# we can get rid of if statement once this is fixed
# https://github.com/NervanaSystems/private-ngraph/issues/2145
if gp_scale:
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
noise_generator = Noise(shape=noise_dim + (args.batch_size,), seed=args.rng_seed)

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

    progress_bar = ProgressBar(unit="iterations", ncols=100, total=args.num_iterations)

    for mb_idx in progress_bar(range(args.num_iterations)):

        k = train_schedule(mb_idx, args)

        # update generator
        for iter_g in range(1):
            z_samp = noise_generator.next()
            output_g = train_computation_g({'noise': z_samp})

        # update discriminator
        for iter_d in range(k):
            z_samp = noise_generator.next()
            datadict = train_set.next()
            image_samp = get_image(datadict)
            output_d = train_computation_d({'noise': z_samp, 'image': image_samp})

        # save loss and gradient data to plot
        if mb_idx % 10 == 0:
            train_data['Discriminator Cost'].append([mb_idx, output_d['batch_cost']])
            train_data['Generator Cost'].append([mb_idx, output_g['batch_cost']])
            train_data['Log Gradient Norm'].append([mb_idx, np.log10(output_d['grad_norm'])])

        # print losses
        msg = ("Disc. loss: {:.2f}, Gen. loss: {:.2f}, Grad Norm: {:.2f}")
        progress_bar.set_description(msg.format(float(output_d['batch_cost']),
                                                float(output_g['batch_cost']),
                                                float(output_d['grad_norm'])))

        # generate plots
        # save some examples from dataset
        if mb_idx == 0:
            save_plots(image_samp, 'mnist-examples.png', train_data, args,
                       title='mnist data examples')

        # save generated images
        if mb_idx % args.plot_interval == 0:
            output_g = train_computation_g({'noise': z_samp})
            generated = output_g['generated']
            pltname = '{}-mnist-iter-{}.png'.format(args.loss_type, mb_idx)
            title = 'iteration {} disc_cost{:.2E} gen_cost{:.2E}'.format(
                mb_idx, float(output_d['batch_cost']), float(output_g['batch_cost']))
            save_plots(generated, pltname, train_data, args, title=title)
