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
WGAN on LSUN dataset

usage(WGAN)::
    python lsun_wgan.py -b gpu -z 32 -t 100000 --loss_type WGAN
usage(WGAN-GP)::
    python lsun_wgan.py -b gpu -z 32 -t 100000 --loss_type WGAN-GP
Builds the WGAN and Improved WGAN (WGAN with gradient penality (GP)) models on LSUN dataset
"""
import numpy as np
import os
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import (Adam, RMSProp, make_bound_computation)
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon.logging import ProgressBar
from ngraph.util.names import name_scope
from lsun_data import make_aeon_loaders
from utils import train_schedule, Noise, save_plots, get_image
from models_lsun import (make_generator_gp, make_generator,
                         make_discriminator_gp, make_discriminator)


def make_optimizer(name=None, weight_clip_value=None, loss_type="WGAN-GP"):
    if loss_type == "WGAN":
        optimizer = RMSProp(learning_rate=5e-5, decay_rate=0.99, epsilon=1e-8,
                            weight_clip_value=weight_clip_value)

    if loss_type == "WGAN-GP":
        optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-8,
                         weight_clip_value=weight_clip_value)

    return optimizer


parser = NgraphArgparser(description='WGAN on LSUN bedroom dataset')
parser.add_argument('--plot_interval', type=int, default=500,
                    help='display generated samples at this frequency')
parser.add_argument('--lsun_dir', default="/dataset/lsun", help='LSUN data directory')
parser.add_argument('--subset_pct', type=float, default=50.0,
                    help='subset of training dataset to use (percentage), default 50.0')
parser.add_argument('--loss_type', default='WGAN-GP',
                    help='Loss Function', choices=['WGAN', 'WGAN-GP'])
parser.add_argument('--im_size', type=int, default=64,
                    help='the size of image')
parser.add_argument('--plot_dir', type=str, default='LSUN_Plots',
                    help='Directory name to save the results')
args = parser.parse_args()

if not os.path.isdir(args.plot_dir):
    os.makedirs(args.plot_dir)

# noise source shape CHW
noise_dim = (100, 1, 1)

# noise placeholder
N = ng.make_axis(name='N', length=args.batch_size)
noise_ax_names = 'CHW'
noise_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                           for nm, l in zip(noise_ax_names, noise_dim)])
z_ax = noise_axes + N
z = ng.placeholder(axes=z_ax)


train_set = make_aeon_loaders(args.lsun_dir, args.batch_size,
                              args.num_iterations, subset_pct=100.0)
noise_generator = Noise(shape=noise_dim + (args.batch_size,), seed=args.rng_seed)

# image placeholder
image = train_set.make_placeholders()['image']

weight_clip_value = None
# build network graph
if args.loss_type == "WGAN":  # Wasserstein GAN
    generator = make_generator(bn=True)
    discriminator = make_discriminator(bn=True, disc_activation=None)

    weight_clip_value = 0.01
    gp_scale = None
elif args.loss_type == "WGAN-GP":  # Improved WGAN
    generator = make_generator_gp(bn=False, n_extra_layers=4)
    discriminator = make_discriminator_gp(bn=False, n_extra_layers=4, disc_activation=None)

    # build gradient penalty
    gp_scale = 10

generated = generator(z)
generated = ng.axes_with_order(generated, image.axes)
D_real = discriminator(image)
D_fake = discriminator(generated)


x = ng.variable(initial_value=0.5, axes=[])
epsilon = ng.uniform(x)
interpolated = epsilon * image + (1 - epsilon) * generated
D3 = discriminator(interpolated)

with name_scope(name="GradientPenalty"):
    gradient = ng.deriv(ng.sum(D3, out_axes=[]), interpolated)
    grad_norm = ng.L2_norm(gradient)
    gradient_penalty = ng.square(grad_norm - 1)

loss_d = D_real - D_fake
loss_g = D_fake
# add gradient penalty
if gp_scale:
    loss_d = loss_d + gp_scale * gradient_penalty

mean_cost_d = ng.mean(loss_d, out_axes=[])
mean_cost_g = ng.mean(loss_g, out_axes=[])
mean_grad_norm = ng.mean(grad_norm, out_axes=[])


# define Losses
optimizer_d = make_optimizer(name='discriminator_optimizer',
                             weight_clip_value=weight_clip_value, loss_type=args.loss_type)
optimizer_g = make_optimizer(name='generator_optimizer', loss_type=args.loss_type)
updates_d = optimizer_d(loss_d, subgraph=discriminator)
updates_g = optimizer_g(loss_g, subgraph=generator)

# compile computations
generator_train_inputs = {'noise': z}
discriminator_train_inputs = {'image': image, 'noise': z}

generator_train_outputs = {'batch_cost': mean_cost_g, 'updates': updates_g,
                           'generated': generated}  # for plots
discriminator_train_outputs = {'batch_cost': mean_cost_d, 'updates': updates_d,
                               'grad_norm': mean_grad_norm}  # check the gradient norm for GP


with closing(ngt.make_transformer()) as transformer:

    train_computation_g = make_bound_computation(transformer,
                                                 generator_train_outputs,
                                                 generator_train_inputs)
    train_computation_d = make_bound_computation(transformer,
                                                 discriminator_train_outputs,
                                                 discriminator_train_inputs)

    train_data = {'Discriminator Cost': [],
                  'Generator Cost': [],
                  'Log Gradient Norm': []}

    # train loop
    print('start train loop')

    progress_bar = ProgressBar(unit="iterations", ncols=100, total=args.num_iterations)
    for mb_idx in progress_bar(range(args.num_iterations)):

        k = train_schedule(mb_idx, args)

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
        msg = ("Discriminator avg loss: {:.5f} Generator avg loss: {:.5f} Gradient Norm: {:.5f}")
        progress_bar.set_description(msg.format(
            float(output_d['batch_cost']),
            float(output_g['batch_cost']),
            float(output_d['grad_norm'])))

        # plot some examples from dataset
        if mb_idx == 0:
            save_plots(image_samp,
                       'lsun-examples.png', train_data, args,
                       title='lsun bedroom data examples')

        # display some generated images
        if mb_idx % args.plot_interval == 0:
            generated = output_g['generated']  # N, C, H, W
            if args.loss_type == "WGAN":
                prefix = 'Originial'
            elif args.loss_type == "WGAN-GP":
                prefix = 'Improved'
            pltname = '{}-wgan-lsun-iter-{}.png'.format(prefix, mb_idx)
            title = 'iteration {} disc_cost{:.2E} gen_cost{:.2E}'.format(
                    mb_idx, float(output_d['batch_cost']), float(output_g['batch_cost']))
            save_plots(generated, pltname, train_data, args, title=title)
