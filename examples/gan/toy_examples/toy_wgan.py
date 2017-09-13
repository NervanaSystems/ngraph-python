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
WGAN and Gradient Penalty toy example
Neon implementation of improved_wgan_training
adapted from https://github.com/igul222/improved_wgan_training

usage: python toy_wgan.py -b gpu -t 100000
"""
from contextlib import closing
import ngraph.transformers as ngt
from ngraph.frontends.neon import Adam, Affine, Rectlin, Sequential
from ngraph.frontends.neon import ConstantInit, KaimingInit
from ngraph.frontends.neon import make_bound_computation, NgraphArgparser
from ngraph.frontends.neon.logging import ProgressBar
import ngraph as ng
import os
import numpy as np
from toy_utils import DataGenerator, NormalNoise, generate_plot


parser = NgraphArgparser()
parser.add_argument('--plot_interval', type=int, default=200,
                    help='Plot results every this many iterations')
parser.add_argument('--loss_type', type=str, default='WGAN-GP',
                    help='Choose loss type', choices=['WGAN-GP', 'WGAN'])
parser.add_argument('--gp_scale', type=int, default=1,
                    help='Scale of the gradient penalty')
parser.add_argument('--w_clip', type=int, default=0.01,
                    help='Weight clipping value for WGAN')
parser.add_argument('--data_type', type=str, default='Roll',
                    help='Choose ground truth distribution', 
                    choices=['Rectangular', 'Circular', 'Roll'])
parser.add_argument('--dim', type=int, default=512,
                    help='Hidden layer dimension for the model')
parser.add_argument('--num_critic', type=int, default=5,
                    help='Number of discriminator iterations per generator iteration')
parser.add_argument('--plot_dir', type=str, default='WGAN_Toy_Plots',
                    help='Directory name to save the results')

args = parser.parse_args()
np.random.seed(args.rng_seed)
dim = args.dim

if not os.path.isdir(args.plot_dir):
    os.makedirs(args.plot_dir)

w_init = KaimingInit()
b_init = ConstantInit()


def make_optimizer(name=None, weight_clip_value=None):
    optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
                     epsilon=1e-8, weight_clip_value=weight_clip_value)

    return optimizer


def make_generator(out_axis):

    generator = [Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(axes=out_axis, weight_init=w_init, bias_init=b_init, activation=None)]

    return Sequential(generator, name="Generator")


def make_discriminator():

    discriminator = [Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=1, weight_init=w_init, bias_init=b_init, activation=None)]

    return Sequential(discriminator, name="Discriminator")


noise_dim = 2
data_dim = 2
N = ng.make_axis(name='N', length=args.batch_size)
W = ng.make_axis(name='W', length=data_dim)
z_ax = ng.make_axes([ng.make_axis(name='H', length=noise_dim), N])
d_ax = ng.make_axes([W, N])

# make placeholders
z = ng.placeholder(axes=z_ax)
data = ng.placeholder(axes=d_ax)

generator = make_generator(out_axis=W)
discriminator = make_discriminator()

generated = generator(z)
D1 = discriminator(data)
D2 = discriminator(generated)

# TODO
# Original Implementation with epsilon - wait till fixed
# https://github.com/NervanaSystems/private-ngraph/issues/2011
# x = ng.variable(initial_value=0.5, axes=[])
# eps = ng.uniform(x)

eps = ng.constant(0.5)  # delete after uniform works
interpolated = eps * data + (1 - eps) * generated

D3 = discriminator(interpolated)
gradient = ng.deriv(ng.sum(D3, out_axes=[]), interpolated)
grad_norm = ng.L2_norm(gradient)
gradient_penalty = ng.square(grad_norm - 1)

if args.loss_type == "WGAN-GP":
    gp = args.gp_scale * gradient_penalty
    weight_clipping = None

elif args.loss_type == "WGAN":  # standard WGAN with no gradient penalty
    gp = None
    weight_clipping = args.w_clip

if gp:
    loss_d = D1 - D2 + gp
else:
    loss_d = D1 - D2

mean_cost_d = ng.mean(loss_d, out_axes=[])

loss_g = D2
mean_cost_g = ng.mean(loss_g, out_axes=[])

mean_grad_norm = ng.mean(grad_norm, out_axes=[])

optimizer_d = make_optimizer(name='discriminator_optimizer', weight_clip_value=weight_clipping)
optimizer_g = make_optimizer(name='generator_optimizer')
updates_d = optimizer_d(loss_d, subgraph=discriminator)
updates_g = optimizer_g(loss_g, subgraph=generator)

# noise and data generators
train_set = DataGenerator((data_dim, args.batch_size), 0, data_type=args.data_type)
noise_gen = NormalNoise((noise_dim, args.batch_size), 0)

# input and output dictionaries
gen_train_inputs = {'noise': z}
dis_train_inputs = {'data': data, 'noise': z}

gen_train_outputs = {'batch_cost': mean_cost_g, 'updates': updates_g,
                     'generated': generated}
dis_train_outputs = {'batch_cost': mean_cost_d, 'updates': updates_d,
                     'grad_norm': mean_grad_norm}

# training
with closing(ngt.make_transformer()) as transformer:

    train_computation_g = make_bound_computation(transformer,
                                                 gen_train_outputs,
                                                 gen_train_inputs)

    train_computation_d = make_bound_computation(transformer,
                                                 dis_train_outputs,
                                                 dis_train_inputs)

    train_data = {'Discriminator Cost': [],
                  'Generator Cost': [],
                  'Log Gradient Norm': []}

    progress_bar = ProgressBar(unit="iterations", ncols=100, total=args.num_iterations)

    for iteration in progress_bar(range(int(args.num_iterations))):

        for iter_g in range(1):
            noise_in = noise_gen.next()
            output_g = train_computation_g({'noise': noise_in})

        for iter_d in range(args.num_critic):
            noise_in = noise_gen.next()
            data_in = train_set.next()
            output_d = train_computation_d({'noise': noise_in, 'data': data_in})

        # save loss and gradient data to plot
        if iteration % 10 == 0:
            train_data['Discriminator Cost'].append([iteration, output_d['batch_cost']])
            train_data['Generator Cost'].append([iteration, output_g['batch_cost']])
            train_data['Log Gradient Norm'].append([iteration, np.log10(output_d['grad_norm'])])

        # report loss every 100 iterations
        msg = ("Disc. loss: {:.2f}, Gen. loss: {:.2f}, Grad Norm: {:.2f}")
        progress_bar.set_description(msg.format(float(output_d['batch_cost']),
                                                float(output_g['batch_cost']),
                                                float(output_d['grad_norm'])))

        if (iteration % args.plot_interval) == 0:
            generate_plot(args.plot_dir, iteration, data_in, output_g, output_d, train_data, args)
