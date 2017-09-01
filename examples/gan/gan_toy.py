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
"""
from contextlib import closing
import ngraph.transformers as ngt
from ngraph.frontends.neon import Adam, Affine, Rectlin, Sequential, Logistic
from ngraph.frontends.neon import ConstantInit, KaimingInit
from ngraph.frontends.neon import make_bound_computation, NgraphArgparser
import ngraph as ng
import os
import numpy as np
from data import *

dim = 512
batch_size = 256
num_iter = 1e5
num_critic = 5

plot_directory = "WGAN_Toy_Example_Plots"
if not os.path.isdir(plot_directory):
    os.makedirs(plot_directory)

data_type = "Roll"  # "Rectangular", "Circular", "Roll"

w_init = KaimingInit()
b_init = ConstantInit()


def make_optimizer(name=None, weight_clip_value=None):
    optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
                     epsilon=1e-8, weight_clip_value=weight_clip_value)

    return optimizer


def make_generator():

    generator = [Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                 Affine(nout=2, weight_init=w_init, bias_init=b_init, activation=None)]

    return Sequential(generator, name="Generator")


def make_discriminator():

    discriminator = [Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=dim, weight_init=w_init, bias_init=b_init, activation=Rectlin()),
                     Affine(nout=1, weight_init=w_init, bias_init=b_init, activation=None)]

    return Sequential(discriminator, name="Discriminator")
 

parser = NgraphArgparser()
parser.add_argument('--plot_interval', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--loss', type=str, default="WGAN-GP")  # WGAN, WGAN-GP
args = parser.parse_args()
args.rng_seed = 0
np.random.seed(args.rng_seed)
args.batch_size = batch_size

# make placeholders
N = ng.make_axis(name='N', length=args.batch_size)
ax_names = 'CDHW'

# noise placeholder
noise_dim = (1, 1, 1, 2)
noise_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                           for nm, l in zip(ax_names, noise_dim)])

z_ax = noise_axes + ng.make_axis(name='N', length=args.batch_size)
z = ng.placeholder(axes=z_ax)

# data placeholder
data_dim = (1, 1, 1, 2)
data_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                          for nm, l in zip(ax_names, data_dim)])
d_ax = data_axes + N
data = ng.placeholder(axes=d_ax)

generator = make_generator()
discriminator = make_discriminator()

generated = generator(z)
D1 = discriminator(data)

# match the axes of the generated data to input data
gen_cast = ng.cast_axes(generated, data.axes[3:])
D2 = discriminator(gen_cast)


grad_scale = 1  # gradient penalty multiplier

# TODO
# Original Implementation with epsilon - wait till fixed
#x = ng.variable(initial_value=0.5, axes=[])
#eps = ng.uniform(x)

eps = ng.constant(0.5)  # delete after uniform works
interpolated = eps * data + (1 - eps) * gen_cast  # delete after uniform works

D3 = discriminator(interpolated)
gradient = ng.deriv(ng.sum(D3, out_axes=[]), interpolated)
grad_norm = ng.L2_norm(gradient)
gradient_penalty = ng.square(grad_norm - 1)

if args.loss == "WGAN-GP":
    gp = grad_scale * gradient_penalty
    weight_clipping = None

elif args.loss == "WGAN":  # standard WGAN with no gradient penalty
    gp = None
    weight_clipping = 0.01

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
train_set = DataGenerator((1, 1, 1, 2, args.batch_size), 0, data_type=data_type)
noise_gen = NormalNoise((1, 1, 1, 2, args.batch_size), 0)

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

    for iteration in range(int(num_iter)):

        for iter_g in range(1):
            noise_in = noise_gen.next()
            output_g = train_computation_g({'noise': noise_in})

        for iter_d in range(num_critic):
            noise_in = noise_gen.next()
            data_in = train_set.next()
            output_d = train_computation_d({'noise': noise_in, 'data': data_in})

        # save loss and gradient data to plot
        if iteration % 10 == 0:
            train_data['Discriminator Cost'].append([iteration, output_d['batch_cost']])
            train_data['Generator Cost'].append([iteration, output_g['batch_cost']])
            train_data['Log Gradient Norm'].append([iteration, np.log10(output_d['grad_norm'])])

        # report loss every 100 iterations
        if iteration % 100 == 0:
            msg = ("Iteration {} complete. \n"
                   "Dis avg loss: {}, Gen avg loss: {}, Grad Norm:{}")
            print(msg.format(iteration, float(output_d['batch_cost']),
                             float(output_g['batch_cost']), float(output_d['grad_norm'])))

        if (iteration % 500) == 0:
            print("Generating Plot")
            generate_plot(plot_directory, iteration, data_in, output_g, output_d, train_data, args)
