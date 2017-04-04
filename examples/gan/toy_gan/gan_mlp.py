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
GAN
following example code from https://github.com/AYLIEN/gan-intro
MLP generator and discriminator
toy example with 1-D Gaussian data distribution
"""

import numpy as np
from contextlib import closing

import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import Layer, Affine, Sequential
from ngraph.frontends.neon import Rectlin, Identity, Tanh, Logistic
from ngraph.frontends.neon import GaussianInit, ConstantInit
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import ArrayIterator
from ngraph.frontends.neon import make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from toygan import ToyGAN


def affine_layer(h_dim, activation, name):
    return Affine(nout=h_dim,
                  activation=activation,
                  weight_init=GaussianInit(var=1.0),
                  bias_init=ConstantInit(val=0.0),
                  name=name)


def make_optimizer(name=None):
    learning_rate = 0.005 if minibatch_discrimination else 0.03
    optimizer = GradientDescentMomentum(learning_rate, momentum_coef=0.0,
                                        wdecay=0.0, gradient_clip_norm=None,
                                        gradient_clip_value=None, name=name)
    return optimizer


parser = NgraphArgparser(description='MLP GAN example')
args = parser.parse_args()

#  model parameters
h_dim = 4
minibatch_discrimination = False
num_iterations = 600
batch_size = 12
num_examples = num_iterations * batch_size

# generator
g_scope = 'generator'
with Layer.variable_scope(g_scope):
    generator_layers = [affine_layer(h_dim, Rectlin(), name='g0'),
                        affine_layer(1, Identity(), name='g1')]
    generator = Sequential(generator_layers)

# discriminator
d_scope = 'discriminator'
with Layer.variable_scope(d_scope):
    discriminator_layers = [affine_layer(2 * h_dim, Tanh(), name='d0'),
                            affine_layer(2 * h_dim, Tanh(), name='d1')]
    if minibatch_discrimination:
        raise NotImplementedError
    else:
        discriminator_layers.append(affine_layer(2 * h_dim, Tanh(), name='d2'))
    discriminator_layers.append(affine_layer(1, Logistic(), name='d3'))
    discriminator = Sequential(discriminator_layers)

# TODO discriminator pre-training

# dataloader
np.random.seed(1)
toy_gan_data = ToyGAN(batch_size, num_iterations)
train_data = toy_gan_data.load_data()
train_set = ArrayIterator(train_data, batch_size, num_iterations)
# reset seed for weights
np.random.seed(2)

# build network graph
inputs = train_set.make_placeholders()

z = inputs['noise_sample']
G = generator(z)  # generated sample

x = inputs['data_sample']
D1 = discriminator(x)  # discriminator output on real data sample

# cast G axes into x
G_t = ng.axes_with_order(G, reversed(G.axes))
G_cast = ng.cast_axes(G_t, x.axes)

D2 = discriminator(G_cast)  # discriminator output on generated sample

loss_d = -ng.log(D1) - ng.log(1 - D2)
mean_cost_d = ng.mean(loss_d, out_axes=[])
loss_g = -ng.log(D2)
mean_cost_g = ng.mean(loss_g, out_axes=[])

optimizer_d = make_optimizer(name='discriminator_optimizer')
optimizer_g = make_optimizer(name='generator_optimizer')
updates_d = optimizer_d(loss_d, variable_scope=d_scope)
updates_g = optimizer_g(loss_g, variable_scope=g_scope)

discriminator_train_outputs = {'batch_cost': mean_cost_d, 'updates': updates_d}
generator_train_outputs = {'batch_cost': mean_cost_g, 'updates': updates_g}

with closing(ngt.make_transformer()) as transformer:

    train_computation_g = make_bound_computation(transformer,
                                                 generator_train_outputs,
                                                 {'noise_sample': inputs['noise_sample']})
    train_computation_d = make_bound_computation(transformer,
                                                 discriminator_train_outputs,
                                                 inputs)

    generator_inference = transformer.computation(G, z)

    # train loop
    k = 1  # variable rate training of discriminator, if k > 1
    for mb_idx, data in enumerate(train_set):
        # update discriminator
        for iter_d in range(k):
            batch_output_d = train_computation_d(data)
        # update generator
        batch_output_g = train_computation_g({'noise_sample': data['noise_sample']})
        # print losses
        if mb_idx % 100 == 0:
            msg = "Iteration {} complete. Discriminator avg loss: {} Generator avg loss: {}"
            print(msg.format(mb_idx,
                             float(batch_output_d['batch_cost']),
                             float(batch_output_g['batch_cost'])))

# visualize results
nrange = toy_gan_data.noise_range
num_points = 10000
num_bins = 100
bins = np.linspace(-nrange, nrange, num_bins)

# data distribution
d = toy_gan_data.data_samples(num_points, 1)
pd, i_pd = np.histogram(d, bins=bins, density=True)

# generated samples
zs = np.linspace(-nrange, nrange, num_points)
g = np.zeros((num_points, 1))
for i in range(num_points // batch_size):
    sl = slice(i * batch_size, (i + 1) * batch_size)
    g[sl] = generator_inference(zs[sl].reshape(batch_size, 1)).reshape(batch_size, 1)
pg, i_pg = np.histogram(g, bins=bins, density=True)

# create and save plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(pd, 'b', label='real data')
    plt.plot(pg, 'g', label='generated data')
    plt.legend(loc='upper left')
    plt.savefig('toygan.png')
    print("png saved")
except ImportError:
    print("needs matplotlib")
