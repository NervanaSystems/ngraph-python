# GAN
# following example code from https://github.com/AYLIEN/gan-intro
# MLP generator and discriminator 
# toy example with 1-D Gaussian data distribution


# issues:
# - cross entropy binary axes mismatch between D1 and D2
# TODO
#  - discriminator pretraining
#  - optimizer schedule    

import numpy as np

import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import Affine, Sequential
from ngraph.frontends.neon import Rectlin, Identity, Tanh, Logistic
from ngraph.frontends.neon import GaussianInit, ConstantInit
from ngraph.frontends.neon import GradientDescentMomentum, Schedule
from ngraph.frontends.neon import ArrayIterator
from ngraph.frontends.neon import make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from toygan import ToyGAN

# define commonly used layer in this example
def affine_layer(h_dim, activation, name):
    return Affine(nout=h_dim, 
                  activation=activation,
                  weight_init=GaussianInit(var=1.0),
                  bias_init=ConstantInit(val=0.0),
                  name=name)


#  model parameters
h_dim = 4  # GAN.mlp_hidden_size
h_dim_G = h_dim
h_dim_D = 2 * h_dim
minibatch_discrimination = False  # for this toy example, seems to be better w/o mb discrim?

num_iterations = 940
batch_size = 12
num_examples = num_iterations*batch_size

# 1. generator
# use relu instead of softplus (focus on porting infrastucture fundamental to GAN)
# fixed 1200 training iterations: relu retains distribution width but box without peak at mean; 
# early stopping at 940 iterations, cost of 0.952, 3.44 looks better
generator = Sequential([affine_layer(h_dim, Rectlin(), name='g0'),
                        affine_layer(1, Identity(), name='g1')])

# 2. discriminator (not implementing minibatch discrimination right now)
discriminator_layers = [affine_layer(2 * h_dim, Tanh(), name='d0'),
                        affine_layer(2 * h_dim, Tanh(), name='d1')]
if minibatch_discrimination:
    raise NotImplementedError
else:
    discriminator_layers.append(affine_layer(2 * h_dim, Tanh(), name='d2'))
discriminator_layers.append(affine_layer(1, Logistic(), name='d3'))
discriminator = Sequential(discriminator_layers)

# 3. TODO discriminator pre-training - skip for now, more concerned with graph infrastructure changes
# TODO: try taking pre-training out from TF example (get worse result shown in blog animation?)

# 4. optimizer
# TODO: set up exponential decay schedule and other optimizer parameters
def make_optimizer(name=None):
    learning_rate = 0.005 if minibatch_discrimination else 0.03
    schedule = Schedule()
    optimizer = GradientDescentMomentum(learning_rate, 
                momentum_coef=0.0,
                stochastic_round=False,
                wdecay=0.0,
                gradient_clip_norm=None,
                gradient_clip_value=None,
                name=name,
                schedule=schedule)	
    return optimizer

# 5. dataloader
toy_gan_data = ToyGAN(num_examples)  # use all default parameters, which are the ones from example TF code
train_data = toy_gan_data.load_data()
train_set = ArrayIterator(train_data, batch_size, num_iterations)  # since num_examples = batch_size*num_iterations, providing num_iterations kw arg is redundant

# 6. create model (build network graph)

# neon frontend interface:
# inputs dict would created by ArrayIterator make_placeholders method
inputs = train_set.make_placeholders()

# this does not work. haven't specified axes correctly (batch)
# (batch, sample)
#inputs = {'data_sample': ng.placeholder(()),
#	      'noise_sample': ng.placeholder(())}

# generated sample
z = inputs['noise_sample']
G = generator.train_outputs(z)  # generated sample

# discriminator
x = inputs['data_sample']
# *** does this work with ngraph, using discriminator for two outputs?
D1 = discriminator.train_outputs(x)  # discriminator output on real data sample
D2 = discriminator.train_outputs(G)  # discriminator output on generated sample

# why does ngraph have both log (LogOp) and safelog?

loss_d = -ng.log(D1) - ng.log(1 - D2)  # use cross_entropy_binary?
# ** cross_entropy_binary causes error - axes of D1 and D2 don't match
#loss_d = ng.cross_entropy_binary(D1, D2)  # TODO: come back to this: this is: - log(D1)*D2 - log(1 - D1)*(1-D2) with sigmoid optimization
					  # TODO: not sure about enable_sig_opt, enable_diff_opt
mean_cost_d = ng.mean(loss_d, out_axes=[])  # difference betw using out_axes and reduction_axes?
loss_g = -ng.log(D2)
mean_cost_g = ng.mean(loss_g, out_axes=[])

optimizer_d = make_optimizer(name='discriminator_optimizer')
optimizer_g = make_optimizer(name='generator_optimizer')
updates_d = optimizer_d(loss_d)
updates_g = optimizer_g(loss_g)

discriminator_train_outputs = {'batch_cost': mean_cost_d,
		 	                   'updates': updates_d}
generator_train_outputs = {'batch_cost': mean_cost_g,
	         	           'updates': updates_g}

transformer = ngt.make_transformer()
train_computation_g = make_bound_computation(transformer, generator_train_outputs, inputs)  # TODO: G inputs just z - does this matter?
train_computation_d = make_bound_computation(transformer, discriminator_train_outputs, inputs)

# with current graph design, inference outputs need to be defined before any computations are run
discriminator_inference_output = discriminator.inference_outputs(x)
generator_inference_output = generator.inference_outputs(z)

discriminator_inference = transformer.computation(discriminator_inference_output, x)  # this syntax feels funny, with x repeated from inference_outputs
generator_inference = transformer.computation(generator_inference_output, z)

# support variable rate training of discriminator and generator
k = 1  # number of discriminator training iterations (in general may be > 1, for example in WGAN paper)

# 7. train loop
# train_set yields data which is a dictionary of named input values ('data_sample' and 'noise_sample')
iter_interval = 100
for mb_idx, data in enumerate(train_set):
    # update discriminator (trained at a different rate than generator, if k > 1)
    for iter_d in range(k):
        # ** if use cross_entropy_binary for loss_d, errors out here, adjoint axes do not match error
        batch_output_d = train_computation_d(data)  # batch_cost and updates for discriminator
    # update generator
    # ? what happens when give an unneeded input to a computation? does it just benignly ignore it? 
    # ? for example, if input given to train_computation_g is just data dict which has both noise_sample and unneeded data_sample
    #batch_output_g = train_computation_g(dict(noise_sample=data['noise_sample']))  # since train_computation_d was created with data_sample key, expects it here too
    batch_output_g = train_computation_g(data)
    if mb_idx % iter_interval == 0:
        msg = "Iteration {} complete. Discriminator avg loss: {} Generator avg loss: {}"
        print(msg.format(mb_idx + 1, float(batch_output_d['batch_cost']), float(batch_output_g['batch_cost'])))

# 8. visualize generator results

# this is basically copied from blog TF code
nrange = toy_gan_data.noise_range 
num_points = 10000
num_bins = 100
bins = np.linspace(-nrange, nrange, num_bins)

# decision boundary - discriminator output on real data distribution samples
xs = np.linspace(-nrange, nrange, num_points)
db = np.zeros((num_points, 1))
for i in range(num_points // batch_size):
    sl = slice(i*batch_size, (i+1)*batch_size)
    inp = xs[sl].reshape(batch_size, 1)
    db[sl] = discriminator_inference(inp).reshape(batch_size, 1)  # * returned in shape (1, batch_size), tripped over this

# data distribution 
d = toy_gan_data.data_samples(num_points)
pd, i_pd = np.histogram(d, bins=bins, density=True)

# generated samples
zs = np.linspace(-nrange, nrange, num_points)
g = np.zeros((num_points, 1))
for i in range(num_points // batch_size):
    sl = slice(i*batch_size, (i+1)*batch_size)
    g[sl] = generator_inference(zs[sl].reshape(batch_size, 1)).reshape(batch_size, 1)
pg, i_pg = np.histogram(g, bins=bins, density=True)

# save off data for plot generation 
import h5py
with h5py.File('simple_gan.h5', 'w') as f:
    f.create_dataset('decision_boundary', (len(db), 1), dtype=float)
    f['decision_boundary'][:] = db
    f.create_dataset('data_distribution', (len(pd), ), dtype=float)
    f['data_distribution'][:] = pd
    f.create_dataset('generated_distribution', (len(pg), ), dtype=float)
    f['generated_distribution'][:] = pg
    # distribution histograms indices
    f.create_dataset('data_dist_index', (len(i_pd), ), dtype=float)
    f['data_dist_index'][:] = i_pd
    f.create_dataset('generated_dist_index', (len(i_pg), ), dtype=float)
    f['generated_dist_index'][:] = i_pg
