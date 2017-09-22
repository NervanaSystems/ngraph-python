#!/usr/bin/env python

# Copyright 2015-2016 Nervana Systems Inc.
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
Usage:
export CUDA_VISIBLE_DEVICES=3; python ./inceptionv3.py -b gpu --mini -z 8 --optimizer_name rmsprop --grad_clip 100.

Inception v3 network based on:
https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py

Imagenet data needs to be downloaded and extracted from:
http://www.image-net.org/
"""
import numpy as np
import pickle
from tqdm import tqdm
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import ax, RMSProp, GradientDescentMomentum
from ngraph.util.names import name_scope
from data import make_aeon_loaders
import inception

def scale_set(image_set):
    """
    Given a batch of images, normalizes each image by subtracting the mean
    And dividing by the standard deviation
    Mean/Std is calculated for each image separately, among its pixels
    Mean/Std is calculated for each channel separately
    image_set: (batch_size, C, H, W)
    returns: scaled image_set (batch_size, C, H, W)
    """
    means = np.mean(image_set, axis=(2,3))
    stds = np.std(image_set, axis=(2,3))
 
    # Matrix that contains per channel means 
    sub_factor = means.reshape(image_set.shape[0],image_set.shape[1],1,1)
    sub_factor = np.repeat(sub_factor, image_set.shape[2], axis=2) 
    sub_factor = np.repeat(sub_factor, image_set.shape[3], axis=3) 

    # Matrix that contains per channel stds
    scale_factor = stds.reshape(image_set.shape[0],image_set.shape[1],1,1)
    scale_factor = np.repeat(scale_factor, image_set.shape[2], axis=2) 
    scale_factor = np.repeat(scale_factor, image_set.shape[3], axis=3) 
    scale_factor = (3*scale_factor + 1e-5)

    scaled_image = (image_set - sub_factor)/ scale_factor
    return scaled_image

def eval_loop(dataset, computation, metric_names):
    """
    Function to calculate the loss metrics on the evaluation set
    dataset: aeon iterator object
    computation: evaluation set computations
    metric_names: metrics to compute for evaluation set
    """
    dataset._dataloader.reset()
    all_results = None
    for data in dataset:
        data['image'] = scale_set(data['image'])
        feed_dict = {inputs[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(np.transpose(res)) for name, res
                           in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))

    reduced_results = {k: np.mean(v[:dataset._dataloader.ndata]) for k, v in
                       all_results.items() if k != 'predictions'}
    return all_results, reduced_results


parser = NgraphArgparser(description=__doc__)
parser.add_argument('--mini', default=False, dest='mini', action='store_true',
                    help='If given, builds a mini version of Inceptionv3')
parser.add_argument("--image_dir", default='/dataset/aeon/I1K/i1k-extracted/',
                    help="Path to extracted imagenet data")
parser.add_argument("--train_manifest_file", default='train-index-tabbed.csv',
                    help="Name of tab separated Aeon training manifest file")
parser.add_argument("--valid_manifest_file", default='val-index-tabbed.csv',
                    help="Name of tab separated Aeon validation manifest file")
parser.add_argument("--optimizer_name", default='rmsprop',
                    help="Name of optimizer (rmsprop or sgd)")
parser.add_argument('--grad_clip', type=float, default=1e9, help="Gradient Clip Value")
parser.set_defaults(batch_size=8, num_iterations=10000000, iter_interval=2000)
args = parser.parse_args()

# Number of outputs of last layer.
ax.Y.length = 1000
ax.N.length = args.batch_size

# Build AEON data loader objects
train_set, valid_set = make_aeon_loaders(train_manifest=args.train_manifest_file,
                                         valid_manifest=args.valid_manifest_file,
                                         batch_size=args.batch_size,
                                         train_iterations=args.num_iterations,
                                         dataset='i1k',
                                         datadir=args.image_dir)
inputs = train_set.make_placeholders(include_iteration=True)

# Input size is 299 x 299 x 3
image_size = 299

# Build the network
inception = inception.Inception(mini=args.mini)

# Declare the optimizer
if args.optimizer_name == 'sgd':
    learning_rate_policy = {'name': 'schedule',
                            'schedule': list(7000*np.arange(1, 10, 1)),
                            'gamma': 0.8,
                            'base_lr': 0.1}

    optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                        momentum_coef=0.9,
                                        gradient_clip_value=args.grad_clip,
                                        wdecay=4e-5,
                                        iteration=inputs['iteration'])
elif args.optimizer_name == 'rmsprop': 
    learning_rate_policy = {'name': 'schedule',
                            'schedule': list(80000*np.arange(1, 10, 1)),
                            'gamma': 0.94,
                            'base_lr': 0.01}
    optimizer = RMSProp(learning_rate=learning_rate_policy, 
                        wdecay=4e-5, decay_rate=0.9,
                        gradient_clip_value=args.grad_clip, epsilon=1.)

else:
    raise NotImplementedError("Unrecognized Optimizer")

# Build the main and auxiliary loss functions
y_onehot = ng.one_hot(inputs['label'][:, 0], axis=ax.Y)
train_prob_main = inception.seq2(inception.seq1(inputs['image']))
train_prob_main = ng.map_roles(train_prob_main, {"C": ax.Y.name})
train_loss_main = ng.cross_entropy_multi(train_prob_main, y_onehot)

train_prob_aux = inception.seq_aux(inception.seq1(inputs['image']))
train_prob_aux = ng.map_roles(train_prob_aux, {"C": ax.Y.name})
train_loss_aux = ng.cross_entropy_multi(train_prob_aux, y_onehot)

# Compute the gradient of output to input
#with name_scope(name="GradientPenalty"):
#    gradient = ng.deriv(ng.sum(train_prob_main, out_axes=[]), inputs['image'])

batch_cost = ng.sequential([optimizer(train_loss_main + 0.4 * train_loss_aux),
                            ng.mean(train_loss_main, out_axes=())])
#train_computation = ng.computation([batch_cost, gradient], 'all')
train_computation = ng.computation([batch_cost], 'all')
label_indices = inputs['label'][:, 0]

# Build the computations for inference (evaluation)
with Layer.inference_mode_on():
    inference_prob = inception.seq2(inception.seq1(inputs['image']))
    slices = [0 if cx.name in ("H", "W") else slice(None) for cx in inference_prob.axes]
    inference_prob = ng.tensor_slice(inference_prob, slices)
    inference_prob = ng.map_roles(inference_prob, {"C": "Y"})
    """
    inference_prob = ng.cast_role(inception.seq2(inception.seq1(inputs['image']))[:, 0, 0, 0, :],
                                  axes=y_onehot.axes)
    """
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, y_onehot)
    eval_loss_names = ['cross_ent_loss', 'misclass', 'predictions']
    eval_computation = ng.computation([eval_loss, errors, inference_prob], "all")

#grads_array = [1e9]*2*args.iter_interval
with closing(ngt.make_transformer()) as transformer:
    train_function = transformer.add_computation(train_computation)
    eval_function = transformer.add_computation(eval_computation)

    if args.no_progress_bar:
        ncols = 0
    else:
        ncols = 100

    tpbar = tqdm(unit="batches", ncols=ncols, total=args.num_iterations)
    interval_cost = 0.0
    saved_losses = {'train_loss': [], 'eval_loss': [],
                    'eval_misclass': [], 'iteration': [], 'grads': []}
    for iter_no, data in enumerate(train_set):
        data = dict(data)
        data['iteration'] = iter_no
        # Scale the image to [0., .1]
        data['image'] = scale_set(data['image'])
        data['label'] = data['label'].reshape((args.batch_size, 1))
        feed_dict = {inputs[k]: data[k] for k in inputs.keys()}
        #output, grads = train_function(feed_dict=feed_dict)
        output = train_function(feed_dict=feed_dict)
        output = float(output[0])
        """
        # Mean grads over channel and batch axis
        grads = np.mean(grads, axis=(0,1)).astype(np.float16)
        grads_array.pop(2*args.iter_interval-1)
        grads_array.insert(0, grads)
        """
        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output))
        interval_cost += output
        if (iter_no + 1) % args.iter_interval == 0 and iter_no > 0:
            interval_cost = interval_cost / args.iter_interval
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f}".format(
                           interval=iter_no // args.iter_interval,
                           iteration=iter_no,
                           cost=interval_cost))
            # Calculate inference on the evaluation set
            # all_results, eval_losses = eval_loop(valid_set, eval_function, eval_loss_names)
            # predictions = all_results['predictions']
            # saved_losses['eval_loss'].append(eval_losses['cross_ent_loss'])
            # saved_losses['eval_misclass'].append(eval_losses['misclass'])

            # Save the training progression
            saved_losses['train_loss'].append(interval_cost)
            saved_losses['iteration'].append(iter_no)
            #saved_losses['grads'] = grads_array
            pickle.dump(saved_losses, open("losses_%s_%s.pkl" % (args.optimizer_name, args.backend), "wb"))
            interval_cost = 0.0

            # If training loss wildly increases in two past iterations, stop training
            if(iter_no > (2*args.iter_interval) ):
                if ((saved_losses['train_loss'][-1] - .1) > saved_losses['train_loss'][-2]):
                    print('Train Loss increased significantly!')

print('\n')
