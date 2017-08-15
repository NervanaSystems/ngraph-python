#!/usr/bin/env python
# ----------------------------------------------------------------------------
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
export CUDA_VISIBLE_DEVICES=3; python -i ./examples/inceptionv3/inceptionv3_mini.py -z 8 -t 10000000 -b gpu --iter_interval 2000
"""
import numpy as np
import pickle
from tqdm import tqdm
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import ax, RMSProp
from data import make_aeon_loaders
import inception

def loop_eval(dataset, computation, metric_names):
    """
    Function to calculate the loss metrics on the evaluation set
    dataset: aeon iterator object
    computation: evaluation set computations
    metric_names: metrics to compute for evaluation set 
    """
    dataset._dataloader.reset()
    all_results = None
    for data in dataset:
        data['image'] = data['image'] / 255. 
        feed_dict = {inputs[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(np.transpose(res)) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))

    reduced_results = {k: np.mean(v[:dataset._dataloader.ndata]) for k, v in all_results.items()if k != 'predictions'}
    return all_results, reduced_results

parser = NgraphArgparser(description=__doc__)
parser.add_argument('--mini', default=False, dest='mini', action='store_true',
                    help='If given, builds a mini version of Inceptionv3')
parser.set_defaults(batch_size=32, num_iterations=1000000, iter_interval=2000)
args = parser.parse_args()

# Number of outputs of last layer.
ax.Y.length = 1000  
# Directory where AEON manifest files (csv's) exist
manifest_dir = './'
# Directory where images exist
data_dir = '/dataset/aeon/I1K/i1k-extracted/'

# Build AEON data loader objects
train_set,valid_set=make_aeon_loaders(manifest_dir, args.batch_size,
                                      args.num_iterations, dataset='i1k',
                                      datadir=data_dir)
inputs=train_set.make_placeholders(include_iteration=True)

# Input size is 299 x 299 x 3
image_size = 299

# Build the network
inception = inception.Inception(mini=args.mini)

# Declare the optimizer
optimizer = RMSProp(learning_rate=.01, decay_rate=0.9, gradient_clip_value=5., epsilon=1.)

# Build the main and auxiliary loss functions
y_onehot = ng.one_hot(inputs['label'][:,0], axis=ax.Y)
train_prob_main = ng.cast_role(inception.seq2(inception.seq1(inputs['image']))[:,0,0,0,:], axes = y_onehot.axes)
train_loss_main = ng.cross_entropy_multi(train_prob_main, y_onehot)

train_prob_aux = ng.cast_role(inception.seq_aux(inception.seq1(inputs['image']))[:,0,0,0,:], axes=y_onehot.axes)
train_loss_aux = ng.cross_entropy_multi(train_prob_aux, y_onehot) 

batch_cost = ng.sequential([optimizer(train_loss_main + 0.4*train_loss_aux), ng.mean(train_loss_main, out_axes=())])
train_computation = ng.computation(batch_cost, 'all')

label_indices = inputs['label'][:, 0]
# Build the computations for inference (evaluation)
with Layer.inference_mode_on():
    inference_prob = ng.cast_role(inception.seq2(inception.seq1(inputs['image']))[:,0,0,0,:], axes = y_onehot.axes)
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, y_onehot)
    eval_loss_names = ['cross_ent_loss', 'misclass', 'predictions']
    eval_computation = ng.computation([eval_loss, errors, inference_prob], "all")

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
                    'eval_misclass': [], 'iteration': []}
    for step, data in enumerate(train_set):
        data['iteration'] = step
        # Scale the image to [0., .1]
        data['image'] = data['image'] / 255.
        feed_dict = {inputs[k]: data[k] for k in inputs.keys()}
        output = train_function(feed_dict=feed_dict)

        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output[()]))
        interval_cost += output[()]
        if (step + 1) % args.iter_interval == 0 and step > 0:
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f}".format(
                           interval=step // args.iter_interval,
                           iteration=step,
                           cost=interval_cost / args.iter_interval))
            #Calculate inference on the evaluation set
            all_results, eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
            predictions = all_results['predictions']

            #Save the training progression
            saved_losses['train_loss'].append(interval_cost / args.iter_interval)
            saved_losses['eval_loss'].append(eval_losses['cross_ent_loss'])
            saved_losses['eval_misclass'].append(eval_losses['misclass'])
            saved_losses['iteration'].append(step)
            pickle.dump( saved_losses, open( "losses.pkl", "wb" ) )
            interval_cost = 0.0

print('\n')
