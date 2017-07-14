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
Approximate Planer Traveling Salesman Problem using Pointer Networks
Reference paper: https://arxiv.org/pdf/1506.03134.pdf
"""

from contextlib import closing
import ngraph as ng
from ngraph.frontends.neon import UniformInit, RMSProp, ax, Tanh
from ngraph.frontends.neon import NgraphArgparser, make_bound_computation
import ngraph.transformers as ngt
import numpy as np
from tsp import TSP
from custom_recurrent import Recurrent
from tsp_seqarrayiter import TSPSequentialArrayIterator
from utils import get_first_example

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--train_file', default='tsp5.txt',
                    choices=['tsp5.txt', 'tsp10.txt'],
                    help='specify training filename')
parser.add_argument('--test_file', default='tsp5_test.txt',
                    choices=['tsp5_test.txt', 'tsp10_test.txt'],
                    help='specify training filename')
parser.set_defaults()
args = parser.parse_args()
args.batch_size = 50
args.num_iterations = 40000

hidden_size = 64
gradient_clip_value = 15
num_features = 2  # for planer TSP, each city's location is represented by 2-d coordinate

# preprocess the TSP dataset
tsp = TSP(train_filename=args.train_file, test_filename=args.test_file)
print('Loading and preprocessing TSP data...')
tsp_data = tsp.load_data()

# take a look at the first TSP input-target example pair
one_input_example, one_target_example = get_first_example(tsp_data)
print('First input example = {}'.format(one_input_example))
print('First target example = {}'.format(one_target_example))

# number of time steps equal to number of points (cities) in a example
time_steps = one_input_example.shape[0]

# create iterator and placeholders for training data
train_set = TSPSequentialArrayIterator(tsp_data['train'],
                                    nfeatures=num_features,
                                    batch_size=args.batch_size,
                                    time_steps=time_steps,
                                    total_iterations=args.num_iterations,
                                    get_prev_target=True)
inputs = train_set.make_placeholders()
ax.Y.length = time_steps + 1

# weight initialization
init = UniformInit(low=-0.08, high=0.08)

# build computational graph
enc = Recurrent(hidden_size, init, activation=Tanh(), reset_cells=True, return_sequence=True)
dec = Recurrent(hidden_size, init, activation=Tanh(), reset_cells=True, return_sequence=True)

enc_out, enc_out_seq = enc(inputs['inp_txt'])
_, dec_out_seq = dec(inputs['prev_tgt'], init_state=enc_out)

tmp_axis1 = ng.make_axis(length=hidden_size, name='feature_axis')
tmp_axis2 = ng.make_axis(length=hidden_size, name='tmp_axis2')

W1 = ng.variable(axes=[tmp_axis1, tmp_axis2], initial_value=init)
W2 = ng.variable(axes=[tmp_axis1, tmp_axis2], initial_value=init)
v = ng.variable(axes=[tmp_axis2, ax.Y], initial_value=init)

score_out = ng.dot(v, ng.tanh(ng.dot(W1, enc_out_seq) + ng.dot(W2, dec_out_seq)))
output_prob = ng.softmax(score_out)

# specify loss function, calculate loss and update weights
one_hot_target = ng.one_hot(inputs['tgt_txt'], axis=ax.Y)
loss = ng.cross_entropy_multi(output_prob,
                              one_hot_target,
                              usebits=True)

mean_cost = ng.mean(loss, out_axes=[])
optimizer = RMSProp(decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                    gradient_clip_value=gradient_clip_value)
updates = optimizer(loss)

# provide outputs for bound computation
train_outputs = dict(batch_cost=mean_cost, updates=updates, output_prob=output_prob)

######################
# Train Loop
with closing(ngt.make_transformer()) as transformer:
    # bind the computations
    train_computation = make_bound_computation(transformer, train_outputs, inputs)

    # import ipdb; ipdb.set_trace
    # iterate over training set
    for idx, data in enumerate(train_set):
        train_output = train_computation(data)
        niter = idx + 1
        if niter % 4000 == 0:
            print('iteration = {}, train loss = {}'.format(niter, train_output['batch_cost']))
            # uncomment lines below to print the predicted target and true target
            print('predicted target = {}'.format(np.argmax(train_output['output_prob'], axis=0).T))
            print('true target = {}'.format(tsp_data['train']['tgt_txt'][niter:(niter + args.batch_size)][:]))
