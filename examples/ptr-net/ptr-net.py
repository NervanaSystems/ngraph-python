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
from ngraph.frontends.neon import UniformInit, RMSProp, ax, Tanh, Logistic
from ngraph.frontends.neon import NgraphArgparser, make_bound_computation
import ngraph.transformers as ngt
import numpy as np
# import argparse
from tsp import TSP
from custom_recurrent import Recurrent, LSTM
from tsp_seqarrayiter import TSPSequentialArrayIterator
from utils import first_example, save_plot

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--train_file', default='tsp5.txt',
                    choices=['tsp5.txt', 'tsp10.txt'],
                    help='specify training filename')
parser.add_argument('--test_file', default='tsp5_test.txt',
                    choices=['tsp5_test.txt', 'tsp10_test.txt'],
                    help='specify testing filename')
parser.add_argument('--lr', type=int, default=0.002, help='learning rate')
parser.add_argument('--hs', type=int, default=128, help='hidden unit size')
parser.set_defaults()
args = parser.parse_args()
args.batch_size = 128
args.num_iterations = 50000

gradient_clip_value = 2
num_features = 2  # for planer TSP, each city's location is represented by 2-d coordinate

# preprocess the TSP dataset
tsp = TSP(train_filename=args.train_file, test_filename=args.test_file)
print('Loading and preprocessing TSP data...')
tsp_data = tsp.load_data(nrows=1000000)

# take a look at the first TSP input-target example pair
one_input_example, one_target_example = first_example(tsp_data)
print('First input example = {}'.format(one_input_example))
print('First target example = {}'.format(one_target_example))


# number of time steps equal to number of points (cities) in each example
time_steps = one_input_example.shape[0]

# create iterator and placeholders for training data
train_set = TSPSequentialArrayIterator(tsp_data['train'],
                                    nfeatures=num_features,
                                    batch_size=args.batch_size,
                                    time_steps=time_steps,
                                    total_iterations=args.num_iterations)
inputs = train_set.make_placeholders()
ax.Y.length = time_steps
ax.N.length = args.batch_size

# weight initialization
init = UniformInit(low=-0.08, high=0.08)

# build computational graph
enc = LSTM(args.hs, init, activation=Tanh(), reset_cells=True,
            gate_activation=Logistic(), return_sequence=True)
dec = LSTM(args.hs, init, activation=Tanh(), reset_cells=True,
            gate_activation=Logistic(), return_sequence=True)

(enc_h_out, enc_c_out), (enc_h_out_seq, enc_c_out_seq) = enc(inputs['inp_txt'], return_cell_state=True)
_, dec_h_out_seq = dec(inputs['teacher_tgt'], init_state=(enc_h_out, enc_c_out), return_cell_state=False)



tmp_axis1 = ng.make_axis(length=args.hs, name='feature_axis')
tmp_axis2 = ng.make_axis(length=args.hs, name='tmp_axis2')
# stack_axis = ng.make_axis(length=time_steps, name='stack_axis')

W1 = ng.variable(axes=[tmp_axis1, tmp_axis2], initial_value=init)
W2 = ng.variable(axes=[tmp_axis1, tmp_axis2], initial_value=init)
v = ng.variable(axes=[tmp_axis2], initial_value=init)

rec_axis = dec_h_out_seq.axes.recurrent_axis()

# ptr-net model
# "two for loop approach"
u_list = []
for i in range(time_steps):
    u_i_list = []
    emb_dec = ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), axis=rec_axis, idx=i)

    for j in range(time_steps):
        emb_enc = ng.slice_along_axis(ng.dot(W1, enc_h_out_seq), axis=rec_axis, idx=j)
        score_out = ng.dot(v, ng.tanh(emb_enc + emb_dec))
        u_i_list.append(score_out)

    output_prob = ng.softmax(ng.stack(u_i_list, axis=ax.Y, pos=1), ax.Y)
    u_list.append(output_prob)

pointer_out = ng.stack(u_list, axis=rec_axis, pos=2)
# axis_0:5, N:10, REC:5

# "one for loop" approach
# u = []
# for i in range(time_steps):
#     score_out2 = ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), axis=rec_axis, idx=i)))
#     # REC: 5, N:10
#
#     output_prob2 = ng.softmax(score_out2, normalization_axes=rec_axis)
#     # REC: 5, N:10
#
#     copy = ng.cast_axes(output_prob2, ng.make_axes([ax.Y, ax.N]))
#
#     u.append(copy)
#
#
# u_stack = ng.stack(u, axis=rec_axis, pos=2)
# axis_0:5, N:10, REC:5

# direct approach
# score_out = ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.dot(W2, dec_h_out_seq)))
# # (6, 5, 128)
#
# test = ng.softmax(score_out)



# specify loss function, calculate loss and update weights
one_hot_target = ng.one_hot(inputs['tgt_txt'], axis=ax.Y)
# axis_0:5, N:10, REC:5

loss = ng.cross_entropy_multi(pointer_out,
                              one_hot_target,
                              usebits=True)

mean_cost = ng.mean(loss, out_axes=[])
optimizer = RMSProp(decay_rate=0.96, learning_rate=args.lr, epsilon=1e-6,
                    gradient_clip_value=gradient_clip_value)
updates = optimizer(loss)

# provide outputs for bound computation
# train_outputs = dict(batch_cost=mean_cost, updates=updates, output_prob=output_prob,
#                     score_out=score_out, W1=W1, W2=W2)

train_outputs = dict(batch_cost=mean_cost, updates=updates, loss = loss,
pointer_out=pointer_out, W1=W1, enc_h_out_seq=enc_h_out_seq,
dec_h_out_seq=dec_h_out_seq, monitor0_0 = ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 0), monitor0_1 = ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 1),
monitor1 = ng.dot(W1, enc_h_out_seq), monitor5_0 = ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 0)),
monitor5_1 = ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 1)),
monitor2_0 = ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 0))),
monitor2_1 = ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 1))),
monitor3_0 = ng.softmax(ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 0))), normalization_axes=rec_axis),
monitor3_1 = ng.softmax(ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 1))), normalization_axes=rec_axis),
monitor4_0 = ng.cast_axes(ng.softmax(ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 0))), normalization_axes=rec_axis), ng.make_axes([ax.Y, ax.N])),
monitor4_1 = ng.cast_axes(ng.softmax(ng.dot(v, ng.tanh(ng.dot(W1, enc_h_out_seq) + ng.slice_along_axis(ng.dot(W2, dec_h_out_seq), rec_axis, 1))), normalization_axes=rec_axis), ng.make_axes([ax.Y, ax.N])),
one_hot_target=one_hot_target)

######################
# Train Loop
with closing(ngt.make_transformer()) as transformer:
    # bind the computations
    train_computation = make_bound_computation(transformer, train_outputs, inputs)

    # iterate over training set
    niters = []
    loss = []
    # import ipdb; ipdb.set_trace()
    for idx, data in enumerate(train_set):
        train_output = train_computation(data)
        niter = idx + 1
        if niter % 2000 == 0:
            # print (train_output['u']-train_output['score_out']).mean()
            print('iteration = {}, train loss = {}'.format(niter, train_output['batch_cost']))
            # uncomment lines below to print the predicted target and true target
            # print('predicted target = {}'.format(np.argmax(train_output['u_stack'], axis=1)))
            # print('true target = {}'.format(tsp_data['train']['tgt_txt'][niter:(niter + args.batch_size)][:]))
            niters.append(niter)
            loss.append(train_output['batch_cost'])

save_plot(niters, loss, args)
