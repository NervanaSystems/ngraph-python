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

import time
from tqdm import tqdm
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import (GaussianInit, GlorotInit, Convolution, Rectlin, Rectlinclip,
                                   BiRNN, GradientDescentMomentum, Affine, Identity)
from ngraph.frontends.neon import ax


parser = NgraphArgparser()
parser.add_argument("--num_rnns", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=20)
parser.add_argument("--sequence_length", type=int, default=30)
parser.add_argument("--lr", type=float, default=1.0e-5)
parser.add_argument("--momentum", type=float, default=0.99)

args = parser.parse_args()

batch_size = args.batch_size
num_rnn_layers = args.num_rnns
num_filters = args.hidden_size
hidden_size = args.hidden_size
rnn_hidden_size = args.hidden_size
max_audio_len = args.sequence_length
num_iterations = args.num_iterations

# Parameters
filter_width = 11  # Width of conv filters
str_w = 1  # Stride of conv filters
nout = 29
learning_rate = args.lr

# data parameters
max_utt_len = ((max_audio_len - 1) // str_w) + 1
max_lbl_len = (max_utt_len - 1) // 2
nbands = 13

# mimic aeon data format
dataset_info = {'valid_pct': {'1': 1, '0': 1}, 
                'trans_length': {'1': 1, '0': 1}, 
                'audio': {'channels': 1, 'frequency': nbands, 'time': max_audio_len},
                'transcription': {'character': 1, 'sequence': max_lbl_len}}

ax.N.length = batch_size
ax.Y.length = nout
ax.Y.name = 'nout'

# prepare inputs 
inputs = dict()
for placeholder_name, axis_info in dataset_info.items():
    p_axes = ng.make_axes([ax.N])
    for nm, sz in axis_info.items():
        p_axes += ng.make_axis(name=nm, length=sz)
    if placeholder_name == 'audio':
        inputs[placeholder_name] = ng.placeholder(p_axes)
    else:
        inputs[placeholder_name] = ng.placeholder(p_axes, np.dtype(np.int32))

# prepare lbls, utt_lens, lbl_lens in format required by warp-ctc
lbls = ng.flatten(inputs['transcription'])
utt_lens = ng.flatten(inputs['valid_pct'])
lbl_lens = ng.flatten(inputs['trans_length'])

# Ensure dtypes are as expected
lbls.dtype = np.dtype(np.int32)
utt_lens.dtype = np.dtype(np.int32)
lbl_lens.dtype = np.dtype(np.int32)

# generate data
ng_sample = dict()
ng_sample['audio'] = np.random.uniform(-1, 1, 
                                       inputs["audio"].axes.lengths
                                       ).astype(np.float32)
ng_sample['transcription'] = np.random.randint(
                                1, nout, inputs['transcription'].axes.lengths, 
                                dtype=np.int32)
ng_sample['valid_pct'] = 100 * np.ones(
                            inputs['valid_pct'].axes.lengths).astype(np.int32)
ng_sample['trans_length'] = np.random.randint(
                                max_lbl_len - 1, max_lbl_len + 1,  
                                inputs['trans_length'].axes.lengths, 
                                dtype=np.int32)

# Initializers
gauss = GaussianInit(0.1)
glorot = GlorotInit()

# 1D Convolution layer
padding = dict(pad_h=0, pad_w=filter_width // 2, pad_d=0)
strides = dict(str_h=1, str_w=str_w, str_d=1)
dilation = dict(dil_d=1, dil_h=1, dil_w=1)

conv_layer = Convolution((nbands, filter_width, num_filters),
                         gauss,
                         padding=padding,
                         strides=strides,
                         dilation=dilation,
                         activation=Rectlin())

output = conv_layer(inputs['audio'])
output = ng.map_roles(output, {"time": "REC"})

# Add one or more BiRNN layers
for ii in range(num_rnn_layers):
    rnn = BiRNN(rnn_hidden_size, init=glorot,
                activation=Rectlinclip(slope=0),
                reset_cells=True, return_sequence=True,
                concat_out=(ii == (num_rnn_layers - 1)))
    output = rnn(output)

# Add a single affine layer
fc = Affine(nout=hidden_size, weight_init=glorot,
            activation=Rectlinclip(slope=0.0))
output = fc(output)

# Add the final affine layer
final = Affine(axes=ax.Y, weight_init=glorot,
               activation=Identity())
output = final(output)

# dimshuffle final layer activations for warp-ctc
time_axis = output.axes.find_by_name('REC')[0]
assert time_axis.is_recurrent
warp_axes = ng.make_axes([time_axis, ax.N, ax.Y]) 
warp_activations = ng.axes_with_order(output, warp_axes)

# set up ctc loss
loss = ng.ctc(warp_activations, lbls, utt_lens, lbl_lens)

optimizer = GradientDescentMomentum(learning_rate, 
                                    momentum_coef=args.momentum,
                                    nesterov=True)
start = time.time()
updates = optimizer(loss)
stop = time.time()
print("Optimizer graph creation took {} seconds".format(stop - start))
mean_cost = ng.sequential([updates, ng.mean(loss, out_axes=())])

# Create a transformer and bind computation
transformer = ngt.make_transformer()
train_computation = transformer.computation(mean_cost,
                                            inputs["audio"],
                                            inputs["transcription"],
                                            inputs["valid_pct"],
                                            inputs["trans_length"])

# Measure how long it takes to initialize transformer
start = time.time()
transformer.initialize()
stop = time.time()
print("Initializing transformer took {} seconds".format(stop - start))

# Setup progress bar for logging, metrics, etc.
tpbar = tqdm(unit=" batches", ncols=100, total=args.num_iterations)
interval_cost = 0.0

# Train loop
for iteration in range(num_iterations):
    av_loss = train_computation(ng_sample["audio"], ng_sample["transcription"],
                                ng_sample["valid_pct"], ng_sample["trans_length"])

    tpbar.update(1)
    tpbar.set_description("Training cost {:0.4f}".format(av_loss[()]))
    interval_cost += av_loss[()]
    if (iteration + 1) % args.iter_interval == 0 and iteration > 0:
        tqdm.write("Interval {interval} Iteration {iteration} complete. "
                   "Avg Train Cost {cost:0.4f}".format(
                        interval=iteration // args.iter_interval,
                        iteration=iteration,
                        cost=interval_cost / args.iter_interval))
        interval_cost = 0.0
