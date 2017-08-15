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
Usage:
    python timeseries_lstm.py -b gpu -z 64
    python timeseries_lstm.py -b gpu -z 64 --predict_seq

Builds an LSTM network to predict the next value in a timeseries
Lissajous curves are used as the timeseries to test the network
In addition, generates future values of the sequence based on an initial seed
    (See the sequence generation section for details, found later in the code)

    --predict_seq option given: Given prior seq_len points, predict next seq_len points
        1) Build sequence [x0, x1, x2, ... , xN]
        2) Split sequence into non-overlapping windows of length seq_len
            S0: [x0, x1, ..., x(seq_len-1)]
            S1: [x(seq_len), ..., x(2*seq_len - 1)]
            ...
        3) Build input sequence - ground truth output sequence pairs
            S0 input, [x1, x2, ..., x(seq_len)] output
            S1 input, [x(seq_len+1), x(seq_len+2), ..., x(2*seq_len)] output
            ...
        4) Use a portion of the input/output pairs for training, remainder for test

    --predict_seq option not given: Given prior seq_len points, predict next point
        1) Build sequence [x0, x1, x2, ... , xN]
        2) Build overlapping sequences of length seq_len
            S0: [x0, x1, ..., x(seq_len-1)]
            S1: [x1, x2, ..., x(seq_len)]
            ...
        3) Build input sequence / output single time point pairs
            S0 input, x(seq_len) output
            S1 input, x(seq_len+1) output
            ...
        4) Use a portion of the input/output pairs for training, remainder for test
"""

from __future__ import division, print_function
from contextlib import closing
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, LSTM, Affine
from ngraph.frontends.neon import UniformInit, Tanh, Logistic, Identity, Adam
from ngraph.frontends.neon import NgraphArgparser, loop_train
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator
import timeseries
import utils
import imp

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--predict_seq', default=False, dest='predict_seq', action='store_true',
                    help='If given, seq_len future timepoints are predicted')
parser.add_argument('--look_ahead', type=int,
                    help="Number of time steps to start predicting from",
                    default=1)
parser.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=32)
parser.add_argument('--epochs', type=int,
                    help="Number of epochs",
                    default=200)
parser.set_defaults()
args = parser.parse_args()

# Plot the inference / generation results
do_plots = True
try:
    imp.find_module('matplotlib')
except ImportError:
    do_plots = False

# Feature dimension of the input (for Lissajous curve, this is 2)
feature_dim = 2
# Output feature dimension (for Lissajous curve, this is 2)
output_dim = 2
# Number of recurrent units in the network
recurrent_units = 32
# Define initialization
init_uni = UniformInit(-0.1, 0.1)

batch_size = args.batch_size
seq_len = args.seq_len
look_ahead = args.look_ahead
predict_seq = args.predict_seq
# Total epochs of training
no_epochs = args.epochs
no_cycles = 2000
no_points = 13

# Calculate how many batches are in data
no_batches = no_cycles * no_points // seq_len // batch_size

# Generate Lissajous Curve
data = timeseries.TimeSeries(train_ratio=0.8,  # ratio of samples to set aside for training
                             seq_len=seq_len,    # length of the sequence in each sample
                             npoints=no_points,  # number of points to take in each cycle
                             ncycles=no_cycles,  # number of cycles in the curve
                             batch_size=batch_size,
                             curvetype='Lissajous2',
                             predict_seq=predict_seq,  # set True if you want sequences as output
                             look_ahead=look_ahead)  # number of time steps to look ahead

# Build input data iterables
# Yields an input array of Shape (batch_size, seq_len, input_feature_dim)
num_iterations = no_epochs * no_batches
train_set = ArrayIterator(data.train, batch_size, total_iterations=num_iterations)
test_set = ArrayIterator(data.test, batch_size)

# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=feature_dim, name="feature_axis")
out_axis = ng.make_axis(length=output_dim, name="output_axis")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
if(predict_seq is True):
    out_axes = ng.make_axes([batch_axis, time_axis, out_axis])
else:
    out_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = {'X': ng.placeholder(in_axes), 'y': ng.placeholder(out_axes),
          'iteration': ng.placeholder(axes=())}

# Network Definition
seq1 = Sequential([LSTM(nout=recurrent_units, init=init_uni, backward=False,
                   activation=Logistic(), gate_activation=Tanh(), return_sequence=predict_seq),
                   Affine(weight_init=init_uni, bias_init=init_uni,
                   activation=Identity(), axes=out_axis)])

# Optimizer
# Following policy will set the initial learning rate to 0.05 (base_lr)
# At iteration (num_iterations // 5), learning rate is multiplied by gamma (new lr = .005)
# At iteration (num_iterations // 2), it will be reduced by gamma again (new lr = .0005)
schedule = [num_iterations // 5, num_iterations // 2]
learning_rate_policy = {'name': 'schedule',
                        'schedule': schedule,
                        'gamma': 0.1,
                        'base_lr': 0.05}
optimizer = Adam(learning_rate=learning_rate_policy,
                 iteration=inputs['iteration'], gradient_clip_value=1)

# Define the loss function (squared L2 loss)
fwd_prop = seq1(inputs['X'])
train_loss = ng.squared_L2(fwd_prop - inputs['y'])

# Cost calculation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

# Forward prop of test set
# Required for correct functioning of batch norm and dropout layers during inference mode
with Layer.inference_mode_on():
    inference_prob = seq1(inputs['X'])
eval_loss = ng.squared_L2(inference_prob - inputs['y'])
eval_outputs = dict(l2_loss=eval_loss)

# Define computations
print('Start training')
eval_computation = ng.computation(inference_prob, "all")
with closing(ngt.make_transformer()) as transformer:
    # transformer = ngt.make_transformer()
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, eval_outputs, inputs)
    eval_function = transformer.add_computation(eval_computation)

    # Printout interval of the validation set loss during training
    iter_interval = num_iterations // 10

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=num_iterations,
                                 eval_set=test_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    # Train the network
    loop_train(train_set, train_computation, cbs)

    # Get predictions for the test set
    predictions = utils.eval_loop(test_set, eval_function, inputs)

    if(do_plots is True):
        # Plot the predictions
        time_points = 8 * no_points
        utils.plot_inference(predictions, predict_seq, data, time_points)

    # Generate a sequence
    # uses the first seq_len samples of the input sequence as seed
    time_points = 8 * no_points
    gen_series, gt_series = utils.generate_sequence(data, time_points, eval_function,
                                                    predict_seq, batch_size, seq_len,
                                                    feature_dim, inputs)

    if(do_plots is True):
        # Plot the generated series vs ground truth series
        utils.plot_generated(gen_series, gt_series, predict_seq)
