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
from builtins import range
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, LSTM, Affine
from ngraph.frontends.neon import UniformInit, Tanh, Logistic, Identity, Adam
from ngraph.frontends.neon import NgraphArgparser, loop_train
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator
import timeseries
import pdb

def loop_eval(dataset, computation):
    """
    Function to return inference results for a given dataset
    """
    dataset.reset()
    results = []
    for data in dataset:
        feed_dict = {inputs[k]: data[k] for k in data.keys() if k != 'iteration'}
        results.append(computation(feed_dict=feed_dict))
    return results


def plot_inference(predictions, predict_seq, gt_data, time_points):
    """
    Plot the ground truth test samples, as well as predictions
    """
    if(predict_seq is False):
        # Flatten the predictions
        # predictions[0][0] is (output_feature_dim, batch_size)
        preds = predictions[0][0]
        for i in range(1, len(predictions)):
            preds = np.concatenate((preds, predictions[i][0]), axis=1)

        # Reshape the prediction axes to batch_axis (samples), feature_axis
        preds = np.swapaxes(preds, 0, 1)

    else:
        # Flatten the predictions
        # predictions[0][0] is (output_feature_dim, seq_len, batch_size)
        preds = predictions[0][0]
        for i in range(1, len(predictions)):
            preds = np.concatenate((preds, predictions[i][0]), axis=2)

        # Reshape the prediction axes to batch_axis (samples), time_axis, feature_axis
        preds = np.swapaxes(preds, 0, 2)

        # Reshape so that samples are concatenated at the end of each other
        # (time_axis, feature_axis)
        preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))

    # If matplotlib is available, plot the results
    # Get ground truth values
    gt_vals = data.test['y']['data'].reshape(preds.shape)

    # Take only up to 8 cycles
    # time_points = min(8 * no_points, preds.shape[0])
    preds = preds[:time_points, ...]
    gt_vals = gt_vals[:time_points, ...]

    # Plot predictions across time
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(range(preds.shape[0]), preds[:, 0],
            linestyle=':',
            marker='s', label='predicted_x')
    ax.plot(range(preds.shape[0]), preds[:, 1],
            linestyle=':',
            marker='o', label='predicted_y')
    ax.plot(range(preds.shape[0]), gt_vals[:, 0],
            linestyle=':',
            marker='d', label='gt_x')
    ax.plot(range(preds.shape[0]), gt_vals[:, 1],
            linestyle=':',
            marker='D', label='gt_y')
    ax.legend()
    ax.grid()
    title = 'Lissajous Curve Predictions and Ground Truth, Predict Sequence:%s' % predict_seq
    ax.set_title(title)
    fig.savefig('PredictedCurve_Time_PredictSeq_%s.png' % predict_seq, dpi=128)
    plt.clf()

    # Plot one feature in x, the other in y axis
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(preds[:, 0], preds[:, 1],
            linestyle=':',
            marker='s', label='predicted')
    ax.plot(gt_vals[:, 0], gt_vals[:, 1],
            linestyle=':',
            marker='o', label='ground truth')
    title = 'Lissajous Curve Predictions and Ground Truth, \
            2D Time Series, Predict Sequence:%s' % predict_seq
    ax.set_title(title)
    ax.legend()
    ax.grid()
    fig.savefig('PredictedCurve_2D_PredictSeq_%s.png' % predict_seq, dpi=128)


def plot_generated(gen_series, gt_series):
    """
    Plots the generated time series over the ground truth series
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(range(gen_series.shape[0]), gen_series[:, 0],
            linestyle=':',
            marker='s', label='generated_x')
    ax.plot(range(gen_series.shape[0]), gen_series[:, 1],
            linestyle=':',
            marker='o', label='generated_y')
    ax.plot(range(gt_series.shape[0]), gt_series[:, 0],
            linestyle=':',
            marker='d', label='gt_x')
    ax.plot(range(gt_series.shape[0]), gt_series[:, 1],
            linestyle=':',
            marker='D', label='gt_y')
    ax.legend()
    ax.grid()
    title = 'Lissajous Curve Generated Series and Ground Truth, \
            Predict Sequence:%s' % predict_seq
    ax.set_title(title)
    fig.savefig('GeneratedCurve_Time_PredictSeq_%s.png' % predict_seq, dpi=128)


def generate_sequence(gt_data, time_points, eval_function, predict_seq,
                      batch_size, seq_len, feature_dim):
    """
    Generates a sequence of length time_points, given ground truth data (gt_data)
    First seq_len points of gt_data is used as the seed
    Returns the generated sequence

    gt_data: ground truth data
    time_points: number of steps to generate the data
    eval_function: forward prop function of the network
    predict_seq: True if network predicts sequences

    Start with first seq_len points in training data, take it as input (call S0)
    S0 = [x0, x1, ..., x(seq_len-1)]
    Given S0, generate next time point x_hat(seq_len), build S1
    S1 = [x1, x2, ..., x(seq_len-1), x_hat(seq_len)]
    Given S1, generate x_hat(seq_len+1)
    Continue generating for a total of time_points
    """
    data = gt_data
    no_gen_time_points = time_points
    input_batch = np.zeros((batch_size, seq_len, feature_dim))
    input_batch[0] = data.train['X']['data'][0]
    gen_series = data.train['X']['data'][0]  # This will hold the generated series
    gt_series = data.train['X']['data'][0]  # This will hold the ground truth series

    output_dim = data.train['y']['data'].shape[-1]
    for tp in range(no_gen_time_points):
        axx = {inputs['X']: input_batch}
        # Get the prediction using seq_len past samples
        result = eval_function(feed_dict=axx)[0]

        if(predict_seq is False):
            # result is of size (output_dim, batch_size)
            # We want the output of the first batch, so get it
            result = result[:, 0]
        else:
            # result is of size (output_dim, seq_len, batch_size)
            # We want the last output of the first batch, so get it
            result = result[:, -1, 0]
        # Now result is (output_dim,)
        # Reshape result to (1,output_dim)
        result = np.reshape(result, (1, output_dim))

        # Get the last (seq_len-1) samples in the past
        # cx is of shape (seq_len-1, output_dim)
        cx = input_batch[0][1:, :]

        # Append the new prediction to the past (seq_len-1) samples
        # Put the result into the first sample in the input batch
        input_batch[0] = np.concatenate((cx, result))

        # Append the current prediction to gen_series
        # This is to keep track of predictions, for plotting purposes only
        gen_series = np.concatenate((gen_series, result))

        # Find the ground truth for this prediction
        if(predict_seq is False):
            gt_outcome = data.train['X']['data'][tp + 1][-1, :]
            # Reshape to (1, output_dim)
            gt_outcome = np.reshape(gt_outcome, (1, output_dim))
        else:
            # When predict_seq is given, input 'X' has non overlapping windows
            # X is of shape (no_samples, seq_len, 2)
            # Thus, find the right index for the ground truth output
            gt_outcome = data.train['X']['data'][(tp + seq_len) // seq_len,
                                                 (tp + seq_len) % seq_len, :]
            # Reshape to (1, output_dim)
            gt_outcome = np.reshape(gt_outcome, (1, output_dim))

        # Append ground truth outcome to gt_series
        # This is to keep track of ground truth, for plotting purposes only
        gt_series = np.concatenate((gt_series, gt_outcome))

    return gen_series, gt_series


# Plot the inference / generation results if matplotlib is available
do_plots = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('No matplotlib installed')
    do_plots = False

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
                    default=500)
parser.set_defaults(num_iterations=12000)
args = parser.parse_args()

# Feature dimension of the input (for Lissajous curve, this is 2)
feature_dim = 2
# Output feature dimension (for Lissajous curve, this is 2)
output_dim = 2
# Number of recurrent units in the network
recurrent_units = 32
# Define initialization
init_uni = UniformInit(-0.1, 0.1)

# Batch size
batch_size = args.batch_size
# Sequence length
seq_len = args.seq_len
# Look Ahead
look_ahead = args.look_ahead
# Set to True if you want to predict seq_len future timepoints
# If set to False, it will only predict the next time point
predict_seq = args.predict_seq

'''
DATA GENERATION
Generate the training data based on Lissajous curve
Total number of samples will be (npoints * ncycles - seq_len)

data.train['X']['data']: will be the input training data.
                         Shape: (no_samples, seq_len, input_feature_dim)
data.train['y']['data']: will be the outputs (labels) for training data.
                         Shape: (no_samples, output_feature_dim)

data.test follows a similar model
'''
# Total epochs of training
no_epochs = args.epochs 
# How many cycles of Lissajous curve to take
no_cycles = 2000
# How many points per cycle to take
no_points = 13
# Calculate how many batches are in data
no_batches = no_cycles * no_points // seq_len // batch_size
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
pdb.set_trace()
eval_computation = ng.computation([inference_prob], "all")
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

    pdb.set_trace()
    # Get predictions for the test set
    predictions = loop_eval(test_set, eval_function)

    if(do_plots is True):
        # Plot the predictions
        time_points = 8 * no_points
        plot_inference(predictions, predict_seq, data, time_points)

    # Generate a sequence
    # uses the first seq_len samples of the input sequence as seed
    time_points = 8 * no_points
    gen_series, gt_series = generate_sequence(data, time_points, eval_function,
                                              predict_seq, batch_size, seq_len,
                                              feature_dim)

    if(do_plots is True):
        # Plot the generated series vs ground truth series
        plot_generated(gen_series, gt_series)
