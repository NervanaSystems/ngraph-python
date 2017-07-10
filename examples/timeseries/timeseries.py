'''
    Usage:
        python timeseries.py -t 10000 -b gpu -z 64

    Builds a recurrent neural network to predict the next value in a timeseries
    Lissajous curves are used as the timeseries to test the network

    Option 1:
        --seq_to_seq option given: Given prior seq_len points, predict next seq_len points
            1) Build sequence [x0, x1, x2, ... , xN]
            2) Split sequence into non-overlapping windows of length seq_len
                S0: [x0, x1, ..., x(seq_len-1)]
                S1: [x(seq_len), ..., x(2*seq_len - 1)]
                ...
            3) Build input sequence - ground truth output sequence pairs
                S0 input, S1 output
                S1 input, S2 output
                ...
            4) Use a portion of the input/output pairs for training, remainder for test

        --seq_to_seq option not given: Given prior seq_len points, predict next point
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
'''

from __future__ import division, print_function
from builtins import range
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, Recurrent, Tanh, Affine
from ngraph.frontends.neon import UniformInit, Identity, Adam
from ngraph.frontends.neon import NgraphArgparser, loop_train
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks # noqa
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator
import utils
# import ngraph.transformers.passes.nviz
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('No matplotlib installed')

# Feature dimension of the input (for Lissajous curve, this is 2)
feature_dim = 2

# Output feature dimension
output_dim = 2

# Number of recurrent units in the network
recurrent_units = 32

# Define initialization
init_uni = UniformInit(-0.1, 0.1)

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--seq_to_seq', default=False, dest='seq_to_seq', action='store_true',
                    help='If given, seq_len future timepoints are predicted')
parser.add_argument('--look_ahead', type=float,
                    help="Number of time steps to start predicting from",
                    default=1)
parser.add_argument('--seq_len', type=float,
                    help="Number of time points in each input sequence",
                    default=30)
parser.set_defaults()
args = parser.parse_args()

# Batch size
batch_size = args.batch_size

# Sequence length
seq_len = args.seq_len

# Look Ahead
look_ahead = args.look_ahead

# Set to True if you want to predict seq_len future timepoints
# If set to False, it will only predict the next time point
seq_to_seq = args.seq_to_seq

'''
Generate the training data based on Lissajous curve
Total number of samples will be (npoints * ncycles - seq_len)

data.train['X']['data']: will be the input training data.
                         Shape: (no_samples, seq_len, input_feature_dim)
data.train['y']['data']: will be the outputs (labels) for training data.
                         Shape: (no_samples, output_feature_dim)

data.test follows a similar model
'''

data = utils.TimeSeries(train_ratio=0.8,  # ratio of samples to set aside for training
                                          # (value between 0. to 1.)
                        seq_len=seq_len,  # length of the sequence in each sample
                        npoints=23,       # number of points to take in each cycle
                        ncycles=1000,     # number of cycles in the curve
                        batch_size=batch_size,
                        curvetype='Lissajous2',
                        seq_to_seq=seq_to_seq,  # set to true if you want sequences as output
                        look_ahead=look_ahead)  # number of time steps to look ahead

# Make an iterable / generator that yields chunks of training data
# Yields an input array of Shape (batch_size, seq_len, input_feature_dim)
train_set = ArrayIterator(data.train, batch_size, total_iterations=args.num_iterations)
test_set = ArrayIterator(data.test, batch_size)

# Name and create the axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=feature_dim, name="feature_axis")
out_axis = ng.make_axis(length=output_dim, name="output_axis")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
if(seq_to_seq is True):
    out_axes = ng.make_axes([batch_axis, time_axis, out_axis])
else:
    out_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = {'X': ng.placeholder(in_axes), 'y': ng.placeholder(out_axes)}

if(seq_to_seq is True):
    seq1 = Sequential([Recurrent(nout=recurrent_units, init=init_uni,
                       activation=Tanh(), return_sequence=True),
                       Affine(weight_init=init_uni, bias_init=init_uni,
                       activation=Identity(), axes=out_axis)])
else:
    seq1 = Sequential([Recurrent(nout=recurrent_units, init=init_uni,
                      activation=Tanh(), return_sequence=False),
                      Affine(weight_init=init_uni, bias_init=init_uni,
                      activation=Identity(), axes=out_axis)])

# Define the optimizer
optimizer = Adam()

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


# Function to return final predicted results for a given dataset
def loop_eval(dataset, computation):
    dataset.reset()
    results = []
    for data in dataset:
        feed_dict = {inputs[k]: data[k] for k in data.keys() if k != 'iteration'}
        results.append(computation(feed_dict=feed_dict))
    return results


print('Start training')
inference_prob = seq1(inputs['X'])
eval_computation = ng.computation([inference_prob], "all")
with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, eval_outputs, inputs)
    eval_function = transformer.add_computation(eval_computation)
    # Make these explicit
    cbs = make_default_callbacks(output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=test_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, train_computation, cbs)
    predictions = loop_eval(test_set, eval_function)

# Plot the ground truth test samples, as well as predictions


if(seq_to_seq is False):
    # Flatten the predictions
    # predictions[0][0] is output_feature_dim, batch_size
    preds = predictions[0][0]
    for i in range(1, len(predictions)):
        preds = np.concatenate((preds, predictions[i][0]), axis=1)

    # Reshape the prediction axes to batch_axis (samples), feature_axis
    preds = np.swapaxes(preds, 0, 1)

else:
    # Flatten the predictions
    # predictions[0][0] is output_feature_dim, seq_len, batch_size
    preds = predictions[0][0]
    for i in range(1, len(predictions)):
        preds = np.concatenate((preds, predictions[i][0]), axis=2)

    # Reshape the prediction axes to batch_axis (samples), time_axis, feature_axis
    preds = np.swapaxes(preds, 0, 2)

    # Reshape so that samples are concatenated at the end of each other (time_axis, feature_axis)
    preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))

# Get ground truth values
gt_vals = data.test['y']['data'].reshape(preds.shape)

# Take only up to 500 time points
time_points = min(500, preds.shape[0])
preds = preds[:time_points, ...]
gt_vals = gt_vals[:time_points, ...]

# Plot predictions across time
fig, ax = plt.subplots(figsize=(20, 10))
line1 = ax.plot(range(preds.shape[0]), preds[:, 0],
                linestyle=':',
                marker='s', label='predicted_x')
line2 = ax.plot(range(preds.shape[0]), preds[:, 1],
                linestyle=':',
                marker='o', label='predicted_y')
line3 = ax.plot(range(preds.shape[0]), gt_vals[:, 0],
                linestyle=':',
                marker='d', label='gt_x')
line4 = ax.plot(range(preds.shape[0]), gt_vals[:, 1],
                linestyle=':',
                marker='D', label='gt_y')
ax.legend()
ax.grid()
title = 'Lissajous Curve Predictions and Ground Truth Across Time, Seq_to_Seq:%s' % seq_to_seq
ax.set_title(title)
fig.savefig('PredictedCurve_Time_Seq_to_Seq_%s.png' % seq_to_seq, dpi=128)
plt.clf()

# Plot one feature in x, the other in y axis
fig, ax = plt.subplots(figsize=(20, 10))
line1 = ax.plot(preds[:, 0], preds[:, 1],
                linestyle=':',
                marker='s', label='predicted')
line2 = ax.plot(gt_vals[:, 0], gt_vals[:, 1],
                linestyle=':',
                marker='o', label='ground truth')
ax.set_title('Lissajous Curve Predictions and Ground Truth, Dual Axes, Seq_to_Seq:%s' % seq_to_seq)
ax.legend()
ax.grid()
fig.savefig('PredictedCurve_DualAxes_Seq_to_Seq_%s.png' % seq_to_seq, dpi=128)
