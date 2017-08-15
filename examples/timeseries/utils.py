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
import copy
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('No matplotlib installed')


def eval_loop(dataset, computation, inputs):
    """
    Function to return inference results for a given dataset
    """
    dataset.reset()
    results = []
    for data in dataset:
        feed_dict = {inputs[k]: data[k] for k in data.keys()}
        results.append(copy.copy(computation(feed_dict=feed_dict)))
    return results


def plot_inference(predictions, predict_seq, gt_data, time_points):
    """
    Plot the ground truth test samples, as well as predictions
    """
    if(predict_seq is False):
        # Flatten the predictions
        # predictions[0] is (output_feature_dim, batch_size)
        preds = predictions[0]
        for i in range(1, len(predictions)):
            preds = np.concatenate((preds, predictions[i]), axis=1)

        # Reshape the prediction axes to batch_axis (samples), feature_axis
        preds = np.swapaxes(preds, 0, 1)

    else:
        # Flatten the predictions
        # predictions[0] is (output_feature_dim, seq_len, batch_size)
        preds = predictions[0]
        for i in range(1, len(predictions)):
            preds = np.concatenate((preds, predictions[i]), axis=2)

        # Reshape the prediction axes to batch_axis (samples), time_axis, feature_axis
        preds = np.swapaxes(preds, 0, 2)

        # Reshape so that samples are concatenated at the end of each other
        # (time_axis, feature_axis)
        preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))

    # If matplotlib is available, plot the results
    # Get ground truth values
    gt_vals = gt_data.test['y']['data'].reshape(preds.shape)

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
    title = 'Lissajous Curve Predictions and Ground Truth,\n' \
            '2D Time Series, Predict Sequence:%s' % predict_seq
    ax.set_title(title)
    ax.legend()
    ax.grid()
    fig.savefig('PredictedCurve_2D_PredictSeq_%s.png' % predict_seq, dpi=128)


def plot_generated(gen_series, gt_series, predict_seq):
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
    title = 'Lissajous Curve Generated Series and Ground Truth\n' \
            'Predict Sequence:%s' % predict_seq
    ax.set_title(title)
    fig.savefig('GeneratedCurve_Time_PredictSeq_%s.png' % predict_seq, dpi=128)


def generate_sequence(gt_data, time_points, eval_function, predict_seq,
                      batch_size, seq_len, feature_dim, inputs):
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
        result = eval_function(feed_dict=axx)

        if(predict_seq is False):
            # result is of size (output_dim, batch_size)
            # We want the output of the first sample, so get it
            result = result[:, 0]
        else:
            # result is of size (output_dim, seq_len, batch_size)
            # We want the last output of the first sample, so get it
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
