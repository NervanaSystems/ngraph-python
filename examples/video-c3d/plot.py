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
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
'''
Helper functions for plotting the trianing and validaiton loss
'''


def moving_average(input_data, window_size):
    '''
    Calculates the moving average to smooth out the data

    input_data(array): Data for smoothing
    window_size(int): Window size for moving average
    '''

    start_idx = 0
    end_idx = 0
    result = []
    while end_idx <= len(input_data):
        end_idx = start_idx + window_size
        avg_res = np.mean(input_data[start_idx:end_idx])
        result.append(avg_res)
        start_idx += 1

    return np.array(result)


def plot_logs(logs=None, log_file=None):
    '''
    Plots the train loss, validation loss, classification error
    If the logs dictionary is not passed in it will attempt to load from
    saved pickle file.

    logs(dict): Dictionary containing training and validation loss information
    log_file(string): Log file name
    '''

    if log_file:
        with open(log_file, 'rb') as f:
            logs = pickle.load(f)

    epoch = 0
    start_points = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for train_loss in logs['train']:
        train_loss = moving_average(train_loss, 100)
        data_len = len(train_loss)
        epoch_start = epoch * data_len
        ax1.plot(epoch_start + np.arange(data_len), train_loss, '-r')
        start_points.append(epoch_start)
        epoch += 1

    ax1.plot(epoch_start + np.arange(data_len), train_loss, '-r', label='train loss')
    ax1.plot(start_points, logs['validation'], '-bo', label='validaiton loss')
    ax2.plot(start_points, np.array(logs['misclass']) * 100, '-go', label='classification error')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('CE loss')
    ax2.set_ylabel('Classificaion Error (%)', color='g')
    ax1.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_plot.png')


if __name__ == "__main__":

    # Load training configuration and parse arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log_file', type=str, default='training_log.pkl',
                        help='name for the trainig log file')
    args = parser.parse_args()
    plot_logs(log_file=args.log_file)
