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
Video-C3D model. Activity classification example from UCF-101 data set.
usage: python video_c3d.py -b gpu
"""
import os
import numpy as np
import ngraph as ng

from ngraph.frontends.neon import GaussianInit, ConstantInit
from ngraph.frontends.neon import Layer, Affine, Convolution, Pooling, Sequential
from ngraph.frontends.neon import Softmax, Rectlin, Dropout, GradientDescentMomentum
from ngraph.frontends.neon import ax, make_bound_computation
from ngraph.frontends.neon import NgraphArgparser

import ngraph.transformers as ngt
from tqdm import tqdm, trange
from contextlib import closing
from data import make_train_loader, make_validation_loader
from plot import plot_logs
import pickle

# TODO Issue raised to have the strides default to the size of the pooling layer
# change the strides after issue is resolved
# https://github.com/NervanaSystems/private-ngraph/issues/2309
# TODO Data loader needs fixing to remove the .reset() calls on the data iterators
# TODO Data loader needs to convert data into dictionary


def create_network():
    '''
    Define 3D convolutional network
    '''

    # Define for weight initialization
    g1 = GaussianInit(mean=0., var=0.01)
    g5 = GaussianInit(mean=0., var=0.005)
    c0 = ConstantInit(val=0.)
    c1 = ConstantInit(val=1.)
    ax.Y.length = 101

    padding = {'D': 1, 'H': 1, 'W': 1, 'C': 0}
    strides = {'D': 2, 'H': 2, 'W': 2, 'C': 1}

    layers = [
        Convolution((3, 3, 3, 64), padding=padding, filter_init=g1, bias_init=c0,
                    activation=Rectlin()),
        Pooling((1, 2, 2), strides={'D': 1, 'H': 2, 'W': 2, 'C': 1}),
        Convolution((3, 3, 3, 128), padding=padding, filter_init=g1, bias_init=c1,
                    activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Convolution((3, 3, 3, 256), padding=padding, filter_init=g1, bias_init=c1,
                    activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Convolution((3, 3, 3, 256), padding=padding, filter_init=g1, bias_init=c1,
                    activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Convolution((3, 3, 3, 256), padding=padding, filter_init=g1, bias_init=c1,
                    activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Affine(nout=2048, weight_init=g5, bias_init=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=2048, weight_init=g5, bias_init=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(axes=ax.Y, weight_init=g1, bias_init=c0, activation=Softmax())
    ]

    return Sequential(layers)


def train_network(model, train_set, valid_set, batch_size, epochs, log_file):
    '''
    Trains the predefined network. Trains the model and saves the progress in
    the log file that is defined in the arguments

    model(object): Defines the model in Neon
    train_set(object): Defines the training set
    valid_set(object): Defines the validation set
    args(object): Training arguments
    batch_size(int): Minibatch size
    epochs(int): Number of training epoch
    log_file(string): File name to store trainig logs for plotting

    '''

    # Form placeholders for inputs to the network
    # Iterations needed for learning rate schedule
    inputs = train_set.make_placeholders(include_iteration=True)

    # Convert labels into one-hot vectors
    one_hot_label = ng.one_hot(inputs['label'], axis=ax.Y)

    learning_rate_policy = {'name': 'schedule',
                            'schedule': list(np.arange(2, epochs, 2)),
                            'gamma': 0.6,
                            'base_lr': 0.001}

    optimizer = GradientDescentMomentum(
        learning_rate=learning_rate_policy,
        momentum_coef=0.9,
        wdecay=0.005,
        iteration=inputs['iteration'])

    # Define graph for training
    train_prob = model(inputs['video'])
    train_loss = ng.cross_entropy_multi(train_prob, one_hot_label)
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])

    with closing(ngt.make_transformer()) as transformer:

        # Define graph for calculating validation set error and misclassification rate
        # Use inference mode for validation to avoid dropout in forward pass
        with Layer.inference_mode_on():
            inference_prob = model(inputs['video'])
            errors = ng.not_equal(ng.argmax(inference_prob), inputs['label'])
            eval_loss = ng.cross_entropy_multi(inference_prob, one_hot_label)
            eval_outputs = {'cross_ent_loss': eval_loss, 'misclass': errors}

            eval_computation = make_bound_computation(transformer, eval_outputs, inputs)

        train_outputs = {'batch_cost': batch_cost}
        train_computation = make_bound_computation(transformer, train_outputs, inputs)

        interval_cost = 0.0

        # Train in epochs
        logs = {'train': [], 'validation': [], 'misclass': []}
        for epoch in trange(epochs, desc='Epochs'):

            # Setup the training bar
            numBatches = train_set.ndata // batch_size
            tpbar = tqdm(unit='batches', ncols=100, total=numBatches, leave=False)

            train_set.reset()
            valid_set.reset()

            train_log = []
            for step, data in enumerate(train_set):
                data = dict(data)
                data['iteration'] = epoch  # learning schedule based on epochs
                output = train_computation(data)
                train_log.append(float(output['batch_cost']))

                tpbar.update(1)
                tpbar.set_description("Training {:0.4f}".format(float(output['batch_cost'])))
                interval_cost += float(output['batch_cost'])
            tqdm.write("Epoch {epch}  complete. "
                       "Avg Train Cost {cost:0.4f}".format(
                           epch=epoch,
                           cost=interval_cost / step))
            interval_cost = 0.0
            tpbar.close()
            validation_loss = run_validation(valid_set, eval_computation)
            tqdm.write("Avg losses: {}".format(validation_loss))
            logs['train'].append(train_log)
            logs['validation'].append(validation_loss['cross_ent_loss'])
            logs['misclass'].append(validation_loss['misclass'])

            # Save log data and plot at the end of each epoch
            with open(log_file, 'wb') as f:
                pickle.dump(logs, f)
            plot_logs(logs=logs)


def run_validation(dataset, computation):
    '''
    Computes the validation error and missclassification rate
    Helper function that is called from the main traning function

    dataset(object): Contains the validation dataset object
    computation(object): Validation function
    metric_names(dict): Names of the metrics calculated by the computation
    inputs(object): Placeholders for inputs
    '''

    dataset.reset()
    all_results = None
    for i, data in enumerate(dataset):
        data = dict(data)
        data['iteration'] = i
        results = computation(data)
        if all_results is None:
            all_results = {k: list(v) for k, v in results.items()}
        else:
            for k, v in results.items():
                all_results[k].extend(list(v))

    reduced_results = {k: np.mean(v[:dataset.ndata]) for k, v in all_results.items()}

    return reduced_results


def get_data(manifest, manifest_root, batch_size, subset_pct, rng_seed):
    '''
    Loads training and validation set using aeon loader

    args(object): Contains function arguments
    manifest(list): Manifest files for traning and validaions
    manifest_root(string): Root directory of manifest file
    batch_size(int): Mini batch size
    subset_pct(float): Subset percentage of the data (0-100)
    rng_seed(int): Seed for random number generator
    '''

    assert 'train' in manifest[1], "Missing train manifest"
    assert 'test' in manifest[0], "Missing validation manifest"

    train_set = make_train_loader(manifest[1], manifest_root, batch_size,
                                  subset_pct, rng_seed)
    valid_set = make_validation_loader(manifest[0], manifest_root, batch_size,
                                       subset_pct)

    return train_set, valid_set


if __name__ == "__main__":

    # Load training configuration and parse arguments
    train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
    config_files = [train_config] if os.path.exists(train_config) else []
    parser = NgraphArgparser(__doc__, default_config_files=config_files)

    parser.add_argument('--subset_pct', type=float, default=100,
                        help='subset of training dataset to use (percentage)')
    parser.add_argument('--log_file', type=str, default='training_log.pkl',
                        help='name for the trainig log file')
    args = parser.parse_args()

    np.random.seed = args.rng_seed

    # Load data
    train_set, valid_set = get_data(args.manifest, args.manifest_root, args.batch_size,
                                    args.subset_pct, args.rng_seed)

    # Define model and train
    model = create_network()
    train_network(model, train_set, valid_set, args.batch_size, args.epochs, args.log_file)
