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
# HETR Usage:
# - set environment variables for HETR_AEON_IP and HETR_AEON_PORT
# - run the aeon-service on antoher linux terminal
#   - cd <PATH_TO_AEON_SERVICE>
#   - ./aeon-service --uri http://<HETR_AEON_IP>:<HETR_AEON_PORT>
# - launch train_resnet.py with the desired arg parameters
# ----------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import ax, NgraphArgparser
from tqdm import tqdm
from data import make_aeon_loaders
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import Layer
from resnet import BuildResnet
from contextlib import closing
from ngraph.frontends.neon import Saver
from utils import get_network_params, set_lr
import os
import logging

logger = logging.getLogger(__name__)


# Calculate metrics over given dataset
def loop_eval(dataset, input_ops, metric_name, computation, en_top5=False):
    # Reset validation set
    dataset.reset()
    all_results = None
    # Iterating over the dataset
    for data in dataset:
        if device_backend == 'hetr' and args.num_devices > 1:
            feed_dict = {input_ops['iteration']: step}
        else:
            data['iteration'] = step
            feed_dict = {input_ops[k]: data[k] for k in input_ops.keys()}
        # Tuple of results from computation
        results = computation(feed_dict=feed_dict)
        # Seperate Results
        results_miss_loss = results[1]
        results_inference = results[0]
        # Collect top5 and top1 results
        top5 = np.argsort(results_inference, axis=0)[-5:]
        top1 = top5[-1:]
        # Get ground truth labels
        correct_label = data['label'].T
        # Compare if any of the top5 matches with labels
        top5_results = np.any(np.equal(correct_label, top5), axis=0)
        # Invert for mis-classification
        top5_results = np.invert(top5_results)
        # Compare which are not equal
        top1_results = np.not_equal(correct_label, top1)
        # Make a list of results
        if(en_top5):
            total_results = [results_miss_loss, top5_results, top1_results]
        else:
            total_results = [results_miss_loss, top1_results]
        # Accumulate results
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, total_results)}
        else:
            for name, res in zip(metric_names, total_results):
                all_results[name].extend(list(res))
    # Take mean of results
    ndata = dataset.ndata
    reduced_results = {k: np.mean(v[:ndata]) for k, v in all_results.items()}
    return reduced_results


if __name__ == "__main__":
    # Hyperparameters
    # Optimizer
    base_lr = 0.1
    gamma = 0.1
    momentum_coef = 0.9
    wdecay = 0.0001
    nesterov = False

    print("HyperParameters")
    print("Learning Rate:     " + str(base_lr))
    print("Momentum:          " + str(momentum_coef))
    print("Weight Decay:      " + str(wdecay))
    print("Nesterov:          " + str(nesterov))

    # Command Line Parser
    parser = NgraphArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('--dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
    parser.add_argument('--tb', action="store_true", help="1- Enables tensorboard")
    parser.add_argument('--logfile', type=str, default=None, help="Name of the csv which \
                        logs different metrics of model")
    parser.add_argument('--hetr_device', type=str, default='cpu', help="hetr device type")
    parser.add_argument('--num_devices', '-m', type=int, default=1, help="num hetr devices")
    parser.add_argument('--disable_batch_norm', action='store_true')
    parser.add_argument('--save_file', type=str, default=None, help="File to save weights")
    parser.add_argument('--inference', type=str, default=None, help="File to load weights")
    args = parser.parse_args()

# Get network parameters
nw_params = get_network_params(args.dataset, args.size, args.batch_size)
metric_names = nw_params['metric_names']
en_top5 = nw_params['en_top5']
num_resnet_mods = nw_params['num_resnet_mods']
args.iter_interval = nw_params['iter_interval']
learning_schedule = nw_params['learning_schedule']
en_bottleneck = nw_params['en_bottleneck']
ax.Y.length = nw_params['num_classes']

# Set batch size
ax.N.length = args.batch_size

# Initialize device
device_backend = args.backend
device_hetr = args.hetr_device
device_id = [str(d) for d in range(args.num_devices)]

aeon_adress = None
aeon_port = None
if device_backend == 'hetr' and args.num_devices > 1:
    if 'HETR_AEON_IP' not in os.environ or 'HETR_AEON_PORT' not in os.environ:
        raise ValueError('To run hetr with more than one device, you need to set an ip address \
            for the HETR_AEON_IP environment variable and a port number for the HETR_AEON_PORT \
            environment variable')

    aeon_adress = os.environ['HETR_AEON_IP']
    aeon_port = int(os.environ['HETR_AEON_PORT'])

# Create training and validation set objects
# two validation dataloaders sets are created,
# one for the input_ops to fetch the data from aeon on the worker process
# the other to get evalutation stats on the result on the hetr master process
train_set, valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                         args.num_iterations, dataset=args.dataset,
                                         num_devices=args.num_devices, device=device_backend,
                                         split_batch=True, address=aeon_adress, port=aeon_port)
master_valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                     args.num_iterations, dataset=args.dataset,
                                     num_devices=1, device=device_backend,
                                     split_batch=False, address=None, port=None,
                                     return_train=False, return_valid=True)
print("Completed loading " + args.dataset + " dataset")
# Randomize seed
np.random.seed(args.rng_seed)

# Make input_ops or placeholders depending on single device or multi device compute
input_ops_train = train_set.make_input_ops("train", aeon_adress, aeon_port, ax.N,
                                           device_hetr, device_id)
input_ops_valid = valid_set.make_input_ops("valid", aeon_adress, aeon_port, ax.N,
                                           device_hetr, device_id)

with ng.metadata(device=device_hetr, device_id=device_id, parallel=ax.N):
    # Build the network
    resnet = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods,
                         batch_norm=not args.disable_batch_norm)

    # Tensorboard
    if(args.tb):
        try:
            from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
        except:
            print("Tensorboard not installed")
        seq1 = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods)
        train = seq1(input_ops_train['image'])
        tb = TensorBoard("/tmp/")
        tb.add_graph(train)
        exit()

    # Learning Rate Placeholder
    lr_ph = ng.placeholder(axes=(), initial_value=base_lr)

    # Optimizer
    # Provided learning policy takes learning rate as input to graph using a placeholder.
    # This allows you to control learning rate based on various factors of network
    learning_rate_policy = {'name': 'provided',
                            'lr_placeholder': lr_ph}

    optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                        momentum_coef=momentum_coef,
                                        wdecay=wdecay,
                                        nesterov=False,
                                        iteration=input_ops_train['iteration'])
    # Make a prediction
    prediction = resnet(input_ops_train['image'])
    # Calculate loss
    train_loss = ng.cross_entropy_multi(prediction,
                                        ng.one_hot(input_ops_train['label'], axis=ax.Y))
    # Average loss over the batch
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_computation = ng.computation(batch_cost, "all")

# Instantiate the Saver object to save weights
weight_saver = Saver()

with ng.metadata(device=device_hetr, device_id=device_id, parallel=ax.N):
    # Inference
    with Layer.inference_mode_on():
        # Doing inference
        inference_prob = resnet(input_ops_valid['image'])
        eval_loss = ng.cross_entropy_multi(inference_prob,
                                           ng.one_hot(input_ops_valid['label'], axis=ax.Y))
        # Computation for inference
        eval_computation = ng.computation([inference_prob, eval_loss], "all")

# Doing inference
if(args.inference is not None):
    # Check if file exists. TODO.
    with closing(ngt.make_transformer()) as transformer:
        restore_eval_function = transformer.add_computation(eval_computation)
        weight_saver.setup_restore(transformer=transformer, computation=eval_computation,
                                   filename=args.inference)
        # Restore weight
        weight_saver.restore()
        # Calculate losses
        eval_losses = loop_eval(master_valid_set, input_ops_valid, metric_names,
                                restore_eval_function, en_top5)
        # Print statistics
        print("From restored weights: Test Avg loss:{tcost}".format(tcost=eval_losses))
        exit()

# Training the network by calling transformer
t_args = {'device': args.hetr_device} if args.backend == 'hetr' else {}
with closing(ngt.make_transformer_factory(args.backend, **t_args)()) as transformer:
    # Trainer
    train_function = transformer.add_computation(train_computation)
    # Inference
    eval_function = transformer.add_computation(eval_computation)
    # Set Saver for saving weights
    weight_saver.setup_save(transformer=transformer, computation=train_computation)
    # Progress bar
    tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
    # Set interval cost to 0.0
    interval_cost = 0.0
    # Declare lists for logging metrics
    if(args.logfile is not None):
        train_result = []
        test_result = []
        err_result = []
    # Iterating over the training set
    for step in range(args.num_iterations):
        # Dictionary for training
        if device_backend == 'hetr' and args.num_devices > 1:
            feed_dict = {input_ops_train['iteration']: step}
        else:
            data = train_set.next()
            data['iteration'] = step
            feed_dict = {input_ops_train[k]: data[k] for k in input_ops_train.keys()}
        # Learning Schedule
        feed_dict[lr_ph] = set_lr(base_lr, step, learning_schedule, gamma)
        # Mean batch cost
        output = train_function(feed_dict=feed_dict)

        # Update progress bar
        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output[()]))
        interval_cost += output[()]
        # Every epoch print test set metrics
        if (step + 1) % args.iter_interval == 0 and step > 0:
            # Call loop_eval to calculate metric over test set
            eval_losses = loop_eval(master_valid_set, input_ops_valid,
                                    metric_names, eval_function, en_top5)
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f} Test Metrics:{tcost}".format(
                           interval=step // args.iter_interval,
                           iteration=step,
                           cost=interval_cost / args.iter_interval, tcost=eval_losses))
            # Log metrics to list
            if(args.logfile is not None):
                # For storing to csv
                train_result.append(interval_cost / args.iter_interval)
                test_result.append(eval_losses['Cross_Ent_Loss'])
                if args.dataset == 'cifar10':
                    err_result.append(eval_losses['Misclass'])
                if args.dataset == 'i1k':
                    err_result.append(eval_losses['Top5_Err'])
            interval_cost = 0.0
            if(args.save_file is not None):
                print("\nSaving Weights")
                weight_saver.save(filename=args.save_file + "_" + str(step))
    # Writing to CSV
    if(args.logfile is not None):
        print("\nSaving results to csv file")
        import csv
        with open(args.logfile + ".csv", 'wb') as train_test_file:
            wr = csv.writer(train_test_file, quoting=csv.QUOTE_ALL)
            wr.writerow(train_result)
            wr.writerow(test_result)
            wr.writerow(err_result)
    print("\nTraining Completed")
    if(args.save_file is not None):
        print("\nSaving Weights")
        weight_saver.save(filename=args.save_file)
