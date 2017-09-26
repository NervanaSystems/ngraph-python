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


# Result collector
def loop_eval(dataset, computation, metric_names):
    dataset.reset()
    all_results = None
    for data in dataset:
        feed_dict = {input_ph[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))
    ndata = dataset.ndata
    reduced_results = {k: np.mean(v[:ndata]) for k, v in all_results.items()}
    return reduced_results


if __name__ == "__main__":
    # Hyperparameters
    # Optimizer
    base_lr = 0.01
    gamma = 0.1
    momentum_coef = 0.9
    wdecay = 0.0001
    nesterov = False

    print("HyperParameters")
    print("Learning Rate:     " + str(base_lr))
    print("Momentum:          " + str(momentum_coef))
    print("Weight Decay:      " + str(wdecay))
    print("Nesterov           " + str(nesterov))

    # Command Line Parser
    parser = NgraphArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('--dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
    parser.add_argument('--tb', action="store_true", help="1- Enables tensorboard")
    parser.add_argument('--name', type=str, default=None, help="Name of the csv which \
                        logs different metrics of model")
    args = parser.parse_args()

    # Checking Command line args are proper
    cifar_sizes = [8, 20, 32, 44, 56, 110]
    i1k_sizes = [18, 34, 50, 101, 152]
    if args.dataset == 'cifar10':
        dataset_sizes = cifar_sizes
        if args.size in dataset_sizes:
            # Num of resnet modules required for cifar10
            num_resnet_mods = (args.size - 2) // 6
            # Change iter_interval to print every epoch
            args.iter_interval = 50000 // args.batch_size
            learning_schedule = [84 * args.iter_interval, 124 * args.iter_interval]
            print("Learning Schedule: " + str(learning_schedule))
            # CIFAR10 doesn't use bottleneck
            en_bottleneck = False
            # There are 10 classes so setting length of label axis to 10
            ax.Y.length = 10
        else:
            raise ValueError("Invalid CIFAR10 size. Select from " + str(dataset_sizes))
    elif args.dataset == 'i1k':
        dataset_sizes = i1k_sizes
        if args.size in dataset_sizes:
            # Enable or disable bottleneck depending on resnet size
            if(args.size in [18, 34]):
                en_bottleneck = False
            else:
                en_bottleneck = True
            # Change iter_interval to print every epoch. TODO
            args.iter_interval = 1281216 // args.batch_size
            learning_schedule = [84 * args.iter_interval, 124 * args.iter_interval]
            print("Learning Schedule: " + str(learning_schedule))
            # For now 200 to prints often will change later
            args.iter_interval = 200
            num_resnet_mods = 0
            # of Classes
            ax.Y.length = 1000
        else:
            raise ValueError("Invalid i1k size. Select from " + str(dataset_sizes))
    else:
        raise NameError("Invalid Dataset. Dataset should be either cifar10 or i1k")
# Set batch size
ax.N.length = args.batch_size
# Create training and validation set objects
train_set, valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                         args.num_iterations, dataset=args.dataset)
print("Completed loading " + args.dataset + " dataset")
# Randomize seed
np.random.seed(args.rng_seed)
# Make placeholders
input_ph = train_set.make_placeholders(include_iteration=True)
# Build the network
resnet = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods)
# Tensorboard
if(args.tb):
    from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
    seq1 = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods)
    train = seq1(input_ph['image'])
    tb = TensorBoard("/tmp/")
    tb.add_graph(train)
    exit()
# Optimizer
learning_rate_policy = {'name': 'schedule',
                        'schedule': learning_schedule,
                        'gamma': gamma,
                        'base_lr': base_lr}

optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                    momentum_coef=momentum_coef,
                                    wdecay=wdecay,
                                    nesterov=False,
                                    iteration=input_ph['iteration'])
label_indices = input_ph['label']
prediction = resnet(input_ph['image'])
train_loss = ng.cross_entropy_multi(prediction, ng.one_hot(label_indices, axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

# Inference Parameters
with Layer.inference_mode_on():
    inference_prob = resnet(input_ph['image'])
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    eval_loss_names = ['cross_ent_loss', 'misclass']
    eval_computation = ng.computation([eval_loss, errors], "all")

# Training the network by calling transformer
with closing(ngt.make_transformer()) as transformer:
    # Trainer
    train_function = transformer.add_computation(train_computation)
    # Inference
    eval_function = transformer.add_computation(eval_computation)
    # Progress bar
    tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
    interval_cost = 0.0
    if(args.name is not None):
        train_result = []
        test_result = []
        err_result = []
    for step, data in enumerate(train_set):
        data['iteration'] = step
        feed_dict = {input_ph[k]: data[k] for k in input_ph.keys()}
        output = train_function(feed_dict=feed_dict)
        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output[()]))
        interval_cost += output[()]
        if (step + 1) % args.iter_interval == 0 and step > 0:
            eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f} Test Avg loss:{tcost}".format(
                           interval=step // args.iter_interval,
                           iteration=step,
                           cost=interval_cost / args.iter_interval, tcost=eval_losses))
            if(args.name is not None):
                # For storing to csv
                train_result.append(interval_cost / args.iter_interval)
                test_result.append(eval_losses['cross_ent_loss'])
                err_result.append(eval_losses['misclass'])
            interval_cost = 0.0
    # Writing to CSV
    if(args.name is not None):
        print("\nSaving results to csv file")
        import csv
        with open(args.name + ".csv", 'wb') as train_test_file:
            wr = csv.writer(train_test_file, quoting=csv.QUOTE_ALL)
            wr.writerow(train_result)
            wr.writerow(test_result)
            wr.writerow(err_result)
    print("\nTraining Completed")
