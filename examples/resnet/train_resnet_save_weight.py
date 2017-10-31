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
from ngraph.frontends.neon import Saver


# Calculate metrics over given dataset
def loop_eval(dataset, input_ph, metric_name, computation, en_top5=False):
    # Reset test set
    dataset.reset()
    all_results = None
    # Iterating over the dataset
    for data in dataset:
        feed_dict = {input_ph[k]: data[k] for k in data.keys()}
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
        total_results = [results_miss_loss, top5_results, top1_results] if en_top5 else [results_miss_loss, top1_results]
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
    parser.add_argument('--name', type=str, default=None, help="Name of the csv which \
                        logs different metrics of model")
    args = parser.parse_args()

    # Checking Command line args are proper
    cifar_sizes = [8, 20, 32, 44, 56, 110]
    i1k_sizes = [18, 34, 50, 101, 152]
    if args.dataset == 'cifar10':
        metric_names = ['Cross_Ent_Loss', 'Misclass']
        en_top5 = False
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
        metric_names = ['Cross_Ent_Loss', 'Top5_Err', 'Top1_Err'] 
        en_top5 = True
        dataset_sizes = i1k_sizes
        if args.size in dataset_sizes:
            # Enable or disable bottleneck depending on resnet size
            if(args.size in [18, 34]):
                en_bottleneck = False
            else:
                en_bottleneck = True
            # Change iter_interval to print every epoch. TODO
            args.iter_interval = 1301000 // args.batch_size
            learning_schedule = [30 * args.iter_interval, 60 * args.iter_interval]
            print("Learning Schedule: " + str(learning_schedule))
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

# Learning Rate Placeholder
lr_ph = ng.placeholder(axes=(), initial_value=base_lr)

# Optimizer
learning_rate_policy = {'name': 'dynamic',
                        'lr_placeholder': lr_ph}

optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                    momentum_coef=momentum_coef,
                                    wdecay=wdecay,
                                    nesterov=False,
                                    iteration=input_ph['iteration'])
label_indices = input_ph['label']
# Make a prediction
prediction = resnet(input_ph['image'])
# Calculate loss
train_loss = ng.cross_entropy_multi(prediction, ng.one_hot(label_indices, axis=ax.Y))
# Average loss over the batch
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

# Inference 
with Layer.inference_mode_on():
    # Doing inference
    inference_prob = resnet(input_ph['image'])
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    # Computation for inference
    eval_computation = ng.computation([inference_prob, eval_loss], "all")

weight_saver = Saver()
# Training the network by calling transformer
with closing(ngt.make_transformer()) as transformer:
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
    if(args.name is not None):
        train_result = []
        test_result = []
        err_result = []
    # Iterating over the training set
    for step, data in enumerate(train_set):
        data['iteration'] = step
        # Dictionary for training
        feed_dict = {input_ph[k]: data[k] for k in input_ph.keys()}
        # Learning Schedule
        feed_dict[lr_ph] = base_lr
        if((step >= learning_schedule[0]) and (step < learning_schedule[1])):
            feed_dict[lr_ph] = base_lr * gamma
        if(step >= learning_schedule[1]):
            feed_dict[lr_ph] = base_lr * gamma * gamma
        # Mean batch cost
        output = train_function(feed_dict=feed_dict)
        # Update progress bar
        tpbar.update(1)
        tpbar.set_description("Training {:0.4f}".format(output[()]))
        interval_cost += output[()]
        # Every epoch print test set metrics
        if (step + 1) % args.iter_interval == 0 and step > 0:
            # Call loop_eval to calculate metric over test set
            eval_losses = loop_eval(valid_set, input_ph, metric_names, eval_function, en_top5)
            tqdm.write("Interval {interval} Iteration {iteration} complete. "
                       "Avg Train Cost {cost:0.4f} Test Metrics:{tcost}".format(
                           interval=step // args.iter_interval,
                           iteration=step,
                           cost=interval_cost / args.iter_interval, tcost=eval_losses))
            # Log metrics to list
            if(args.name is not None):
                # For storing to csv
                train_result.append(interval_cost / args.iter_interval)
                test_result.append(eval_losses['Cross_Ent_Loss'])
                if args.dataset == 'cifar10':
                    err_result.append(eval_losses['Misclass'])
                if args.dataset == 'i1k':
                    err_result.append(eval_losses['Top5_Err'])
            interval_cost = 0.0
    tpbar.close()
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

    print("\nTesting weight save/loading")
    # Save weights at end of training
    weight_saver.save(filename="weights")

with Layer.inference_mode_on():
    # Doing inference post weight restore
    restore_inference_prob = resnet(input_ph['image'])
    restore_eval_loss = ng.cross_entropy_multi(restore_inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    # Computation for inference
    restore_eval_computation = ng.computation([restore_inference_prob, restore_eval_loss], "all")

with closing(ngt.make_transformer()) as transformer:
    restore_eval_function = transformer.add_computation(restore_eval_computation)
    # Restore weight
    weight_saver.setup_restore(transformer=transformer, vomputation=restore_eval_computation)
    weight_saver.restore()

    restore_eval_losses = loop_eval(valid_set, input_ph, metric_names, restore_eval_function, en_top5)
    print("From restored weights: Test Avg loss:{tcost}".format(tcost=restore_eval_losses))
    print("\nComplete: Testing weight save/loading")
