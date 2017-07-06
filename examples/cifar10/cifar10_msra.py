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
CIFAR MSRA with spelled out neon model framework in one file

Run it using

python examples/cifar10/cifar10_msra.py --stage_depth 2 --data_dir /usr/local/data/CIFAR

For full training, the number of iterations should be 64000 with batch size 128.

"""
from __future__ import division, print_function
from builtins import range
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, NgraphArgparser
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks, loop_train  # noqa
from tqdm import tqdm
import ngraph.transformers as ngt

parser = NgraphArgparser(description='Train deep residual network on cifar10 dataset')
parser.add_argument('--stage_depth', type=int, default=2,
                    help='depth of each stage (network depth will be 9n+2)')
parser.add_argument('--use_aeon', action='store_true', help='whether to use aeon dataloader')
args = parser.parse_args()

np.random.seed(args.rng_seed)

# Create the dataloader
if args.use_aeon:
    from data import make_aeon_loaders
    train_set, valid_set = make_aeon_loaders(args.data_dir, args.batch_size, args.num_iterations)
else:
    from ngraph.frontends.neon import ArrayIterator  # noqa
    from ngraph.frontends.neon import CIFAR10  # noqa
    train_data, valid_data = CIFAR10(args.data_dir).load_data()
    train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
    valid_set = ArrayIterator(valid_data, args.batch_size)

# we need to ask the dataset to create an iteration placeholder for our learning rate schedule
inputs = train_set.make_placeholders(include_iteration=True)
ax.Y.length = 10

######################
# Model specification


def cifar_mean_subtract(x):
    # Assign roles
    bgr_mean = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([104., 119., 127.]))

    return (x - bgr_mean) / 255.


def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm),
                strides=strides,
                padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                filter_init=KaimingInit(),
                batch_norm=batch_norm)


class f_module(object):
    def __init__(self, nfm, first=False, strides=1):

        self.trunk = None
        self.side_path = None

        main_path = [Convolution(**conv_params(1, nfm, strides=strides)),
                     Convolution(**conv_params(3, nfm)),
                     Convolution(**conv_params(1, nfm * 4, relu=False, batch_norm=False))]

        if first or strides == 2:
            self.side_path = Convolution(
                **conv_params(1, nfm * 4, strides=strides, relu=False, batch_norm=False))
        else:
            main_path = [BatchNorm(), Activation(Rectlin())] + main_path

        if strides == 2:
            self.trunk = Sequential([BatchNorm(), Activation(Rectlin())])

        self.main_path = Sequential(main_path)

    def __call__(self, x):
        t_x = self.trunk(x) if self.trunk else x
        s_y = self.side_path(t_x) if self.side_path else t_x
        m_y = self.main_path(t_x)
        return s_y + m_y


class residual_network(Sequential):

    def __init__(self, stage_depth):
        nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * stage_depth)]
        print(nfms)
        strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

        layers = [Preprocess(functor=cifar_mean_subtract),
                  Convolution(**conv_params(3, 16)),
                  f_module(nfms[0], first=True)]

        for nfm, stride in zip(nfms[1:], strides):
            layers.append(f_module(nfm, strides=stride))

        layers.append(BatchNorm())
        layers.append(Activation(Rectlin()))
        layers.append(Pool2D(8, op='avg'))
        layers.append(Affine(axes=ax.Y,
                             weight_init=KaimingInit(),
                             activation=Softmax()))
        super(residual_network, self).__init__(layers=layers)


def loop_eval(dataset, computation, metric_names):
    dataset.reset()
    all_results = None
    for data in dataset:

        feed_dict = {inputs[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))

    reduced_results = {k: np.mean(v[:dataset.ndata]) for k, v in all_results.items()}
    return reduced_results


resnet = residual_network(args.stage_depth)

learning_rate_policy = {'name': 'schedule',
                        'schedule': [32000, 48000],
                        'gamma': 0.1,
                        'base_lr': 0.1}

optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                    momentum_coef=0.9,
                                    wdecay=0.0001,
                                    iteration=inputs['iteration'])
label_indices = inputs['label']
train_loss = ng.cross_entropy_multi(resnet(inputs['image']),
                                    ng.one_hot(label_indices, axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

with Layer.inference_mode_on():
    inference_prob = resnet(inputs['image'])
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    eval_loss_names = ['cross_ent_loss', 'misclass']
    eval_computation = ng.computation([eval_loss, errors], "all")

# Now bind the computations we are interested in
transformer = ngt.make_transformer()
train_function = transformer.add_computation(train_computation)
eval_function = transformer.add_computation(eval_computation)

tpbar = tqdm(unit="batches", ncols=100, total=args.num_iterations)
interval_cost = 0.0

for step, data in enumerate(train_set):
    data['iteration'] = step
    feed_dict = {inputs[k]: data[k] for k in inputs.keys()}
    output = train_function(feed_dict=feed_dict)

    tpbar.update(1)
    tpbar.set_description("Training {:0.4f}".format(output[()]))
    interval_cost += output[()]
    if (step + 1) % args.iter_interval == 0 and step > 0:
        tqdm.write("Interval {interval} Iteration {iteration} complete. "
                   "Avg Train Cost {cost:0.4f}".format(
                       interval=step // args.iter_interval,
                       iteration=step,
                       cost=interval_cost / args.iter_interval))
        interval_cost = 0.0
        eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
        tqdm.write("Avg losses: {}".format(eval_losses))

print("Training complete.")
