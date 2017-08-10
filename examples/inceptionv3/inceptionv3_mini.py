#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
export CUDA_VISIBLE_DEVICES=3; python ./examples/inceptionv3/inceptionv3_mini.py -z 8 -t 10000 -b gpu --iter_interval 1000
"""
import numpy as np
import pickle
import ngraph as ng
import ngraph.transformers as ngt
from tqdm import tqdm
from contextlib import closing
from ngraph.frontends.neon import NgraphArgparser, ArrayIterator
from ngraph.frontends.neon import XavierInit, UniformInit, Layer
from ngraph.frontends.neon import Affine, Convolution, Pool2D, Sequential, Dropout
from ngraph.frontends.neon import Rectlin, Softmax, Identity, GradientDescentMomentum
from ngraph.frontends.neon import ax, Adam, RMSProp
from data import make_aeon_loaders
from inception_blocks import Inceptionv3_b1, Inceptionv3_b2, Inceptionv3_b3
from inception_blocks import Inceptionv3_b4, Inceptionv3_b5

def loop_eval(dataset, computation, metric_names):
    dataset._dataloader.reset()
    all_results = None
    for data in dataset:

        feed_dict = {inputs[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))

    reduced_results = {k: np.mean(v[:dataset._dataloader.ndata]) for k, v in all_results.items()}
    return reduced_results

np.seterr(all='raise')

parser = NgraphArgparser(description=__doc__)
# Default batch_size for convnet-googlenet is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 299
'''
X_train = np.random.uniform(-1, 1, (args.batch_size, 3, image_size, image_size))
y_train = np.ones(shape=(args.batch_size), dtype=np.int32)
train_data = {'image': {'data': X_train,
                        'axes': ('batch', 'C', 'height', 'width')},
              'label': {'data': y_train,
                        'axes': ('batch',)}}
train_set = ArrayIterator(train_data,
                          batch_size=args.batch_size,
                          total_iterations=args.num_iterations)
inputs = train_set.make_placeholders(include_iteration=True)
'''
ax.Y.length = 1000  # number of outputs of last layer.

# weight initialization
bias_init = UniformInit(low=-0.08, high=0.08)

manifest_dir = '/nfs/site/home/gkeskin/work/ngraph/dataset/i1k/'
train_set,valid_set=make_aeon_loaders(manifest_dir, args.batch_size,
                                      args.num_iterations, dataset='i1k')
inputs=train_set.make_placeholders(include_iteration=True)
# Input size is 299 x 299 x 3
# Root branch of the tree
seq1 = Sequential([Convolution((3, 3, 32), padding=0, strides=2, batch_norm=True,
                               activation=Rectlin(), bias_init=bias_init,
                               filter_init=XavierInit()),  # conv2d_1a_3x3
                   Convolution((3, 3, 16), activation=Rectlin(), padding=0, batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),  # conv2d_2a_3x3
                   Convolution((3, 3, 16), activation=Rectlin(), padding=1, batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),  # conv2d_2b_3x3
                   Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_3a_3x3 
                   Convolution((1, 1, 16), activation=Rectlin(), batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),  # conv2d_3b_1x1
                   Convolution((3, 3, 32), activation=Rectlin(), padding=1, batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),  # conv2d_4a_3x3
                   Pool2D(fshape=3, padding=0, strides=2, op='max'),  # maxpool_5a_3x3
                   Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5b 
                   Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5c 
                   Inceptionv3_b1([(32,), (32, 32), (32, 32, 32), (32, )]),  # mixed_5d 
                   Inceptionv3_b2([(32,), (32, 32, 32)]),  # mixed_6a 
                   Inceptionv3_b3([(32,), (32, 32, 32),
                                   (32, 32, 32, 32, 32), (32,)]),  # mixed_6b 
                   Inceptionv3_b3([(32,), (32, 32, 32),
                                   (32, 32, 32, 32, 32), (32,)]),  # mixed_6c 
                   Inceptionv3_b3([(32,), (32, 32, 32),
                                   (32, 32, 32, 32, 32), (32,)]),  # mixed_6d 
                   Inceptionv3_b3([(32,), (32, 32, 32),
                                   (32, 32, 32, 32, 32), (32,)])])  # mixed_6e 

# Branch of main classifier
seq2 = Sequential([Inceptionv3_b4([(32, 32), (32, 32, 32, 32)]),  # mixed_7a
                   Inceptionv3_b5([(32,), (32, 32, 32),
                                   (32, 32, 32, 32), (32,)]),  # mixed_7b
                   Inceptionv3_b5([(32,), (32, 32, 32),
                                   (32, 32, 32, 32), (32,)]),  # mixed_7c
                   Pool2D(fshape=8, padding=0, strides=2, op='avg'),  # Last Avg Pool 
                   Convolution((1, 1, 1000), activation=Softmax(),
                               bias_init=bias_init, filter_init=XavierInit())])

# Auxiliary classifier
seq_aux = Sequential([Pool2D(fshape=5, padding=0, strides=3, op='avg'), 
                      Convolution((1, 1, 32), activation=Rectlin(), batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),
                      Convolution((5, 5, 32), padding=0, activation=Rectlin(), batch_norm=True,
                               bias_init=bias_init, filter_init=XavierInit()),
                      Convolution((1, 1, 1000), activation=Softmax(),
                               bias_init=bias_init, filter_init=XavierInit())])

# To match tensorflow learning rate
optimizer = RMSProp(learning_rate=.01, decay_rate=0.9)

y_onehot = ng.one_hot(inputs['label'][:,0], axis=ax.Y)
train_prob_main = ng.cast_role(seq2(seq1(inputs['image']))[:,0,0,0,:], axes = y_onehot.axes)
train_loss_main = ng.cross_entropy_multi(train_prob_main, y_onehot)

train_prob_aux = ng.cast_role(seq_aux(seq1(inputs['image']))[:,0,0,0,:], axes=y_onehot.axes)
train_loss_aux = ng.cross_entropy_multi(train_prob_aux, y_onehot) 

batch_cost = ng.sequential([optimizer(train_loss_main + 0.4*train_loss_aux), ng.mean(train_loss_main, out_axes=())])
train_computation = ng.computation(batch_cost, 'all')

label_indices = inputs['label'][:, 0]
with Layer.inference_mode_on():
    inference_prob = ng.cast_role(seq2(seq1(inputs['image']))[:,0,0,0,:], axes = y_onehot.axes)
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
    eval_loss = ng.cross_entropy_multi(inference_prob, y_onehot)
    eval_loss_names = ['cross_ent_loss', 'misclass']
    eval_computation = ng.computation([eval_loss, errors], "all")

with closing(ngt.make_transformer()) as transformer:
    train_function = transformer.add_computation(train_computation)
    eval_function = transformer.add_computation(eval_computation)

    if args.no_progress_bar:
        ncols = 0
    else:
        ncols = 100

    tpbar = tqdm(unit="batches", ncols=ncols, total=args.num_iterations)
    interval_cost = 0.0

    saved_losses = {'train_loss': [], 'eval_loss': [], 'eval_misclass': []}
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
            saved_losses['train_loss'].append(interval_cost / args.iter_interval)
            interval_cost = 0.0
            eval_losses = loop_eval(valid_set, eval_function, eval_loss_names)
            saved_losses['eval_loss'].append(eval_losses['cross_ent_loss'])
            saved_losses['eval_misclass'].append(eval_losses['misclass'])
            pickle.dump( saved_losses, open( "losses.pkl", "wb" ) )
            tqdm.write("Avg losses: {}".format(eval_losses))
print('\n')
