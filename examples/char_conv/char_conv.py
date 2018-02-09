# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from contextlib import closing
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import ax
from ngraph.frontends.neon import NgraphArgparser, ArrayIterator
from ngraph.frontends.neon import GaussianInit, ConstantInit
from ngraph.frontends.neon import Sequential, Layer, Convolution, \
    Pooling, Affine, LookupTable, Dropout
from ngraph.frontends.neon import Rectlin, Softmax
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import make_bound_computation, loop_train, \
    loop_eval, make_default_callbacks
from dataset import CrepeDataset


def conv_params(ksize, nout, init):
    return dict(
        filter_shape=(ksize, nout),
        activation=Rectlin(),
        filter_init=init,
        bias_init=ConstantInit(0.)
    )


def make_iterators(dbpedia_data, num_iterations, batch_size):
    train_set = ArrayIterator(dbpedia_data['train'], batch_size=batch_size,
                              total_iterations=num_iterations, shuffle=True)
    test_set = ArrayIterator(dbpedia_data['test'], batch_size=batch_size)
    return train_set, test_set


def make_embedding_layer(vocab_size):
    vectors = []
    vectors.append(np.zeros((1, vocab_size)))
    vectors.append(np.eye(vocab_size))
    vectors = np.concatenate(vectors)

    embed_init = ConstantInit(vectors)
    embed_layer = LookupTable(vocab_size + 1,
                              vocab_size,
                              embed_init,
                              update=False,
                              pad_idx=0)
    return embed_layer


def make_layers(use_large, vocab_size):

    if use_large:
        init = GaussianInit(0., 0.02)
    else:
        init = GaussianInit(0., 0.05)

    layers = []
    layers.append(make_embedding_layer(vocab_size))
    layers.append(lambda op: ng.map_roles(op, {'REC': 'W', 'F': 'C'}))

    kernel_sizes = [7, 7, 3, 3, 3, 3]
    pool_layer_idxs = [0, 1, 5]
    conv_nout = 1024 if use_large else 256
    fc_nout = 2048 if use_large else 1024
    for i in range(6):
        conv_layer = Convolution(**conv_params(kernel_sizes[i], conv_nout, init))
        layers.append(conv_layer)
        if i in pool_layer_idxs:
            pool_layer = Pooling(pool_shape=(3,), strides=3)
            layers.append(pool_layer)
    layers.append(Affine(nout=fc_nout,
                         weight_init=init,
                         bias_init=ConstantInit(0.),
                         activation=Rectlin()))
    layers.append(Dropout(keep=0.5))
    layers.append(Affine(nout=fc_nout,
                         weight_init=init,
                         bias_init=ConstantInit(0.),
                         activation=Rectlin()))
    layers.append(Dropout(keep=0.5))
    layers.append(Affine(axes=(ax.Y,),
                         weight_init=init,
                         bias_init=ConstantInit(0.),
                         activation=Softmax()))

    return layers


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--sentence_length', type=int, default=1014,
                    help='the number of characters in a sentence')
parser.add_argument('--use_uppercase', action='store_true', default=False,
                    help='whether to use uppercase characters in the vocabulary')
parser.add_argument('--use_large', action='store_true', default=False,
                    help='whether to use the large model')
parser.add_argument('-e', '--num_epochs', type=int, default=10,
                    help='the number of epochs to train')
parser.add_argument('--num_classes', type=int, required=True,
                    help='the number of classes')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='the weight decay')
parser.add_argument('--use_lr_decay', action='store_true', default=False,
                    help='whether to use a decaying lr schedule')
parser.add_argument('--lr', type=float, default=0.01,
                    help='the base learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='the momentum coefficient in the optimizer')
args = parser.parse_args()

args.batch_size = 128
dbpedia_dataset = CrepeDataset(path=args.data_dir,
                               sentence_length=args.sentence_length,
                               use_uppercase=args.use_uppercase)
dbpedia_data = dbpedia_dataset.load_data()
train_nexamples = len(dbpedia_data['train']['label']['data'])
args.num_iterations = (args.num_epochs * train_nexamples) // args.batch_size

train_set, test_set = make_iterators(dbpedia_data, args.num_iterations, args.batch_size)

inputs = train_set.make_placeholders(include_iteration=args.use_lr_decay)
ax.Y.length = args.num_classes

layers = make_layers(args.use_large, dbpedia_dataset.vocab_size)
seq = Sequential(layers)

if args.use_lr_decay:
    lr_schedule = [(i + 1) * 3 * train_set.nbatches for i in range(10)]
    lr_policy = {'name': 'schedule', 'base_lr': args.lr, 'schedule': lr_schedule, 'gamma': 0.5}
    optimizer = GradientDescentMomentum(lr_policy,
                                        momentum_coef=args.momentum,
                                        iteration=inputs['iteration'],
                                        wdecay=args.weight_decay)
else:
    optimizer = GradientDescentMomentum(args.lr,
                                        momentum_coef=args.momentum,
                                        wdecay=args.weight_decay)

train_prob = seq(inputs['text'])
train_loss = ng.cross_entropy_multi(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with Layer.inference_mode_on():
    inference_prob = seq(inputs['text'])

errors = ng.not_equal(ng.argmax(inference_prob, reduction_axes=[ax.Y]), inputs['label'])
eval_loss = ng.cross_entropy_multi(inference_prob,
                                   ng.one_hot(inputs['label'], axis=ax.Y))
eval_outputs = dict(cross_ent_loss=eval_loss, misclass_pct=errors)

# Now bind the computations we are interested in
with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, eval_outputs, inputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, train_computation, cbs)
    print("Testing...")
    print(loop_eval(test_set, loss_computation))
