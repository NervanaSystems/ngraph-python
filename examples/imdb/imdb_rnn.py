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
from __future__ import division
from __future__ import print_function
import ngraph as ng
from ngraph.frontends.neon import (Layer, Sequential, BiRNN, Recurrent, Affine,
                                   Softmax, Tanh, LookupTable)
from ngraph.frontends.neon import UniformInit, RMSProp
from ngraph.frontends.neon import ax, loop_train, make_bound_computation, make_default_callbacks
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
import ngraph.transformers as ngt

from imdb import IMDB


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--layer_type', default='rnn', choices=['rnn', 'birnn'],
                    help='type of recurrent layer to use (rnn or birnn)')
parser.set_defaults(gen_be=False)
args = parser.parse_args()

# these hyperparameters are from the paper
args.batch_size = 128
time_steps = 128
hidden_size = 10
gradient_clip_value = 15
embed_size = 128
vocab_size = 20000
pad_idx = 0

# download IMDB
imdb_dataset = IMDB(path=args.data_dir, sentence_length=time_steps, pad_idx=pad_idx)
imdb_data = imdb_dataset.load_data()

train_set = ArrayIterator(imdb_data['train'], batch_size=args.batch_size,
                          total_iterations=args.num_iterations)
valid_set = ArrayIterator(imdb_data['valid'], batch_size=args.batch_size)

inputs = train_set.make_placeholders()
ax.Y.length = imdb_dataset.nclass

# weight initialization
init = UniformInit(low=-0.08, high=0.08)

if args.layer_type == "rnn":
    rlayer = Recurrent(hidden_size, init, activation=Tanh(),
                       reset_cells=True, return_sequence=False)
else:
    rlayer = BiRNN(hidden_size, init, activation=Tanh(),
                   reset_cells=True, return_sequence=False, sum_out=True)

# model initialization
seq1 = Sequential([LookupTable(vocab_size, embed_size, init, update=True, pad_idx=pad_idx),
                   rlayer,
                   Affine(init, activation=Softmax(), bias_init=init, axes=(ax.Y,))])

optimizer = RMSProp(decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                    gradient_clip_value=gradient_clip_value)

train_prob = seq1(inputs['review'])
train_loss = ng.cross_entropy_multi(train_prob,
                                    ng.one_hot(inputs['label'], axis=ax.Y),
                                    usebits=True)

batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=[])])
train_outputs = dict(batch_cost=batch_cost)

with Layer.inference_mode_on():
    inference_prob = seq1(inputs['review'])
eval_loss = ng.cross_entropy_multi(inference_prob,
                                   ng.one_hot(inputs['label'], axis=ax.Y),
                                   usebits=True) #suspicious
eval_outputs = dict(cross_ent_loss=eval_loss)

# Now bind the computations we are interested in
transformer = ngt.make_transformer()
train_computation = make_bound_computation(transformer, train_outputs, inputs)
loss_computation = make_bound_computation(transformer, eval_outputs, inputs)

cbs = make_default_callbacks(output_file=args.output_file,
                             frequency=args.iter_interval,
                             train_computation=train_computation,
                             total_iterations=args.num_iterations,
                             eval_set=valid_set,
                             loss_computation=loss_computation,
                             use_progress_bar=args.progress_bar)

loop_train(train_set, train_computation, cbs)
