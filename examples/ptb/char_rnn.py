# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
import ngraph as ng
from ngraph.frontends.neon import Sequential, nnPreprocess, nnRecurrent, nnAffine, Softmax, Tanh
from ngraph.frontends.neon import UniformInit, Callbacks, RMSProp, ax, make_keyed_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import SequentialArrayIterator
import ngraph.transformers as ngt

from ptb import PTB


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.set_defaults(gen_be=False)
args = parser.parse_args()

# these hyperparameters are from the paper
args.batch_size = 50
time_steps = 5
hidden_size = 10
gradient_clip_value = None

# download penn treebank
tree_bank_data = PTB(path=args.data_dir)
ptb_data = tree_bank_data.load_data()
train_set = SequentialArrayIterator(ptb_data['train'], batch_size=args.batch_size,
                                    time_steps=time_steps, total_iterations=args.num_iterations)

valid_set = SequentialArrayIterator(ptb_data['train'], batch_size=args.batch_size,
                                    time_steps=time_steps)

# weight initialization
init = UniformInit(low=-0.08, high=0.08)

# model initialization
seq1 = Sequential([nnPreprocess(functor=lambda x: ng.one_hot(x, axis=ax.Y)),
                   nnRecurrent(hidden_size, init, activation=Tanh()),
                   nnAffine(init, activation=Softmax(), bias=init, axes=(ax.Y, ax.REC))])

# Bind axes lengths:
ax.Y.length = len(tree_bank_data.vocab)
ax.REC.length = time_steps
ax.N.length = args.batch_size

# placeholders with descriptive names
inputs = dict(inp=ng.placeholder([ax.REC, ax.N]),
              tgt=ng.placeholder([ax.REC, ax.N]),
              idx=ng.placeholder([]))

optimizer = RMSProp(decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6)
output_prob = seq1.train_outputs(inputs['inp'])
train_cost = ng.cross_entropy_multi(output_prob,
                                    ng.one_hot(inputs['tgt'], axis=ax.Y),
                                    usebits=True)
mean_cost = ng.mean(train_cost, out_axes=[])
updates = optimizer(train_cost, inputs['idx'])

# Now bind the computations we are interested in
transformer = ngt.make_transformer()
train_computation = make_keyed_computation(transformer, [mean_cost, updates], inputs)

cb = Callbacks(seq1, args.output_file, args.iter_interval, show_progress=args.progress_bar)

######################
# Train Loop
cb.on_train_begin(args.num_iterations)

for mb_idx, data in enumerate(train_set):
    cb.on_minibatch_begin(mb_idx)
    batch_cost, _ = train_computation(dict(inp=data[0], tgt=data[1], idx=mb_idx))
    seq1.current_batch_cost = float(batch_cost)
    cb.on_minibatch_end(mb_idx)

cb.on_train_end()

# callbacks = Callbacks(rnn, eval_set=valid_set, **args.callback_args)
# rnn.initialize(
#     dataset=train_set,
#     input_axes=make_axes((ax.C, ax.REC)),
#     target_axes=make_axes((ax.Y, ax.REC)),
#     optimizer=opt,
#     cost=cost
# )

# rnn.fit(
#     train_set,
#     num_epochs=args.epochs,
#     callbacks=callbacks
# )
