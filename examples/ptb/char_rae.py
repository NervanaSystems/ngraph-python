#!/usr/bin/env python
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
"""
Character-level recurrent autoencoder. This model shows how to build an Encoder-Decoder style RNN.

The model uses a sequence from the PTB dataset as input, and learns to output
the same sequence in reverse order.
"""

import numpy as np
from contextlib import closing
import ngraph as ng
from ngraph.frontends.neon import Layer, Preprocess, Recurrent, Affine, Softmax, Tanh
from ngraph.frontends.neon import UniformInit, RMSProp
from ngraph.frontends.neon import ax, loop_train
from ngraph.frontends.neon import NgraphArgparser, make_bound_computation, make_default_callbacks
from ngraph.frontends.neon import SequentialArrayIterator
import ngraph.transformers as ngt

from ngraph.frontends.neon import PTB

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.set_defaults(batch_size=128, num_iterations=2000)
args = parser.parse_args()

# model parameters
time_steps = 5
hidden_size = 256
gradient_clip_value = 5

# download penn treebank
# set shift_target to be False, since it is going to predict the same sequence
tree_bank_data = PTB(path=args.data_dir, shift_target=False)
ptb_data = tree_bank_data.load_data()
train_set = SequentialArrayIterator(ptb_data['train'],
                                    batch_size=args.batch_size,
                                    time_steps=time_steps,
                                    total_iterations=args.num_iterations,
                                    reverse_target=True,
                                    get_prev_target=True)
valid_set = SequentialArrayIterator(ptb_data['valid'],
                                    batch_size=args.batch_size,
                                    time_steps=time_steps,
                                    total_iterations=10,
                                    reverse_target=True,
                                    get_prev_target=True)

inputs = train_set.make_placeholders()
ax.Y.length = len(tree_bank_data.vocab)


def generate_samples(inputs, encode, decode, num_time_steps):
    """
    Inference
    """
    encoding = encode(inputs)
    decoder_input = np.zeros(decode.computation_op.parameters[0].axes.lengths)
    state = encoding
    tokens = list()
    for step in range(num_time_steps):
        output, state = decode(decoder_input, state.squeeze())
        index = np.argmax(output, axis=0)
        decoder_input[:] = 0
        decoder_input[index] = 1
        tokens.append(index)

    return np.squeeze(np.array(tokens)).T


def expand_onehot(x):
    return ng.one_hot(x, axis=ax.Y)


# weight initialization
init = UniformInit(low=-0.08, high=0.08)

# model initialization
one_hot_enc = Preprocess(functor=expand_onehot)
enc = Recurrent(hidden_size, init, activation=Tanh(), reset_cells=True, return_sequence=False)
one_hot_dec = Preprocess(functor=expand_onehot)
dec = Recurrent(hidden_size, init, activation=Tanh(), reset_cells=True, return_sequence=True)
linear = Affine(init, activation=Softmax(), bias_init=init, axes=(ax.Y))

optimizer = RMSProp(decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                    gradient_clip_value=gradient_clip_value)

# build network graph
one_hot_enc_out = one_hot_enc(inputs['inp_txt'])
one_hot_dec_out = one_hot_dec(inputs['prev_tgt'])
enc_out = enc(one_hot_enc_out)
dec_out = dec(one_hot_dec_out, init_state=enc_out)
output_prob = linear(dec_out)

loss = ng.cross_entropy_multi(output_prob,
                              ng.one_hot(inputs['tgt_txt'], axis=ax.Y),
                              usebits=True)
mean_cost = ng.mean(loss, out_axes=[])
updates = optimizer(loss)

train_outputs = dict(batch_cost=mean_cost, updates=updates)
loss_outputs = dict(cross_ent_loss=loss)

# inference graph
with Layer.inference_mode_on():
    enc_out_inference = enc(one_hot_enc_out)

    # Create decoder placeholders
    axes = one_hot_dec_out.axes
    axes = axes - axes.recurrent_axis() + ng.make_axis(length=1, name="REC")
    decoder_input_inference = ng.placeholder(axes, name="input")
    decoder_state_inference = ng.placeholder(enc_out_inference.axes, name="state")
    dec_out_inference = dec(decoder_input_inference, init_state=decoder_state_inference)
    inference_out = linear(dec_out_inference)

encoder_computation = ng.computation(enc_out_inference, inputs["inp_txt"])
decoder_computation = ng.computation([inference_out, dec_out_inference],
                                     decoder_input_inference,
                                     decoder_state_inference)


######################
# Train Loop

# Now bind the computations we are interested in
with closing(ngt.make_transformer()) as transformer:
    # training computations
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, loss_outputs, inputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=valid_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    # inference computations
    encoder_function = transformer.add_computation(encoder_computation)
    decoder_function = transformer.add_computation(decoder_computation)

    # training
    loop_train(train_set, train_computation, cbs)

    # inference
    valid_set.reset()
    num_errors = 0
    for mb_idx, data in enumerate(valid_set):
        tokens = generate_samples(data["inp_txt"], encoder_function, decoder_function, time_steps)
        num_errors += len(np.argwhere(tokens != data["tgt_txt"]))
    num_total = valid_set.total_iterations * (time_steps * args.batch_size)
    print('Misclassification error: {} %'.format(float(num_errors) / num_total * 100))

    # print some samples
    for sample_idx in range(5):
        print(''.join([tree_bank_data.vocab[i] for i in tokens[sample_idx, :]])[::-1])
