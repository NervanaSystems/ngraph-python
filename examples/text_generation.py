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
Usage:
python text_generation.py -z 128 -b gpu -t 100000 --predict_seq

Uses Shakespeare text sample to train an LSTM based model to predict the next character
Uses the trained model to generate own text

Parameters:
--recurrent_units : Number of cells in the LSTM (default 128)
--seq_len : Length of each string input to the LSTM (number of chars, default 32)
--predict_seq : If given in command line, the output of the LSTM will be a sequence
                If not, only the next character is predicted
--use_embedding : If given in command line, first layer of the network is Embedding Layer
                  If not, one hot encoding of characters is used
    Example:
        predict_seq True:
            Input ['My name is Georg'], Ground Truth Output: ['y name is George']
        predict_seq False:
            Input ['My name is Georg'], Ground Truth Output: ['e']
"""
from __future__ import division, print_function
from builtins import range
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, LSTM, Affine, Softmax, Preprocess, LookupTable
from ngraph.frontends.neon import UniformInit, Tanh, Logistic, RMSProp
from ngraph.frontends.neon import NgraphArgparser
import ngraph.transformers as ngt
from ngraph.frontends.neon import Shakespeare, RollingWindowIterator


def eval_loop(inputs, eval_set, eval_function):
    """
    Evaluates the trained model on the evaluation text sample
    """
    shakes_test.reset()
    eval_cost = 0
    for eval_iter, eval_sample in enumerate(eval_set):
        feed_dict = {inputs[k]: eval_sample[k] for k in inputs.keys()}
        [batch_eval_cost, batch_eval_outs] = eval_function(feed_dict=feed_dict)
        eval_cost += np.mean(batch_eval_cost)

    eval_cost = eval_cost / (eval_iter + 1)
    return eval_cost


def train_loop(inputs, train_set, train_function, eval_set,
               eval_function, gen_function,
               index_to_token, token_to_index):
    """
    Trains the model using train_set text sample
    Evaluates the model at regular intervals (iter_interval) on the eval_set
    At every iter_interval, generates its own 250 character long text
    For generation, a seed sentence is used
    """
    train_cost_hist = []
    interval_cost = 0.
    for iter_val, sample in enumerate(train_set):
        # Train the network for one batch
        feed_dict = {inputs[k]: sample[k] for k in inputs.keys()}
        [batch_train_cost, batch_train_outs] = train_function(feed_dict=feed_dict)
        # Keep a record of the mean cost of this batch
        interval_cost += batch_train_cost

        # At regular intervals, evaluate the model on the eval_set
        # Also generate text based on the model
        if ((iter_val + 1) % iter_interval == 0):
            # Find average batch in this interval
            interval_cost = interval_cost / iter_interval

            # Store the cost of this interval, to keep track of model evolution
            train_cost_hist.append(interval_cost)

            # Iterate over the evaluation set and get evaluation cost
            eval_cost = eval_loop(inputs, eval_set, eval_function)
            print('****\nIteration: %d, Train Cost: %1.2e, Eval Cost: %1.2e\n****'
                  % (iter_val + 1, interval_cost, eval_cost))

            # Reset the interval cost
            interval_cost = 0.

            # Take the first sample in eval_set as the seed sentence
            eval_set.reset()
            first_batch = next(eval_set.__iter__())
            sentence = first_batch['X'][0, ...]

            # We will compute the forward pass of the seed text
            # Since we defined the axes, model expects batch_size samples at each forward pass
            # We will use the first sample in the batch as our seed text
            # Output of other samples will be computed, but simply discarded
            seed_txt = sample['X']  # Initialize the full batch to the current training batch
            seed_txt[0, :] = sentence  # Set the first sample of the batch to our seed text

            # Generate the text
            gen_text = generate_text(inputs, gen_function, seed_txt, index_to_token)
            print('\nSample Text:')
            print(''.join(gen_text))
            print('\nEnd Sample Text\n')

    return train_cost_hist


def generate_text(inputs, generation_function, seed_txt, index_to_token,
                  gen_txt_length=250):
    """
    Generates a text given the seed sentence (seed_txt) for gen_txt_length chars
    Uses the trained model and the generation_function
    Generation function is the forward pass of the network

    We start with the seed, and forward pass it
    Fwd pass gives predicted probabilities of each token in the vocabulary
    We sample this distribution, the sample gives the next character
    This next character is appended to the seed text
    Earliest char in the seed is discarded
    Process is repeated gen_txt_length times
    """
    feed_dict = {inputs['X']: seed_txt}
    gen_txt = []
    for gen_iter in range(gen_txt_length):
        # Get the probability of each character
        gen_chars = generation_function(feed_dict=feed_dict)

        # Get the probability for each character for the first sample
        if(predict_seq is False):
            gen_chars = gen_chars[:, 0]
        else:
            gen_chars = gen_chars[:, -1, 0]
        # Due to rounding errors, sum of softmax output could be very slightly above 1
        # This throws an error in np.random.multinomial
        # Scale softmax outputs so they sum up to 1
        gen_chars = gen_chars / (gen_chars.sum() + 1e-6)
        # Sample the next character from the scaled distribution
        pred_char = np.argmax(np.random.multinomial(1, gen_chars, 1))

        # Append the sampled char to the seed_txt's first sample
        seed_txt[0, :-1] = seed_txt[0, 1:]
        seed_txt[0, -1] = pred_char
        feed_dict = {inputs['X']: seed_txt}

        # Append the sampled character to generated text
        gen_txt.append(pred_char)

    # Convert integer index of tokens to actual tokens
    gen_txt = [index_to_token[i] for i in gen_txt]
    return gen_txt


def expand_onehot(x):
    """
    Simply converts an integer to a one-hot vector of the same size as out_axis
    """
    return ng.one_hot(x, axis=out_axis)


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--use_embedding', default=False, dest='use_embedding', action='store_true',
                    help='If given, embedding layer is used as the first layer')
parser.add_argument('--predict_seq', default=False, dest='predict_seq', action='store_true',
                    help='If given, seq_len future timepoints are predicted')
parser.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=32)
parser.add_argument('--recurrent_units', type=int,
                    help="Number of recurrent units in the network",
                    default=256)
parser.set_defaults(num_iterations=40000)
args = parser.parse_args()

use_embedding = args.use_embedding
predict_seq = args.predict_seq
recurrent_units = args.recurrent_units
batch_size = args.batch_size
seq_len = args.seq_len
num_iterations = args.num_iterations

# Ratio of the text to use for training
train_ratio = 0.95
# Define initialization method of neurons in the network
init_uni = UniformInit(-0.1, 0.1)

# Create the object that includes the sample text
shakes = Shakespeare(train_split=train_ratio)

# Build an iterator that gets seq_len long chunks of characters
# Stride is by how many characters the window moves in each step
# if stride is set to seq_len, windows are non-overlapping
stride = seq_len // 8
shakes_train = RollingWindowIterator(data_array=shakes.train, total_iterations=num_iterations,
                                     seq_len=seq_len, batch_size=batch_size, stride=stride,
                                     return_sequences=predict_seq)
shakes_test = RollingWindowIterator(data_array=shakes.test,
                                    seq_len=seq_len, batch_size=batch_size, stride=stride,
                                    return_sequences=predict_seq)

# Our input is of size (batch_size, seq_len)
# batch_axis must be named N
batch_axis = ng.make_axis(length=batch_size, name="N")
# time_axis must be named REC
time_axis = ng.make_axis(length=seq_len, name="REC")

# Output is of size (vocab_size + 1,1)
# +1 is for unknown token
out_axis = ng.make_axis(length=len(shakes.vocab) + 1, name="out_feature_axis")
in_axes = ng.make_axes([batch_axis, time_axis])

# RollingWindowIterator gives an output of (batch_size, 1) for each iteration
# We will later convert this output to onehot
if(predict_seq is True):
    out_axes = ng.make_axes([batch_axis, time_axis])
else:
    out_axes = ng.make_axes([batch_axis])

# Build placeholders for the created axes
inputs = {'X': ng.placeholder(in_axes), 'y': ng.placeholder(out_axes),
          'iteration': ng.placeholder(axes=())}

# Network Definition
if(use_embedding is False):
    seq1 = Sequential([Preprocess(functor=expand_onehot),
                       LSTM(nout=recurrent_units, init=init_uni, backward=False, reset_cells=True,
                            activation=Logistic(), gate_activation=Tanh(),
                            return_sequence=predict_seq),
                       Affine(weight_init=init_uni, bias_init=init_uni,
                              activation=Softmax(), axes=out_axis)])
else:
    embedding_dim = 8
    seq1 = Sequential([LookupTable(len(shakes.vocab) + 1, embedding_dim, init_uni, update=True),
                       LSTM(nout=recurrent_units, init=init_uni, backward=False, reset_cells=True,
                            activation=Logistic(), gate_activation=Tanh(),
                            return_sequence=predict_seq),
                       Affine(weight_init=init_uni, bias_init=init_uni,
                              activation=Softmax(), axes=out_axis)])

# Optimizer
# Initial learning rate is 0.01 (base_lr)
# At iteration (num_iterations // 75), lr is multiplied by gamma (new lr = .95 * .01)
# At iteration (num_iterations * 2 // 75), it is reduced by gamma again
# So on..
no_steps = 75
step = num_iterations // no_steps
schedule = list(np.arange(step, num_iterations, step))
learning_rate_policy = {'name': 'schedule',
                        'schedule': schedule,
                        'gamma': 0.95,
                        'base_lr': 0.01}
optimizer = RMSProp(gradient_clip_value=1, learning_rate=learning_rate_policy,
                    iteration=inputs['iteration'])

# Define the loss function (Cross entropy loss)
# Note that we convert the integer values of input['y'] to one hot here
fwd_prop = seq1(inputs['X'])
train_loss = ng.cross_entropy_multi(fwd_prop,
                                    ng.one_hot(inputs['y'], axis=out_axis),
                                    usebits=True)

# Train cost computation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation([batch_cost, fwd_prop], "all")
train_outputs = dict(batch_cost=batch_cost)

# Forward prop of evaluation set
# Required for correct functioning of batch norm and dropout layers during inference mode
with Layer.inference_mode_on():
    inference_prop = seq1(inputs['X'])
eval_loss = ng.cross_entropy_multi(inference_prop,
                                   ng.one_hot(inputs['y'], axis=out_axis),
                                   usebits=True)
eval_computation = ng.computation([eval_loss, inference_prop], "all")
eval_outputs = dict(x_ent_loss=eval_loss)

# Computation for text generation - this is pure inference (fwd prop)
gen_computation = ng.computation(inference_prop, "all")

print('Start training ...')
with closing(ngt.make_transformer()) as transformer:
    # Add computations to the transformer
    train_function = transformer.add_computation(train_computation)
    eval_function = transformer.add_computation(eval_computation)
    generate_function = transformer.add_computation(gen_computation)

    # Determine printout interval of the validation set loss during training
    iter_interval = min(4000, num_iterations // 20)

    # Training Loop
    train_cost = train_loop(inputs, shakes_train, train_function, shakes_test,
                            eval_function, generate_function,
                            shakes.index_to_token, shakes.token_to_index)
