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
'''
 Usage:
    python text_generation.py -z 128 -b gpu -t 100000

 Uses Shakespeare text sample to train an LSTM based model to predict the next character
 Uses the trained model to generate own text
'''
from __future__ import division, print_function
from builtins import range
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, LSTM, Affine, Softmax, Preprocess
from ngraph.frontends.neon import UniformInit, Tanh, Logistic, RMSProp
from ngraph.frontends.neon import NgraphArgparser
import ngraph.transformers as ngt
from contextlib import closing
from rollingWindowIterator import RollingWindowIterator
from ngraph.frontends.neon import Shakespeare


def eval_loop(eval_set, eval_function):
    # Evaluates the trained model on the evaluation text sample
    shakes_test.reset()
    eval_cost = 0
    for eval_iter, eval_sample in enumerate(eval_set):
        feed_dict = {inputs[k]: eval_sample[k] for k in inputs.keys()}
        [batch_eval_cost, batch_eval_outs] = eval_function(feed_dict=feed_dict)
        eval_cost += np.mean(batch_eval_cost)

    eval_cost = eval_cost / (eval_iter + 1)
    return eval_cost


def train_loop(train_set, train_function, eval_set,
               eval_function, gen_function,
               index_to_token, token_to_index):
    # Trains the model using train_set text sample
    # Evaluates the model at regular intervals (iter_interval) on the eval_set
    # At every iter_interval, generates its own 250 character long text
    # For generation, a seed sentence is used
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
        if (((iter_val + 1) % iter_interval) == 0):
            # Find average batch in this interval
            interval_cost = interval_cost / iter_interval

            # Store the cost of this interval, to keep track of model evolution
            train_cost_hist.extend([interval_cost])

            # Iterate over the evaluation set and get evaluation cost
            eval_cost = eval_loop(eval_set, eval_function)
            print('****\nIteration: %d, Train Cost: %1.2e, Eval Cost: %1.2e\n****'
                  % (iter_val + 1, interval_cost, eval_cost))

            # Reset the interval cost
            interval_cost = 0.

            # Generate text based on the following seed sentence
            sentence = 'We are accounted poor citizens, the patricians good. \
                        What authority surfeits on would relieve us: if they'
            # Convert seed text to a sequence of integers
            sentence = np.asarray([token_to_index[sentence[i]] for i in range(seq_len)])
            sentence = sentence[:, np.newaxis]

            # We will compute the forward pass of the seed text
            # Since we defined the axes, model expects batch_size samples at each forward pass
            # We will use the first sample in the batch as our seed text
            # Output of other samples will be computed, but simply discarded
            seed_txt = sample['X']  # Initialize the full batch to the current training batch
            seed_txt[0, :] = sentence  # Set the first sample of the batch to our seed text

            # Generate the text
            gen_text = generate_text(gen_function, seed_txt)
            # Convert integer index of tokens to actual tokens
            gen_text = [index_to_token[i] for i in gen_text]
            print('****\nSample Text:')
            print(''.join(gen_text))
            print('****\nEnd Sample Text\n')

    return train_cost_hist


def generate_text(generation_function, seed_txt, gen_txt_length=250):
    # Generates a text given the seed sentence (seed_txt)
    # Uses the trained model and the generation_function
    # Generation function is simply the forward pass of the network

    # We start with the seed, and forward pass it
    # Fwd pass gives predicted probabilities of each token in the vocabulary
    # We sample this distribution, the sample gives the next character
    # This next character is appended to the seed text
    # Earliest char in the seed is discarded
    # Process is repeated gen_txt_length times
    feed_dict = {inputs['X']: seed_txt}
    gen_txt = []
    for gen_iter in range(gen_txt_length):
        # Get the probability of each character
        gen_chars = generation_function(feed_dict=feed_dict)

        # Get the probability for each character for the first sample
        gen_chars = gen_chars[:, 0]
        # Scale the probabilites of all characters so that they sum up to 1
        gen_chars = gen_chars / (gen_chars.sum() + 1e-6)

        # Sample the next character from the scaled distribution
        pred_char = np.argmax(np.random.multinomial(1, gen_chars, 1))

        # Append the sampled char to the seed_txt's first sample
        seed_txt[0, :-1, 0] = seed_txt[0, 1:, 0]
        seed_txt[0, -1, 0] = pred_char
        feed_dict = {inputs['X']: seed_txt}

        # Append the sampled character to generated text
        gen_txt.append(pred_char)
    return gen_txt


def expand_onehot(x):
    # Simply converts an integer to a one-hot vector of the same size as out_axis
    return ng.one_hot(x, axis=out_axis)


if __name__ == "__main__":

    # Ratio of the text to use for training
    train_ratio = 0.95

    # Define initialization method of neurons in the network
    init_uni = UniformInit(-0.1, 0.1)

    # parse the command line arguments
    parser = NgraphArgparser(__doc__)
    parser.add_argument('--predict_seq', default=False, dest='predict_seq', action='store_true',
                        help='If given, seq_len future timepoints are predicted')
    parser.add_argument('--seq_len', type=int,
                        help="Number of time points in each input sequence",
                        default=32)
    parser.add_argument('--recurrent_units', type=int,
                        help="Number of recurrent units in the network",
                        default=128)
    parser.set_defaults()
    args = parser.parse_args()

    # Recurrent units
    recurrent_units = args.recurrent_units

    # Batch size
    batch_size = args.batch_size

    # Sequence length
    seq_len = args.seq_len

    # Iterations
    num_iterations = args.num_iterations

    # Create the object that includes the sample text
    shakes = Shakespeare(train_split=train_ratio)

    # Build an iterator that gets seq_len long chunks of characters
    # Stride is by how many characters the window moves in each step
    # if stride is set to seq_len, windows are non-overlapping
    stride = seq_len // 2
    shakes_train = RollingWindowIterator(data_array=shakes.train, total_iterations=num_iterations,
                                         seq_len=seq_len, batch_size=batch_size, stride=stride)
    shakes_test = RollingWindowIterator(data_array=shakes.test,
                                        seq_len=seq_len, batch_size=batch_size, stride=stride)

    # ********************
    # NAME AND CREATE AXES
    # ********************
    # Our input is of size (batch_size, seq_len)
    # Create two axis, with each having corresponding sizes
    # batch_axis must be named N
    batch_axis = ng.make_axis(length=batch_size, name="N")
    # time_axis must be named REC
    time_axis = ng.make_axis(length=seq_len, name="REC")

    # Output is of size (number of unique tokens + 1,1)
    # Unique tokens is equal to the vocabulary size
    # We add one more output element just in case we come across an unknown token
    out_axis = ng.make_axis(length=len(shakes.vocab) + 1, name="out_feature_axis")

    in_axes = ng.make_axes([batch_axis, time_axis])

    # RollingWindowIterator gives an output of (batch_size, 1) for each iteration
    # Thus create an axis of that size
    # We will later convert this output to onehot
    out_axes = ng.make_axes([batch_axis])

    # Build placeholders for the created axes
    inputs = {'X': ng.placeholder(in_axes), 'y': ng.placeholder(out_axes),
              'iteration': ng.placeholder(axes=())}

    # ******************
    # NETWORK DEFINITION
    # ******************
    seq1 = Sequential([Preprocess(functor=expand_onehot),
                       LSTM(nout=recurrent_units, init=init_uni, backward=False,
                       reset_cells=True,
                       activation=Logistic(), gate_activation=Tanh(), return_sequence=False),
                       Affine(weight_init=init_uni, bias_init=init_uni,
                       activation=Softmax(), axes=out_axis)])
    '''
    # Below is an alternate topology that uses an embedding (LookupTable) layer as the first layer
    embedding_dim = 8
    seq1 = Sequential([LookupTable(len(shakes.vocab) + 1, embedding_dim, init_uni, update=True),
                       LSTM(nout=recurrent_units, init=init_uni, backward=True,
                       activation=Logistic(), gate_activation=Tanh(), return_sequence=False),
                       Affine(weight_init=init_uni, bias_init=init_uni,
                       activation=Softmax(), axes=out_axis)])
    '''

    # ***************************
    # OPTIMIZER AND LOSS FUNCTION
    # ***************************

    learning_rate_policy = {'name': 'exp',
                            'gamma': 0.98,
                            'base_lr': 0.01}
    optimizer = RMSProp(gradient_clip_value=1, learning_rate=learning_rate_policy,
                        iteration=inputs['iteration'] // 1000)
    '''
    # Below is an alternate optimization procedure that reduces learning rate at a given schedule
    # Initial learning rate is 0.02 (base_lr)
    # At iteration (num_iterations * .1), learning rate is multiplied by gamma (new lr = .002)
    # At iteration (num_iterations * .2), it is reduced by gamma again (new lr = .0002)
    # So on..
    schedule = [.1* num_iterations * i for i in range(1,10,1)]
    learning_rate_policy = {'name': 'schedule',
                            'schedule': schedule,
                            'gamma': 0.1,
                            'base_lr': 0.01}
    optimizer = GradientDescentMomentum(momentum_coef=.9,
                                        learning_rate=learning_rate_policy,
                                        iteration=inputs['iteration'],
                                        gradient_clip_value=5)
    '''

    # Define the loss function (Cross entropy loss)
    # Note that we convert the integer values of input['y'] to one hot here
    # The one-hot size is determined by out_axis
    fwd_prop = seq1(inputs['X'])
    train_loss = ng.cross_entropy_multi(fwd_prop,
                                        ng.one_hot(inputs['y'], axis=out_axis),
                                        usebits=True)

    # Cost calculation
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
    # *******************
    # DEFINE COMPUTATIONS
    # *******************
    print('Start training ...')
    with closing(ngt.make_transformer()) as transformer:
        train_function = transformer.add_computation(train_computation)
        eval_function = transformer.add_computation(eval_computation)
        generate_function = transformer.add_computation(gen_computation)

        # Determine printout interval of the validation set loss during training
        iter_interval = min(4000, num_iterations // 20)

        # ***************
        # TRAINING LOOP
        # ***************
        train_cost = train_loop(shakes_train, train_function, shakes_test,
                                eval_function, generate_function,
                                shakes.index_to_token, shakes.token_to_index)
