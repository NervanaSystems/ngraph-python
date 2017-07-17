"""
Copyright (c) 2017, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

LeNet model on MNIST dataset.
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

Example:
    python examples/mnist/mnist_lenet.py -b gpu -z 64
        Run the example using the NervanaGPU backend and batch size 64
"""

from __future__ import division
from __future__ import print_function
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Affine, Preprocess, Convolution, Pool2D, Sequential
from ngraph.frontends.neon import XavierInit, Rectlin, Softmax, GradientDescentMomentum
from ngraph.frontends.neon import ax, loop_train
from ngraph.frontends.neon import NgraphArgparser, make_bound_computation, make_default_callbacks
from ngraph.frontends.neon import ArrayIterator

from ngraph.frontends.neon import MNIST
import ngraph.transformers as ngt

parser = NgraphArgparser(description='Train LeNet topology on Mnist dataset')
args = parser.parse_args()

np.random.seed(args.rng_seed)

# Create the dataloader
train_data, valid_data = MNIST(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
valid_set = ArrayIterator(valid_data, args.batch_size)

inputs = train_set.make_placeholders()
ax.Y.length = 10

######################
# Model specification

init_xav = XavierInit()

seq1 = Sequential([Preprocess(functor=lambda x: x / 255.),
                   Convolution((5, 5, 16), filter_init=init_xav, activation=Rectlin()),
                   Pool2D(2, strides=2),
                   Convolution((5, 5, 32), filter_init=init_xav, activation=Rectlin()),
                   Pool2D(2, strides=2),
                   Affine(nout=500, weight_init=init_xav, activation=Rectlin()),
                   Affine(axes=ax.Y, weight_init=init_xav, activation=Softmax())])

optimizer = GradientDescentMomentum(0.01, 0.9)
train_prob = seq1(inputs['image'])
train_loss = ng.cross_entropy_binary(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with Layer.inference_mode_on():
    inference_prob = seq1(inputs['image'])
errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), inputs['label'])
eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
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
                                 eval_set=valid_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, train_computation, cbs)