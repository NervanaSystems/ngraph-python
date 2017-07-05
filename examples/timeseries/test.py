from __future__ import division, print_function
from builtins import range
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, Recurrent, BiRNN, Tanh
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import UniformInit, KaimingInit, Rectlin, Identity, Softmax, GradientDescentMomentum, RMSProp
from ngraph.frontends.neon import ax, NgraphArgparser, loop_train
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks # noqa
from tqdm import tqdm
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator, SequentialArrayIterator
import pdb
import math
import utils
import ngraph.transformers.passes.nviz

batch_size  = 128
feature_dim = 3
seq_len     = 30

output_dim  = 2

recurrent_units = 32
#Define initialization
init_uni = UniformInit(-0.1, 0.1)

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--layer_type', default='rnn', choices=['rnn', 'birnn'],
                    help='type of recurrent layer to use (rnn or birnn)')
parser.set_defaults()
args = parser.parse_args()

'''
Generate the training data based on Lissajous curve
Number of samples will be (npoints * ncycles)

data.train['X']['data']: will be the input training data
data.train['X']['axes']: will be axes names for input data
data.train['y']['data']: will be the outputs/labels for training data
data.train['y']['axes']: will be axes names for outputs/labels

data.test follows a similar model

'''

data    = utils.TimeSeries( train_ratio = 0.8,  #ratio of samples to set aside for training 
                            seq_len = seq_len,  #length of the sequence in each sample
                            npoints = 100,      #number of points to take in each cycle
                            ncycles = 10)       #number of cycles in the curve

#Make an iterable / generator that receives chunks of training data (chunk = batch_size)
#train_set   = SequentialArrayIterator(data.train, seq_len, batch_size) 
#test_set    = SequentialArrayIterator(data.test, seq_len, batch_size) 
train_set   = ArrayIterator(data.train, batch_size) 
test_set    = ArrayIterator(data.test, batch_size) 

#Make placeholders of training data (these are temporary variables used for training)
inputs      = train_set.make_placeholders()

#Define the number of output classes
#ax.Y.length = output_dim


#Define the network
ax.Y.length = output_dim
ax.Y.name   = 'Fo'
seq1    = Sequential([Recurrent(nout = recurrent_units, init = init_uni, activation=Tanh(), return_sequence=False),
#                      Affine(weight_init = init_uni, bias_init=init_uni, activation=Identity(), nout = output_dim)] )
                      Affine(weight_init = init_uni, bias_init=init_uni, activation=Identity(), axes = (ax.Y,) )] )

optimizer   = RMSProp()

#Define the loss function
fwd_prop    = seq1(inputs['X'])
train_loss  = ng.squared_L2(fwd_prop - inputs['y'])

#Not sure what this is
batch_cost      = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs   = dict(batch_cost=batch_cost)


with Layer.inference_mode_on():
    inference_prob = seq1(inputs['X'])
eval_loss       = ng.squared_L2(inference_prob - inputs['y'])
eval_outputs    = dict(l2_loss=eval_loss)


# Now bind the computations we are interested in
with closing(ngt.make_transformer()) as transformer:
    #transformer.register_graph_pass(ngraph.transformers.passes.nviz.VizPass(show_all_metadata=True, show_axes=True, view= False))
    train_computation   = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation    = make_bound_computation(transformer, eval_outputs, inputs)

    cbs = make_default_callbacks(output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=test_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, train_computation, cbs)
