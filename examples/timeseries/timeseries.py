'''
    Usage:
        python timeseries.py -t 10000 -b gpu

    Builds a neural network to predict the next value in a continuous timeseries
    Input: 
        Each input sample has seq_len time steps, each time step is a two-long input vector (two features per time step)
    Output:
        Two-long vector, expected value at time point (seq_len + 1) 
'''


from __future__ import division, print_function
from builtins import range
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Layer, Sequential, Recurrent, BiRNN, Tanh
from ngraph.frontends.neon import Affine, Preprocess, Convolution, Pool2D, BatchNorm, Activation
from ngraph.frontends.neon import UniformInit, KaimingInit, Rectlin, Identity, Softmax, GradientDescentMomentum, RMSProp, Adam
from ngraph.frontends.neon import ax, NgraphArgparser, loop_train
from ngraph.frontends.neon import make_bound_computation, make_default_callbacks # noqa
from tqdm import tqdm
import ngraph.transformers as ngt
from ngraph.frontends.neon import ArrayIterator, SequentialArrayIterator
import math
import utils
import ngraph.transformers.passes.nviz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size  = 128

#Feature dimension of the input (for Lissajous curve, this is 2)
feature_dim = 2

#Number of time steps in the input sequence for each sample
seq_len     = 30

#Output feature dimension
output_dim  = 2

#Number of recurrent units in the network
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
Total number of samples will be (npoints * ncycles - seq_len)

data.train['X']['data']: will be the input training data. Shape: (no_samples, seq_len, input_feature_dim)
data.train['y']['data']: will be the outputs (labels) for training data. Shape: (no_samples, output_feature_dim)

data.test follows a similar model
'''

data    = utils.TimeSeries( train_ratio = 0.9,  #ratio of samples to set aside for training (value between 0. to 1.) 
                            seq_len = seq_len,  #length of the sequence in each sample
                            npoints = 37,       #number of points to take in each cycle
                            ncycles = 500,      #number of cycles in the curve
                            batch_size = batch_size,
                            curvetype = 'Lissajous2')

#Make an iterable / generator that yields chunks of training data
#Yields an input array of Shape (batch_size, seq_len, input_feature_dim)
train_set   = ArrayIterator(data.train, batch_size, total_iterations=args.num_iterations) 
test_set    = ArrayIterator(data.test, batch_size) 

#Name and create the axes
batch_axis      = ng.make_axis(length=batch_size, name="N")
time_axis       = ng.make_axis(length=seq_len, name="REC")
feature_axis    = ng.make_axis(length=feature_dim, name="feature_axis")
out_axis        = ng.make_axis(length=output_dim, name="output_axis")

in_axes         = ng.make_axes([batch_axis, time_axis, feature_axis])
out_axes        = ng.make_axes([batch_axis,out_axis])

#Build placeholders for the created axes
inputs          = {'X': ng.placeholder(in_axes),'y': ng.placeholder(out_axes) }

seq1    = Sequential([Recurrent(nout = recurrent_units, init = init_uni, activation=Tanh(), return_sequence=False),
                      Affine(weight_init = init_uni, bias_init=init_uni, activation=Identity(), axes = out_axis )] )
#Define the optimizer
optimizer   = Adam()

#Define the loss function (squared L2 loss)
fwd_prop    = seq1(inputs['X'])
train_loss  = ng.squared_L2(fwd_prop - inputs['y'])

#Cost calculation
batch_cost      = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs   = dict(batch_cost=batch_cost)

#Forward prop of test set
#Required for correct functioning of batch norm and dropout layers during inference mode
with Layer.inference_mode_on():
    inference_prob = seq1(inputs['X'])
eval_loss       = ng.squared_L2(inference_prob - inputs['y'])
eval_outputs    = dict(l2_loss=eval_loss)

#Function to return final predicted results for a given dataset
def loop_eval(dataset, computation):
    dataset.reset()
    results = []
    for data in dataset:
        feed_dict   = {inputs[k]: data[k] for k in data.keys() if k !='iteration'}
        results.append(computation(feed_dict=feed_dict))
    return results

print('Start training')
inference_prob      = seq1(inputs['X'])
eval_computation    = ng.computation([inference_prob], "all")

with closing(ngt.make_transformer()) as transformer:
    train_computation   = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation    = make_bound_computation(transformer, eval_outputs, inputs)
    eval_function       = transformer.add_computation(eval_computation)
    #Make these explicit
    cbs = make_default_callbacks(output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=test_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, train_computation, cbs)
    predictions = loop_eval(test_set, eval_function)

#Plot the ground truth test samples, as well as predictions
#Flatten the predictions
preds = predictions[0][0].transpose()
for i in range(1,len(predictions)):
    preds = np.concatenate( (preds,predictions[i][0].transpose()),axis = 0)

fig, ax = plt.subplots()
#Plot predictions
line1 = ax.plot(preds[:,0], preds[:,1],
                linestyle='None', 
                marker ='s', label ='predicted')
#Plot ground truth values
line2 = ax.plot(data.test['y']['data'][:,0], data.test['y']['data'][:,1],
                linestyle='None', linewidth = 1,
                label ='ground truth', marker='D',
                markerfacecolor='None')
ax.legend()
ax.grid()
fig.savefig('PredictedCurve.png', dpi=128)
