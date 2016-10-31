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
from neon.backends import gen_backend
from ngraph.frontends.neon import *  # noqa
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.data import PTB
from neon.initializers import Uniform

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(gen_be=False)
args = parser.parse_args()

# these hyperparameters are from the paper
args.batch_size = 50
args.backend = 'dataloader'
time_steps = 2
hidden_size = 15
gradient_clip_value = None

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# download penn treebank
dataset = PTB(time_steps, path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# weight initialization
init = Uniform(low=-0.08, high=0.08)

# model initialization
layers = [
    Recurrent(hidden_size, init, activation=Tanh(), time_axis=ax.REC),
    Affine(
        len(train_set.vocab), init,
        activation=Softmax(), bias=init, axes=(ax.C, ax.REC)
    )
]

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
opt_rms = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)

rnn = Model(layers=layers)
callbacks = Callbacks(rnn, eval_set=valid_set, **args.callback_args)
rnn.initialize(
    dataset=train_set,
    input_axes=Axes((ax.C, ax.REC)),
    target_axes=Axes((ax.C, ax.REC)),
    optimizer=opt_rms,
    cost=cost
)

rnn.fit(
    train_set,
    num_epochs=args.epochs,
    callbacks=callbacks
)
