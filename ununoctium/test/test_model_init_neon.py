#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
#
# Then run the example:
#
# test_model_init_neon.py -v -r 0 --backend cpu --epochs 10 --eval_freq 1
#
# should produce sth like this:
#
# 2016-06-10 09:42:51,635 - neon - DISPLAY - Misclassification error = 89.26%
# 2016-06-10 09:42:51,635 - neon - DISPLAY - epoch: 0 learning_rate: 0.01 initial_train_loss: -1.39674e-06
# Epoch 0   [Train |████████████████████|  391/391  batches, 1.94 cost, 4.67s] [CrossEntropyMulti Loss 1.95, 0.62s]
# 2016-06-10 09:42:57,624 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6953
# 2016-06-10 09:42:57,625 - neon.callbacks.callbacks - INFO - Epoch 0 complete.  Train Cost 2.065901.  Eval Cost 1.951186
# 2016-06-10 09:42:57,626 - neon - DISPLAY - epoch: 1 learning_rate: 0.01 initial_train_loss: 2.0659
# Epoch 1   [Train |████████████████████|  391/391  batches, 1.90 cost, 4.06s] [CrossEntropyMulti Loss 1.90, 0.63s]
# 2016-06-10 09:43:03,030 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6659
# 2016-06-10 09:43:03,030 - neon.callbacks.callbacks - INFO - Epoch 1 complete.  Train Cost 1.928025.  Eval Cost 1.899957
# 2016-06-10 09:43:03,031 - neon - DISPLAY - epoch: 2 learning_rate: 0.01 initial_train_loss: 1.92803
# Epoch 2   [Train |████████████████████|  390/390  batches, 1.88 cost, 3.90s] [CrossEntropyMulti Loss 1.87, 0.59s]
# 2016-06-10 09:43:08,047 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6572
# 2016-06-10 09:43:08,048 - neon.callbacks.callbacks - INFO - Epoch 2 complete.  Train Cost 1.876261.  Eval Cost 1.870557
# 2016-06-10 09:43:08,048 - neon - DISPLAY - epoch: 3 learning_rate: 0.01 initial_train_loss: 1.87626
# Epoch 3   [Train |████████████████████|  391/391  batches, 1.86 cost, 4.24s] [CrossEntropyMulti Loss 1.86, 0.66s]
# 2016-06-10 09:43:13,615 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6457
# 2016-06-10 09:43:13,616 - neon.callbacks.callbacks - INFO - Epoch 3 complete.  Train Cost 1.853629.  Eval Cost 1.855121
# 2016-06-10 09:43:13,617 - neon - DISPLAY - epoch: 4 learning_rate: 0.01 initial_train_loss: 1.85363
# Epoch 4   [Train |████████████████████|  391/391  batches, 1.84 cost, 4.38s] [CrossEntropyMulti Loss 1.84, 0.67s]
# 2016-06-10 09:43:19,312 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6419
# 2016-06-10 09:43:19,313 - neon.callbacks.callbacks - INFO - Epoch 4 complete.  Train Cost 1.842031.  Eval Cost 1.835479
# 2016-06-10 09:43:19,313 - neon - DISPLAY - epoch: 5 learning_rate: 0.01 initial_train_loss: 1.84203
# Epoch 5   [Train |████████████████████|  390/390  batches, 1.81 cost, 4.40s] [CrossEntropyMulti Loss 1.83, 0.71s]
# 2016-06-10 09:43:25,068 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6339
# 2016-06-10 09:43:25,069 - neon.callbacks.callbacks - INFO - Epoch 5 complete.  Train Cost 1.818277.  Eval Cost 1.830219
# 2016-06-10 09:43:25,069 - neon - DISPLAY - epoch: 6 learning_rate: 0.01 initial_train_loss: 1.81828
# Epoch 6   [Train |████████████████████|  391/391  batches, 1.80 cost, 4.50s] [CrossEntropyMulti Loss 1.81, 0.65s]
# 2016-06-10 09:43:30,896 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6285
# 2016-06-10 09:43:30,897 - neon.callbacks.callbacks - INFO - Epoch 6 complete.  Train Cost 1.812860.  Eval Cost 1.812943
# 2016-06-10 09:43:30,898 - neon - DISPLAY - epoch: 7 learning_rate: 0.01 initial_train_loss: 1.81286
# Epoch 7   [Train |████████████████████|  390/390  batches, 1.80 cost, 4.18s] [CrossEntropyMulti Loss 1.80, 0.70s]
# 2016-06-10 09:43:36,446 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6289
# 2016-06-10 09:43:36,447 - neon.callbacks.callbacks - INFO - Epoch 7 complete.  Train Cost 1.794399.  Eval Cost 1.802400
# 2016-06-10 09:43:36,447 - neon - DISPLAY - epoch: 8 learning_rate: 0.01 initial_train_loss: 1.7944
# Epoch 8   [Train |████████████████████|  391/391  batches, 1.78 cost, 4.12s] [CrossEntropyMulti Loss 1.79, 0.70s]
# 2016-06-10 09:43:41,960 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6216
# 2016-06-10 09:43:41,960 - neon.callbacks.callbacks - INFO - Epoch 8 complete.  Train Cost 1.791364.  Eval Cost 1.794109
# 2016-06-10 09:43:41,961 - neon - DISPLAY - epoch: 9 learning_rate: 0.01 initial_train_loss: 1.79136
# Epoch 9   [Train |████████████████████|  391/391  batches, 1.77 cost, 6.58s] [CrossEntropyMulti Loss 1.80, 0.85s]
# 2016-06-10 09:43:50,098 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6195
# 2016-06-10 09:43:50,098 - neon.callbacks.callbacks - INFO - Epoch 9 complete.  Train Cost 1.786760.  Eval Cost 1.797422
# 2016-06-10 09:43:50,708 - neon - DISPLAY - Misclassification error = 62.27%

"""
Small CIFAR10 based MLP with fully connected layers.
"""

from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Misclassification, CrossEntropyMulti, Softmax, Rectlin, Tanh
from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks, Callback, LossCallback
from neon import logger as neon_logger
import numpy as np

class myCallback(Callback):
    """
    Callback for printing the initial weight value before training.

    Arguments:
        eval_set (NervanaDataIterator): dataset to evaluate
        epoch_freq (int, optional): how often (in epochs) to log info.
                                    Defaults to every 1 epoch.
    """

    def __init__(self, eval_set, epoch_freq=1):
        super(myCallback, self).__init__(epoch_freq=epoch_freq)
        self.eval_set = eval_set
        self.loss = self.be.zeros((1, 1), dtype=np.float32)

    def on_train_begin(self, callback_data, model, epochs):
        print(model.layers.layers[0].W.shape)
        print(model.layers.layers[0].W.get())
        print(model.layers.layers[2].W.shape)
        print(model.layers.layers[2].W.get())

        init_error = model.eval(self.eval_set, metric=Misclassification()) * 100
        neon_logger.display('Misclassification error = %.2f%%' % init_error)

    def on_epoch_begin(self, callback_data, model, epoch):
        lrate = model.optimizer.schedule.get_learning_rate(0.01, epoch)
        neon_logger.display('epoch: ' + str(epoch) + ' learning_rate: ' + str(lrate)
                            + ' initial_train_loss: ' + str(model.total_cost[0][0]))

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()
args.epochs = 10

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=False, do_transforms=False, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0)

# set up the model layers
init_uni1 = Uniform(low=-0.1, high=0.1)
layers = [Affine(nout=200, init=init_uni1, activation=Tanh()),
          Affine(nout=10, init=init_uni1, activation=Softmax())]
mlp = Model(layers=layers)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test, metric=Misclassification(), **args.callback_args)
callbacks.add_callback(myCallback(eval_set=test, epoch_freq=1))

# callbacks.add_callback(LossCallback(eval_set=test,epoch_freq=1)) # TODO: adding this line (from tutorial) just breaks
# callbacks.add_hist_callback(plot_per_mini=True) # TODO: did not see any effect with this line

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
# print(mlp.params) # TODO: None value and confusing naming, should be removed

# print(layers[0][0].W.shape)
# print(layers[0][0].W.get())

final_error = mlp.eval(test, metric=Misclassification())*100
neon_logger.display('Misclassification error = %.2f%%' % final_error)

# TODO: the final Misclassification error does not match the Top1Misclass of the last epoch
