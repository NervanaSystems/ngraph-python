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
# 2016-06-16 16:20:41,472 - neon - DISPLAY - Misclassification error = 89.26%
# 2016-06-16 16:20:41,472 - neon - DISPLAY - epoch: 0 learning_rate: 0.01 initial_train_loss: 7.1575e-33
# 2016-06-16 16:20:48,942 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6895
# 2016-06-16 16:20:48,944 - neon.callbacks.callbacks - INFO - Epoch 0 complete.  Train Cost 2.062410.  Eval Cost 1.952405
# 2016-06-16 16:20:48,945 - neon - DISPLAY - epoch: 1 learning_rate: 0.01 initial_train_loss: 2.06241
# 2016-06-16 16:20:54,028 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6686
# 2016-06-16 16:20:54,029 - neon.callbacks.callbacks - INFO - Epoch 1 complete.  Train Cost 1.917983.  Eval Cost 1.897193
# 2016-06-16 16:20:54,030 - neon - DISPLAY - epoch: 2 learning_rate: 0.01 initial_train_loss: 1.91798
# 2016-06-16 16:20:59,107 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6578
# 2016-06-16 16:20:59,109 - neon.callbacks.callbacks - INFO - Epoch 2 complete.  Train Cost 1.862682.  Eval Cost 1.868695
# 2016-06-16 16:20:59,110 - neon - DISPLAY - epoch: 3 learning_rate: 0.01 initial_train_loss: 1.86268
# 2016-06-16 16:21:04,690 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6481
# 2016-06-16 16:21:04,691 - neon.callbacks.callbacks - INFO - Epoch 3 complete.  Train Cost 1.845137.  Eval Cost 1.847145
# 2016-06-16 16:21:04,692 - neon - DISPLAY - epoch: 4 learning_rate: 0.01 initial_train_loss: 1.84514
# 2016-06-16 16:21:09,788 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6393
# 2016-06-16 16:21:09,788 - neon.callbacks.callbacks - INFO - Epoch 4 complete.  Train Cost 1.834346.  Eval Cost 1.835796
# 2016-06-16 16:21:09,789 - neon - DISPLAY - epoch: 5 learning_rate: 0.01 initial_train_loss: 1.83435
# 2016-06-16 16:21:15,231 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6441
# 2016-06-16 16:21:15,232 - neon.callbacks.callbacks - INFO - Epoch 5 complete.  Train Cost 1.808218.  Eval Cost 1.835048
# 2016-06-16 16:21:15,232 - neon - DISPLAY - epoch: 6 learning_rate: 0.01 initial_train_loss: 1.80822
# 2016-06-16 16:21:20,232 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.633
# 2016-06-16 16:21:20,233 - neon.callbacks.callbacks - INFO - Epoch 6 complete.  Train Cost 1.809532.  Eval Cost 1.820541
# 2016-06-16 16:21:20,233 - neon - DISPLAY - epoch: 7 learning_rate: 0.01 initial_train_loss: 1.80953
# 2016-06-16 16:21:25,392 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6278
# 2016-06-16 16:21:25,393 - neon.callbacks.callbacks - INFO - Epoch 7 complete.  Train Cost 1.789490.  Eval Cost 1.811423
# 2016-06-16 16:21:25,394 - neon - DISPLAY - epoch: 8 learning_rate: 0.01 initial_train_loss: 1.78949
# 2016-06-16 16:21:30,609 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6304
# 2016-06-16 16:21:30,610 - neon.callbacks.callbacks - INFO - Epoch 8 complete.  Train Cost 1.782343.  Eval Cost 1.799538
# 2016-06-16 16:21:30,610 - neon - DISPLAY - epoch: 9 learning_rate: 0.01 initial_train_loss: 1.78234
# 2016-06-16 16:21:36,211 - neon.callbacks.callbacks - INFO - Top1Misclass: 0.6261
# 2016-06-16 16:21:36,212 - neon.callbacks.callbacks - INFO - Epoch 9 complete.  Train Cost 1.778128.  Eval Cost 1.790867
# 2016-06-16 16:21:36,823 - neon - DISPLAY - Misclassification error = 62.61%


"""
Small CIFAR10 based MLP with fully connected layers.
"""
from __future__ import print_function
from builtins import str

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
