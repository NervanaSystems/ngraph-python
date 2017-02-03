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

from __future__ import print_function, division
import argparse
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
from tensorflow.examples.tutorials.mnist import input_data
import ngraph.transformers as ngt
import ngraph.frontends.common.utils as util
from caffe2.python import core, cnn, workspace
import numpy as np
import ngraph as ng
import os

c2_on = 01
ng_on = 01

def mnist_mlp(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=False)

    train_x, train_y = mnist.train.next_batch(args.batch)
    workspace.FeedBlob('train_x', train_x)  # TODO change
    workspace.FeedBlob('train_y', train_y.astype('int32'))

    init_net = core.Net("init")
    main_net = core.Net("main")

    fc_size = [784, 512, 128, 10]
    init_net.UniformFill([], 'fc_w1', shape=[fc_size[1], fc_size[0]], min=-0.1, max=0.1)
    init_net.UniformFill([], 'fc_w2', shape=[fc_size[2], fc_size[1]], min=-0.1, max=0.1)
    init_net.UniformFill([], 'fc_w3', shape=[fc_size[3], fc_size[2]], min=-0.1, max=0.1)
    init_net.UniformFill([], 'fc_b1', shape=[fc_size[1]], min=0.1, max=0.1)
    init_net.UniformFill([], 'fc_b2', shape=[fc_size[2]], min=0.1, max=0.1)
    init_net.UniformFill([], 'fc_b3', shape=[fc_size[3]], min=0.1, max=0.1)
    # init_net.UniformFill([], 'fc_w1', shape=[fc_size[3], fc_size[0]], min=-1., max=1.) # TODO single leayer mlp
    # init_net.UniformFill([], 'fc_b1', shape=[fc_size[3]], min=-1., max=1.)

    main_net.FC(['train_x', 'fc_w1', 'fc_b1'], 'FC1', dim_in=fc_size[0], dim_out=fc_size[1])
    main_net.Relu('FC1', 'activ1')
    main_net.FC(['activ1', 'fc_w2', 'fc_b2'], 'FC2', dim_in=fc_size[1], dim_out=fc_size[2])
    main_net.Relu('FC2', 'activ2')
    main_net.FC(['activ2', 'fc_w3', 'fc_b3'], 'FC3', dim_in=fc_size[2], dim_out=fc_size[3])
    main_net.Softmax('FC3', 'softmax')

    # main_net.FC(['train_x', 'fc_w1', 'fc_b1'], 'FC1', dim_in=fc_size[0], dim_out=fc_size[3])
    # main_net.Softmax('FC1', 'softmax')

    main_net.LabelCrossEntropy(['softmax', 'train_y'], 'xent') # TODO should be xent
    main_net.AveragedLoss('xent', 'loss')

    # Ngraph part
    if ng_on:
        print(">>>>>>>>>>>>>> Ngraph")
        # import graph_def
        importer = C2Importer()
        importer.parse_net_def(net_def=main_net.Proto(),
                               init_net_def=init_net.Proto(),
                               c2_workspace=workspace)

        # get handle of ngraph ops
        x_train_ng, y_train_ng, fc_w1_ng, fc_w2_ng, fc_w3_ng, fc_b1_ng, fc_b2_ng, fc_b3_ng, loss_ng = \
            importer.get_op_handle(['train_x', 'train_y', 'fc_w1', 'fc_w2', 'fc_w3', 'fc_b1', 'fc_b2', 'fc_b3', 'loss'])

        # x_train_ng, y_train_ng, fc_w1_ng, fc_b1_ng, softmax_ng, loss_ng = \
        #     importer.get_op_handle(['train_x', 'train_y', 'fc_w1', 'fc_b1', 'softmax', 'loss'])

        # setting learning rate for ngraph, that matches the one that it will be used for caffe2 below
        alpha = ng.placeholder(axes=(), initial_value=[args.lrate])

        # transformer and computations
        # parallel_update = util.CommonSGDOptimizer(args.lrate).minimize(loss_ng, [fc_w1_ng, fc_b1_ng])
        parallel_update = util.CommonSGDOptimizer(args.lrate).minimize(loss_ng, [fc_w1_ng, fc_b1_ng])

        transformer = ngt.make_transformer()
        # update_fun = transformer.computation(
        #     [loss_ng, w_ng, b_ng, parallel_update], alpha, x_ng, y_gt_ng)

        # update_fun = transformer.computation(
        #     [loss_ng, fc_w1_ng, fc_b1_ng, parallel_update], alpha, x_train_ng, y_train_ng)
        update_fun = transformer.computation(
            [loss_ng, fc_w1_ng, fc_b1_ng, parallel_update], alpha, x_train_ng, y_train_ng)

        # train
        true_iter = [0]
        # ngraph actual computation
        for i in range(args.max_iter // args.batch):
            for b in range(args.batch):
                train_x, train_y = mnist.train.next_batch(args.batch)
                lr = args.lrate * (1 + 0.0001 * i) ** (-0.75)
                loss_val, _, _, _ = update_fun(lr, train_x, train_y)
                # print("N it: %s W: %s, B: %s loss %s " % (i, w_val, b_val, loss_val))
                if i % 200 == 0: print("iter %s, loss %s " % (i, loss_val))
                true_iter[0] += 1
    # ======================================
    if c2_on:
        print(">>>>>>>>>>>>>> Caffe")
        # caffe2 backward pass and computation to compare results with ngraph
        ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
        ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)
        gradient_map = main_net.AddGradientOperators(['loss'])

        # Increment the iteration by one.
        main_net.Iter(ITER, ITER)

        # Caffe2 backward pass and computation
        # Get gradients for all the computations above and do the weighted sum
        # LR = main_net.LearningRate(ITER, "LR", base_lr=-lrate, policy="step",
        #                             stepsize=step_size, gamma=gamma)
        # LR = main_net.LearningRate(ITER, "LR", base_lr=args.lrate, policy="inv", power=-0.75, gamma=0.0001)
        LR = main_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999)

        main_net.WeightedSum(['fc_w1', 'ONE', gradient_map['fc_w1'], 'LR'], 'fc_w1')
        main_net.WeightedSum(['fc_w2', 'ONE', gradient_map['fc_w2'], 'LR'], 'fc_w2')
        main_net.WeightedSum(['fc_w3', 'ONE', gradient_map['fc_w3'], 'LR'], 'fc_w3')
        main_net.WeightedSum(['fc_b1', 'ONE', gradient_map['fc_b1'], 'LR'], 'fc_b1')
        main_net.WeightedSum(['fc_b2', 'ONE', gradient_map['fc_b2'], 'LR'], 'fc_b2')
        main_net.WeightedSum(['fc_b3', 'ONE', gradient_map['fc_b3'], 'LR'], 'fc_b3')
        # main_net.WeightedSum(['activ1', 'ONE', gradient_map['activ1'], 'LR'], 'activ1')
        # main_net.WeightedSum(['activ2', 'ONE', gradient_map['activ2'], 'LR'], 'activ2')
        # main_net.WeightedSum(['softmax', 'ONE', gradient_map['softmax'], 'LR'], 'softmax')
        workspace.RunNetOnce(init_net)
        workspace.CreateNet(main_net)

        mnist = input_data.read_data_sets(args.data_dir, one_hot=False)
        for i in range(args.max_iter):
            train_x, train_y = mnist.train.next_batch(args.batch)
            workspace.FeedBlob('train_x', train_x)
            workspace.FeedBlob('train_y', train_y.astype('int32'))
            # print("x: {}".format(workspace.FetchBlob("train_x")))
            workspace.RunNet(main_net.Proto().name)

            # print('train_y: {}'.format(workspace.FetchBlob('train_y')))
            # print('FC3: {}'.format(workspace.FetchBlob('FC3')))
            # print('softmax: {}'.format(workspace.FetchBlob('softmax')))
            # print('loss: {}'.format(workspace.FetchBlob('loss')))
            # print('loss_autogen_grad: {}'.format(workspace.FetchBlob('loss_autogen_grad')))
            # print('softmax_grad: {}'.format(workspace.FetchBlob('softmax_grad')))

            if i % 200 == 0:
                print("Iter: {}, C2 loss is: {}".format(i, workspace.FetchBlob("loss")))
            #     print("y: {}".format(workspace.FetchBlob("train_y")))

        print("Caffe2 loss is: {}".format(workspace.FetchBlob("loss")))
        # end of caffe2 part

        # printing out results
        # print("Caffe2 after training, W3 is: {}".format(workspace.FetchBlob("fc_w3")))
        # print("Caffe2 after training, B3 is: {}".format(workspace.FetchBlob("fc_b3")))
        # print("Ngraph after training, W3 is: {}".format(fc_w3_ng))
        # print("Ngraph after training, B3 is: {}".format(fc_b3_ng))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='/tmp/data')
    parser.add_argument('-i', '--max_iter', type=int, default=10000)
    parser.add_argument('-l', '--lrate', type=float, default=0.05,
                        help="Learning rate")
    parser.add_argument('-b', '--batch', type=int, default=1)  # TODO
    args = parser.parse_args()
    mnist_mlp(args)

