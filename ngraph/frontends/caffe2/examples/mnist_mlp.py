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


def mnist_mlp(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    train_x, train_y = mnist.train.next_batch(args.batch)
    workspace.FeedBlob('train_x', train_x)  # TODO change
    workspace.FeedBlob('train_y', train_y)

    init_net = core.Net("init")
    main_net = core.Net("main")
    # init_net.ConstantFill([], "ONE", shape=[1], value=1.)
    # init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.train_xType.INT32)

    fc_size = [784, 512, 128, 10]
    # init_net.UniformFill([], 'fc_w1', shape=[fc_size[1], fc_size[0]], min=-1., max=1.)      # TODO shapes
    # init_net.UniformFill([], 'fc_w2', shape=[fc_size[2], fc_size[1]], min=-1., max=1.)
    # init_net.UniformFill([], 'fc_w3', shape=[fc_size[3], fc_size[2]], min=-1., max=1.)
    # init_net.UniformFill([], 'fc_b1', shape=[1], min=-1., max=1.)
    # init_net.UniformFill([], 'fc_b2', shape=[1], min=-1., max=1.)
    # init_net.UniformFill([], 'fc_b3', shape=[1], min=-1., max=1.)
    init_net.UniformFill([], 'fc_w1', shape=[fc_size[3], fc_size[0]], min=-1., max=1.) # TODO single leayer mlp
    init_net.UniformFill([], 'fc_b1', shape=[fc_size[3]], min=-1., max=1.)

    # main_net.FC(['train_x', 'fc_w1', 'fc_b1'], 'FC1', dim_in=fc_size[0], dim_out=fc_size[1])
    # main_net.Relu('FC1', 'activ1')
    # main_net.FC(['activ1', 'fc_w2', 'fc_b1'], 'FC2', dim_in=fc_size[1], dim_out=fc_size[2])
    # main_net.Relu('FC2', 'activ2')
    # main_net.FC(['activ2', 'fc_w3', 'fc_b1'], 'FC3', dim_in=fc_size[2], dim_out=fc_size[3])
    # main_net.Softmax('FC3', 'softmax')

    main_net.FC(['train_x', 'fc_w1', 'fc_b1'], 'FC1', dim_in=fc_size[0], dim_out=fc_size[3])
    main_net.Softmax('FC1', 'softmax')
    main_net.LabelCrossEntropy(['softmax', 'train_y'], 'loss')


    # Ngraph part
    # import graph_def
    importer = C2Importer()
    importer.parse_net_def(net_def=main_net.Proto(),
                           init_net_def=init_net.Proto(),
                           c2_workspace=workspace)

    # get handle of ngraph ops
    # fc_w1_ng, fc_w2_ng, fc_w3_ng, fc_b1_ng, fc_b2_ng, fc_b3_ng, softmax_ng, loss_ng = \
    #     importer.get_op_handle(['fc_w1', 'fc_w2', 'fc_w3', 'fc_b1', 'fc_b2', 'fc_b3', 'softmax', 'loss'])

    fc_w1_ng, fc_b1_ng, softmax_ng, loss_ng = \
        importer.get_op_handle(['fc_w1', 'fc_b1', 'softmax', 'loss'])


    # setting learning rate for ngraph, that matches the one that it will be used for caffe2 below
    alpha = ng.placeholder(axes=(), initial_value=[args.lrate])

    # transformer and computations

    parallel_update = util.CommonSGDOptimizer(args.lrate).minimize(loss_ng, [fc_w1_ng, fc_b1_ng])
    # parallel_update = util.CommonSGDOptimizer(lrate).minimize(loss_ng, [w_ng, b_ng])
    transformer = ngt.make_transformer()
    # update_fun = transformer.computation(
    #     [loss_ng, w_ng, b_ng, parallel_update], alpha, x_ng, y_gt_ng)

    update_fun = transformer.computation(
        [loss_ng, fc_w1_ng, fc_b1_ng, parallel_update], alpha, train_x, train_y)

    # train
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='/tmp/data')
    parser.add_argument('-i', '--max_iter', type=int, default=10)
    parser.add_argument('-l', '--lrate', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('-b', '--batch', type=int, default=1)  # TODO
    args = parser.parse_args()
    mnist_mlp(args)

