# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import print_function, division
import argparse
import ngraph as ng
import ngraph.transformers as ngt
import ngraph.frontends.common.utils as util
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
from caffe2.python import core, workspace
from tensorflow.examples.tutorials.mnist import input_data

c2_on = 1
ng_on = 1

log_interval = 200

def mnist_mlp(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=False)

    train_x, train_y = mnist.train.next_batch(args.batch)
    # we have to feed blobs with some data, to give them valid shape,
    # because ngraph will import this shape
    workspace.FeedBlob('train_x', train_x)
    # currently caffe2 accepts only int32 data type
    workspace.FeedBlob('train_y', train_y.astype('int32'))

    init_net = core.Net('init')
    main_net = core.Net('main')

    # definition of number of neurons for each hidden layer
    fc_size = [784, 512, 128, 10]
    init_net.UniformFill([], 'fc_w1', shape=[fc_size[1], fc_size[0]], min=-.5, max=.5)
    init_net.UniformFill([], 'fc_w2', shape=[fc_size[2], fc_size[1]], min=-.5, max=.5)
    init_net.UniformFill([], 'fc_w3', shape=[fc_size[3], fc_size[2]], min=-.5, max=.5)
    init_net.UniformFill([], 'fc_b1', shape=[fc_size[1]], min=-.5, max=.5)
    init_net.UniformFill([], 'fc_b2', shape=[fc_size[2]], min=-.5, max=.5)
    init_net.UniformFill([], 'fc_b3', shape=[fc_size[3]], min=-.5, max=.5)

    main_net.FC(['train_x', 'fc_w1', 'fc_b1'], 'FC1')
    main_net.Relu('FC1', 'activ1')
    main_net.FC(['activ1', 'fc_w2', 'fc_b2'], 'FC2')
    main_net.Relu('FC2', 'activ2')
    main_net.FC(['activ2', 'fc_w3', 'fc_b3'], 'FC3')
    main_net.Softmax('FC3', 'softmax')
    main_net.LabelCrossEntropy(['softmax', 'train_y'], 'xent')
    main_net.AveragedLoss('xent', 'loss')

    # Ngraph part
    if ng_on:
        print('>>>>>>>>>>>>>> Ngraph')
        # import graph_def
        importer = C2Importer()
        importer.parse_net_def(net_def=main_net.Proto(),
                               init_net_def=init_net.Proto(),
                               c2_workspace=workspace)

        # get handle of ngraph ops
        x_train_ng, y_train_ng, fc_w1_ng, fc_w2_ng, fc_w3_ng, fc_b1_ng, fc_b2_ng, fc_b3_ng, loss_ng = \
            importer.get_op_handle(['train_x', 'train_y', 'fc_w1', 'fc_w2', 'fc_w3', 'fc_b1', 'fc_b2', 'fc_b3', 'loss'])

        # setting learning rate for ngraph, that matches the one that it will be used for caffe2 below
        alpha = ng.placeholder(axes=(), initial_value=[args.lrate])

        # transformer and computations
        parallel_update = util.CommonSGDOptimizer(args.lrate).minimize(loss_ng, [fc_w1_ng, fc_w2_ng,fc_w3_ng, fc_b1_ng, fc_b2_ng, fc_b3_ng])
        transformer = ngt.make_transformer()
        update_fun = transformer.computation(
            [loss_ng, fc_w1_ng, fc_b1_ng, parallel_update], alpha, x_train_ng, y_train_ng)

        # train
        # ngraph actual computation
        for i in range(args.max_iter):
            train_x, train_y = mnist.train.next_batch(args.batch)
            lr = args.lrate * (1 + args.gamma * i) ** (-args.power)
            loss_val, _, _, _ = update_fun(lr, train_x, train_y)
            if args.verbose and i % log_interval == 0:
                print('iter %s, loss %s ' % (i, loss_val))
    # ======================================
    if c2_on:
        mnist = input_data.read_data_sets(args.data_dir, one_hot=False)
        print('>>>>>>>>>>>>>> Caffe')
        # caffe2 backward pass and computation to compare results with ngraph
        init_net.ConstantFill([], 'ONE', shape=[1], value=1.)
        init_net.ConstantFill([], 'ITER', shape=[1], value=0, dtype=core.DataType.INT32)
        gradient_map = main_net.AddGradientOperators(['loss'])

        # Increment the iteration by one.
        main_net.Iter('ITER', 'ITER')

        # Caffe2 backward pass and computation
        # Get gradients for all the computations above and do the weighted sum
        main_net.LearningRate('ITER', 'LR', base_lr=-args.lrate, policy='inv',
                              power=args.power, gamma=args.gamma)

        main_net.WeightedSum(['fc_w1', 'ONE', gradient_map['fc_w1'], 'LR'], 'fc_w1')
        main_net.WeightedSum(['fc_w2', 'ONE', gradient_map['fc_w2'], 'LR'], 'fc_w2')
        main_net.WeightedSum(['fc_w3', 'ONE', gradient_map['fc_w3'], 'LR'], 'fc_w3')
        main_net.WeightedSum(['fc_b1', 'ONE', gradient_map['fc_b1'], 'LR'], 'fc_b1')
        main_net.WeightedSum(['fc_b2', 'ONE', gradient_map['fc_b2'], 'LR'], 'fc_b2')
        main_net.WeightedSum(['fc_b3', 'ONE', gradient_map['fc_b3'], 'LR'], 'fc_b3')
        workspace.RunNetOnce(init_net)
        workspace.CreateNet(main_net)

        for i in range(args.max_iter):
            train_x, train_y = mnist.train.next_batch(args.batch)
            workspace.FeedBlob('train_x', train_x)
            workspace.FeedBlob('train_y', train_y.astype('int32'))
            workspace.RunNet(main_net.Proto().name)
            if args.verbose and i % log_interval == 0:
                print('Iter: {}, C2 loss is: {}'.format(i, workspace.FetchBlob('loss')))
        # end of caffe2 part

    if ng_on:
        print('Ngraph loss is: %s' % loss_val)
    if c2_on:
        print('Caffe2 loss is: {}'.format(workspace.FetchBlob('loss')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='/tmp/data')
    parser.add_argument('-i', '--max_iter', type=int, default=10000)
    parser.add_argument('-l', '--lrate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-v', '--verbose', type=int, default=1)
    # fixed inv policy
    parser.add_argument('-p', '--power', type=float, default=0.75)
    parser.add_argument('-g', '--gamma', type=float, default=0.0001)

    args = parser.parse_args()
    mnist_mlp(args)

