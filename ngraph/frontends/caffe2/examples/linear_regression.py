from __future__ import print_function, division
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
import ngraph.transformers as ngt
import ngraph.frontends.common.utils as util
from caffe2.python import core, workspace
import numpy as np


def linear_regression(iter_num, lrate, gamma, step_size, noise_scale):
    # data multiplier
    m = 3
    # batch_len and data
    xs_np = np.array([[0, 0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [-1.0, -1.0]], dtype='f')
    ys_np = np.array([[0.5 * m], [2.5 * m], [4.5 * m], [6.5 * m], [-1.5 * m]], dtype='f')
    batch_len = len(ys_np)

    # with these values we have the following target weight and bias
    # to be approximated after computation:
    target_b = 0.5 * m
    target_w = np.array([1.0, 1.0]) * m

    # noise amplitude and noise generation
    noise_l = np.array(noise_scale * np.random.randn(batch_len), dtype='f')
    noise = [[i] for i in noise_l]

    # caffe2 init network
    init_net = core.Net("init")
    ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
    ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)

    # for the parameters to be learned: we randomly initialize weight
    # being output scalar, and two variables, W is 1x2, X is 2x1
    W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
    B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
    print('Created init net.')

    # caffe2 train net
    train_net = core.Net("train")

    # definition of external inputs: X, ground truth and noisy version of truth
    workspace.FeedBlob('X', xs_np)
    workspace.FeedBlob('Y_gt', ys_np)
    workspace.FeedBlob('Y_noise', ys_np + noise)
    train_net.AddExternalInput("X")
    train_net.AddExternalInput("Y_noise")
    train_net.AddExternalInput("Y_gt")

    # now, for the normal linear regression prediction, this is all we need.
    Y_pred = train_net.FC(["X", W, B], "Y_pred")

    # when it will be computing the loss, we want to refer to the noisy version of the truth:
    dist = train_net.SquaredL2Distance(["Y_noise", Y_pred], "dist")
    loss = dist.AveragedLoss([], ["loss"])

    # Caffe2 creation of the initialization and training nets, needed to have objects created
    # and therefore handlers can be obtained by the importer
    workspace.CreateNet(init_net)
    workspace.CreateNet(train_net)

    # importing in ngraph caffe2 network
    print("\n\n---------------------ngraph behaviour:")
    importer = C2Importer()
    importer.parse_net_def(net_def=train_net.Proto(), init_net_def=init_net.Proto(),
                           c2_workspace=workspace)

    # Get handles to the various objects we are interested to for ngraph computation
    y_gt_ng, x_ng, w_ng, b_ng, y_pred_ng, dist_ng, loss_ng =  \
        importer.get_op_handle(['Y_noise', 'X', 'W', 'B', 'Y_pred', 'dist', 'loss'])

    # setting learning rate for ngraph, that matches the one that it will be used for caffe2 below
    lr_params = {'name': 'step', 'base_lr': lrate, 'gamma': gamma, 'step': step_size}

    SGD = util.CommonSGDOptimizer(lr_params)
    parallel_update = SGD.minimize(loss_ng, [w_ng, b_ng])
    transformer = ngt.make_transformer()
    update_fun = transformer.computation(
        [loss_ng, w_ng, b_ng, parallel_update], x_ng, y_gt_ng, SGD.get_iter_buffer())

    true_iter = [0]
    # ngraph actual computation
    for i in range(iter_num // batch_len):
        for xs, ys in zip(xs_np, ys_np + noise):
            loss_val, w_val, b_val, _ = update_fun(xs, ys, i)
            # print("N it: %s W: %s, B: %s loss %s " % (i, w_val, b_val, loss_val))
            true_iter[0] += 1

    print("Ngraph loss %s " % (loss_val))

    # end of ngraph part

    # caffe2 backward pass and computation to compare results with ngraph
    gradient_map = train_net.AddGradientOperators([loss])

    # Increment the iteration by one.
    train_net.Iter(ITER, ITER)

    # Caffe2 backward pass and computation
    # Get gradients for all the computations above and do the weighted sum
    LR = train_net.LearningRate(ITER, "LR", base_lr=-lrate, policy="step",
                                stepsize=step_size, gamma=gamma)
    train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
    train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
    workspace.RunNetOnce(init_net)
    workspace.CreateNet(train_net)

    for i in range(iter_num):
        workspace.RunNet(train_net.Proto().name)
        # print("During training, loss is: {}".format(workspace.FetchBlob("loss")))

    print("Caffe2 loss is: {}".format(workspace.FetchBlob("loss")))
    # end of caffe2 part

    # printing out results
    print("Done {} iterations over the batch data, with noise coefficient set to {}".
          format(iter_num, noise_scale))
    print("Caffe2 after training, W is: {}".format(workspace.FetchBlob("W")))
    print("Caffe2 after training, B is: {}".format(workspace.FetchBlob("B")))
    print("Ngraph after training, W is: {}".format(w_val))
    print("Ngraph after training, B is: {}".format(b_val))
    print("Target W was: {}".format(target_w))
    print("Target B was: {}".format(target_b))

    assert(workspace.FetchBlob("loss") < 0.01)
    assert(loss_val < 0.01)


if __name__ == "__main__":
    iter_num, lrate, gamma, step_size, noise_scale = 200, 0.01, 0.9, 20, 0.01
    linear_regression(iter_num, lrate, gamma, step_size, noise_scale)
