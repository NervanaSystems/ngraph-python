#!/usr/bin/env python
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
import time
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import (GaussianInit, GlorotInit, ConstantInit, Convolution, Rectlin,
                                   Rectlinclip, BiRNN, GradientDescentMomentum, Affine, Softmax,
                                   Sequential, Layer)
from ngraph.frontends.neon import ax
from ngraph.frontends.neon.data import Librispeech
from data import SpeechTranscriptionLoader
from decoder import ArgMaxDecoder


class DeepBiRNN(Sequential):

    def __init__(self, num_layers, hidden_size, init, activation, batch_norm=False,
                 sum_final=False):

        rnns = list()
        sum_out = concat_out = False
        for ii in range(num_layers):
            if ii == (num_layers - 1):
                sum_out = sum_final
                concat_out = not sum_final

            rnn = BiRNN(hidden_size, init=init,
                        activation=activation,
                        batch_norm=batch_norm,
                        reset_cells=True, return_sequence=True,
                        concat_out=concat_out, sum_out=sum_out)
            rnns.append(rnn)

        super(DeepBiRNN, self).__init__(layers=rnns)

    def __call__(self, in_obj, *args, **kwargs):

        # Also accept "time" axis as a recurrent axis
        if in_obj.axes.recurrent_axis() is None:
            in_obj = ng.map_roles(in_obj, {"time": "REC"})
            assert in_obj.axes.recurrent_axis() is not None, "in_obj has no recurrent or time axis"

        return super(DeepBiRNN, self).__call__(in_obj, *args, **kwargs)


class Deepspeech(Sequential):

    def __init__(self, nfilters, filter_width, str_w, nbands, depth, hidden_size,
                 batch_norm=False, batch_norm_affine=False, batch_norm_conv=False, to_ctc=True):

        self.to_ctc = to_ctc

        # Initializers
        gauss = GaussianInit(0.01)
        glorot = GlorotInit()

        # 1D Convolution layer
        padding = dict(pad_h=0, pad_w=filter_width // 2, pad_d=0)
        strides = dict(str_h=1, str_w=str_w, str_d=1)
        dilation = dict(dil_d=1, dil_h=1, dil_w=1)

        conv_layer = Convolution((nbands, filter_width, nfilters),
                                 gauss,
                                 bias_init=ConstantInit(0),
                                 padding=padding,
                                 strides=strides,
                                 dilation=dilation,
                                 activation=Rectlin(),
                                 batch_norm=batch_norm_conv)

        # Add BiRNN layers
        deep_birnn = DeepBiRNN(depth, hidden_size, glorot, Rectlinclip(), batch_norm=batch_norm)

        # Add a single affine layer
        fc = Affine(nout=hidden_size, weight_init=glorot,
                    activation=Rectlinclip(),
                    batch_norm=batch_norm_affine)

        # Add the final affine layer
        # Softmax output is computed within the CTC cost function, so no activation is needed here.
        if self.to_ctc is False:
            activation = Softmax()
        else:
            activation = None
        final = Affine(axes=ax.Y, weight_init=glorot, activation=activation)

        layers = [conv_layer,
                  deep_birnn,
                  fc,
                  final]

        super(Deepspeech, self).__init__(layers=layers)

    def __call__(self, *args, **kwargs):

        output = super(Deepspeech, self).__call__(*args, **kwargs)

        # prepare activations/gradients for warp-ctc
        # TODO: This should be handled in a graph pass
        if self.to_ctc is True:
            warp_axes = ng.make_axes([output.axes.recurrent_axis(),
                                      output.axes.batch_axis()])
            warp_axes = warp_axes | output.axes.feature_axes()
            output = ng.axes_with_order(output, warp_axes)
            output = ng.ContiguousOp(output)

        return output


def decode_outputs(probs, inds, decoder):
    """
    Decode from network probabilities and compute CER
    Arguments:
        probs: Tensor of character probabilities
        inds: List of character indices for ground truth
        decoder: instance of a Decoder

    Returns:
        Tuple of (ground truth transcript, decoded transcript, CER)
    """
    ground_truth = decoder.process_string(decoder.convert_to_string(inds),
                                          remove_repetitions=False)
    decoded_string = decoder.decode(probs)
    cer = decoder.cer(ground_truth, decoded_string) / float(len(ground_truth))

    return ground_truth, decoded_string, cer


if __name__ == "__main__":
    import logging
    from ngraph.frontends.neon.logging import ProgressBar, PBStreamHandler
    from ngraph.frontends.neon import NgraphArgparser

    parser = NgraphArgparser()
    structure = parser.add_argument_group("Network Structure")
    structure.add_argument('--nfilters', type=int,
                           help='Number of convolutional filters in the first layer',
                           default=256)
    structure.add_argument('--filter_width', type=int,
                           help='Width of 1D convolutional filters',
                           default=11)
    structure.add_argument('--str_w', type=int,
                           help='Stride in time',
                           default=3)
    structure.add_argument('--depth', type=int,
                           help='Number of RNN layers',
                           default=3)
    structure.add_argument('--hidden_size', type=int,
                           help='Number of hidden units in the RNN and affine layers',
                           default=256)
    structure.add_argument('--batch_norm', action='store_true')

    learning = parser.add_argument_group("Learning Hyperparameters")
    learning.add_argument('--lr', type=float,
                          help='learning rate',
                          default=2e-5)
    learning.add_argument('--momentum', type=float,
                          help='momentum',
                          default=0.99)
    learning.add_argument("--gradient_clip_norm", type=float,
                          help="maximum norm for gradients",
                          default=400)
    learning.add_argument("--nesterov", action="store_true",
                          help="Use Nesterov accelerated gradient")

    data_params = parser.add_argument_group("Data Parameters")
    data_params.add_argument('--max_length', type=float,
                             help="max duration for each audio sample",
                             default=7.5)
    data_params.add_argument("--manifest_train",
                             help="Path to training manifest file")
    data_params.add_argument("--manifest_val",
                             help="Path to validation manifest file")

    # TODO: Remove this once testing is further along
    parser.add_argument("--small", action="store_true",
                        help="Use a small version of the model with fake data")
    args = parser.parse_args()

    if args.small is True:
        args.nfilters = 20
        args.depth = 3
        args.hidden_size = 20
        args.max_length = .3
        args.fake = True

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(PBStreamHandler(level=logging.DEBUG))

    # Data parameters
    alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    nout = 29
    frame_stride = 0.01
    max_utt_len = ((int(args.max_length / frame_stride) - 1) // args.str_w) + 1
    # max_lbl_len = 409
    max_lbl_len = (max_utt_len - 1) // 2
    nbands = 13

    # Initialize the decoder
    decoder = ArgMaxDecoder(alphabet=alphabet,
                            blank_index=alphabet.index("_"),
                            space_index=alphabet.index(" "))

    # TODO: Add this back. Right now it's fine to default to smaller datasets
    # if (args.manifest_train is None) or (args.manifest_val is None):
    #     raise ValueError("For real data, you must provide 'manifest_train' and ""
    #                      ""'manifest_val'.")

    logger.debug("Getting librispeech data")
    train_manifest = Librispeech(manifest_file=args.manifest_train, path=args.data_dir,
                                 version="dev-clean").load_data()

    train_set = SpeechTranscriptionLoader(train_manifest,
                                          args.max_length,
                                          max_lbl_len,
                                          frame_stride=frame_stride,
                                          num_filters=nbands,
                                          alphabet=alphabet,
                                          batch_size=args.batch_size,
                                          num_batches=args.num_iterations,
                                          seed=args.rng_seed)

    inference = False
    if args.manifest_val is not None:
        eval_manifest = Librispeech(manifest_file=args.manifest_val, path=args.data_dir,
                                    version="test-clean").load_data()

        eval_set = SpeechTranscriptionLoader(train_manifest,
                                             args.max_length,
                                             max_lbl_len,
                                             frame_stride=frame_stride,
                                             num_filters=nbands,
                                             alphabet=alphabet,
                                             batch_size=args.batch_size,
                                             single_iteration=True,
                                             seed=args.rng_seed)
        inference = True

    inputs = train_set.make_placeholders()

    ax.Y.length = nout
    ax.Y.name = "characters"

    # Create the network
    logger.debug("Creating deepspeech2 model")
    ds2 = Deepspeech(args.nfilters, args.filter_width, args.str_w, nbands,
                     args.depth, args.hidden_size, batch_norm=args.batch_norm)
    output = ds2(inputs["audio"])

    # set up ctc loss
    loss = ng.ctc(output,
                  ng.flatten(inputs["char_map"]),
                  ng.flatten(inputs["audio_length"]),
                  ng.flatten(inputs["char_map_length"]))

    optimizer = GradientDescentMomentum(args.lr,
                                        momentum_coef=args.momentum,
                                        gradient_clip_norm=args.gradient_clip_norm,
                                        nesterov=args.nesterov)

    start = time.time()
    updates = optimizer(loss)
    stop = time.time()
    logger.debug("Optimizer graph creation took {} seconds".format(stop - start))
    mean_cost = ng.sequential([updates, ng.mean(loss, out_axes=())])

    # Create computation and initialize the transformer to allocate weights
    train_computation = ng.computation([mean_cost, output], "all")
    if inference is True:
        with Layer.inference_mode_on():
            eval_output = ds2(inputs["audio"])
        eval_computation = ng.computation(eval_output, "all")

    # Now bind the computations we are interested in
    with closing(ngt.make_transformer()) as transformer:
        train_function = transformer.add_computation(train_computation)
        if inference is True:
            eval_function = transformer.add_computation(eval_computation)

        start = time.time()
        transformer.initialize()
        stop = time.time()
        logger.debug("Initializing transformer took {} seconds".format(stop - start))

        progress_bar = ProgressBar(unit="batches", ncols=100, total=args.num_iterations)
        interval_cost = 0.0

        timing = list()
        for step, sample in progress_bar(enumerate(train_set)):
            feed_dict = {inputs[k]: sample[k] for k in inputs.keys()}
            start = time.time()
            [cost_val, net_val] = train_function(feed_dict=feed_dict)
            stop = time.time()
            timing.append(stop - start)
            cost_val = cost_val[()]

            progress_bar.set_description("Training {:0.4f}".format(cost_val))
            interval_cost += cost_val
            if (step + 1) % args.iter_interval == 0 and step > 0:
                logger.info("Interval {interval} Iteration {iteration} complete. "
                            "Avg Train Cost {cost:0.4f}".format(
                                interval=step // args.iter_interval,
                                iteration=step,
                                cost=interval_cost / args.iter_interval))
                interval_cost = 0.0
                if inference is True:
                    eval_total_cost = 0.0
                    logger.debug("Starting eval loop")
                    for eval_step, eval_sample in enumerate(eval_set):
                        feed_dict = {inputs[k]: eval_sample[k] for k in inputs.keys()}
                        p_out = eval_function(feed_dict=feed_dict)

                        flat_labels = eval_sample["char_map"].ravel()
                        start = batch_cost = 0
                        for ii in range(args.batch_size):
                            stop = start + eval_sample["char_map_length"].squeeze()[ii]
                            inds = flat_labels[start: stop]
                            ground_truth, decoded_string, cer = decode_outputs(p_out[:, ii, :].T,
                                                                               inds,
                                                                               decoder)
                            batch_cost += cer
                        eval_total_cost += batch_cost / args.batch_size

                    eval_set.reset()
                    eval_total_cost = eval_total_cost / (eval_step + 1)
                    logger.info("Validation Avg. CER: {}".format(eval_total_cost))
                else:
                    flat_labels = sample["char_map"].ravel()
                    inds = flat_labels[:sample["char_map_length"].squeeze()[0]]
                    ground_truth, decoded_string, _ = decode_outputs(net_val[:, 0, :].T,
                                                                     inds,
                                                                     decoder)
                logger.info("Example decodings")
                logger.info("\tGround truth: {}".format(ground_truth))
                logger.info("\tPredicted:    {}".format(decoded_string))
