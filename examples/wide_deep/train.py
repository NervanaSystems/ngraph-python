# Implementation of the Wide and Deep model tutorial from Google.
#
# https://www.tensorflow.org/tutorials/wide_and_deep
#
#
# In order to simplify the presentation we choose to use Adagrad directly over both
# #streams.
import ngraph as ng
from contextlib import closing
import ngraph.transformers as ngt
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import Adagrad, Rectlin
from model import WideDeepClassifier
from tqdm import tqdm
import numpy as np

import data

np.random.seed(123456)


def make_placeholders(batch_size, data):

    placeholders = {}

    placeholders['N'] = ng.make_axis(length=batch_size, name='N')

    # Concat the embedding with the continuous features.
    # Using same axes.
    placeholders['C'] = ng.make_axis(length=data.parameters['continuous_features']
                                     + data.parameters['indicators_features'], name="F")
    placeholders['WF'] = ng.make_axis(length=data.parameters['linear_features'], name="WF")
    placeholders['X_d'] = ng.placeholder(
        axes=[placeholders['C'], placeholders['N']],
        name="X_d")
    placeholders['X_w'] = ng.placeholder(
        axes=[placeholders['WF'], placeholders['N']],
        name="X_w")
    placeholders['Y'] = ng.placeholder(axes=[placeholders['N']], name="Y")

    embeddings_placeholders = []
    for lut in range(len(data.parameters['dimensions_embeddings'])):
        embedding_placeholder = ng.placeholder(ng.make_axes([placeholders['N']]), name="EMB")
        embeddings_placeholders.append(embedding_placeholder)

    placeholders['embeddings_placeholders'] = embeddings_placeholders

    return placeholders


parser = NgraphArgparser(description=__doc__)
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="Learning rate")
parser.add_argument("--epochs", type=int, default=41,
                    help="Number of epochs")
parser.add_argument("--deep_parameters", default='100,50', type=str,
                    help="Comma separated list of hidden neurons on the deep section of the model")
parser.set_defaults(batch_size=40)

args = parser.parse_args()

fc_layers_deep = [int(s) for s in args.deep_parameters.split(',')]

cs_loader = data.CensusDataset(args.batch_size)

inputs = make_placeholders(args.batch_size, cs_loader)

model = WideDeepClassifier(cs_loader.parameters['dimensions_embeddings'],
                           cs_loader.parameters['tokens_in_embeddings'],
                           fc_layers_deep, deep_activation_fn=Rectlin())

wide_deep = model(args.batch_size, inputs)

loss = ng.cross_entropy_binary(wide_deep, inputs['Y'])

optimizer = Adagrad(args.learning_rate)

# recall that optimizer does not generate output

batch_cost = ng.sequential([optimizer(loss), ng.sum(loss, out_axes=())])


def compute_accuracy(data):
    accuracy = 0.0
    total = 0.0

    for value in data.values():

        x_d = value[0]
        x_w = value[1]
        x_e = value[2]
        y = value[3]

        wide_features = x_w
        deep_features = x_d
        embedding_index_occupation = x_e[0]
        embedding_index_native_country = x_e[1]
        inference = eval_fun(wide_features, deep_features, embedding_index_occupation,
                             embedding_index_native_country, y)

        for label, i in zip(y, inference[0]):
            if int(label) == int(round(i)):
                accuracy += 1.0
            total += 1.0

    accuracy = accuracy / total * 100.0
    return accuracy


with closing(ngt.make_transformer()) as transformer:
    update_fun = transformer.computation([batch_cost], inputs['X_w'],
                                         inputs['X_d'],
                                         inputs['embeddings_placeholders'][0],
                                         inputs['embeddings_placeholders'][1],
                                         inputs['Y'])
    eval_fun = transformer.computation([wide_deep], inputs['X_w'],
                                       inputs['X_d'],
                                       inputs['embeddings_placeholders'][0],
                                       inputs['embeddings_placeholders'][1],
                                       inputs['Y'])

    print("Starting training ...")
    test_data = cs_loader.gendata(cs_loader.test_size / args.batch_size,
                                  cs_loader.deep_data_test,
                                  cs_loader.embeddings_index_test,
                                  cs_loader.wide_data_test,
                                  cs_loader.labels_test)

    train_data = cs_loader.gendata(cs_loader.train_size / args.batch_size, cs_loader.deep_data,
                                   cs_loader.embeddings_index,
                                   cs_loader.wide_data, cs_loader.labels)

    tpbar = tqdm(unit="epochs", total=args.epochs)
    # 2000 epochs the same as the Google Tutorial.
    for i in range(args.epochs):

        avg_loss = 0.0
        for value in train_data.values():
            xs_d = value[0]
            xs_w = value[1]
            xs_e = value[2]
            ys = value[3]
            loss_val = update_fun(xs_w, xs_d, xs_e[0], xs_e[1], ys)
            avg_loss += loss_val[0]
        avg_loss /= float(len(train_data))

        tpbar.update(1)
        accuracy = compute_accuracy(test_data)
        tpbar.set_description(
            "Avg loss %s , accuracy: %s " % (avg_loss, accuracy))
