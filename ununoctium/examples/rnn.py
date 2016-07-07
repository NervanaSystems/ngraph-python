from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
import geon.backends.graph.dataloaderbackend

import geon.backends.graph.defmod as be
import geon.backends.graph.graph as graph
import geon.backends.graph.pycudatransform as evaluation
import numpy as np


# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()


# noinspection PyPep8Naming
def L2(x):
    return be.dot(x, x)


class MyRnn(be.Model):

    def __init__(self, **kargs):
        super(MyRnn, self).__init__(**kargs)
        # g: graph node root namespace
        g = self.graph

        # Define the axes
        g.N = be.Axis()
        g.T = be.Axis(dependents=(g.N,))
        g.X = be.Axis()
        g.Y = be.Axis()
        g.H = be.Axis()

        # Define the inputs.
        # Length of the sequences in the batch
        g.t = be.Tensor(axes=(g.N), dtype=np.int32)
        g.T.length = g.t

        # There are a number of ways we could store variable-length sequences
        # 1) Pick a maximal length for T for all the data (Neon does this)
        # 2) Pick a maximal length for T per batch; batch size might vary
        # 3) Pack the data (not supported by NumPy arrays)

        # Input batch of sequences
        g.x = be.Tensor(axes=(g.X, g.T, g.N))
        # Output batch of sequences for training/evaluation
        g.y_ = be.Tensor(axes=(g.Y, g.T, g.N))

        # Recursive computation of the hidden state.
        # Axes for defining position roles
        h = be.RecursiveTensor(axes=(g.H, g.T, g.N))
        h[:, 0, :] = be.Variable(axes=(g.H,))
        HWh = be.Variable(axes=(g.H, g.H))
        HWx = be.Variable(axes=(g.X, g.H))
        Hb = be.Variable(axes=(g.H,))

        g.t = be.Var(g.T)
        h[:, g.t+1] = be.sig(be.dot(h[:, g.t], HWh)+be.dot(g.x[g.T], HWx)+Hb)

        YW = be.Variable(axes=(g.H, g.Y))
        Yb = be.Variable(axes=(g.Y))
        # This is the value we want for inference
        g.y = be.tanh(be.dot(h, YW)+Yb)

        # This is what we want for training, perhaps added to a parameter regularization
        e = g.y-g.y_
        g.error = be.dot(e, e)/e.size

        # L2 regularizer of parameters
        reg = None
        for param in be.find_all(types=be.Variable, used_by=g.error):
            l2 = L2(param)
            if reg is None:
                reg = l2
            else:
                reg = reg + l2
        g.loss = g.error + .01 * reg

    @be.with_graph_scope
    @be.with_environment
    def train(self, epochs):
        g  = self.graph

        # setup data provider
        imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                              repo_dir=args.data_dir, subset_pct=args.subset_pct)

        train = ImageLoader(set_name='train', shuffle=True, **imgset_options)

        g.N.length = train.bsz
        g.Y.length = train.nclasses

        learning_rate = be.input(axes=())

        params = g.loss.parameters()
        derivs = [be.deriv(g.loss, param) for param in params]

        updates = be.doall(all=[be.decrement(param, learning_rate * deriv) for param, deriv in zip(params, derivs)])

        enp = evaluation.NumPyEvaluator(results=[self.graph.value, g.error, updates]+derivs)

        # Some future data loader
        for epoch_no, batches in train:
            for batch_no, batch in batches:
                g.N.length = batch.batch_size
                g.t[:] = batch.sample_lengths
                g.x[:] = batch.x
                g.y_[:] = batch.y

                learning_rate[:] = be.ArrayWithAxes(.001, shape=(), axes=())

                vals = enp.evaluate()
                print(vals[g.error])

    @be.with_graph_scope
    @be.with_environment
    def dump(self):
        for _ in be.get_all_defs():
            print('{s} # {file_info}'.format(s=_, file_info=_.file_info))


MyRnn().dump()